"""
real_rag.py - 工业级混合检索增强生成系统 (Hybrid RAG)
============================================================================
架构设计：双路检索 + RRF融合 + 熔断门卫 + 流式生成
- 语义路：ChromaDB + embedding-3 (捕捉语义相似)
- 关键词路：BM25Okapi (捕捉精确匹配)
- 融合算法：RRF简化版去重重排
- 安全机制：相似度阈值熔断拦截
============================================================================
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path

# 第三方依赖检查
try:
    from pypdf import PdfReader
    import chromadb
    from chromadb.config import Settings
    from openai import OpenAI
    from rank_bm25 import BM25Okapi
    import jieba
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("pip install chromadb openai pypdf rank-bm25 jieba numpy")
    sys.exit(1)


class Config:
    """工业级配置中心 - 所有超参数集中管控"""
    
    # 安全层：API密钥从环境变量注入（零明文泄漏）
    ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
    if not ZHIPU_API_KEY:
        raise RuntimeError("❌ ZHIPU_API_KEY环境变量未设置！")
    
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    
    # 模型层
    EMBEDDING_MODEL = "embedding-3"      # 1024维高精度向量
    LLM_MODEL = "glm-4-flash"            # 免费快速推理
    
    # 切片层：600字块 + 120字重叠（20%重叠率，工业验证最佳实践）
    CHUNK_SIZE = 600
    OVERLAP_SIZE = 120
    
    # 检索层
    TOP_K_VECTOR = 2                     # 向量路检索数量
    TOP_K_BM25 = 2                       # 关键词路检索数量
    TOP_K_FINAL = 2                      # 融合后最终返回数量
    
    # 熔断层：余弦距离阈值（>0.7触发拦截，0.7=约30%相似度）
    SIMILARITY_THRESHOLD = 0.7
    
    # 存储层
    CHROMA_PATH = "./chroma_db"
    COLLECTION_NAME = "paper_hybrid_vectors"
    
    # 文件层
    PDF_FILENAME = "paper.pdf"


class PDFChunker:
    """PDF处理器：提取 + 智能切片（滑动窗口重叠）"""
    
    def __init__(self, pdf_path: str, chunk_size: int, overlap: int):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap
    
    def extract_text(self) -> str:
        """从PDF提取全文并清洗"""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"❌ PDF不存在: {self.pdf_path}")
        
        reader = PdfReader(self.pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                texts.append(' '.join(text.split()))
        
        full_text = '\n'.join(texts)
        print(f"📄 PDF提取完成: {len(full_text)}字符, {len(reader.pages)}页")
        return full_text
    
    def chunk(self, text: str) -> List[Dict]:
        """
        滑动窗口切片算法
        步长=stride, 相邻块共享overlap字符保证语义连续
        """
        chunks = []
        text_len = len(text)
        
        if text_len <= self.chunk_size:
            return [{"id": "chunk_000", "text": text, "start": 0, "end": text_len}]
        
        idx = 0
        start = 0
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append({
                "id": f"chunk_{idx:03d}",
                "text": text[start:end],
                "start": start,
                "end": end
            })
            start += self.stride
            idx += 1
            if end >= text_len:
                break
        
        print(f"🔪 切片完成: {len(chunks)}块, 平均{self.chunk_size}字, 重叠{self.overlap}字")
        return chunks


class HybridRetriever:
    """
    混合检索引擎：向量语义 + BM25关键词 = 双路召回
    核心创新：RRF简化融合 + 相似度熔断门卫
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = OpenAI(api_key=config.ZHIPU_API_KEY, base_url=config.BASE_URL)
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
        self.bm25_index = None       # BM25检索引擎实例
        self.bm25_chunks = []        # BM25原始文本块（用于索引构建）
        self.all_chunks = []         # 所有chunk元数据
    
    def embed(self, text: str) -> List[float]:
        """调用embedding-3生成1024维语义向量"""
        resp = self.openai_client.embeddings.create(
            model=self.config.EMBEDDING_MODEL, input=text
        )
        return resp.data[0].embedding
    
    def build_indices(self, chunks: List[Dict]):
        """
        构建双路索引：
        1. ChromaDB向量索引（语义路）
        2. BM25关键词索引（精准路）
        """
        self.all_chunks = chunks
        
        # === 删除旧集合，创建新集合 ===
        try:
            self.chroma_client.delete_collection(self.config.COLLECTION_NAME)
        except:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度空间
        )
        
        # === 向量路：批量生成embedding并存入ChromaDB ===
        print(f"🔄 向量化入库 {len(chunks)} 块...")
        for i, chunk in enumerate(chunks):
            emb = self.embed(chunk["text"])
            self.collection.add(
                ids=[chunk["id"]],
                embeddings=[emb],
                documents=[chunk["text"]],
                metadatas=[{"start": chunk["start"], "end": chunk["end"]}]
            )
            if (i+1) % 10 == 0:
                print(f"  向量化: {i+1}/{len(chunks)}")
        
        # === 关键词路：构建BM25索引 ===
        # 使用jieba分词处理中文文本
        print(f"🔨 构建BM25关键词索引...")
        tokenized_chunks = []
        self.bm25_chunks = [chunk["text"] for chunk in chunks]
        
        for text in self.bm25_chunks:
            # jieba精确模式分词，过滤单字（提升检索精度）
            tokens = [w for w in jieba.lcut(text) if len(w) > 1]
            tokenized_chunks.append(tokens)
        
        self.bm25_index = BM25Okapi(tokenized_chunks)
        print(f"✅ 双路索引构建完成: 向量{len(chunks)}条 + BM25{len(tokenized_chunks)}条")
    
    def vector_retrieve(self, query: str) -> Tuple[List[Dict], bool]:
        """
        语义路检索 + 熔断门卫
        返回: (检索结果列表, 是否触发熔断)
        """
        query_emb = self.embed(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=self.config.TOP_K_VECTOR,
            include=["documents", "metadatas", "distances"]
        )
        
        # === 熔断门卫：检查最大距离（余弦距离越小越相关） ===
        distances = results["distances"][0]
        max_distance = max(distances) if distances else 1.0
        
        # 如果连最相关的块距离都>0.7，说明语义完全不匹配
        triggered = max_distance > self.config.SIMILARITY_THRESHOLD
        
        # 构造标准返回格式
        vector_results = []
        if not triggered:
            for i in range(len(results["documents"][0])):
                vector_results.append({
                    "text": results["documents"][0][i],
                    "source": "vector",
                    "distance": distances[i],
                    "score": 1.0 - distances[i],  # 转换为相似度分数
                    "metadata": results["metadatas"][0][i]
                })
        
        return vector_results, triggered
    
    def bm25_retrieve(self, query: str) -> List[Dict]:
        """
        关键词路检索：BM25精确匹配
        """
        if not self.bm25_index:
            return []
        
        # 分词查询
        query_tokens = [w for w in jieba.lcut(query) if len(w) > 1]
        scores = self.bm25_index.get_scores(query_tokens)
        
        # 获取Top-K索引
        top_indices = np.argsort(scores)[::-1][:self.config.TOP_K_BM25]
        
        bm25_results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤0分结果
                bm25_results.append({
                    "text": self.bm25_chunks[idx],
                    "source": "bm25",
                    "score": float(scores[idx]),
                    "metadata": {"start": self.all_chunks[idx]["start"], 
                                "end": self.all_chunks[idx]["end"]}
                })
        
        return bm25_results
    
    def rrf_fusion(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """
        RRF简化融合算法 (Reciprocal Rank Fusion)
        原理：对不同来源的结果按排名位置加权融合，去重后重新排序
        
        公式：RRF_score = Σ 1/(k + rank_i)
        其中k=60（平滑参数），rank从1开始
        """
        k = 60  # RRF平滑常数（行业标准值）
        fused_scores = {}
        fused_data = {}
        
        # === 向量路排名加权 ===
        for rank, item in enumerate(vector_results, 1):
            text = item["text"]
            rrf_score = 1.0 / (k + rank)
            
            if text not in fused_scores:
                fused_scores[text] = rrf_score
                fused_data[text] = item
            else:
                fused_scores[text] += rrf_score
        
        # === BM25路排名加权 ===
        for rank, item in enumerate(bm25_results, 1):
            text = item["text"]
            rrf_score = 1.0 / (k + rank)
            
            if text not in fused_scores:
                fused_scores[text] = rrf_score
                fused_data[text] = item
            else:
                fused_scores[text] += rrf_score  # 双路命中加权
        
        # === 按融合分数降序排列，取Top-K ===
        sorted_texts = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = []
        
        for text, score in sorted_texts[:self.config.TOP_K_FINAL]:
            item = fused_data[text]
            item["rrf_score"] = score  # 添加融合分数
            final_results.append(item)
        
        return final_results
    
    def hybrid_retrieve(self, query: str) -> List[Dict]:
        """
        混合检索主流程：双路召回 + 熔断检查 + RRF融合
        """
        # === 1. 语义路检索（内置熔断） ===
        vector_results, is_blocked = self.vector_retrieve(query)
        
        if is_blocked:
            print("\n" + "="*60)
            print("⚠️  警告：您输入的内容与本论文无任何语义关联，已被向量网关安全拦截！")
            print(f"   触发阈值: 余弦距离 > {self.config.SIMILARITY_THRESHOLD}")
            print("="*60)
            return []
        
        # === 2. 关键词路检索 ===
        bm25_results = self.bm25_retrieve(query)
        
        # === 3. RRF融合去重重排 ===
        final_results = self.rrf_fusion(vector_results, bm25_results)
        
        # === 4. 输出检索诊断信息 ===
        print(f"\n🔍 混合检索诊断:")
        print(f"   向量路召回: {len(vector_results)}条")
        print(f"   BM25路召回: {len(bm25_results)}条")
        print(f"   RRF融合后: {len(final_results)}条")
        for i, item in enumerate(final_results, 1):
            print(f"   结果{i}: 来源={item['source']}, RRF分数={item['rrf_score']:.4f}, "
                  f"位置={item['metadata']['start']}-{item['metadata']['end']}")
        
        return final_results


class LLMStreamer:
    """流式大模型生成器：防御性Prompt + 流式输出"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.ZHIPU_API_KEY, base_url=config.BASE_URL)
    
    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        构建防御性Prompt并流式生成
        Prompt设计：系统角色约束 + 参考资料注入 + 引用要求
        """
        # 拼接上下文
        context = "\n\n---\n\n".join([
            f"[参考资料{i+1} 来源:{chunk['source']} 位置:{chunk['metadata']['start']}-{chunk['metadata']['end']}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        system_prompt = """你是严格的学术论文分析专家。规则：
1. 仅基于提供的参考资料回答，禁止引入外部知识
2. 参考资料不足时明确说明"依据现有资料无法确定"
3. 回答需标注引用来源（如[参考资料1]）
4. 保持学术严谨，不推测不编造"""
        
        user_prompt = f"""论文参考资料：
{context}

用户问题：{query}

请严格基于以上资料回答："""
        
        print(f"\n🤖 GLM-4-Flash流式生成:\n{'='*60}")
        
        full_response = ""
        try:
            stream = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                temperature=0.2,    # 低温度确保准确性
                max_tokens=1500,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print(f"\n{'='*60}")
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
        
        return full_response


class HybridRAGSystem:
    """工业级混合RAG系统主控制器"""
    
    def __init__(self):
        self.config = Config()
        self.chunker = PDFChunker(
            self.config.PDF_FILENAME, 
            self.config.CHUNK_SIZE, 
            self.config.OVERLAP_SIZE
        )
        self.retriever = HybridRetriever(self.config)
        self.llm = LLMStreamer(self.config)
    
    def initialize(self):
        """系统初始化：PDF处理 + 双路索引构建"""
        print("\n" + "🚀"*30)
        print("工业级混合检索RAG系统启动")
        print("🚀"*30 + "\n")
        
        # 检查是否已有数据库
        try:
            existing = self.retriever.chroma_client.get_collection(
                self.config.COLLECTION_NAME
            )
            if existing.count() > 0:
                print(f"📚 发现已有数据库({existing.count()}条)")
                choice = input("重建? (y/n): ").strip().lower()
                if choice != 'y':
                    self.retriever.collection = existing
                    # 重建BM25索引
                    docs = existing.get()["documents"]
                    tokenized = [[w for w in jieba.lcut(doc) if len(w)>1] for doc in docs]
                    self.retriever.bm25_index = BM25Okapi(tokenized)
                    self.retriever.bm25_chunks = docs
                    self.retriever.all_chunks = [
                        {"id": f"chunk_{i:03d}", "text": doc, "start": i*600, "end": (i+1)*600}
                        for i, doc in enumerate(docs)
                    ]
                    print("✅ 使用现有数据库")
                    return
        except:
            pass
        
        # 完整处理流程
        text = self.chunker.extract_text()
        chunks = self.chunker.chunk(text)
        self.retriever.build_indices(chunks)
        print("\n✅ 系统初始化完成，进入问答模式")
    
    def query(self, question: str):
        """执行一次完整的混合检索增强生成"""
        # 混合检索（含熔断）
        relevant = self.retriever.hybrid_retrieve(question)
        
        # 熔断触发时不生成
        if not relevant:
            return
        
        # 流式生成
        self.llm.generate(question, relevant)
    
    def run(self):
        """交互式主循环"""
        self.initialize()
        
        print("\n" + "💬"*30)
        print("混合检索问答模式 (quit退出)")
        print("💬"*30)
        
        while True:
            try:
                q = input("\n🔍 问题: ").strip()
                if q.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见")
                    break
                if not q:
                    continue
                self.query(q)
            except KeyboardInterrupt:
                print("\n👋 中断退出")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


if __name__ == "__main__":
    # 启动工业级混合RAG系统
    system = HybridRAGSystem()
    system.run()