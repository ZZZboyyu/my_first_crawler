"""
rerank_rag.py - 工业级两阶段RAG系统 (纯本地重排序版)
================================================================
核心方案：使用本地Cross-Encoder模型进行重排序
- 阶段一：智谱embedding-3向量粗筛 (Top-10)
- 阶段二：本地sentence-transformers Cross-Encoder精排 (Top-2)
- 完全避免API依赖，100%可运行

安装依赖：
pip install chromadb openai pypdf sentence-transformers numpy torch
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict

# ===========================================================================
# 依赖检查
# ===========================================================================
try:
    from pypdf import PdfReader
    import chromadb
    from chromadb.config import Settings
    from openai import OpenAI
    from sentence_transformers import CrossEncoder
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install chromadb openai pypdf sentence-transformers numpy torch")
    sys.exit(1)


class Config:
    """全局配置"""
    
    # API密钥（环境变量读取）
    ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
    if not ZHIPU_API_KEY:
        raise RuntimeError("❌ 请设置 ZHIPU_API_KEY 环境变量")
    
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    EMBEDDING_MODEL = "embedding-3"
    LLM_MODEL = "glm-4-flash"
    
    # 切片配置
    CHUNK_SIZE = 600
    OVERLAP_SIZE = 150
    
    # 检索配置
    TOP_K_COARSE = 10  # 粗筛数量
    TOP_K_FINE = 2     # 精排数量
    
    # 本地Cross-Encoder模型（自动下载，约1.2GB）
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    
    # 数据库配置
    CHROMA_PATH = "./chroma_db_rerank"
    COLLECTION_NAME = "paper_rerank_vectors"
    
    # PDF文件
    PDF_FILENAME = "paper.pdf"


class OverlapChunker:
    """
    带重叠的文本切片器
    
    为什么需要Overlap？
    ===================
    假设论文原文："实验结果表明，Transformer模型的注意力机制
    在处理长序列时表现优异，特别是通过多头注意力可以捕获不同
    子空间的特征表示。"
    
    无重叠切片（600字/块）：
    块1: "...实验结果表明，Transformer模型的注意力机制在处理"
    块2: "长序列时表现优异，特别是通过多头注意力..."
    
    ❌ "在处理"和"长序列"被切断！检索时两块都拿不到完整语义。
    
    有重叠切片（600字/块，150字重叠）：
    块1: "...实验结果表明，Transformer模型的注意力机制在处理长序列时表现优异，特别是..."
    块2: "...注意力机制在处理长序列时表现优异，特别是通过多头注意力可以捕获..."
    
    ✅ "注意力机制在处理长序列时表现优异"在两块中都完整保留！
    """
    
    def __init__(self, chunk_size=600, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap
    
    def extract_pdf(self, pdf_path):
        """提取PDF文本"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ 找不到 {pdf_path}")
        
        print(f"📖 读取PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = []
        for i, page in enumerate(reader.pages, 1):
            content = page.extract_text()
            if content.strip():
                text.append(' '.join(content.split()))
            if i % 5 == 0:
                print(f"   进度: {i}/{len(reader.pages)}")
        
        full_text = '\n'.join(text)
        print(f"✅ 提取完成: {len(full_text)} 字符\n")
        return full_text
    
    def chunk(self, text):
        """滑动窗口切片"""
        print(f"🔪 开始切片 (块={self.chunk_size}字, 重叠={self.overlap}字)")
        
        chunks = []
        text_len = len(text)
        
        # 短文本直接返回
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
        
        print(f"✅ 生成 {len(chunks)} 个文本块\n")
        return chunks


class VectorStore:
    """向量数据库 - 阶段一粗筛"""
    
    def __init__(self, config):
        self.config = config
        self.openai_client = None
        self.chroma_client = None
        self.collection = None
    
    def init(self):
        """初始化连接"""
        self.openai_client = OpenAI(
            api_key=self.config.ZHIPU_API_KEY,
            base_url=self.config.BASE_URL
        )
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        print(f"💾 ChromaDB已连接\n")
    
    def embed(self, text):
        """向量化"""
        resp = self.openai_client.embeddings.create(
            model=self.config.EMBEDDING_MODEL,
            input=text
        )
        return resp.data[0].embedding
    
    def build(self, chunks):
        """构建索引"""
        try:
            self.chroma_client.delete_collection(self.config.COLLECTION_NAME)
        except:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"🔄 向量化入库 {len(chunks)} 块...")
        for i, chunk in enumerate(chunks, 1):
            emb = self.embed(chunk["text"])
            self.collection.add(
                ids=[chunk["id"]],
                embeddings=[emb],
                documents=[chunk["text"]],
                metadatas=[{"start": chunk["start"], "end": chunk["end"]}]
            )
            if i % 5 == 0:
                print(f"   进度: {i}/{len(chunks)}")
        print(f"✅ 索引构建完成\n")
    
    def retrieve(self, query, top_k=10):
        """粗筛检索"""
        print(f"🔍 阶段一：向量粗筛 (Top-{top_k})")
        query_emb = self.embed(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        print(f"✅ 召回 {len(chunks)} 个候选块\n")
        return chunks


class LocalReranker:
    """
    本地Cross-Encoder重排序器
    
    为什么Cross-Encoder比向量检索更准？
    =====================================
    向量检索(Bi-Encoder)：
    Query → [Encoder] → 向量A
    Doc   → [Encoder] → 向量B
    然后计算 cos(A,B)
    ❌ Query和Doc从未真正"见面"，只是比较它们的"照片"
    
    Cross-Encoder：
    [Query + Doc] → [Transformer] → 相似度分数
    ✅ Query和Doc一起输入模型，通过注意力机制深度交互
    就像让它们"面对面交流"，真正理解彼此的关系
    
    举例：
    Query: "苹果的股价怎么样？"
    Doc1: "苹果公司今日股价上涨3%"  → Cross-Encoder: 0.95分 ✅
    Doc2: "苹果富含维生素C"       → Cross-Encoder: 0.12分 ❌
    
    向量检索可能给Doc2打0.7分（都提到"苹果"）
    Cross-Encoder能精确区分"苹果公司"和"水果苹果"
    """
    
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """初始化本地Cross-Encoder模型"""
        print(f"📥 加载本地重排序模型: {model_name}")
        print(f"   (首次运行会自动下载，约1.2GB，请耐心等待...)")
        
        try:
            self.model = CrossEncoder(model_name)
            print(f"✅ 模型加载成功\n")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print(f"   降级策略：使用简单文本匹配排序")
            self.model = None
    
    def rerank(self, query, candidates, top_k=2):
        """
        精排重排序
        
        Cross-Encoder工作原理（大白话）：
        1. 把问题和每个候选文档拼在一起
        2. 一起喂给BERT模型
        3. BERT通过注意力机制让问题中的每个词和文档中的每个词"对话"
        4. 输出一个0-1的相关性分数
        5. 按分数排序，选最高的Top-K
        """
        print(f"🎯 阶段二：Cross-Encoder精排 (Top-{top_k})")
        
        if self.model is None:
            # 降级：按向量距离排序
            print("   使用向量距离降级排序")
            candidates.sort(key=lambda x: x["distance"])
            results = candidates[:top_k]
        else:
            # 构建[query, doc]对
            pairs = [[query, doc["text"][:500]] for doc in candidates]
            
            # Cross-Encoder打分
            scores = self.model.predict(pairs)
            
            # 添加分数并排序
            for i, doc in enumerate(candidates):
                doc["rerank_score"] = float(scores[i])
            
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = candidates[:top_k]
        
        # 输出结果
        for i, res in enumerate(results, 1):
            score = res.get("rerank_score", 1 - res["distance"])
            print(f"   {i}. 相关性={score:.4f} | "
                  f"位置={res['metadata']['start']}-{res['metadata']['end']}")
        print()
        
        return results


class LLMGenerator:
    """大模型流式生成器"""
    
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            api_key=config.ZHIPU_API_KEY,
            base_url=config.BASE_URL
        )
    
    def generate(self, query, contexts):
        """流式生成回答"""
        context_text = "\n\n---\n\n".join([
            f"[参考{i+1}] (得分:{ctx.get('rerank_score', 0):.4f})\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        system_prompt = """你是严谨的学术论文分析专家。严格规则：
1. 只基于提供的参考资料回答，不引入外部知识
2. 资料不足时说"基于现有资料无法确定"
3. 标注引用来源如"[参考1]"
4. 直接引用原文数据，不编造不推测"""
        
        user_prompt = f"""论文参考资料：
{context_text}

问题：{query}

请严格基于资料回答："""
        
        print(f"🤖 GLM-4-Flash 流式生成:\n{'='*50}")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                temperature=0.1,
                max_tokens=2000
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            
            print(f"\n{'='*50}\n")
        except Exception as e:
            print(f"\n❌ 生成失败: {e}\n")


class TwoStageRAG:
    """两阶段RAG系统主控"""
    
    def __init__(self):
        self.config = Config()
        self.chunker = OverlapChunker(self.config.CHUNK_SIZE, self.config.OVERLAP_SIZE)
        self.vector_store = VectorStore(self.config)
        self.reranker = LocalReranker(self.config.RERANKER_MODEL)
        self.llm = LLMGenerator(self.config)
    
    def setup(self):
        """初始化系统"""
        print("\n" + "="*50)
        print("两阶段RAG系统 (本地Cross-Encoder版)")
        print("="*50 + "\n")
        
        self.vector_store.init()
        
        # 检查已有数据库
        try:
            existing = self.vector_store.chroma_client.get_collection(
                self.config.COLLECTION_NAME
            )
            if existing.count() > 0:
                print(f"📚 已有数据库 ({existing.count()}条)")
                if input("重建? (y/n): ").strip() != 'y':
                    self.vector_store.collection = existing
                    print("✅ 使用现有数据库\n")
                    return
        except:
            pass
        
        # 新建数据库
        text = self.chunker.extract_pdf(self.config.PDF_FILENAME)
        chunks = self.chunker.chunk(text)
        self.vector_store.build(chunks)
        print("✅ 初始化完成\n")
    
    def ask(self, question):
        """处理一个问题"""
        print(f"\n📝 问题: {question}\n")
        
        # 阶段一：粗筛
        t1 = time.time()
        coarse = self.vector_store.retrieve(question, self.config.TOP_K_COARSE)
        
        # 阶段二：精排
        fine = self.reranker.rerank(question, coarse, self.config.TOP_K_FINE)
        
        t2 = time.time()
        print(f"⚡ 检索耗时: {(t2-t1)*1000:.0f}ms\n")
        
        # 生成
        self.llm.generate(question, fine)
    
    def run(self):
        """交互循环"""
        self.setup()
        
        print("💬 输入问题开始 (quit退出)\n")
        while True:
            try:
                q = input("🔍 问题: ").strip()
                if q.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见")
                    break
                if q:
                    self.ask(q)
            except KeyboardInterrupt:
                print("\n👋 退出")
                break


if __name__ == "__main__":
    rag = TwoStageRAG()
    rag.run()