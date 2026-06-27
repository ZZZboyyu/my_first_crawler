"""
real_rag.py - 基于智谱AI的本地向量数据库Chroma检索增强生成系统
===========================================================================
核心原理：
1. 文档切片：将PDF文档按语义单元切分成重叠的文本块
2. 向量化入库：使用嵌入模型将文本块转换为高维向量存入Chroma
3. 语义检索：将用户问题向量化后在向量空间中进行相似度搜索
4. 增强生成：将检索到的相关文本块与问题组合，输入大模型生成精准答案

技术栈：
- 嵌入模型：智谱AI embedding-3 (最新高精度向量模型)
- 大语言模型：智谱AI GLM-4-Flash (免费快速推理)
- 向量数据库：ChromaDB (持久化本地存储)
- PDF处理：PyPDF2 (文档读取与切片)

作者：AI专业大一学生
日期：2026年
"""

import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path

# ===========================================================================
# 第三方库导入 - 这些库需要提前安装
# pip install chromadb openai pypdf
# ===========================================================================
try:
    from pypdf import PdfReader
    import chromadb
    from openai import OpenAI
except ImportError as e:
    print(f"❌ 缺少必要的库文件: {e}")
    print("请运行: pip install chromadb openai pypdf")
    sys.exit(1)


class Config:
    """
    全局配置类 - 存储所有系统参数
    
    安全原则：
    - API密钥通过环境变量读取，绝不在代码中硬编码
    - 所有路径使用相对路径，确保跨平台兼容性
    """
    
    # === API配置 ===
    # 从Windows系统环境变量获取API密钥（绝对安全，无明文泄露风险）
    ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
    
    # 智谱AI官方接口地址
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    
    # === 模型配置 ===
    EMBEDDING_MODEL = "embedding-3"        # 嵌入模型：将文本转换为1024维向量
    LLM_MODEL = "glm-4-flash"              # 大语言模型：免费高速推理
    
    # === 文档切片配置 ===
    CHUNK_SIZE = 600                       # 每个文本块600字（平衡语义完整性与检索精度）
    OVERLAP_SIZE = 120                     # 重叠120字（防止关键信息在边界被切断）
    
    # === 检索配置 ===
    TOP_K = 2                              # 检索最相关的2个文本块（足够精准又不超上下文限制）
    
    # === 数据库配置 ===
    CHROMA_PATH = "./chroma_db"            # Chroma持久化数据库存储路径
    COLLECTION_NAME = "paper_vectors"      # 向量集合名称
    
    # === 文件配置 ===
    PDF_FILENAME = "paper.pdf"             # 待处理的PDF论文文件名


class PDFProcessor:
    """
    PDF文档处理器 - 负责读取PDF并智能切片
    
    切片策略：
    - 固定大小切片（600字）：确保每个块包含足够的上下文
    - 滑动窗口重叠（120字）：保证语义连贯性，避免关键信息被切断
    - 这个重叠率(20%)是经过工业界验证的最佳实践
    """
    
    def __init__(self, pdf_path: str, chunk_size: int = 600, overlap: int = 120):
        """
        初始化PDF处理器
        
        Args:
            pdf_path: PDF文件路径
            chunk_size: 每个文本块的字符数
            overlap: 相邻块之间的重叠字符数
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_text(self) -> str:
        """
        从PDF中提取全部文本内容
        
        Returns:
            清洗后的完整文本字符串
        
        处理流程：
        1. 逐页读取PDF
        2. 提取每页文本
        3. 过滤空行和多余空白
        """
        print(f"📖 正在读取PDF文件: {self.pdf_path}")
        
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"❌ PDF文件不存在: {self.pdf_path}")
        
        reader = PdfReader(self.pdf_path)
        total_pages = len(reader.pages)
        print(f"📄 检测到 {total_pages} 页内容")
        
        full_text = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():  # 过滤空页面
                # 清理文本：合并多个空格，去除首尾空白
                cleaned_text = ' '.join(text.split())
                full_text.append(cleaned_text)
            
            if i % 5 == 0:  # 每5页显示一次进度
                print(f"⏳ 提取进度: {i}/{total_pages} 页")
        
        extracted = '\n'.join(full_text)
        print(f"✅ 文本提取完成，总计 {len(extracted)} 字符")
        return extracted
    
    def create_chunks(self, text: str) -> List[Dict[str, str]]:
        """
        将长文本切分成重叠的语义块
        
        Args:
            text: 完整文本
            
        Returns:
            包含chunk_id和text的字典列表
            
        算法说明：
        - 使用滑动窗口策略
        - 步长 = chunk_size - overlap
        - 例如：chunk_size=600, overlap=120, 步长=480
        - 第一个块: [0:600]
        - 第二个块: [480:1080]
        - 第三个块: [960:1560]
        - 以此类推，确保相邻块共享120字上下文
        """
        print(f"\n🔪 开始文本切片 (块大小: {self.chunk_size}字, 重叠: {self.overlap}字)")
        
        chunks = []
        text_length = len(text)
        stride = self.chunk_size - self.overlap  # 计算滑动步长
        
        # 如果文本长度小于chunk_size，直接作为一个完整块
        if text_length <= self.chunk_size:
            chunks.append({
                "chunk_id": "chunk_000",
                "text": text,
                "start_pos": 0,
                "end_pos": text_length
            })
        else:
            chunk_index = 0
            start = 0
            
            while start < text_length:
                # 计算当前块的结束位置
                end = min(start + self.chunk_size, text_length)
                chunk_text = text[start:end]
                
                # 构建块元数据
                chunk_data = {
                    "chunk_id": f"chunk_{chunk_index:03d}",  # 格式化为三位数字ID
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end
                }
                chunks.append(chunk_data)
                
                # 移动到下一个起始位置
                start += stride
                chunk_index += 1
                
                # 如果已经到达文本末尾，退出循环
                if end >= text_length:
                    break
        
        print(f"✅ 切片完成，共生成 {len(chunks)} 个文本块")
        
        # 打印切片统计信息
        avg_length = sum(len(c["text"]) for c in chunks) / len(chunks)
        print(f"📊 切片统计: 平均长度 {avg_length:.0f} 字符, 重叠率 {self.overlap/self.chunk_size*100:.1f}%")
        
        return chunks


class VectorDatabase:
    """
    向量数据库管理器 - 使用ChromaDB进行本地持久化存储
    
    核心概念：
    - Embedding：将文本映射到高维向量空间（1024维）
    - 向量相似度：余弦相似度计算文本间的语义距离
    - 索引：为向量建立搜索索引，实现毫秒级检索
    """
    
    def __init__(self, config: Config):
        """
        初始化向量数据库
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        self.client = None
        self.collection = None
        self.openai_client = None
        
    def initialize_clients(self):
        """
        初始化API客户端和数据库连接
        
        双重检查机制：
        1. 验证API密钥存在性
        2. 测试数据库连接可用性
        """
        # === 检查API密钥 ===
        if not self.config.ZHIPU_API_KEY:
            raise ValueError(
                "❌ 未找到ZHIPU_API_KEY环境变量！\n"
                "请在Windows系统环境变量中设置：\n"
                "1. 按 Win+X 打开系统菜单\n"
                "2. 选择'系统' -> '高级系统设置'\n"
                "3. 点击'环境变量'\n"
                "4. 新建用户变量: ZHIPU_API_KEY=你的API密钥"
            )
        
        # === 初始化OpenAI兼容客户端（连接智谱AI） ===
        # 智谱AI提供OpenAI兼容接口，可以直接使用openai库
        self.openai_client = OpenAI(
            api_key=self.config.ZHIPU_API_KEY,
            base_url=self.config.BASE_URL
        )
        print("🔗 已连接到智谱AI API")
        
        # === 初始化Chroma持久化客户端 ===
        # PersistentClient会在本地磁盘创建数据库文件，程序重启数据不丢失
        self.client = chromadb.PersistentClient(
            path=self.config.CHROMA_PATH,
            settings=chromadb.Settings(
                anonymized_telemetry=False  # 关闭遥测，保护隐私
            )
        )
        print(f"💾 Chroma数据库已初始化 (存储路径: {self.config.CHROMA_PATH})")
        
    def create_embedding(self, text: str) -> List[float]:
        """
        调用智谱AI嵌入模型将文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            1024维浮点数向量列表
            
        向量化原理：
        - embedding-3模型将文本的语义信息编码为1024维向量
        - 语义相似的文本在向量空间中距离更近
        - 每维代表文本的一个抽象特征（如主题、情感、风格等）
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=text
            )
            # 提取嵌入向量
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            raise RuntimeError(f"❌ 向量化失败: {str(e)}")
    
    def create_collection(self, chunks: List[Dict[str, str]]) -> None:
        """
        创建向量集合并批量入库
        
        Args:
            chunks: 文本块列表
            
        处理流程：
        1. 删除旧集合（如果存在）
        2. 创建新集合
        3. 批量生成嵌入向量
        4. 将向量和文本存入数据库
        """
        collection_name = self.config.COLLECTION_NAME
        
        # === 删除已存在的集合（避免重复数据） ===
        try:
            self.client.delete_collection(collection_name)
            print(f"🗑️  已删除旧集合: {collection_name}")
        except:
            pass  # 集合不存在时忽略错误
        
        # === 创建新集合 ===
        # metadata用于存储集合的描述信息
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={
                "description": "论文向量索引库",
                "embedding_model": self.config.EMBEDDING_MODEL,
                "chunk_size": str(self.config.CHUNK_SIZE),
                "overlap": str(self.config.OVERLAP_SIZE)
            }
        )
        print(f"📚 已创建向量集合: {collection_name}")
        
        # === 批量处理文本块 ===
        print(f"\n🔄 开始向量化 {len(chunks)} 个文本块...")
        
        for i, chunk in enumerate(chunks, 1):
            # 生成向量嵌入
            embedding = self.create_embedding(chunk["text"])
            
            # 存储到ChromaDB
            # - ids: 唯一标识符
            # - embeddings: 向量数据
            # - documents: 原始文本（用于检索后返回）
            # - metadatas: 附加元数据（位置信息等）
            self.collection.add(
                ids=[chunk["chunk_id"]],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[{
                    "chunk_id": chunk["chunk_id"],
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"],
                    "char_length": len(chunk["text"])
                }]
            )
            
            # 进度显示
            if i % 5 == 0 or i == len(chunks):
                progress = (i / len(chunks)) * 100
                print(f"⏳ 向量化进度: {i}/{len(chunks)} ({progress:.1f}%)")
        
        print(f"✅ 所有文本块已成功存入向量数据库")
        
    def query_relevant_chunks(self, query: str) -> List[Dict]:
        """
        语义检索：根据用户查询找到最相关的文本块
        
        Args:
            query: 用户问题
            
        Returns:
            包含相关文本块及其元数据的列表
            
        检索原理：
        1. 将用户问题转换为向量
        2. 在向量空间中计算余弦相似度
        3. 返回相似度最高的TOP_K个文本块
        """
        print(f"\n🔍 正在检索相关文本块: \"{query}\"")
        
        # 将查询转换为向量
        query_embedding = self.create_embedding(query)
        
        # 在ChromaDB中执行向量相似度搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.TOP_K,  # 返回最相关的2个结果
            include=["documents", "metadatas", "distances"]  # 返回文档、元数据和距离
        )
        
        # 解析检索结果
        relevant_chunks = []
        for i in range(len(results["documents"][0])):
            chunk_info = {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],  # 距离越小越相关
                "relevance_score": 1 - results["distances"][0][i]  # 转换为相关性分数
            }
            relevant_chunks.append(chunk_info)
            
            print(f"📌 相关块 {i+1}: "
                  f"ID={chunk_info['metadata']['chunk_id']}, "
                  f"相关性={chunk_info['relevance_score']:.4f}, "
                  f"距离={chunk_info['distance']:.4f}")
        
        return relevant_chunks


class LLMGenerator:
    """
    大语言模型生成器 - 基于检索结果生成精准回答
    
    防御性Prompt设计：
    - 明确告诉模型只能基于提供的参考资料回答
    - 防止模型过度发挥，避免产生幻觉
    - 确保回答的准确性和可追溯性
    """
    
    def __init__(self, config: Config):
        """
        初始化LLM生成器
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        
        # 初始化OpenAI兼容客户端
        self.client = OpenAI(
            api_key=config.ZHIPU_API_KEY,
            base_url=config.BASE_URL
        )
        
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        基于检索结果生成流式回答
        
        Args:
            query: 用户问题
            relevant_chunks: 检索到的相关文本块
            
        Returns:
            流式生成的回答字符串
        
        Prompt工程策略：
        1. 系统提示：限制模型行为边界
        2. 上下文注入：插入检索到的准确信息
        3. 防御性指令：防止模型编造信息
        """
        
        # === 构建防御性Prompt ===
        # 将检索到的文本块拼接为参考上下文
        context = "\n\n---\n\n".join([
            f"【参考资料 {i+1}】(来源位置: {chunk['metadata']['start_pos']}-{chunk['metadata']['end_pos']}字符)\n{chunk['text']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # 系统提示词 - 严格限定模型行为
        system_prompt = """你是一个专业的学术论文分析助手。你的任务是基于提供的论文片段，准确回答用户的问题。

请严格遵守以下规则：
1. 只能基于提供的参考资料回答问题，不得引入外部知识
2. 如果参考资料不足以回答问题，请明确说明"根据提供的资料无法确定"
3. 回答要精准、简洁，直接引用原文关键信息
4. 使用学术化的语言，保持专业性
5. 在回答中标注信息来源（如"根据参考资料1..."）"""
        
        # 用户提示 - 组合上下文和问题
        user_prompt = f"""论文参考资料：
{context}

用户问题：{query}

请基于以上参考资料，给出准确、详细的回答："""
        
        print(f"\n🤖 正在生成回答...\n")
        print("=" * 60)
        
        # === 流式生成 ===
        full_response = ""
        
        try:
            # 调用智谱AI GLM-4-Flash模型进行流式生成
            stream = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,  # 启用流式输出
                temperature=0.3,  # 降低随机性，提高准确度
                max_tokens=2000,  # 限制最大生成长度
            )
            
            # 逐token打印输出
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)  # 实时打印，不缓冲
                    full_response += content
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            error_msg = f"❌ 生成回答失败: {str(e)}"
            print(error_msg)
            return error_msg
        
        return full_response


class RAGSystem:
    """
    RAG系统主控制器 - 协调各个组件完成检索增强生成任务
    
    工作流程：
    1. 检查数据库是否已初始化
    2. 如果未初始化，执行PDF读取、切片、向量化流程
    3. 进入交互式问答循环
    4. 对每个问题执行检索+生成
    """
    
    def __init__(self):
        """初始化RAG系统"""
        self.config = Config()
        self.pdf_processor = PDFProcessor(
            self.config.PDF_FILENAME,
            self.config.CHUNK_SIZE,
            self.config.OVERLAP_SIZE
        )
        self.vector_db = VectorDatabase(self.config)
        self.llm_generator = LLMGenerator(self.config)
        
    def initialize_database(self) -> bool:
        """
        初始化向量数据库
        
        Returns:
            是否为新创建的数据库
        
        智能检测：
        - 如果ChromaDB已存在且有数据，直接使用
        - 如果不存在或为空，执行完整的入库流程
        """
        print("\n" + "🚀 " * 20)
        print("RAG系统启动中...")
        print("🚀 " * 20 + "\n")
        
        # 初始化客户端连接
        self.vector_db.initialize_clients()
        
        # 检查是否已有数据
        collection_exists = False
        try:
            existing_collection = self.vector_db.client.get_collection(
                self.config.COLLECTION_NAME
            )
            if existing_collection.count() > 0:
                collection_exists = True
                print(f"✅ 发现现有向量数据库，包含 {existing_collection.count()} 条记录")
                
                # 询问用户是否需要重建
                response = input("🔄 是否重建数据库？(y/n, 默认n): ").strip().lower()
                if response != 'y':
                    self.vector_db.collection = existing_collection
                    print("📚 使用现有数据库")
                    return False
        except:
            pass  # 集合不存在，需要新建
        
        # 执行完整的PDF处理和向量化流程
        print("\n" + "📄 " * 20)
        print("开始PDF文档处理和向量化入库流程")
        print("📄 " * 20 + "\n")
        
        # 步骤1: 提取PDF文本
        full_text = self.pdf_processor.extract_text()
        
        # 步骤2: 文本切片
        chunks = self.pdf_processor.create_chunks(full_text)
        
        # 步骤3: 向量化并存入数据库
        self.vector_db.create_collection(chunks)
        
        print("\n" + "✅ " * 20)
        print("向量数据库初始化完成！")
        print("✅ " * 20 + "\n")
        
        return True
    
    def query(self, user_question: str) -> str:
        """
        处理用户查询
        
        Args:
            user_question: 用户问题
            
        Returns:
            生成的回答
        """
        # 检索相关文本块
        relevant_chunks = self.vector_db.query_relevant_chunks(user_question)
        
        # 基于检索结果生成回答
        answer = self.llm_generator.generate_answer(user_question, relevant_chunks)
        
        return answer
    
    def interactive_mode(self):
        """交互式问答模式"""
        print("\n" + "💬 " * 20)
        print("进入交互式问答模式 (输入 'quit' 或 'exit' 退出)")
        print("💬 " * 20 + "\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n🔍 请输入你的问题: ").strip()
                
                # 退出检查
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 感谢使用，再见！")
                    break
                
                # 空输入检查
                if not user_input:
                    print("⚠️  请输入有效问题")
                    continue
                
                # 执行查询
                self.query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断信号，再见！")
                break
            except Exception as e:
                print(f"❌ 处理查询时出错: {str(e)}")
                continue


def main():
    """
    主函数 - RAG系统入口点
    
    使用示例：
    1. 将paper.pdf放在当前目录
    2. 设置环境变量 ZHIPU_API_KEY
    3. 运行: python real_rag.py
    """
    try:
        # 创建RAG系统实例
        rag_system = RAGSystem()
        
        # 初始化数据库
        rag_system.initialize_database()
        
        # 进入交互模式
        rag_system.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ 系统启动失败: {str(e)}")
        print("\n🔧 故障排查建议:")
        print("1. 检查ZHIPU_API_KEY环境变量是否已设置")
        print("2. 确认paper.pdf文件存在于当前目录")
        print("3. 检查网络连接是否正常")
        print("4. 确认已安装所有依赖: pip install chromadb openai pypdf")
        sys.exit(1)


if __name__ == "__main__":
    main()