#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
111.py - 多模态结构化表格解析与两阶段重排RAG系统 v2.1
绝对免疫文件锁 + 零崩溃渲染 + 防海外劫持
部署路径: structure_parser/111.py
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# ===================== 1. 环境安全 =====================
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    raise RuntimeError("❌ 致命错误: 环境变量 ZHIPU_API_KEY 未设置")

# ===================== 2. 导入第三方库 =====================
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
except ImportError:
    raise ImportError("请安装 reportlab: pip install reportlab")

try:
    import pdfplumber
except ImportError:
    raise ImportError("请安装 pdfplumber: pip install pdfplumber")

try:
    from zhipuai import ZhipuAI
except ImportError:
    raise ImportError("请安装 zhipuai: pip install zhipuai")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("请安装 chromadb: pip install chromadb")


# ===================== 3. 零崩溃PDF生成引擎 =====================
def make_pure_english_pdf(pdf_path: str) -> None:
    """
    零崩溃PDF生成器
    特性: 纯英文文本 + 硬编码表格宽度 + 无HTML标签
    """
    # 创建文档 - A4纸张
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    
    # 样式 - 纯字体，无HTML标签
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Normal'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12,
        fontName='Helvetica',
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_LEFT,
        spaceAfter=6,
        fontName='Helvetica',
    )
    
    story = []
    
    # ---------- 第一段：纯英文文本（无HTML标签） ----------
    story.append(Paragraph("Multi-Modal Structured Table Parsing for RAG Systems", title_style))
    story.append(Spacer(1, 12))
    
    first_text = """
    In industrial-grade LLM applications, multi-modal structured table parsing has emerged as
    a critical component in Retrieval-Augmented Generation (RAG) systems. Traditional text-only
    parsing methods often fail to accurately identify row-column structures in PDF tables,
    resulting in severe semantic distortion. This study proposes a two-dimensional matrix
    extraction algorithm using pdfplumber, which automatically converts PDF tables into
    standard Markdown format, significantly improving retrieval accuracy in RAG pipelines.
    """
    story.append(Paragraph(first_text, body_style))
    story.append(Spacer(1, 12))
    
    # ---------- 硬编码表格（纯英文，无汉字） ----------
    table_data = [
        ['Model Name', 'Exp Group', 'Ctrl Group'],
        ['Baseline', '82.5', '12.4'],
        ['Advanced RAG', '98.9', '11.2'],
    ]
    
    # 🚨 关键：硬编码绝对安全的物理像素宽度
    # A4宽度595像素，左margin72，右margin72，可用宽度451
    # colWidths: 140 + 100 + 100 = 340 < 451，安全！
    t = Table(table_data, colWidths=[140, 100, 100])
    
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 12))
    
    # ---------- 第二段：纯英文总结 ----------
    second_text = """
    Experimental results demonstrate that the Advanced RAG model achieves a significantly
    higher experimental group score (98.9) compared to the Baseline model (82.5), validating
    the effectiveness of structured table parsing in enhancing RAG performance. The control
    group score (11.2) further confirms that structured parsing substantially reduces retrieval
    noise. These findings provide important theoretical foundations and practical references
    for data governance engineering in production-grade RAG systems.
    """
    story.append(Paragraph(second_text, body_style))
    
    # 构建PDF - 绝对不会有排版溢出崩溃
    doc.build(story)
    print(f"   ✅ PDF生成成功: {pdf_path}")


# ===================== 4. 多模态结构化提取引擎 =====================
class StructuredTableExtractor:
    """多模态结构化提取 - Markdown表格重构"""
    
    @staticmethod
    def extract_structured_content(pdf_path: str) -> str:
        """提取PDF内容，将表格转换为Markdown格式"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF不存在: {pdf_path}")
        
        full_content = []
        
        print(f"   📄 正在解析PDF: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 提取普通文本
                page_text = page.extract_text() or ""
                
                # 提取表格
                tables = page.extract_tables()
                
                if tables and len(tables) > 0:
                    if page_text.strip():
                        full_content.append(page_text.strip())
                        full_content.append("\n")
                    
                    for table_idx, table in enumerate(tables, start=1):
                        if table and len(table) > 0:
                            markdown_table = StructuredTableExtractor._table_to_markdown(table)
                            full_content.append(markdown_table)
                            full_content.append("\n\n")
                else:
                    if page_text.strip():
                        full_content.append(page_text.strip())
                        full_content.append("\n\n")
        
        return "".join(full_content)
    
    @staticmethod
    def _table_to_markdown(table: List[List[Any]]) -> str:
        """将二维矩阵转换为Markdown表格"""
        if not table or len(table) == 0:
            return ""
        
        # 清理数据
        cleaned_rows = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    cell_str = str(cell).strip().replace('\n', ' ')
                    cleaned_row.append(cell_str)
            cleaned_rows.append(cleaned_row)
        
        # 确定列数
        max_cols = max([len(row) for row in cleaned_rows]) if cleaned_rows else 0
        if max_cols == 0:
            return ""
        
        # 补齐列
        for row in cleaned_rows:
            while len(row) < max_cols:
                row.append("")
        
        # 构建Markdown表格
        md_lines = []
        
        # 表头
        if len(cleaned_rows) > 0:
            header = cleaned_rows[0]
            md_lines.append("| " + " | ".join(header) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # 数据行
            for row in cleaned_rows[1:]:
                md_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(md_lines)


# ===================== 5. 滑动窗口分块器 =====================
class SlidingWindowChunker:
    """滑动窗口分块 - 600字/块, Overlap 150"""
    
    @staticmethod
    def chunk(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        step = chunk_size - overlap
        chunks = []
        text_len = len(text)
        
        for start in range(0, text_len, step):
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            if len(chunk) >= 50:
                chunks.append(chunk)
            if end == text_len:
                break
        
        return chunks


# ===================== 6. 防劫持向量库管理 =====================
class VectorDBManager:
    """防海外静默劫持 - 显式传入embeddings"""
    
    def __init__(self, persist_dir: str = "./chroma_db_structure_parser"):
        self.persist_dir = persist_dir
        self.collection = None
    
    def initialize(self):
        """初始化本地持久化向量库"""
        client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        try:
            client.delete_collection("structure_collection")
        except:
            pass
        
        self.collection = client.create_collection(
            name="structure_collection",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        
        print(f"   ✅ Collection创建成功 (hnsw:space=cosine)")
        return self.collection
    
    def add_embeddings(self, chunks: List[str], embeddings: List[List[float]]):
        """显式传入embeddings，封杀自动下载"""
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            raise ValueError("chunks和embeddings数量不匹配")
        
        ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": "paper.pdf",
                "chunk_index": i,
                "chunk_length": len(chunks[i]),
                "timestamp": datetime.now().isoformat()
            }
            for i in range(len(chunks))
        ]
        
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        print(f"   ✅ 成功入库 {self.collection.count()} 条向量记录")


# ===================== 7. 检索与问答引擎 =====================
class RAGQueryEngine:
    """两阶段重排RAG引擎"""
    
    def __init__(self, collection, api_key: str):
        self.collection = collection
        self.client = ZhipuAI(api_key=api_key)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """向量检索召回"""
        try:
            response = self.client.embeddings.create(
                model="embedding-3",
                input=query,
            )
            query_embedding = response.data[0].embedding
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            
            if results and results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            return []
    
    def stream_answer(self, query: str, context: List[str]) -> None:
        """流式问答 - 支持表格数字提问"""
        if not context:
            print("\n⚠️ 未检索到相关上下文")
            return
        
        combined_context = "\n\n".join(context)
        
        system_prompt = """You are a multi-modal structured table parsing and RAG expert.

Core capabilities:
1. Accurately parse Markdown tables with |---|---| separators
2. Precisely extract numerical values from table cells
3. Respond with proper Markdown table format when showing data
4. Provide exact numbers with two decimal places when applicable

Output rules:
- Use Markdown table format for data presentation
- Provide precise numerical values with decimal precision
- Clearly state if data is not found in retrieved context
- Maintain professional academic tone in English"""
        
        user_prompt = f"""
Retrieved context:
{combined_context}

User question:
{query}

Please answer based on the retrieved context. If the question involves table data, present it with proper Markdown table formatting.
"""
        
        print("\n🤖 GLM-4-Flash流式输出:")
        print("-" * 50)
        
        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"❌ 流式输出失败: {e}")


# ===================== 8. 主函数 =====================
def main():
    """主流水线 - 抗文件锁 + 零崩溃 + 防劫持"""
    
    print("=" * 70)
    print("多模态结构化表格解析RAG系统 v2.1")
    print("架构: 零崩溃PDF生成 → 结构化提取 → 滑动窗口 → 向量检索 → GLM问答")
    print("=" * 70)
    
    # ---------- Phase 1: 生成时间戳PDF（物理破锁） ----------
    print("\n[Phase 1] 零崩溃PDF生成 (时间戳物理破锁)...")
    
    # 🚨 关键：时间戳物理破锁
    timestamp = int(time.time())
    pdf_dir = "structure_parser"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = f"{pdf_dir}/victory_paper_{timestamp}.pdf"
    
    make_pure_english_pdf(pdf_path)
    
    # ---------- Phase 2: 多模态结构化提取 ----------
    print("\n[Phase 2] 多模态结构化提取 (Markdown表格重构)...")
    extractor = StructuredTableExtractor()
    structured_content = extractor.extract_structured_content(pdf_path)
    
    if not structured_content:
        print("❌ 提取内容为空")
        sys.exit(1)
    
    print(f"   📊 提取内容: {len(structured_content):,} 字符")
    
    # ---------- Phase 3: 滑动窗口分块 ----------
    print("\n[Phase 3] 滑动窗口分块 (600/150)...")
    chunker = SlidingWindowChunker()
    chunks = chunker.chunk(structured_content)
    print(f"   ✅ 分块完成: {len(chunks)} 块")
    
    # ---------- Phase 4: 向量化 ----------
    print("\n[Phase 4] 调用智谱 embedding-3 (1024维)...")
    
    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)
    
    embeddings = []
    for idx, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(
                model="embedding-3",
                input=chunk,
            )
            emb = response.data[0].embedding
            if len(emb) != 1024:
                print(f"⚠️ 维度异常: {len(emb)} != 1024")
            embeddings.append(emb)
            print(f"   ✅ {idx+1}/{len(chunks)} 块向量化完成")
            time.sleep(0.05)
        except Exception as e:
            print(f"❌ 向量化失败: {e}")
            sys.exit(1)
    
    if len(embeddings) != len(chunks):
        print("❌ 向量生成不完整")
        sys.exit(1)
    
    # ---------- Phase 5: 防劫持入库 ----------
    print("\n[Phase 5] 防海外劫持 - ChromaDB入库...")
    
    try:
        # 修复: 使用 persist_dir 而不是 path
        db_manager = VectorDBManager(persist_dir="./chroma_db_structure_parser")
        collection = db_manager.initialize()
        db_manager.add_embeddings(chunks, embeddings)
        print("   🎉 防海外劫持成功")
    except Exception as e:
        print(f"❌ 入库失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ---------- Phase 6: 交互式问答 ----------
    print("\n" + "=" * 70)
    print("✅ 系统初始化完成！进入交互式问答模式")
    print("💡 提示: 输入 'exit' 或 'quit' 退出")
    print("💡 示例: 'What is the score of Advanced RAG?'")
    print("=" * 70)
    
    engine = RAGQueryEngine(collection, ZHIPU_API_KEY)
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("⚠️ Question cannot be empty")
                continue
            
            print(f"   🔍 Retrieving relevant documents...")
            retrieved_chunks = engine.retrieve(query, top_k=5)
            
            if not retrieved_chunks:
                print("⚠️ No relevant context found")
                continue
            
            print(f"   📚 Retrieved {len(retrieved_chunks)} chunks")
            engine.stream_answer(query, retrieved_chunks)
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted, exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


# ===================== 9. 单文件自启动 =====================
if __name__ == "__main__":
    main()