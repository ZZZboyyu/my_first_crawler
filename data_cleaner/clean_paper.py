#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_paper.py - 大厂级RAG生产环境数据治理重构 - 工业级黄金版 v5.0
技术栈: PyPDF + Regex + ChromaDB + ZhipuAI (Embedding-3 + GLM-4-Flash)
架构原则: 环境安全 / 七层物理脱水 / 对账报告 / 防海外劫持 / 优雅降级 / 人设注入
版本特性: 强力正则抹除冷门乱码 + 审稿专家人设注入
"""

import os
import re
import sys
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# ===================== 1. 环境安全 - 从环境变量读取密钥 =====================
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    raise RuntimeError("致命错误: 环境变量 ZHIPU_API_KEY 未设置，程序终止。")

# ===================== 2. 导入第三方库 (单文件自包含) =====================
try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError("请安装 pypdf: pip install pypdf")

try:
    from zhipuai import ZhipuAI
except ImportError:
    raise ImportError("请安装 zhipuai: pip install zhipuai")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("请安装 chromadb: pip install chromadb")


# ===================== 3. 七层物理脱水机 (Cleaner) - 工业级强化版 =====================
class PaperCleaner:
    """
    PDF赛博垃圾物理脱水过滤器 - 七层递进式清洗架构
    第1层: 字面量控制字符清除
    第2层: 实际控制字符清除
    第3层: 换行连字符修复
    第4层: 乱码/标记/装饰符号清除
    第5层: 空白字符压缩
    第6层: 标点符号修复
    第7层: 冷门乱码强力抹除 (只保留中英文/数字/空格/常用标点)
    """

    @staticmethod
    def clean(raw_text: str) -> str:
        # ---------- 第1层: 清除字面量形式的控制字符 ----------
        # 匹配字面量的 \x00-\x1f 形式（即反斜杠+x+两位十六进制）
        clean_text = re.sub(r'\\x[0-9a-fA-F]{2}', '', raw_text)
        # 匹配字面量的 \uXXXX 形式
        clean_text = re.sub(r'\\u[0-9a-fA-F]{4}', '', clean_text)
        # 匹配字面量的 \amp； \quot； 等LaTeX转义
        clean_text = re.sub(r'\\amp[；;]', '&', clean_text)
        clean_text = re.sub(r'\\quot[；;]', '"', clean_text)
        clean_text = re.sub(r'\\bsp[；;]', '', clean_text)
        clean_text = re.sub(r'\\t[；;]', ' ', clean_text)
        
        # ---------- 第2层: 清除实际的控制字符（字节值） ----------
        # ASCII控制字符: 0x00-0x1F (保留: \n=0x0A, \r=0x0D, \t=0x09)
        clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean_text)
        # Unicode特殊控制字符: 零宽字符, 特殊空格等
        clean_text = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff]', '', clean_text)
        
        # ---------- 第3层: 修复换行连字符 ----------
        # PDF中常见: "trans- \nformer" 或 "trans-\nformer"
        clean_text = re.sub(r'(\w+)-[ \t]*\n[ \t]*(\w+)', r'\1\2', clean_text)
        clean_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', clean_text)
        clean_text = re.sub(r'(\w+)- \n(\w+)', r'\1\2', clean_text)
        
        # ---------- 第4层: 清理乱码和标记 ----------
        # 清理 ◆◆◆ ■●● ▲▲▲ 等装饰符号
        clean_text = re.sub(r'[◆■●▲▼★☆]+', '', clean_text)
        # 清理重复符号: =====, !!!!!, ????, .....
        clean_text = re.sub(r'[=!?]{2,}', '', clean_text)
        clean_text = re.sub(r'[\.]{4,}', '', clean_text)
        clean_text = re.sub(r'[~]{3,}', '', clean_text)
        clean_text = re.sub(r'[-]{4,}', '', clean_text)
        # 清理 "test123", "乱码" 等标记
        clean_text = re.sub(r'test\d+', '', clean_text)
        clean_text = re.sub(r'乱码[◆■●▲▼]*', '', clean_text)
        clean_text = re.sub(r'数据丢失[！!]*', '', clean_text)
        clean_text = re.sub(r'未完待续[\.。…]*', '', clean_text)
        # 清理 【...】 标记
        clean_text = re.sub(r'【.*?】', '', clean_text)
        # 清理残留的x0b标记
        clean_text = re.sub(r'x[0-9a-fA-F]{1,2}\s*', '', clean_text)
        clean_text = re.sub(r'x[0-9a-fA-F]{1,2}', '', clean_text)
        
        # ---------- 第5层: 压缩空白字符 ----------
        # 多个空格/制表符合并为单个空格
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        # 多个换行符合并为两个（保留段落分隔）
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        
        # ---------- 第6层: 修复标点 ----------
        # 压缩重复标点
        clean_text = re.sub(r'[,，]{3,}', ',', clean_text)
        clean_text = re.sub(r'[.。]{3,}', '.', clean_text)
        clean_text = re.sub(r'[;；]{3,}', ';', clean_text)
        # 修复多余空格导致的 "。 " -> "。"
        clean_text = re.sub(r'([。！？；：])\s+', r'\1', clean_text)
        
        # ---------- 第7层: 冷门乱码强力抹除 (核心) ----------
        # 只保留: 中文字符、英文字母、数字、空格、常用标点符号
        # 常用标点: . , ! ? : - " ( ) 以及中文标点 。 ， ！ ？ ： ； “ ” ‘ ’
        clean_text = re.sub(
            r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!\?:\-\"\(\)\u3002\uff0c\uff01\uff1f\uff1a\uff1b\u201c\u201d\u2018\u2019]',
            '',
            clean_text
        )
        
        # 最终trim
        clean_text = clean_text.strip()
        
        return clean_text


# ===================== 4. 清洗数据持久化 =====================
def save_cleaned_data(clean_text: str, metadata: Dict[str, Any]) -> tuple:
    """
    保存清洗后的数据到文件
    Returns: (clean_file_path, meta_file_path)
    """
    # 创建清洗数据存储目录
    clean_dir = "./cleaned_data"
    os.makedirs(clean_dir, exist_ok=True)
    
    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_file = os.path.join(clean_dir, f"cleaned_paper_{timestamp}.txt")
    meta_file = os.path.join(clean_dir, f"metadata_{timestamp}.json")
    
    # 保存清洗后的文本（UTF-8编码）
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write(clean_text)
    
    # 保存元数据
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 清洗数据已保存: {clean_file}")
    print(f"📋 元数据已保存: {meta_file}")
    
    # 额外保存一份带行号的预览文件（方便调试）
    preview_file = os.path.join(clean_dir, f"preview_{timestamp}.txt")
    with open(preview_file, "w", encoding="utf-8") as f:
        lines = clean_text.split('\n')
        for i, line in enumerate(lines[:50], 1):  # 只保存前50行
            f.write(f"{i:3d} | {line}\n")
    
    return clean_file, meta_file


# ===================== 5. 构建高级人设Prompt =====================
def build_expert_prompt(retrieved_context: str, user_query: str = "") -> str:
    """
    构建带有高级清洗人设的Prompt
    人设: 审稿专家 + 数据修复引擎
    自动忽略PDF排版噪声，还原通顺学术干货
    """
    system_prompt = """你不仅是审稿专家，还是一个顶级的数据修复引擎。

当你阅读 <retrieved_context> 标签内的资料时，你的大脑请全自动在后台执行语义连连看：
1. 自动忽略由于PDF排版导致的错位空格
2. 自动修复断行连字符（如 "trans-\nformer" → "transformer"）
3. 自动在脑海中将其还原为通顺的工业级中英文学术干货
4. 绝对不准被残缺的排版噪声带偏！

你的输出要求：
- 基于检索到的资料，给出专业、准确、有深度的回答
- 如果资料不足以回答，请明确指出
- 保持中英文混合输出（专业术语保留英文）
- 输出结构清晰，逻辑严密"""

    user_prompt = f"""
<retrieved_context>
{retrieved_context}
</retrieved_context>

{user_query if user_query else "请基于以上资料，给出专业的学术性总结和分析。"}
"""
    
    return system_prompt, user_prompt


# ===================== 6. 优雅降级兜底机制 (Fallback) =====================
def fallback_to_llm(text: str, user_query: str = "") -> None:
    """
    降级方案: 直接使用纯文本调用智谱 GLM-4-Flash 流式输出
    带有高级人设注入，确保即使降级也能产出高质量回答
    确保核心业务 24h 不闪退
    """
    print("\n" + "🛡️ " * 10)
    print("【优雅降级模式】直接连线 GLM-4-Flash 流式处理 (带高级人设)")
    print("🛡️ " * 10)

    if not text or len(text) < 10:
        print("⚠️ 输入文本过短，无法进行有效推理。")
        return

    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        
        # 使用高级人设构建Prompt
        system_prompt, user_prompt = build_expert_prompt(text[:4000], user_query)

        print("🤖 正在调用 GLM-4-Flash 流式生成答案（审稿专家模式）...\n")
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
        )

        print("📝 降级输出结果 (流式):")
        print("-" * 40)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
        print("\n" + "-" * 40)
        print("✅ 降级推理完成，业务可用性保持100%")

    except Exception as e:
        print(f"❌ 降级调用也失败: {e}")
        print("💀 终极兜底: 仅打印清洗后文本前200字符供人工查看")
        print("\n--- 清洗后文本预览 ---")
        print(text[:200])
        print("... (截断)")
        print("--- 结束 ---")


# ===================== 7. 主函数流程 =====================
def main():
    print("=" * 60)
    print("大厂级RAG数据治理重构 - 启动黄金流水线 v5.0")
    print("架构: 七层物理脱水 + 防海外劫持 + 优雅降级 + 人设注入")
    print("=" * 60)

    # ---------- 7.1 PDF真实吞噬 ----------
    pdf_path = "paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ 错误: 文件 {pdf_path} 不存在，请检查路径。")
        sys.exit(1)

    print(f"\n📄 正在吞噬PDF: {pdf_path}")
    raw_paper_text = ""
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            raw_paper_text += page_text + "\n"
            print(f"   ✅ 已吞噬第 {page_num}/{total_pages} 页")
    except Exception as e:
        print(f"❌ PDF读取失败: {e}")
        sys.exit(1)

    # 显示原始文本样本（显示真实控制字符）
    print(f"\n📝 原始文本样本 (前300字符，显示转义):")
    print(repr(raw_paper_text[:300]))
    print()

    # ---------- 7.2 七层物理脱水 (Cleaner) ----------
    raw_len = len(raw_paper_text)
    print(f"🔧 清洗前原始文字长度: {raw_len:,} 字符")

    cleaner = PaperCleaner()
    clean_text = cleaner.clean(raw_paper_text)

    clean_len = len(clean_text)
    reduction = ((raw_len - clean_len) / raw_len) * 100 if raw_len > 0 else 0

    # ---------- 7.3 垃圾清理对账报告 ----------
    print(f"🧹 清洗后文字长度: {clean_len:,} 字符")
    print(f"📉 自动脱水赛博废话: {reduction:.2f}%")
    print(f"📊 节省存储量: {raw_len - clean_len:,} 字符")
    
    # 显示清洗后样本
    print(f"\n📝 清洗后文本样本 (前300字符):")
    print(clean_text[:300])
    print()

    # ---------- 7.4 保存清洗后的数据 ----------
    metadata = {
        "raw_length": raw_len,
        "clean_length": clean_len,
        "reduction_percent": round(reduction, 2),
        "reduction_chars": raw_len - clean_len,
        "timestamp": datetime.now().isoformat(),
        "cleaner_version": "v5.0",
        "pdf_path": pdf_path,
        "total_pages": total_pages
    }
    save_cleaned_data(clean_text, metadata)

    # 如果清洗后文本为空或过短，直接降级
    if not clean_text or len(clean_text) < 50:
        print("⚠️ 警告: 清洗后文本过短，触发优雅降级...")
        fallback_to_llm(clean_text if clean_text else "清洗后文本为空")
        return

    # ---------- 7.5 滑动窗口分块 (600字, Overlap 150) ----------
    chunk_size = 600
    overlap = 150
    step = chunk_size - overlap
    chunks = []
    text_len = len(clean_text)

    for start in range(0, text_len, step):
        end = min(start + chunk_size, text_len)
        chunk = clean_text[start:end]
        # 过滤太短的尾部块（至少保留50字符）
        if len(chunk) >= 50:
            chunks.append(chunk)
        if end == text_len:
            break

    print(f"\n📦 滑动窗口分块完成: 共 {len(chunks)} 块 (块大小{chunk_size}, Overlap{overlap})")

    # ---------- 7.6 调用智谱 embedding-3 生成 1024维向量 ----------
    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
    except Exception as e:
        print(f"❌ 初始化ZhipuAI客户端失败: {e}")
        fallback_to_llm(clean_text[:3000])
        return

    embeddings: List[List[float]] = []
    print("\n🧠 正在调用智谱 embedding-3 生成向量 (1024维)...")
    for idx, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(
                model="embedding-3",
                input=chunk,
            )
            emb = response.data[0].embedding
            if len(emb) != 1024:
                print(f"⚠️ 警告: 第{idx+1}块向量维度 {len(emb)} != 1024, 继续...")
            embeddings.append(emb)
            print(f"   ✅ 第 {idx+1}/{len(chunks)} 块向量化完成")
            time.sleep(0.1)  # 礼貌限流，避免API限频
        except Exception as e:
            print(f"❌ 第 {idx+1} 块向量化失败: {e}")
            fallback_to_llm(clean_text[:3000])
            return

    if not embeddings or len(embeddings) != len(chunks):
        print("❌ 向量生成不完整，触发降级...")
        fallback_to_llm(clean_text[:3000])
        return

    # ---------- 7.7 防海外静默下载劫持 - ChromaDB 本地入库 ----------
    print("\n💾 初始化本地ChromaDB (防海外劫持模式)...")
    try:
        # 使用PersistentClient，路径固定，关闭遥测
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db_clean",
            settings=Settings(
                anonymized_telemetry=False,  # 关闭遥测，防止外连
                allow_reset=True,
            )
        )
        
        # 删除旧collection以保持干净
        try:
            chroma_client.delete_collection("paper_clean_collection")
        except:
            pass

        # 创建collection，强制锁死 hnsw:space = cosine
        collection = chroma_client.create_collection(
            name="paper_clean_collection",
            metadata={"hnsw:space": "cosine"},  # ⚠️ 关键: 封杀amazonaws下载英文模型
            embedding_function=None,  # 手动传入embeddings，不用内置函数
        )
        print("   ✅ Collection创建成功 (hnsw:space=cosine)")

        # 准备数据: ids, documents, embeddings, metadatas
        ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": "paper.pdf",
                "chunk_index": i,
                "timestamp": datetime.now().isoformat(),
                "chunk_length": len(chunks[i])
            } 
            for i in range(len(chunks))
        ]

        # ⚠️ 关键: 显式传入自己算好的embeddings列表，彻底绕过内置下载
        print("📤 正在写入向量库 (显式传入embeddings, 无外连)...")
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,  # 显式传入，禁止自动下载模型
            metadatas=metadatas,
        )
        print(f"✅ 成功入库 {collection.count()} 条向量记录")
        print("🎉 向量化入库完成，业务稳定运行中")

    except Exception as e:
        print(f"❌ ChromaDB入库失败: {e}")
        print("🔄 触发优雅降级兜底机制...")
        # 降级时使用高级人设
        fallback_to_llm(clean_text[:3000], "请对以上学术资料进行全面总结和分析")
        return

    # ---------- 7.8 最终状态报告 ----------
    print("\n" + "=" * 60)
    print("✅ 大厂级RAG数据治理重构流水线执行完毕")
    print(f"   - 原始字符数: {raw_len:,}")
    print(f"   - 清洗后字符数: {clean_len:,}")
    print(f"   - 脱水率: {reduction:.2f}%")
    print(f"   - 分块数: {len(chunks)}")
    print(f"   - 向量维度: 1024")
    print(f"   - 存储路径: ./chroma_db_clean")
    print(f"   - 清洗数据: ./cleaned_data/")
    print(f"   - 清洗版本: v5.0 (七层脱水 + 冷门乱码抹除)")
    print(f"   - 人设模式: 审稿专家 + 数据修复引擎")
    print("=" * 60)


# ===================== 8. 单文件自启动 =====================
if __name__ == "__main__":
    main()