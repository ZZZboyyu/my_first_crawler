#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================
最简单的 Python RAG 系统 - 单文件实现
============================================
功能：从 PDF 读取文本 → 分块 → 关键词检索 → 智谱 AI 生成回答
作者：AI 专业大一学生
日期：2026-06-24
"""

# ==================== 第一部分：导入必要的库 ====================

import os
import re
from typing import List, Tuple

# PDF 读取库：用于解析 PDF 文件
from pypdf import PdfReader

# OpenAI 标准库：用于连接智谱 AI（兼容 OpenAI API 格式）
from openai import OpenAI

# ==================== 第二部分：配置信息 ====================

# 智谱 AI 的配置
ZHIPU_API_KEY = "d1445eb55b8a49d0a1b2ad3d8b812f03.w3sAU5x6n5tKJNU3"
# 重要：智谱 AI 的正确 Base URL
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # ✅ 修正为正确的地址
ZHIPU_MODEL = "glm-4-flash"  # 智谱的免费模型

# PDF 文件路径（放在当前目录下）
PDF_PATH = "./paper.pdf"

# 文本分块大小（按字符数计算，大约 600 字）
CHUNK_SIZE = 600

# 检索时返回的最相关文本块数量
TOP_K = 2


# ==================== 第三部分：读取 PDF 并提取文本 ====================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 文件中提取所有文本内容
    
    参数:
        pdf_path: PDF 文件的路径
        
    返回:
        提取出的全部文本（字符串）
    """
    print(f"📄 正在读取 PDF 文件: {pdf_path}")
    
    try:
        # 创建 PDF 阅读器对象
        reader = PdfReader(pdf_path)
        
        # 存储所有页面的文本
        all_text = ""
        
        # 遍历每一页
        for page_num, page in enumerate(reader.pages, start=1):
            # 提取当前页的文本
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"  # 每页之间加换行
            print(f"   ✅ 已读取第 {page_num} 页")
        
        print(f"   📊 总共提取了 {len(all_text)} 个字符")
        return all_text
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {pdf_path}")
        print("   请确保 'paper.pdf' 文件在当前目录下")
        return ""
    except Exception as e:
        print(f"❌ 读取 PDF 时出错: {e}")
        return ""


# ==================== 第四部分：文本分块 ====================

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    将长文本按指定字数切分成小文本块
    
    参数:
        text: 原始长文本
        chunk_size: 每个块的目标字符数
        
    返回:
        文本块列表
    """
    print(f"✂️  正在将文本切分成块（每块约 {chunk_size} 字）...")
    
    # 如果文本为空，直接返回空列表
    if not text:
        return []
    
    # 按句子分割（以句号、问号、感叹号、换行等作为分隔）
    # 这样能尽量保持语义完整性
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前块加上新的句子后不超过 chunk_size，则加入
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            # 如果当前块不为空，保存它
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 如果这个句子本身就超过 chunk_size，需要进一步切割
            if len(sentence) > chunk_size:
                # 按空格或标点切分
                sub_sentences = re.split(r'(?<=[，,、 ])', sentence)
                current_chunk = ""
                for sub in sub_sentences:
                    if len(current_chunk) + len(sub) <= chunk_size:
                        current_chunk += sub
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sub
            else:
                current_chunk = sentence
    
    # 处理最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # 过滤掉空块
    chunks = [chunk for chunk in chunks if chunk]
    
    print(f"   ✅ 成功切分成 {len(chunks)} 个文本块")
    return chunks


# ==================== 第五部分：自定义检索函数（关键词匹配） ====================

def simple_retrieve(chunks: List[str], query: str, top_k: int = TOP_K) -> List[Tuple[str, int]]:
    """
    使用简单的关键词匹配从文本块中检索最相关的片段
    
    原理：计算用户问题中的关键词在文本块中出现的次数，出现越多越相关。
    这是一种"词袋"方法，虽然简单但有效。
    
    参数:
        chunks: 所有文本块的列表
        query: 用户的问题
        top_k: 返回最相关的前 K 个块
        
    返回:
        一个列表，包含 (文本块, 得分) 的元组，按得分从高到低排序
    """
    print(f"🔍 正在检索与问题最相关的 {top_k} 段文本...")
    print(f"   问题: {query[:50]}..." if len(query) > 50 else f"   问题: {query}")
    
    # 对查询进行预处理：分词、去停用词、转小写
    # 这里我们简单地把查询拆分成单词
    query_words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query.lower())
    
    # 去除非常常见的停用词（简单版）
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '它', '他', '她', '们'}
    query_words = [w for w in query_words if w not in stopwords]
    
    if not query_words:
        # 如果查询没有有效关键词，返回前 top_k 个块
        print("   ⚠️ 查询中没有有效关键词，返回前几个块")
        return [(chunks[i], 0) for i in range(min(top_k, len(chunks)))]
    
    print(f"   🔑 提取的关键词: {query_words}")
    
    # 对每个文本块计算得分
    scored_chunks = []
    for chunk in chunks:
        # 把文本块也转小写
        chunk_lower = chunk.lower()
        
        # 计算每个关键词在文本块中出现的次数
        score = 0
        for word in query_words:
            # 使用 count 计算出现次数
            count = chunk_lower.count(word)
            score += count
            
            # 额外奖励：如果词在开头或结尾出现，稍微加分（增强效果）
            if count > 0:
                # 检查是否在句首或句尾
                if chunk_lower.startswith(word) or chunk_lower.endswith(word):
                    score += 1
        
        scored_chunks.append((chunk, score))
    
    # 按得分从高到低排序
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # 取前 top_k 个
    top_chunks = scored_chunks[:top_k]
    
    print(f"   ✅ 检索完成，找到 {len(top_chunks)} 个相关段落")
    for i, (chunk, score) in enumerate(top_chunks, start=1):
        preview = chunk[:80].replace('\n', ' ') + "..." if len(chunk) > 80 else chunk.replace('\n', ' ')
        print(f"      #{i} 得分: {score} | 预览: {preview}")
    
    return top_chunks


# ==================== 第六部分：调用智谱 AI 生成回答 ====================

def generate_answer_with_zhipu(query: str, contexts: List[str]) -> None:
    """
    使用智谱 AI 的 GLM-4-Flash 模型生成回答（流式输出）
    
    参数:
        query: 用户的问题
        contexts: 检索到的相关文本块列表（作为背景资料）
    """
    print("\n" + "="*60)
    print("🤖 正在调用智谱 AI 生成回答...")
    print("="*60)
    
    # 1. 构建提示词（Prompt）
    # 把检索到的文本块组合成背景资料
    background = "\n\n---\n\n".join(contexts)
    
    system_prompt = """你是一个专业的知识问答助手。请基于提供的【背景资料】回答用户的问题。

【重要规则】：
1. 如果背景资料中有相关信息，请用清晰、有条理的语言回答。
2. 如果背景资料中没有相关信息，请明确告知用户"根据提供的资料无法回答该问题"。
3. 不要编造或添加背景资料中没有的信息。
4. 回答要简洁、准确、易懂。
"""

    user_prompt = f"""
【背景资料】：
{background}

【用户问题】：
{query}

请基于上述背景资料回答用户的问题：
"""
    
    # 2. 创建 OpenAI 客户端（连接智谱 AI）
    try:
        print(f"   🌐 连接地址: {ZHIPU_BASE_URL}")
        print(f"   🤖 使用模型: {ZHIPU_MODEL}")
        
        client = OpenAI(
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL  # ✅ 直接使用正确的地址，不需要额外加 /v1
        )
        
        # 3. 调用模型 API（流式）
        print("\n📝 回答：\n")
        print("-" * 60)
        
        response = client.chat.completions.create(
            model=ZHIPU_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,  # 开启流式输出
            temperature=0.7,  # 控制随机性，0.7 比较适中
            max_tokens=4096  # 最大输出长度
        )
        
        # 4. 逐块输出流式响应
        full_response = ""
        for chunk in response:
            # 从响应中提取文本内容
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)  # 实时打印
                full_response += content
        
        print("\n" + "-" * 60)
        print("\n✅ 回答完成！")
        
    except Exception as e:
        print(f"\n❌ 调用智谱 AI 时出错: {e}")
        print("\n   🔧 请检查：")
        print("   1. ✅ API Key 格式是否正确（已包含）")
        print("   2. ✅ Base URL 是否正确（已修正为 open.bigmodel.cn）")
        print("   3. ⚠️  网络是否能够访问智谱 AI 服务")
        print("   4. ⚠️  API Key 是否有效且账户有余额")
        print("\n   💡 如果使用代理，请设置环境变量：")
        print("      export HTTP_PROXY=http://proxy:port")
        print("      export HTTPS_PROXY=http://proxy:port")


# ==================== 第七部分：主函数 ====================

def main():
    """
    主函数：RAG 系统的完整流程
    """
    print("\n" + "="*60)
    print("🚀 最简单的 Python RAG 系统")
    print("="*60)
    print("流程：读取 PDF → 分块 → 关键词检索 → 智谱 AI 回答")
    print("="*60 + "\n")
    
    # ----- 步骤 1：读取 PDF 文件 -----
    raw_text = extract_text_from_pdf(PDF_PATH)
    if not raw_text:
        print("❌ 无法读取 PDF，程序退出")
        return
    
    # ----- 步骤 2：将文本切分成块 -----
    chunks = split_text_into_chunks(raw_text, CHUNK_SIZE)
    if not chunks:
        print("❌ 文本块为空，程序退出")
        return
    
    # ----- 步骤 3：获取用户输入 -----
    print("\n" + "="*60)
    print("💬 请输入您的问题（输入 'quit' 或 'exit' 退出）")
    print("="*60)
    
    while True:
        print("\n" + ">"*40)
        query = input("👤 您: ").strip()
        
        # 检查退出条件
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not query:
            print("⚠️ 问题不能为空，请重新输入")
            continue
        
        # ----- 步骤 4：检索最相关的文本块 -----
        top_chunks = simple_retrieve(chunks, query, TOP_K)
        
        # 提取文本内容（忽略得分）
        contexts = [chunk for chunk, _ in top_chunks]
        
        # ----- 步骤 5：调用智谱 AI 生成回答 -----
        generate_answer_with_zhipu(query, contexts)
    
    print("\n" + "="*60)
    print("🏁 RAG 系统运行结束")
    print("="*60)


# ==================== 第八部分：程序入口 ====================

if __name__ == "__main__":
    """
    这是 Python 程序的入口点。
    当直接运行这个文件时（而不是作为模块导入），会执行 main() 函数。
    """
    main()