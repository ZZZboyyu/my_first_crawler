markdown
# 🤖 最简单的 Python RAG 系统

> 一个从零构建的检索增强生成（RAG）系统，实现本地 PDF 智能问答

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![智谱AI](https://img.shields.io/badge/智谱AI-GLM--4--Flash-orange.svg)](https://open.bigmodel.cn/)

---

## 📖 项目简介

本项目实现了一个**完整的 RAG（Retrieval-Augmented Generation）架构**，包含三大核心模块：

| 模块 | 功能 | 实现方式 |
|------|------|----------|
| **Chunking（文本切片）** | 将 PDF 文档按 600 字切分成语义完整的文本块 | 基于标点符号的智能分句 |
| **Retrieval（检索）** | 从文本块中找出与用户问题最相关的片段 | 关键词匹配 + 词频统计 |
| **Generation（生成）** | 基于检索结果生成精准、流畅的回答 | 智谱 AI GLM-4-Flash 流式生成 |

### 🎯 核心特性

- ✅ **单文件实现** — 所有逻辑在 `my_rag.py` 中完成，零配置开箱即用
- ✅ **本地 PDF 处理** — 使用 `pypdf` 解析，无需外部服务
- ✅ **自定义检索器** — 不依赖向量数据库，纯 Python 实现关键词匹配
- ✅ **流式输出** — 大模型回答实时逐字显示，体验流畅
- ✅ **交互式对话** — 支持连续问答，输入 `quit` 退出

---

## 🏗️ 系统架构
┌─────────────────────────────────────────────────────────────────┐
│ RAG 完整工作流程 │
└─────────────────────────────────────────────────────────────────┘

PDF 文档
│
▼
┌───────────────────────────────────────────────────────────────┐
│ 1. Chunking（文本切片） │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ "本文研究了深度学习在医疗图像中的应用...（600字）" │ │
│ │ "实验结果表明模型准确率达到95%...（600字）" │ │
│ │ "未来工作方向包括多模态融合...（600字）" │ │
│ └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
│
▼
用户问题："论文的主要贡献是什么？"
│
▼
┌───────────────────────────────────────────────────────────────┐
│ 2. Retrieval（检索） │
│ 关键词提取 → 词频统计 → 排序 → Top-K 筛选 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Top-1: "本文主要贡献是提出了新的网络架构..." 得分: 8 │ │
│ │ Top-2: "实验验证了该方法在多个数据集上有效..." 得分: 5│ │
│ └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────────────────┐
│ 3. Generation（生成） │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ System: 你是一个知识问答助手... │ │
│ │ User: 【背景资料】Top-1 + Top-2 │ │
│ │ 【问题】论文的主要贡献是什么？ │ │
│ └─────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🤖 流式输出: "根据提供的资料，论文的主要贡献..." │ │
│ └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘

text

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip（Python 包管理器）

### 安装依赖

```bash
# 进入项目目录
cd ai/

# 安装必需的 Python 库
pip install pypdf openai
准备文件
将你的 PDF 文档命名为 paper.pdf，放在 ai/ 目录下：

text
ai/
├── my_rag.py          # 主程序
├── paper.pdf          # 你的 PDF 文档
└── README.md          # 本文档
运行程序
bash
python my_rag.py
交互示例
bash
$ python my_rag.py

============================================================
🚀 最简单的 Python RAG 系统
============================================================
流程：读取 PDF → 分块 → 关键词检索 → 智谱 AI 回答
============================================================

📄 正在读取 PDF 文件: ./paper.pdf
   ✅ 已读取第 1 页
   ✅ 已读取第 2 页
   📊 总共提取了 15234 个字符

✂️  正在将文本切分成块（每块约 600 字）...
   ✅ 成功切分成 25 个文本块

============================================================
💬 请输入您的问题（输入 'quit' 或 'exit' 退出）
============================================================

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
👤 您: 这篇论文主要研究了什么？

🔍 正在检索与问题最相关的 2 段文本...
   问题: 这篇论文主要研究了什么？
   🔑 提取的关键词: ['论文', '主要', '研究']
   ✅ 检索完成，找到 2 个相关段落
      #1 得分: 6 | 预览: 本文主要研究了深度学习在医疗图像分割中的应用...
      #2 得分: 4 | 预览: 实验结果表明，该方法在多个数据集上表现优异...

============================================================
🤖 正在调用智谱 AI 生成回答...
============================================================
   🌐 连接地址: https://open.bigmodel.cn/api/paas/v4
   🤖 使用模型: glm-4-flash

📝 回答：
------------------------------------------------------------
根据提供的论文资料，该研究主要聚焦于深度学习技术
在医疗图像分割领域的应用。作者提出了一种新的网络架构...
（流式输出，逐字显示）
------------------------------------------------------------

✅ 回答完成！
🔧 配置说明
智谱 AI 配置
python
# my_rag.py 中的配置项
ZHIPU_API_KEY = "your-api-key-here"          # 智谱 AI API 密钥
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # API 端点
ZHIPU_MODEL = "glm-4-flash"                  # 使用的模型（免费）
文档处理配置
python
CHUNK_SIZE = 600    # 文本块大小（字符数）
TOP_K = 2          # 检索返回的相关片段数量
生成参数配置
python
temperature=0.7    # 控制回答的随机性（0-1）
max_tokens=4096    # 最大输出长度
🐛 排错心得：405 网关报错解决纪实
问题现象
程序运行时持续报错：

text
❌ 调用智谱 AI 时出错: <html>
<head><title>405 Not Allowed</title>...</head>
<body><center><h1>405 Not Allowed</h1></center></body>
</html>
排查过程
第一阶段：怀疑 API Key 问题
❌ 尝试：检查 API Key 是否正确

❌ 结果：405 错误依然存在，说明不是认证问题

第二阶段：怀疑账户余额不足
❌ 尝试：检查智谱控制台余额

❌ 结果：glm-4-flash 是免费模型，且余额充足

❌ 结论：405 ≠ 429（余额不足应返回 429）

第三阶段：定位根本原因
错误代码：

python
# ❌ 错误的 Base URL
ZHIPU_BASE_URL = "https://bigmodel.cn"
client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL + "/v1"  # 拼接到错误地址
)
# 实际请求：https://bigmodel.cn/v1/chat/completions
问题分析：

405 Not Allowed 表示请求方法（POST）不被服务器支持

实际原因：请求发送到了错误的 API 端点

智谱 AI 的正确端点是 https://open.bigmodel.cn/api/paas/v4/chat/completions

错误端点是 https://bigmodel.cn/v1/chat/completions（不存在该接口）

第四阶段：验证与修复
验证方法 - 创建测试脚本：

python
import requests
import json

url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
headers = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "glm-4-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": False
}
response = requests.post(url, headers=headers, json=data)
print(response.status_code)  # 200 表示成功
修复代码：

python
# ✅ 正确的 Base URL
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL  # 直接使用，不加额外后缀
)
# 实际请求：https://open.bigmodel.cn/api/paas/v4/chat/completions
经验总结
错误码	含义	排查方向
405	Method Not Allowed	检查 URL 端点是否正确，而不是证书或权限问题
429	Too Many Requests	检查账户余额或限流设置
401	Unauthorized	检查 API Key 是否正确
404	Not Found	检查 URL 路径是否完整
关键启示：

405 错误 ≠ 余额不足 — 不同错误码对应不同问题，不要混淆

先验证 API 再集成 — 用最小测试脚本确认 API 可用

仔细阅读官方文档 — 智谱 AI 的 Base URL 包含 /api/paas/v4 路径

对比成功和失败请求 — 用 curl 或测试脚本抓取完整请求对比

📁 项目结构
text
ai/
├── my_rag.py          # 主程序（单文件实现）
│   ├── 导入依赖
│   ├── 配置信息
│   ├── extract_text_from_pdf()   # PDF 读取
│   ├── split_text_into_chunks()  # 文本切片
│   ├── simple_retrieve()         # 关键词检索
│   ├── generate_answer_with_zhipu() # AI 生成
│   └── main()                    # 主循环
├── paper.pdf          # 待处理的 PDF 文档
└── README.md          # 项目文档
🧠 核心算法解析
1. 文本切片（Chunking）
python
def split_text_into_chunks(text: str, chunk_size: int = 600) -> List[str]:
    # 按句子分割（保持语义完整）
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    
    # 贪心合并：不超过 chunk_size 的句子合并为一个块
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
设计思路：

以标点符号（。！？；\n）作为句子边界

尽可能保持句子完整，避免截断语义

如果单个句子超过 chunk_size，按空格或逗号二次切分

2. 关键词检索（Retrieval）
python
def simple_retrieve(chunks, query, top_k=2):
    # 1. 提取关键词（去除停用词）
    words = extract_keywords(query)
    
    # 2. 计算词频得分
    for chunk in chunks:
        score = sum(chunk.count(word) for word in words)
        scored_chunks.append((chunk, score))
    
    # 3. 排序返回 Top-K
    return sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:top_k]
设计思路：

无向量数据库：纯文本匹配，适合学习 RAG 原理

词袋模型：统计关键词出现次数，简单有效

停用词过滤：去除"的、了、在"等无意义词

3. 生成增强（Generation）
python
def generate_answer_with_zhipu(query, contexts):
    # 构建增强提示词
    prompt = f"""
    【背景资料】: {contexts}
    【用户问题】: {query}
    请基于背景资料回答...
    """
    
    # 流式调用智谱 AI
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[...],
        stream=True
    )
    
    # 逐字输出
    for chunk in response:
        print(chunk.content, end="")
设计思路：

上下文注入：检索结果作为背景资料输入

指令约束：要求模型基于资料回答，防止幻觉

流式输出：提升用户体验

🔬 性能优化方向
当前瓶颈
瓶颈	原因	优化方案
检索准确率	关键词匹配无法理解语义	升级为向量检索（Embedding + 余弦相似度）
处理速度	PDF 解析和分块串行处理	异步 I/O + 多进程处理
上下文长度	600 字分块可能截断重要信息	增加重叠窗口（Overlapping Chunks）
进阶改进建议
python
# 1. 使用向量检索（需要智谱 Embedding API）
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="...")
embeddings = client.embeddings.create(
    model="embedding-2",
    input=["文本内容"]
)

# 2. 添加重叠窗口
def split_with_overlap(text, chunk_size=600, overlap=50):
    # 每个块与前一个块有 50 字重叠
    pass

# 3. 使用 Redis 缓存检索结果
import redis
cache = redis.Redis()
cached = cache.get(query_hash)
📚 参考资料
智谱 AI 官方文档

RAG 论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

OpenAI Python Library

📄 License
MIT License — 自由使用、修改、分发

🙏 致谢
本项目是 AI 专业大一学生的课程实践项目，感谢智谱 AI 提供的免费 GLM-4-Flash 模型支持。

Happy Coding! 🚀

text

这份 README 包含了：
1. ✅ 完整的 RAG 架构说明（Chunking → Retrieval → Generation）
2. ✅ 详细的 405 错误排查过程
3. ✅ Base URL 对齐的完整记录
4. ✅ 代码注释和核心算法解析
5. ✅ 性能优化建议和进阶方向