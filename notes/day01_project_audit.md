# Day 1 项目体检

检查日期：2026-07-22

项目位置：`D:\langchain\my_first_crawler`

学习环境：`D:\langchain\langchain-learning\.venv`

## 1. 项目里已有模块

- `crawler.py`：Hacker News 简单爬虫，目标是抓取首页新闻标题和链接，并保存到 CSV。
- `paper_ai`：早期论文 RAG 实验目录，包含 PDF 和 `my_rag.py`。
- `vector_rag`：混合检索 RAG 实现，包含 PDF、ChromaDB 本地向量库和 `real_rag.py`。
- `advanced_rag`：更高级的 RAG 实验，方向是混合检索 + reranker。
- `data_cleaner`：PDF 文本清洗、chunk、embedding、Chroma 入库流水线。
- `agent_team`：基于 CrewAI 的多 Agent 论文总结项目。
- `secure_agent`：安全聊天/安全网关方向的实验代码。
- `llm_red_team`：红队越狱测试与模型安全评估脚本。
- `prompt_optimizer`：自动优化 prompt 的实验脚本。
- `sft_dataset_builder`：把论文文本转换成 SFT JSONL 数据集。
- `structure_parser`：结构解析相关实验，包含 ChromaDB 和 PDF 产物。
- `ai`：包含论文 PDF 和助手脚本文本，像是早期临时实验目录。

## 2. 当前最重要的问题

- 中文注释和 README 在当前终端读取时出现明显乱码，后续需要统一编码或重写关键文档。
- 仓库缺少统一的 `requirements.txt` 或 `pyproject.toml`，依赖管理分散在各个脚本里。
- API key 管理分散，部分脚本直接依赖 `ZHIPU_API_KEY`，部分使用 `OPENAI_API_KEY`。
- 向量库、PDF、JSONL、报告文件等产物混在仓库里，后续应该用 `.gitignore` 和 `data/outputs/` 约定隔离。
- 多个 RAG 模块重复实现 PDF 读取、文本切分、embedding、Chroma 入库和检索逻辑。
- 目前脚本多是单文件实验形态，缺少统一入口、可复用模块和最小测试。

## 3. 我最想先跑通的功能

第一优先级：Hacker News 爬虫。

原因：

- 它是仓库最清晰的入口，适合从基础 Python 工程化开始。
- 可以自然接到 LangChain：爬取新闻 -> 清洗结构化 -> 总结热点 -> 分类主题 -> 输出 JSONL/Markdown。
- 不依赖 PDF、向量库或大型模型本地资源，适合作为第 1 周主线。

第二优先级：论文 RAG。

原因：

- 仓库里已经有多个 RAG 实验，可以通过重构学会 LangChain 的 loader、splitter、embedding、retriever 和 chain。
- 后续可以扩展成“论文知识库问答助手”。

## 4. 今天确认的依赖

当前虚拟环境：`D:\langchain\langchain-learning\.venv`

已安装：

- `requests`
- `langchain`
- `langchain_openai`
- `chromadb`
- `pypdf`
- `openai`
- `numpy`

缺少：

- `bs4` / `beautifulsoup4`：`crawler.py` 需要。
- `zhipuai`：`secure_agent`、`prompt_optimizer`、`llm_red_team`、`data_cleaner` 等需要。
- `crewai`：`agent_team` 需要。
- `PyPDF2`：`agent_team` 和部分旧 RAG 代码需要。
- `rank_bm25`：混合检索需要。
- `jieba`：中文 BM25 分词需要。

建议后续补充的依赖安装命令：

```powershell
D:\langchain\langchain-learning\.venv\Scripts\python.exe -m pip install beautifulsoup4 zhipuai PyPDF2 rank-bm25 jieba
```

`crewai` 依赖较重，建议等到第 4 周 Agent 阶段再安装和锁版本。

## 5. 明天要做

修复并跑通 `crawler.py`：

- 整理乱码注释，保留清晰中文说明。
- 补齐爬虫依赖 `beautifulsoup4`。
- 跑通 Hacker News 首页抓取。
- 输出 `hacker_news.csv`。
- 为后续 AI 总结准备结构化字段：`rank`、`title`、`url`、`source`、`fetched_at`。

## 6. 今日结论

这个仓库已经具备 AI 工程学习的完整材料：爬虫、RAG、Agent、安全、Prompt 优化、SFT 数据集都有雏形。接下来最好的路线不是重新开一个玩具项目，而是把现有实验一步步整理成可运行、可维护、可展示的项目。

Day 2 从最稳的入口开始：先让 `crawler.py` 成为一个干净、可靠的项目入口。
