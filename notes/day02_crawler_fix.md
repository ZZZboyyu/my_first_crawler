# Day 2 修复并跑通 Hacker News 爬虫

日期：2026-07-22

项目位置：`D:\langchain\my_first_crawler`

运行环境：`D:\langchain\langchain-learning\.venv`

## 1. 今日目标

修复 `crawler.py`，让它成为一个干净、稳定、可运行的项目入口。

完成内容：

- 安装爬虫依赖 `beautifulsoup4`。
- 重写乱码和存在语法风险的旧版 `crawler.py`。
- 把脚本拆成 `fetch_html`、`parse_stories`、`save_csv`、`main`。
- 成功抓取 Hacker News 首页。
- 成功输出 `hacker_news.csv`。

## 2. 安装的依赖

```powershell
D:\langchain\langchain-learning\.venv\Scripts\python.exe -m pip install beautifulsoup4
```

安装结果：

- `beautifulsoup4`
- `soupsieve`

## 3. 今日代码结构

`crawler.py` 当前包含 4 个主要函数：

- `fetch_html(url)`：请求 Hacker News 首页 HTML。
- `parse_stories(html)`：用 BeautifulSoup 提取新闻标题和链接。
- `save_csv(stories, output_path)`：把结构化结果写入 CSV。
- `main()`：串起完整流程。

当前输出字段：

- `rank`
- `title`
- `url`

## 4. 运行命令

```powershell
cd D:\langchain\my_first_crawler
D:\langchain\langchain-learning\.venv\Scripts\python.exe crawler.py
```

## 5. 运行结果

本次成功抓取：

- 30 条 Hacker News 新闻
- 输出文件：`D:\langchain\my_first_crawler\hacker_news.csv`

前 5 条示例：

1. OverpAId - Fire your CEO. Hire the future
2. Back to Kagi
3. Businesses with ugly AI menu redesigns
4. OpenAI and Hugging Face address security incident during model evaluation
5. Introduction to Formal Verification with Lean Part 1

## 6. 遇到的问题

`requests` 在当前环境访问 `https://news.ycombinator.com/` 时出现过 SSL EOF 错误。

处理方式：

- 保留 `requests` 作为主请求方式。
- 增加 `urllib` fallback。
- 当 `requests` 失败时，自动切换到 `urllib` 继续抓取。

这个处理让脚本在网络/证书环境有波动时更稳。

## 7. 今天学到的点

- 爬虫不只是 `requests.get()`，还要考虑超时、异常和 fallback。
- HTML 解析要先定位稳定结构，本例中 Hacker News 新闻行是 `tr.athing`。
- 结构化输出比直接打印更重要，因为后续 LangChain 总结、分类、JSONL 都依赖干净数据。
- 小脚本也应该拆函数，否则后面接 AI 功能会越来越乱。

## 8. 明天 Day 3 要做

把爬虫数据进一步结构化：

- 增加 `source` 字段。
- 增加 `fetched_at` 字段。
- 保留 CSV 输出。
- 新增 JSONL 输出。
- 为 Day 4 的 LangChain 新闻总结准备输入数据。

目标产物：

- `hacker_news.csv`
- `hacker_news.jsonl`
- 更清晰的新闻数据结构。
