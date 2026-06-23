📰 Hacker News 简易爬虫
一个专为 Python 爬虫初学者设计的实战项目，优雅地抓取 Hacker News 首页新闻

https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/license-MIT-green.svg
https://img.shields.io/badge/PRs-welcome-brightgreen.svg

<p align="center"> <img src="https://news.ycombinator.com/favicon.ico" alt="Hacker News Logo" width="80"> </p>
📖 项目简介
Hacker News 简易爬虫 是一个精心设计的 Python 教学项目，能够自动抓取 Hacker News 首页的所有技术新闻，并将标题和链接保存为结构化的 CSV 文件。

✨ 为什么选择这个项目？
作为计算机系大一新生的实践作业，这个项目完美诠释了"做中学"的理念。它不仅仅是一个爬虫工具，更是一份精心注释的 Python 学习手册——每行代码都配有详细的中文说明，帮助你理解：

🌐 HTTP 协议基础：理解浏览器与服务器的对话方式

🏗️ HTML 结构解析：学会用 BeautifulSoup 解剖网页骨架

📊 数据处理与存储：掌握 CSV 文件的读写操作

🛡️ 防御性编程思想：培养健壮的代码习惯

🚀 功能特性
✅ 自动抓取 Hacker News 首页全部新闻（通常 30 条）

✅ 智能处理相对链接，自动补全为完整 URL

✅ 模拟浏览器请求头，优雅绕过反爬机制

✅ 输出 UTF-8 编码的 CSV 文件，Excel 打开无乱码

✅ 终端实时显示抓取进度

✅ 完善的错误处理，程序健壮不崩溃

🛠️ 技术栈
技术	用途	版本要求
Python	编程语言	≥ 3.8
Requests	HTTP 请求库	≥ 2.28
BeautifulSoup4	HTML 解析器	≥ 4.11
CSV	数据存储模块	Python 内置
OS	文件路径处理	Python 内置
📦 安装与运行
第一步：克隆项目
bash
git clone https://github.com/yourusername/hacker-news-crawler.git
cd hacker-news-crawler
第二步：创建虚拟环境（推荐）
bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
第三步：安装依赖
bash
pip install -r requirements.txt
requirements.txt 内容：

text
requests>=2.28.0
beautifulsoup4>=4.11.0
第四步：运行爬虫
bash
python hacker_news_crawler.py
预期输出
text
正在请求网页：https://news.ycombinator.com
服务器返回状态码：200
成功获取 HTML，共 45231 个字符
在页面上找到了 30 条新闻
  [1] Show HN: My First Open Source Project
  [2] The Future of WebAssembly
  [3] A Deep Dive into Rust Macros
  [4] Why PostgreSQL Is Better Than MySQL
  [5] Building a CLI Tool in Go
  ...

爬取完成！共保存 30 条新闻到：/path/to/hacker_news.csv
📁 项目结构
text
hacker-news-crawler/
├── hacker_news_crawler.py   # 主程序文件
├── requirements.txt         # 依赖列表
├── hacker_news.csv          # 输出文件（运行后生成）
├── README.md                # 项目说明书
└── .gitignore               # Git 忽略规则
💡 代码精读：它如何工作？
🎯 核心流程






🔍 关键代码片段
python
# 1. 伪装成浏览器，避免被反爬
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."
}

# 2. 发送 GET 请求并检查状态
response = requests.get(TARGET_URL, headers=headers, timeout=15)
response.raise_for_status()  # 如果失败，立即抛出异常

# 3. 解析 HTML，定位新闻标签
soup = BeautifulSoup(response.text, "html.parser")
news_rows = soup.find_all("tr", class_="athing")  # athing = "a thing"

# 4. 提取标题和链接
title = title_link.get_text(strip=True)  # 纯文本标题
url = title_link.get("href", "")         # 链接地址

# 5. 写入 CSV（UTF-8 BOM，Excel 友好）
with open(csv_path, "w", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Title", "URL"])
🔮 未来迭代规划
"优秀的项目永远不会真正完成，它们只是在不断进化。"

🌱 Phase 1：基础增强（学习重点）
功能	描述	学习目标
命令行参数	支持 --output custom.csv 自定义输出文件名	argparse 模块
配置文件支持	用 config.yaml 管理爬取参数	YAML 解析、配置管理
日志系统	用 logging 替代 print，记录运行日志	日志级别、文件日志
单元测试	为解析函数编写 pytest 测试用例	TDD 思想、测试驱动开发
🌿 Phase 2：功能扩展（能力提升）
功能	描述	技术栈
📊 数据分析面板	统计新闻域名分布、关键词词云	Pandas + Matplotlib
🗄️ 数据库存储	支持 SQLite/PostgreSQL 持久化存储	SQLAlchemy
📧 邮件推送	每日自动抓取并发送 Top 10 新闻	SMTP + Schedule
🌐 多页爬取	抓取前 N 页新闻（翻页功能）	URL 参数构造
⚡ 异步并发	使用 asyncio + aiohttp 加速爬取	异步编程
🌳 Phase 3：架构升级（工程化思维）
功能	描述	亮点
🐳 Docker 化部署	一键容器化，跨平台运行	Dockerfile 编写
🔄 RESTful API	用 FastAPI 提供实时查询接口	API 设计、OpenAPI 文档
🎨 Web 可视化	React 前端展示新闻瀑布流	前后端分离
🧠 AI 摘要生成	调用 GPT API 生成新闻摘要	LLM 应用开发
☁️ 云原生部署	部署到 AWS Lambda + S3	Serverless 架构
🚀 Phase 4：星辰大海（探索前沿）
python
# 未来某天，也许这个项目会变成...

class HackerNewsAI:
    """
    🤖 智能新闻助手
    
    功能：
    - 自动抓取并分类技术新闻
    - 基于用户兴趣的个性化推荐
    - 生成中英双语摘要
    - 趋势预测与热点追踪
    - 与 Slack/Discord/微信集成
    """
    
    def __init__(self):
        self.crawler = AsyncDistributedCrawler()  # 分布式爬虫
        self.nlp_model = load_pretrained_bert()   # NLP 模型
        self.recommender = CollaborativeFilter()  # 推荐算法
        
    async def intelligent_digest(self, user_profile):
        """为每个用户生成个性化技术日报"""
        news = await self.crawler.fetch_latest()
        filtered = self.recommender.filter(news, user_profile)
        return self.nlp_model.summarize(filtered)
🎯 你的成长路线图
完成以上迭代，你将掌握：

✅ 第1个月：Python 基础 + 爬虫原理

✅ 第3个月：数据处理 + 自动化运维

✅ 第6个月：Web 开发 + 云服务部署

✅ 第1年：成为能独立开发全栈项目的工程师

🤝 贡献指南
欢迎提交 Issue 和 Pull Request！对于初学者，我们特别推荐从 good first issue 标签开始。

贡献流程
Fork 本仓库

创建特性分支：git checkout -b feature/amazing-feature

提交更改：git commit -m 'feat: add amazing feature'

推送分支：git push origin feature/amazing-feature

提交 Pull Request

📝 学习资源推荐
Python 官方文档 - 最好的学习材料

Requests 库中文文档 - HTTP 请求的艺术

BeautifulSoup 文档 - HTML 解析利器

Hacker News API - 官方 API 文档

📄 许可证
本项目采用 MIT 许可证 - 详见 LICENSE 文件

🌟 致谢
感谢 Hacker News 提供优质的技术内容

感谢 Y Combinator 创造了这个伟大的技术社区

感谢所有为开源世界贡献智慧的开发者们

<p align="center"> <b>⭐ 如果这个项目对你有帮助，请给一个 Star！</b><br> <i>Made with ❤️ by a passionate CS freshman</i> </p>
📮 联系我
GitHub Issues：技术问题请在此讨论

Email：boyuzheng22@gmail.com


"学习编程最好的方式不是阅读，而是动手。这个项目就是你的第一块敲门砖。"