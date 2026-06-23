# ============================================================
# Hacker News 简易爬虫
# 功能：抓取首页所有新闻的标题和链接，并保存到 hacker_news.csv
# 适合：Python 爬虫初学者练习
# ============================================================

# --- 第 1 步：导入需要用到的 Python 库 ---

# requests：第三方库，专门用来发送 HTTP 请求（就像浏览器访问网页一样）
import requests

# BeautifulSoup（来自 bs4 库）：用来解析 HTML 源码，方便我们从网页里"抠"出想要的数据
from bs4 import BeautifulSoup

# csv：Python 自带的模块，用来读写 CSV 表格文件，不需要额外安装
import csv

# os：Python 自带的模块，用来处理文件路径（比如把 CSV 保存到脚本所在目录）
import os


# --- 第 2 步：定义要爬取的目标网址 ---

# Hacker News（黑客新闻）是 Y Combinator 旗下的技术新闻聚合网站
# 它的首页地址是 news.ycombinator.com（注意：不是 ycombinator.com 主站）
# ycombinator.com 是公司官网；news.ycombinator.com 才是新闻列表页面
TARGET_URL = "https://news.ycombinator.com"

# 定义输出 CSV 文件的文件名（会保存在与本脚本相同的目录下）
OUTPUT_CSV = "hacker_news.csv"


# --- 第 3 步：发送 HTTP 请求，获取网页 HTML 源码 ---

# 设置请求头（Headers）：模拟浏览器访问，避免被网站拒绝
# User-Agent 告诉服务器"我是一个正常的浏览器"，而不是可疑的爬虫脚本
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 使用 requests.get() 向目标网址发送 GET 请求
# GET 是最常见的请求方式，相当于在浏览器地址栏输入网址后按回车
print(f"正在请求网页：{TARGET_URL}")
response = requests.get(TARGET_URL, headers=headers, timeout=15)

# response.status_code 是服务器返回的状态码
# 200 表示"请求成功"；404 表示"页面不存在"；500 表示"服务器内部错误"
print(f"服务器返回状态码：{response.status_code}")

# 如果状态码不是 200，说明请求出了问题，抛出异常并终止程序
response.raise_for_status()

# response.text 是网页的 HTML 源码（字符串格式）
# 这就是你在浏览器里按 F12 → 查看源代码 看到的内容
html_content = response.text
print(f"成功获取 HTML，共 {len(html_content)} 个字符")


# --- 第 4 步：用 BeautifulSoup 解析 HTML，提取新闻标题和链接 ---

# BeautifulSoup(源码, 解析器)
# "html.parser" 是 Python 内置的 HTML 解析器，不需要额外安装
soup = BeautifulSoup(html_content, "html.parser")

# 在 Hacker News 的 HTML 中，每条新闻对应一个 <tr> 标签
# 这个 <tr> 带有 class="athing"（"a thing" 的缩写，表示一条新闻条目）
# find_all() 会找出页面上所有符合条件的标签，返回一个列表
news_rows = soup.find_all("tr", class_="athing")

print(f"在页面上找到了 {len(news_rows)} 条新闻")

# 创建一个空列表，用来存放解析出来的新闻数据
# 每条新闻会是一个字典：{"title": "标题", "url": "链接"}
news_list = []

# 用 for 循环遍历每一条新闻所在的 <tr> 行
for index, row in enumerate(news_rows, start=1):
    # row.find() 在当前 <tr> 内部查找第一个匹配的子标签
    # 新闻标题包裹在 <span class="titleline"> 里面
    titleline_span = row.find("span", class_="titleline")

    # 防御性编程：如果某一行没有找到 titleline，就跳过，避免程序崩溃
    if titleline_span is None:
        continue

    # 在 titleline 里面找到 <a> 标签，它同时包含标题文字和链接地址
    title_link = titleline_span.find("a")

    # 如果找不到 <a> 标签，同样跳过这一行
    if title_link is None:
        continue

    # .get_text() 提取标签内的纯文本，即新闻标题
    # strip=True 会去掉文字首尾多余的空格和换行符
    title = title_link.get_text(strip=True)

    # .get("href") 读取 <a> 标签的 href 属性，即链接地址
    url = title_link.get("href", "")

    # Hacker News 有些链接是相对路径（如 item?id=12345）
    # 相对路径需要拼接完整域名才能正常访问
    if url.startswith("item?"):
        url = "https://news.ycombinator.com/" + url

    # 把标题和链接打包成字典，追加到 news_list 列表末尾
    news_list.append({"title": title, "url": url})

    # 在控制台打印进度，方便观察爬取过程（只打印前 5 条，避免刷屏）
    if index <= 5:
        print(f"  [{index}] {title}")


# --- 第 5 步：把抓取结果写入 CSV 文件 ---

# os.path.dirname(__file__) 获取当前脚本文件所在的目录路径
# os.path.join() 把目录和文件名拼接成完整路径（自动处理 Windows/Mac 的路径差异）
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, OUTPUT_CSV)

# open() 打开（或创建）文件
# mode="w" 表示写入模式（如果文件已存在会被覆盖）
# newline="" 是 csv 模块官方推荐的写法，防止 Windows 下出现多余空行
# encoding="utf-8-sig" 使用 UTF-8 编码，sig 会在文件开头加 BOM，Excel 打开中文不会乱码
with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as csv_file:
    # csv.writer() 创建一个 CSV 写入器，负责把数据格式化成表格行
    writer = csv.writer(csv_file)

    # writerow() 写入一行数据
    # 第一行通常是"表头"，告诉读者每一列代表什么
    writer.writerow(["Title", "URL"])

    # 遍历 news_list，把每条新闻的标题和链接写入 CSV
    for news in news_list:
        writer.writerow([news["title"], news["url"]])

# with 语句块结束后，文件会自动关闭，不需要手动调用 close()

print(f"\n爬取完成！共保存 {len(news_list)} 条新闻到：{csv_path}")
