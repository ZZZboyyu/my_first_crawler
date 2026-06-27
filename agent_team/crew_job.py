"""
============================================================
多智能体论文编译团队 - 完整实现
============================================================
功能：自动读取英文PDF论文，提取核心内容，生成中文趣味报告
框架：CrewAI + 智谱AI (GLM-4-Flash)
作者：AI专业大一学生练习项目
============================================================
"""

# ============================================================
# 第一步：导入所有需要用到的库
# ============================================================

import os
import sys
from pathlib import Path

# CrewAI 核心组件
from crewai import Agent, Task, Crew, Process, LLM

# 智谱AI的OpenAI兼容客户端（本质就是OpenAI的SDK）
from openai import OpenAI

# 用于读取PDF文件的库
import PyPDF2

# 用于处理中文和文本格式
import re


# ============================================================
# 第二步：配置大模型大脑（使用智谱AI）
# ============================================================

# 【重要】把你的智谱AI API Key 粘贴到下面的字符串里
# 获取地址：https://bigmodel.cn/usercenter/proj-mgmt/apikeys
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")  # ← 替换这里！！！

# 创建智谱AI的客户端（使用OpenAI标准格式）
# 这相当于给CrewAI配置了一个"大脑"
zhupu_llm = LLM(
    model="openai/glm-4-flash",
    base_url="https://bigmodel.cn/api/paas/v4/",  # 智谱AI的API地址
    api_key=ZHIPU_API_KEY,                        # 你的密钥
    default_headers={                             # 智谱AI要求的额外头信息
        "Authorization": f"Bearer {ZHIPU_API_KEY}"
    }
)

# 但是我建议用更标准的方式——直接创建一个CrewAI能认的LLM对象
# CrewAI推荐使用ChatOpenAI包装器，这样更规范
from langchain_openai import ChatOpenAI

# 这才是CrewAI官方推荐的大模型配置方式！
zhipu_llm = LLM(
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱AI的Base URL
    api_key=ZHIPU_API_KEY,                        # 你的API Key
    model="openai/glm-4-flash",                          # 免费模型，速度快
    temperature=0.7,                              # 温度控制创造性 (0~1)
    max_tokens=4096,                              # 最大输出长度
)


# ============================================================
# 第三步：招聘特工A - 英文论文调查员 (Researcher)
# ============================================================

# Agent（智能体）就是我们的"虚拟员工"，它有角色、目标和背景故事

researcher = Agent(
    role="英文论文调查员 (Academic Detective)",  # 角色名称
    goal="""从英文PDF论文中精准提取：
    1. 研究背景（这篇论文为什么做这个研究）
    2. 核心创新点（这篇论文有什么新发现/新方法）
    3. 核心实验数据（关键的数字、图表结论）
    要求：只提取干货，不要废话""",  # 目标任务
    backstory="""你是一位世界顶尖的学术侦探，拥有10年以上的科研文献分析经验。
    你精通从原始PDF文本中嗅探出最关键的创新点和实验数据。
    你擅长快速扫描、精确定位、提炼精华。
    你的座右铭是："每篇论文都藏着一颗宝石，我的任务就是把它挖出来。" """,  # 人设背景
    llm=zhipu_llm,                                      # 使用上面配置的大脑
    verbose=True,                                 # 打印详细运行日志
    allow_delegation=False,                       # 不允许委派任务给其他Agent
    max_iter=3,                                   # 最多迭代3次，防止无限循环
)


# ============================================================
# 第四步：招聘特工B - 中文文案主笔 (Writer)
# ============================================================

writer = Agent(
    role="中文文案主笔 (Science Communicator)",  # 角色名称
    goal="""把特工A提供的学术素材，改写成一篇：
    1. 中文写成的
    2. 大一新生能秒懂的
    3. 排版精美、带趣味小图标
    4. 结构清晰、一目了然
    的趣味学术总结报告""",  # 目标任务
    backstory="""你是一位科技报的大牌主编，有15年的科技写作经验。
    你的绝活是把Nature级别的硬核论文，写成中学生都能看懂的有趣故事。
    你擅长使用比喻、拟人、小图标来降低理解门槛。
    你的名言是："看不懂不是读者笨，是作者没讲清楚。" """,  # 人设背景
    llm=zhipu_llm,                                      # 使用同一个大脑
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)


# ============================================================
# 第五步：定义任务 (Task) - 给Agent分配具体工作
# ============================================================

# 我们先写一个函数，用来读取本地PDF文件的内容
def read_pdf_file(file_path):
    """
    读取PDF文件，提取所有文本内容
    参数：file_path - PDF文件的路径
    返回：提取出的全部文本（字符串）
    """
    try:
        # 用Path检查文件是否存在
        if not Path(file_path).exists():
            return f"错误：找不到文件 {file_path}"
        
        # 打开PDF文件（二进制模式）
        with open(file_path, 'rb') as file:
            # 创建PDF读取器
            reader = PyPDF2.PdfReader(file)
            
            # 获取总页数
            num_pages = len(reader.pages)
            print(f"📄 检测到PDF共 {num_pages} 页")
            
            # 提取每一页的文本
            all_text = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                all_text += text + "\n\n"
            
            # 如果提取的文本太短，可能是PDF是扫描件（图片格式）
            if len(all_text.strip()) < 100:
                return "警告：PDF文本内容较少，可能是扫描件。建议使用OCR工具处理。"
            
            return all_text
        
    except Exception as e:
        return f"读取PDF时出错：{str(e)}"


# 定义Task A - 调查任务
task_a = Task(
    description=f"""
    【任务A：论文素材提取】
    
    请阅读以下PDF论文全文，并提取关键信息。
    
    PDF论文内容：
    {read_pdf_file("paper.pdf")}
    
    提取要求：
    1. 研究背景：用3-5句话概括，说明"为什么要做这个研究"
    2. 核心创新点：列出2-4个关键创新，说明"这个研究有什么新东西"
    3. 核心实验数据：提取关键的数值、性能指标、对比结果
    
    输出格式：请用清晰的Markdown标题分类输出。
    """,
    agent=researcher,
    expected_output="""一份结构清晰的学术素材摘要，包含：
    ## 研究背景
    ## 核心创新点  
    ## 核心实验数据""",
    verbose=True,
)


task_b = Task(
    description="""
    【任务B：中文趣味报告撰写】
    
    你收到了特工A（调查员）提取的论文素材，现在请你：
    
    1. 把它翻译成中文（如果原文是英文）
    2. 用大一新生能看懂的语言重写（不要用晦涩的术语）
    3. 添加趣味小图标（比如：🔬 📊 💡 🎯 ⭐ 🔥）
    4. 重新组织结构，让它一目了然
    
    报告结构建议：
    📌 一句话总结（论文在做什么）
    🔬 为什么做这个研究（背景）
    💡 他们发现了什么（创新点）
    📊 数据有多硬核（实验数据）
    🎯 对我们有什么用（意义）
    
    要求：语言生动活泼，排版精美，让非专业人士也能看懂！
    """,
    agent=writer,
    expected_output="""一篇排版精美，带小图标的中文趣味学术报告，
    结构清晰，语言通俗易懂，适合大一新生阅读。""",
    verbose=True,
    context=[task_a],
)


# ============================================================
# 第六步：成立项目组 (Crew) - 把Agent们组织起来
# ============================================================

# Crew就是"项目组"，它负责管理所有Agent和Task，协调它们的工作

crew = Crew(
    agents=[researcher, writer],      # 所有Agent的列表
    tasks=[task_a, task_b],           # 所有Task的列表（按顺序执行）
    process=Process.sequential,       # 顺序执行：先A后B
    verbose=True,                     # 打印详细执行日志
    full_output=True,                 # 输出完整结果
    share_crew=False,                 # 不分享数据到云端
)


# ============================================================
# 第七步：启动运行！
# ============================================================

def main():
    """
    主函数：启动Crew执行任务
    """
    print("=" * 60)
    print("🚀 多智能体论文编译团队 启动中...")
    print("=" * 60)
    print("")
    
    # 检查paper.pdf是否存在
    if not Path("paper.pdf").exists():
        print("❌ 错误：当前目录下找不到 paper.pdf 文件！")
        print("   请把要分析的论文PDF文件放在本程序同一目录下，并命名为 paper.pdf")
        print("")
        print("   运行示例：python crew_job.py")
        return
    
    # 检查API Key是否已配置
    if ZHIPU_API_KEY == "在这里粘贴你那串智谱AI的sk-钥匙":
        print("❌ 错误：请先配置智谱AI的API Key！")
        print("   打开 crew_job.py 文件，找到 ZHIPU_API_KEY 变量")
        print("   把 '在这里粘贴你那串智谱AI的sk-钥匙' 替换成你的真实Key")
        print("")
        print("   获取Key地址：https://bigmodel.cn/usercenter/proj-mgmt/apikeys")
        return
    
    print("✅ 配置检查通过，开始执行任务...")
    print("")
    print("📋 任务流程：")
    print("   [特工A] 英文论文调查员 → 提取核心素材")
    print("   [特工B] 中文文案主笔 → 撰写趣味报告")
    print("")
    print("⏳ 正在运行，请稍候（可能需要1-3分钟）...")
    print("-" * 60)
    print("")
    
    try:
        # 执行Crew任务（这是核心调用！）
        result = crew.kickoff()
        
        print("")
        print("=" * 60)
        print("✅ 任务完成！最终报告如下：")
        print("=" * 60)
        print("")
        print(result)
        print("")
        print("=" * 60)
        print("🎉 论文编译完成！")
        print("   完整结果已打印在上方，你也可以将其保存为 .md 文件")
        print("=" * 60)
        
        # 可选：自动保存结果到文件
        with open("论文总结报告.md", "w", encoding="utf-8") as f:
            f.write("# 🤖 AI论文编译团队 生成报告\n\n")
            f.write(str(result))
            f.write("\n\n---\n")
            f.write("*本报告由 CrewAI + 智谱AI GLM-4-Flash 自动生成*")
        print("📁 报告已自动保存到：论文总结报告.md")
        
    except Exception as e:
        print("")
        print("=" * 60)
        print(f"❌ 运行出错：{str(e)}")
        print("=" * 60)
        print("")
        print("可能的原因和解决方案：")
        print("1. API Key无效或已过期 → 请检查智谱AI控制台")
        print("2. 网络连接问题 → 检查能否访问 https://bigmodel.cn")
        print("3. 余额不足 → 智谱AI的glm-4-flash是免费的，但需要账户有余额")
        print("4. 依赖库未安装 → 运行：pip install crewai langchain-openai PyPDF2")
        raise


# ============================================================
# 第八步：程序入口
# ============================================================

if __name__ == "__main__":
    """
    这是Python的标准入口写法。
    只有直接运行这个文件时（python crew_job.py），main()才会执行。
    如果被别的文件import导入，则不会自动运行。
    """
    main()