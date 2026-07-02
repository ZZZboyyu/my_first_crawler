"""
╔══════════════════════════════════════════════════════════════╗
║       工业级大模型 Prompt 全自动提示词进化与迭代调优引擎     ║
║              prompt_optimizer/auto_tune.py                  ║
║         优化器注入 → 考官打分 → 迭代进化 → 锁死最优解       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import re
import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# 第三方依赖: pip install zhipuai
from zhipuai import ZhipuAI

# ============================== 安全环境变量 ==============================
# 🔒 密钥硬隔离：绝不硬编码，从系统环境变量注入
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    raise RuntimeError("[硬中断] 环境变量 ZHIPU_API_KEY 未设置！请先 export ZHIPU_API_KEY='your_key'")


# ============================== 数据结构定义 ==============================
@dataclass
class PromptVersion:
    """单轮优化产出的提示词版本"""
    round: int                              # 当前优化轮次
    prompt_text: str                        # 提示词全文
    score: int                              # 考官打分 (1-100)
    evolution_type: str                     # 进化类型标签
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================== 提示词优化器 ==============================
class PromptOptimizer:
    """
    🔧 提示词优化器：注入高级人设，将初级提示词升级为完全体结构化 Prompt
    核心能力：XML标签包装、人设注入、边界约束、思维链引导、输出格式锁定
    """

    # 优化器的元提示词 —— 这是引擎的灵魂
    OPTIMIZER_SYSTEM_PROMPT = """你是一位世界顶级科技公司（如OpenAI/Google DeepMind）的首席提示词架构师。
你拥有博士学位级别的自然语言处理和人机交互专业知识。
你的任务是将用户提供的粗糙、低效的初级提示词，重构成一个完美的、工业级的、结构化的完全体Prompt。

你必须严格遵循以下【结构化优化准则】：

1. 【人设注入】：赋予AI一个精准的专家身份（如"你是Nature期刊的资深审稿人"）。
2. 【XML标签化】：使用 <task>、<context>、<constraints>、<output_format> 等XML标签组织信息。
3. 【思维链引导】：在 <workflow> 标签内植入 step-by-step 的思考流程。
4. 【边界约束】：在 <constraints> 标签内设置硬性禁区（如字数、禁止行为、合规红线）。
5. 【输出格式锁死】：在 <output_format> 标签内精确规定返回结构（JSON/Markdown/表格）。
6. 【Few-shot示例】：如果适用，提供1-2个高质量范例。

【输出规则】：
- 只输出优化后的完全体Prompt全文，不要任何额外解释。
- 优化的Prompt必须自包含，可直接复制使用。
- 保持语言与用户输入一致（中/英文）。"""

    def __init__(self, api_key: str):
        """初始化优化器，建立智谱客户端连接"""
        self.client = ZhipuAI(api_key=api_key)
        self.model = "glm-4-flash"  # 目标优化模型

    def evolve_prompt(self, current_prompt: str, round_num: int, previous_best: Optional[PromptVersion] = None) -> str:
        """
        执行单轮提示词进化
        参数:
            current_prompt - 当前待优化的提示词
            round_num - 当前优化轮次
            previous_best - 上一轮最优版本（用于迭代改进）
        返回:
            进化后的完全体Prompt文本
        """
        # 构建进化上下文：如果存在上一轮结果，要求在此基础上进一步优化
        if previous_best and previous_best.score > 0:
            evolution_context = f"""
【上一轮最优版本 - 得分: {previous_best.score}/100】
{previous_best.prompt_text}

【考官反馈】：上一轮得分未达完美，请根据以下方向继续进化：
- 增强约束的精准度
- 提升输出格式的严谨性
- 丰富思维链的引导深度
- 强化人设的专业感和权威性
"""
        else:
            evolution_context = "这是第一轮优化，请从零构建完美Prompt。"

        # 组装完整的进化请求
        evolution_request = f"""{evolution_context}

【用户原始粗糙提示词】：
"{current_prompt}"

请将其重构为完全体结构化Prompt。记住：只输出优化后的完整Prompt，不要任何额外说明。"""

        # 调用智谱API进行提示词进化
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.OPTIMIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": evolution_request},
                ],
                temperature=0.7,  # 适度创造性
                max_tokens=2048,  # 足够输出完整的结构化Prompt
            )
            evolved_prompt = response.choices[0].message.content.strip()
            return evolved_prompt
        except Exception as e:
            print(f"  ⚠️ [优化器API异常] 第{round_num}轮进化失败: {e}")
            # 容错回退：返回当前提示词的简单增强版
            return f"【紧急回退版本】\n<task>\n{current_prompt}\n</task>\n<constraints>\n请给出专业、详细的回答\n</constraints>"


# ============================== 提示词考官 ==============================
class PromptEvaluator:
    """
    📊 提示词考官：对优化后的Prompt进行自动化模拟打分
    评分维度：结构性、清晰度、约束力、人设感、输出控制力
    """

    # 考官的系统提示词 —— 模拟严苛的评审标准
    EVALUATOR_SYSTEM_PROMPT = """你是一位严苛的大模型提示词质量评估专家。
你需要对给定的提示词进行1-100分的综合评分，并给出简短的理由。

【评分维度 - 每项20分】：
1. 结构性 (20分)：是否使用了XML标签、层级分明、信息组织有序
2. 人设精准度 (20分)：专家身份定义是否清晰、权威、场景感强
3. 约束力 (20分)：边界条件是否明确、硬性限制是否到位、合规红线是否划定
4. 思维链引导 (20分)：是否有step-by-step工作流、推理路径是否清晰
5. 输出控制力 (20分)：输出格式是否锁死、结构化程度、可解析性

【输出格式 - 严格遵守】：
你只能输出以下格式，不要有任何额外文字：
SCORE: [1-100的整数]
REASON: [一句话简短理由，不超过30个字]"""

    def __init__(self, api_key: str):
        """初始化考官，建立智谱客户端连接"""
        self.client = ZhipuAI(api_key=api_key)
        self.model = "glm-4-flash"  # 使用同一模型进行评分（也可换成其他模型）

    def evaluate(self, prompt_text: str) -> Tuple[int, str]:
        """
        对给定的提示词进行自动化打分
        参数:
            prompt_text - 待评估的提示词全文
        返回:
            (得分: 1-100, 评分理由: 字符串)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"请对以下提示词进行评分：\n\n{prompt_text}"},
                ],
                temperature=0.3,  # 低温度，保证评分稳定性
                max_tokens=150,
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # 正则解析器：从考官输出中提取分数和理由
            score_match = re.search(r'SCORE:\s*(\d+)', raw_output, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.+)', raw_output, re.IGNORECASE)
            
            score = int(score_match.group(1)) if score_match else 50  # 解析失败默认50
            score = max(1, min(100, score))  # 强制裁剪到1-100区间
            
            reason = reason_match.group(1).strip() if reason_match else "评分解析异常"
            
            return score, reason
            
        except Exception as e:
            print(f"  ⚠️ [考官API异常] 评分失败: {e}")
            return 30, f"评分系统异常: {str(e)[:20]}"


# ============================== 全自动进化流水线 ==============================
class AutoEvolutionPipeline:
    """
    🧬 全自动提示词进化流水线
    工作流：初始Prompt → 优化器进化 → 考官打分 → 迭代N轮 → 锁死最优解
    """

    def __init__(self, api_key: str, max_rounds: int = 3):
        """
        初始化进化流水线
        参数:
            api_key - 智谱API密钥
            max_rounds - 最大进化轮次
        """
        self.optimizer = PromptOptimizer(api_key)
        self.evaluator = PromptEvaluator(api_key)
        self.max_rounds = max_rounds
        
        # 进化历史记录
        self.evolution_history: List[PromptVersion] = []

    def run_evolution(self, initial_prompt: str) -> PromptVersion:
        """
        执行全自动进化主循环
        参数:
            initial_prompt - 最初始的粗糙提示词
        返回:
            进化完成后的最优PromptVersion
        """
        print(f"\n{'='*60}")
        print(f"🧬 提示词全自动进化引擎启动")
        print(f"🔧 优化器: glm-4-flash (首席提示词架构师)")
        print(f"📊 考官: glm-4-flash (严苛质量评估专家)")
        print(f"🔄 进化轮次: {self.max_rounds}")
        print(f"📝 初始提示词: \"{initial_prompt}\"")
        print(f"{'='*60}\n")
        
        # 第0轮：先对初始提示词进行裸奔打分，建立基线
        print(f"📏 [基线测试] 对初始粗糙提示词进行裸奔打分...")
        initial_score, initial_reason = self.evaluator.evaluate(initial_prompt)
        
        baseline_version = PromptVersion(
            round=0,
            prompt_text=initial_prompt,
            score=initial_score,
            evolution_type="BASELINE"
        )
        self.evolution_history.append(baseline_version)
        
        print(f"  📊 初始基线得分: {initial_score}/100 | 原因: {initial_reason}\n")
        
        # 当前最优版本追踪器
        best_version = baseline_version
        current_prompt = initial_prompt
        
        # 主进化循环
        for round_num in range(1, self.max_rounds + 1):
            print(f"🧬 [第{round_num}轮进化] " + "=" * 40)
            
            # 阶段1：优化器进化提示词
            print(f"  🔧 优化器工作中... (注入首席架构师人设，结构化重构)")
            evolved_prompt = self.optimizer.evolve_prompt(
                current_prompt=current_prompt,
                round_num=round_num,
                previous_best=best_version if best_version.score < 95 else None
            )
            
            # 截断过长输出，防止终端刷屏
            display_prompt = evolved_prompt[:200] + "..." if len(evolved_prompt) > 200 else evolved_prompt
            print(f"  📝 进化产出预览: {display_prompt}")
            
            # 阶段2：考官对进化后的提示词打分
            print(f"  📊 考官评估中...")
            time.sleep(0.5)  # 短暂冷却，避免API限流
            score, reason = self.evaluator.evaluate(evolved_prompt)
            
            # 阶段3：记录本轮结果
            version = PromptVersion(
                round=round_num,
                prompt_text=evolved_prompt,
                score=score,
                evolution_type="STRUCTURED_EVOLUTION"
            )
            self.evolution_history.append(version)
            
            # 终端流式输出本轮战报
            improvement = score - best_version.score
            trend_icon = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"  {trend_icon} 本轮得分: {score}/100 | 相较最优提升: {improvement:+d}分")
            print(f"  💬 考官评语: {reason}")
            
            # 阶段4：更新最优版本
            if score > best_version.score:
                best_version = version
                print(f"  🏆 新最优版本诞生！锁定第{round_num}轮产出")
                current_prompt = evolved_prompt  # 以最优版本为基础继续进化
            elif score == best_version.score:
                print(f"  ➡️ 得分持平，保留当前最优版本")
                # 仍然尝试在最优基础上继续优化
                current_prompt = best_version.prompt_text
            else:
                print(f"  ⚠️ 得分下降，回退到最优版本继续尝试")
                current_prompt = best_version.prompt_text
            
            # 提前收敛检测：如果连续满分或接近满分，提前锁死
            if best_version.score >= 98:
                print(f"\n🎯 检测到接近满分 ({best_version.score}/100)，提前锁死最优解！")
                break
            
            print()  # 轮次间空行
        
        return best_version

    def generate_report(self, final_version: PromptVersion) -> str:
        """
        生成自动化迭代对账报告
        参数:
            final_version - 最终锁死的最优版本
        返回:
            格式化的报告字符串
        """
        baseline = self.evolution_history[0]
        final = final_version
        
        # 计算进化收益
        score_gain = final.score - baseline.score
        gain_percentage = (score_gain / baseline.score * 100) if baseline.score > 0 else 0
        
        # 进化轨迹字符串
        trajectory = " → ".join([f"R{v.round}:{v.score}" for v in self.evolution_history])
        
        report = f"""
{'='*60}
📊 Prompt自动优化报告
{'='*60}
  初始Prompt:  "{baseline.prompt_text[:60]}{'...' if len(baseline.prompt_text)>60 else ''}"
  进化轮次:    {self.max_rounds} 轮
  进化轨迹:    {trajectory}
{'='*60}
  初始基线得分:   {baseline.score}/100
  最终完全体得分: {final.score}/100
  进化收益:       +{score_gain} 分 ({gain_percentage:+.1f}%)
{'='*60}
  最终完全体 Prompt 已诞生，已就地锁死！
{'='*60}
  最优Prompt全文:
{'-'*60}
{final.prompt_text}
{'-'*60}
"""
        return report


# ============================== 主函数入口 ==============================
def main():
    """主函数：驱动全自动提示词进化引擎 —— 支持用户自定义输入"""

    print(f"\n{'='*60}")
    print(f"🧬 欢迎使用 Prompt 全自动进化引擎")
    print(f"{'='*60}")
    print(f"💡 请输入你想要优化的初始提示词（粗糙版本即可）")
    print(f"📝 示例: 帮我分析论文 / 写一个爬虫 / 翻译这段话")
    print(f"{'='*60}")
    
    # ========== 交互式输入：用户自定义提示词 ==========
    initial_prompt = input("\n🎯 请输入你的初始提示词:\n> ").strip()
    
    # 输入校验：防止空输入
    while not initial_prompt:
        print("⚠️ 提示词不能为空，请重新输入")
        initial_prompt = input("> ").strip()
    
    # 确认进化轮次
    try:
        rounds_input = input("\n🔄 请输入进化轮次 (默认3轮，直接回车跳过):\n> ").strip()
        max_rounds = int(rounds_input) if rounds_input else 3
        max_rounds = max(1, min(10, max_rounds))  # 限制在1-10轮之间
    except ValueError:
        print("⚠️ 输入无效，使用默认3轮")
        max_rounds = 3
    
    print(f"\n⚡ 确认配置: 提示词=\"{initial_prompt[:50]}{'...' if len(initial_prompt)>50 else ''}\" | 进化轮次={max_rounds}")
    confirm = input("🚀 确认启动进化引擎？(Y/n): ").strip().lower()
    
    if confirm and confirm != 'y':
        print("❌ 已取消操作")
        return None
    # =================================================
    
    # 初始化进化流水线
    pipeline = AutoEvolutionPipeline(
        api_key=ZHIPU_API_KEY,
        max_rounds=max_rounds
    )
    
    # 执行全自动进化
    final_version = pipeline.run_evolution(initial_prompt)
    
    # 生成并打印最终报告
    report = pipeline.generate_report(final_version)
    print(report)
    
    # 保存进化历史到JSON文件
    log_filename = f"prompt_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evolution_log = {
        "initial_prompt": initial_prompt,
        "total_rounds": pipeline.max_rounds,
        "final_score": final_version.score,
        "final_prompt": final_version.prompt_text,
        "history": [
            {
                "round": v.round,
                "score": v.score,
                "prompt_preview": v.prompt_text[:200],
                "evolution_type": v.evolution_type,
                "timestamp": v.timestamp,
            }
            for v in pipeline.evolution_history
        ],
    }
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(evolution_log, f, ensure_ascii=False, indent=2)
    print(f"📁 完整进化日志已保存至: {log_filename}")
    
    return final_version


if __name__ == "__main__":
    """
    使用方式:
        export ZHIPU_API_KEY="your_api_key_here"
        python prompt_optimizer/auto_tune.py
    
    依赖安装:
        pip install zhipuai
    
    运行后:
        - 终端会提示你输入初始提示词
        - 可选自定义进化轮次（默认3轮）
        - 流式打印每轮进化的得分和考官评语
        - 最终输出完全体Prompt全文
        - JSON进化日志自动保存
    """
    main()