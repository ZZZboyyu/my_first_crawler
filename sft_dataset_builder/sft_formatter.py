#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
工业级大模型监督微调（SFT）JSONL 数据集自动清洗与格式化转换系统
【论文段落→QA自动生成版】
===============================================================================
功能：
  1. 环境变量安全配置（ZHIPU_API_KEY）
  2. 读取TXT论文段落，自动拆分为QA问答对
  3. SFT格式转换流水线（多轮对话三元组）
  4. 物理级JSONL写入（一行一JSON）
  5. 数据质量量化质检（Token阈值、空值检测）
  
作者：AI特种兵
日期：2026-07-01
版本：v2.0.0 - 论文段落版
"""

import os
import json
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib


# ============================================================================
# 第一部分：环境变量安全配置
# ============================================================================

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")

if not ZHIPU_API_KEY:
    print("⚠️  [安全警告] ZHIPU_API_KEY 未在环境变量中设置！")
    print("💡 请执行: export ZHIPU_API_KEY='your_api_key_here'")
    print("🚀 系统将以离线规则模式运行（基于NLP规则抽取QA）\n")
else:
    masked_key = ZHIPU_API_KEY[:8] + "*" * (len(ZHIPU_API_KEY) - 12) + ZHIPU_API_KEY[-4:]
    print(f"✅ [安全审计] ZHIPU_API_KEY 已加载: {masked_key}\n")


# ============================================================================
# 第二部分：数据模型定义
# ============================================================================

class MessageRole(str, Enum):
    """消息角色枚举 - 严格遵循OpenAI/智谱官方标准"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class SFTMessage:
    """单条消息数据结构"""
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}
    
    def validate(self) -> Tuple[bool, str]:
        if not self.content or not self.content.strip():
            return False, f"角色 '{self.role}' 的content为空"
        if self.role not in [r.value for r in MessageRole]:
            return False, f"非法角色类型: {self.role}"
        return True, "OK"


@dataclass
class SFTDataPoint:
    """单条SFT数据点"""
    messages: List[SFTMessage]
    
    def to_sft_format(self) -> Dict[str, Any]:
        return {"messages": [msg.to_dict() for msg in self.messages]}
    
    def estimate_tokens(self) -> int:
        total_chars = sum(len(msg.content) for msg in self.messages)
        return int(total_chars / 3.0)


# ============================================================================
# 第三部分：论文段落解析与QA自动生成引擎
# ============================================================================

class PaperParagraphParser:
    """
    论文段落解析器 - 将纯文本论文段落拆分为结构化段落
    识别章节标题、段落边界、关键术语
    """
    
    def __init__(self):
        # 论文常见章节标题模式
        self.section_patterns = [
            r'^(?:第[一二三四五六七八九十\d]+章|[一二三四五六七八九十\d]+[\.、．])',
            r'^(?:摘要|Abstract|引言|Introduction|相关工作|Related Work|方法|Method|实验|Experiment|结论|Conclusion|参考文献|Reference)',
            r'^\d+\.\d*\s',  # 1.1 1.2 编号
            r'^[A-Z][A-Za-z\s]+$',  # 全大写英文标题
        ]
        
        # 关键术语提取模式（中英文混合）
        self.key_term_patterns = [
            r'(?:提出|定义|称为|即|所谓)(?:了|的)?([^，。,\.；;]+?(?:模型|算法|方法|架构|机制|理论|定理|框架|网络|系统))',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,5})',  # 英文专有名词
            r'「([^」]+)」',  # 引号内术语
            r'"([^"]+)"',
        ]
    
    def parse_txt_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        读取并解析TXT论文文件
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            结构化段落列表 [{"text": "...", "keywords": [...], "section": "..."}, ...]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"论文文件不存在: {file_path}")
        
        print(f"📖 正在读取论文文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        print(f"📊 文件大小: {len(raw_text)} 字符\n")
        
        # Step 1: 按双换行拆分段落（保留章节结构）
        raw_paragraphs = re.split(r'\n\s*\n', raw_text)
        
        # Step 2: 清洗并结构化每个段落
        structured_paragraphs = []
        for para in raw_paragraphs:
            para = para.strip()
            if len(para) < 30:  # 跳过太短的段落（标题、图表说明等）
                continue
            
            # 识别章节
            section = self._detect_section(para)
            
            # 提取关键术语
            keywords = self._extract_keywords(para)
            
            structured_paragraphs.append({
                "text": para,
                "keywords": keywords,
                "section": section,
                "char_count": len(para)
            })
        
        print(f"✅ 解析完成: {len(structured_paragraphs)} 个有效段落")
        print(f"📝 识别章节: {set(p['section'] for p in structured_paragraphs if p['section'])}\n")
        
        return structured_paragraphs
    
    def _detect_section(self, text: str) -> str:
        """检测段落所属章节"""
        first_line = text.split('\n')[0].strip()
        for pattern in self.section_patterns:
            if re.match(pattern, first_line):
                return first_line[:30]
        return "正文"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从段落中提取关键术语"""
        keywords = set()
        for pattern in self.key_term_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                keyword = match.strip()
                if 2 <= len(keyword) <= 50:  # 过滤太短或太长的
                    keywords.add(keyword)
        return list(keywords)[:5]  # 最多5个关键词


class QAGenerator:
    """
    论文段落→QA问答对生成器
    基于规则模板 + 关键词抽取，自动生成3类问题
    """
    
    def __init__(self):
        # 问题模板库（按论文学术场景分类）
        self.question_templates = {
            "定义型": [
                "什么是{term}？请给出详细定义。",
                "{term}的概念是什么？它有哪些核心特征？",
                "如何理解{term}？请用简洁的语言解释。",
                "{term}的定义是什么？它在文中的具体含义是什么？",
            ],
            "原理解释型": [
                "{term}的工作原理是什么？请详细阐述。",
                "{term}是如何实现的？涉及哪些关键技术？",
                "请解释{term}的机制和流程。",
                "{term}的核心思想是什么？为什么它有效？",
            ],
            "对比应用型": [
                "{term}相比传统方法有什么优势？",
                "{term}在什么场景下适用？有哪些局限性？",
                "{term}与相关技术的主要区别是什么？",
                "在实际应用中，{term}面临哪些挑战？如何解决？",
            ]
        }
        
        # 答案抽取模板
        self.answer_patterns = [
            r'{term}(?:是|指|即|为|定义为)([^。]+。)',  # 定义句
            r'{term}(?:的|可以|能够|通过)([^。]+。)',  # 功能句
            r'{term}(?:具有|拥有|包括|包含)([^。]+。)',  # 特征句
        ]
    
    def generate_qa_pairs(self, paragraph: Dict[str, Any], max_pairs: int = 3) -> List[Dict[str, str]]:
        """
        从单个段落生成QA问答对
        
        Args:
            paragraph: 结构化段落 {"text": "...", "keywords": [...], ...}
            max_pairs: 每个段落最多生成多少对QA
            
        Returns:
            [{"question": "...", "answer": "..."}, ...]
        """
        text = paragraph["text"]
        keywords = paragraph.get("keywords", [])
        
        if not keywords:
            # 如果没有提取到关键词，使用段落首句主题
            first_sentence = text.split('。')[0]
            if len(first_sentence) > 10:
                # 提取主语作为伪关键词
                subject_match = re.search(r'([^，,]{2,20})(?:是|的|可以|能够|具有)', first_sentence)
                if subject_match:
                    keywords = [subject_match.group(1).strip()]
        
        if not keywords:
            return []
        
        qa_pairs = []
        used_terms = set()
        
        for term in keywords:
            if term in used_terms or len(qa_pairs) >= max_pairs:
                continue
            
            # 选择一个问题类型（轮流使用不同类型）
            question_type = list(self.question_templates.keys())[len(qa_pairs) % 3]
            templates = self.question_templates[question_type]
            
            # 随机选择一个模板
            template = templates[hash(f"{term}{len(qa_pairs)}") % len(templates)]
            question = template.format(term=term)
            
            # 抽取答案
            answer = self._extract_answer(text, term)
            if not answer:
                answer = self._generate_fallback_answer(text, term)
            
            if answer and len(answer) > 10:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
                used_terms.add(term)
        
        return qa_pairs
    
    def _extract_answer(self, text: str, term: str) -> Optional[str]:
        """从段落中抽取包含关键词的句子作为答案"""
        # 转义特殊字符
        escaped_term = re.escape(term)
        
        for pattern in self.answer_patterns:
            full_pattern = pattern.format(term=escaped_term)
            match = re.search(full_pattern, text)
            if match:
                answer = match.group(1).strip()
                if len(answer) > 15:
                    return answer
        
        # 如果没有匹配到模板，找包含关键词的最长句子
        sentences = re.split(r'[。！？]', text)
        best_sentence = None
        max_length = 0
        
        for sentence in sentences:
            if term in sentence and len(sentence) > max_length:
                max_length = len(sentence)
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence and len(best_sentence) > 15 else None
    
    def _generate_fallback_answer(self, text: str, term: str) -> str:
        """兜底策略：使用段落前150字作为答案"""
        return text[:150].strip() + "..."


# ============================================================================
# 第四部分：SFT格式转换流水线
# ============================================================================

class SFTFormatter:
    """SFT格式转换器"""
    
    MAX_TOKEN_THRESHOLD = 2048
    
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt or "你是一个专业的AI学术助手，请基于论文内容准确、详细地回答学术问题。"
        self.conversion_stats = {
            "total_input": 0,
            "success": 0,
            "skipped_empty": 0,
            "skipped_too_long": 0,
            "skipped_invalid": 0
        }
    
    def to_sft_format(self, qa_pairs: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        将QA对转换为SFT标准三元组格式
        支持单轮和多轮对话
        """
        self.conversion_stats["total_input"] += 1
        
        try:
            messages = [SFTMessage(role="system", content=self.system_prompt)]
            
            for pair in qa_pairs:
                question = pair.get("question", "").strip()
                answer = pair.get("answer", "").strip()
                
                if not question or not answer:
                    self.conversion_stats["skipped_empty"] += 1
                    return None
                
                messages.append(SFTMessage(role="user", content=question))
                messages.append(SFTMessage(role="assistant", content=answer))
            
            data_point = SFTDataPoint(messages=messages)
            
            # Token阈值检测
            estimated_tokens = data_point.estimate_tokens()
            if estimated_tokens > self.MAX_TOKEN_THRESHOLD:
                self.conversion_stats["skipped_too_long"] += 1
                return None
            
            self.conversion_stats["success"] += 1
            return data_point.to_sft_format()
            
        except Exception as e:
            print(f"  ❌ 转换异常: {str(e)}")
            self.conversion_stats["skipped_invalid"] += 1
            return None
    
    def print_stats(self):
        """打印转换统计"""
        total = self.conversion_stats["total_input"]
        success = self.conversion_stats["success"]
        print(f"\n📊 格式转换统计:")
        print(f"  输入: {total} | 成功: {success} | 跳过: {total - success}")


# ============================================================================
# 第五部分：JSONL物理级写入器
# ============================================================================

class JSONLWriter:
    """JSONL格式写入器"""
    
    def __init__(self, output_path: str = "sft_dataset.jsonl"):
        self.output_path = output_path
        self.written_count = 0
    
    def write_batch(self, data_list: List[Dict[str, Any]]) -> int:
        """批量写入JSONL"""
        if not data_list:
            print("⚠️  警告: 数据列表为空，跳过写入")
            return 0
        
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for idx, data in enumerate(data_list):
                    if not data:
                        continue
                    
                    json_line = json.dumps(
                        data,
                        ensure_ascii=False,
                        indent=None,
                        separators=(',', ':')
                    )
                    
                    f.write(json_line + '\n')
                    self.written_count += 1
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  📝 已写入 {idx + 1}/{len(data_list)} 条...")
            
            print(f"\n✅ JSONL文件已生成: {self.output_path}")
            print(f"📦 文件大小: {os.path.getsize(self.output_path) / 1024:.2f} KB")
            return self.written_count
            
        except IOError as e:
            print(f"❌ 文件写入失败: {str(e)}")
            raise


# ============================================================================
# 第六部分：数据质量量化质检引擎
# ============================================================================

class DataAuditor:
    """数据质量审计器"""
    
    def __init__(self):
        self.audit_results = {
            "total": 0, "passed": 0,
            "failed_empty": 0, "failed_token_limit": 0,
            "failed_role_check": 0, "failed_structure": 0
        }
        self.TOKEN_LIMIT = 2048
    
    def audit_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量审计"""
        print("\n" + "=" * 60)
        print("🔍 启动数据质量量化质检...")
        print("=" * 60)
        
        clean_data = []
        for idx, data in enumerate(data_list):
            self.audit_results["total"] += 1
            
            # 结构检查
            if "messages" not in data or not data["messages"]:
                self.audit_results["failed_structure"] += 1
                print(f"  ❌ [第{idx+1}条] 结构异常")
                continue
            
            # 逐条消息验证
            valid_roles = {"system", "user", "assistant"}
            total_length = 0
            is_valid = True
            
            for msg in data["messages"]:
                if not isinstance(msg, dict):
                    is_valid = False
                    self.audit_results["failed_structure"] += 1
                    break
                
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role not in valid_roles:
                    self.audit_results["failed_role_check"] += 1
                    is_valid = False
                    break
                
                if not content or not content.strip():
                    self.audit_results["failed_empty"] += 1
                    is_valid = False
                    break
                
                total_length += len(content)
            
            if not is_valid:
                print(f"  ❌ [第{idx+1}条] 校验失败")
                continue
            
            # Token检查
            estimated_tokens = int(total_length / 3.0)
            if estimated_tokens > self.TOKEN_LIMIT:
                self.audit_results["failed_token_limit"] += 1
                print(f"  ❌ [第{idx+1}条] Token超标: {estimated_tokens}")
                continue
            
            self.audit_results["passed"] += 1
            clean_data.append(data)
        
        self._print_report()
        return clean_data
    
    def _print_report(self):
        """打印质检报告"""
        total = self.audit_results["total"]
        passed = self.audit_results["passed"]
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 SFT数据集质检报告")
        print("=" * 60)
        print(f"  总样本数:        {total}")
        print(f"  ✅ 合格:          {passed}")
        print(f"  ❌ 空值淘汰:      {self.audit_results['failed_empty']}")
        print(f"  ❌ Token超限:     {self.audit_results['failed_token_limit']}")
        print(f"  ❌ 角色异常:      {self.audit_results['failed_role_check']}")
        print(f"  ❌ 结构损坏:      {self.audit_results['failed_structure']}")
        print(f"  📈 合格率:        {pass_rate:.1f}%")
        print("=" * 60)
        
        if pass_rate >= 95.0:
            print(f"🎉 成功导出 {passed} 条黄金微调数据，合格率 {pass_rate:.1f}%，已就地锁死！")
        else:
            print(f"⚠️  合格率 {pass_rate:.1f}%，建议检查数据源质量")
        print("=" * 60 + "\n")


# ============================================================================
# 第七部分：主控流水线
# ============================================================================

def main():
    """
    主控流水线 - 从TXT论文到SFT JSONL的一条龙服务
    """
    print("=" * 60)
    print("🚀 工业级SFT数据集自动清洗与格式化转换系统 v2.0.0")
    print("    【论文段落→QA生成版】")
    print("=" * 60)
    print(f"📅 执行日期: 2026-07-01 (星期三)")
    print(f"🔑 API状态: {'已配置' if ZHIPU_API_KEY else '离线规则模式'}")
    print("=" * 60 + "\n")
    
    # ==============================
    # Phase 1: 读取TXT论文文件
    # ==============================
    print("=" * 60)
    print("📥 Phase 1: 读取论文TXT文件")
    print("=" * 60)
    
    # 【配置】在这里指定你的TXT文件路径！
    TXT_FILE_PATH = "paper.txt"  # <--- 改成你的文件路径
    
    if not os.path.exists(TXT_FILE_PATH):
        print(f"\n❌ 找不到文件: {TXT_FILE_PATH}")
        print("💡 请将论文TXT文件放在同目录下，或修改 TXT_FILE_PATH 变量\n")
        
        # 创建演示文件
        print("📝 正在创建演示论文文件 'paper.txt' 供测试...")
        demo_paper = """
Transformer: 一种基于自注意力机制的神经网络架构

摘要
Transformer是一种完全基于注意力机制的神经网络架构，由Vaswani等人在2017年提出。它摒弃了传统的循环神经网络和卷积神经网络结构，通过自注意力机制实现了序列到序列的并行计算。

引言
在自然语言处理领域，序列建模长期被RNN和LSTM主导。然而这些模型存在无法并行计算、长距离依赖难以捕捉等问题。Transformer的提出彻底改变了这一局面。

核心机制
Transformer的核心是自注意力机制。该机制通过计算Query、Key、Value三个矩阵，实现对输入序列的全局建模。具体来说，对于输入序列中的每个位置，模型会计算它与所有其他位置的注意力权重，然后加权求和得到输出表示。

多头注意力
多头注意力机制允许模型在不同的表示子空间中学习不同的注意力模式。每个注意力头独立计算注意力，最后将所有头的结果拼接起来。这种设计使得模型能够同时关注输入的不同方面。

位置编码
由于Transformer没有循环结构，它需要通过位置编码来保留序列信息。原始论文使用正弦和余弦函数来生成位置编码，这些编码与词嵌入相加后输入模型。

实验结论
在WMT 2014英德翻译任务上，Transformer取得了28.4的BLEU分数，超越了当时所有其他模型。更重要的是，它的训练速度比传统模型快了一个数量级。
"""
        with open(TXT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(demo_paper)
        print(f"✅ 演示文件已创建: {TXT_FILE_PATH}\n")
    
    # 解析论文
    parser = PaperParagraphParser()
    paragraphs = parser.parse_txt_file(TXT_FILE_PATH)
    
    # ==============================
    # Phase 2: 段落→QA自动生成
    # ==============================
    print("\n" + "=" * 60)
    print("🤖 Phase 2: 论文段落→QA问答对自动生成")
    print("=" * 60)
    
    qa_generator = QAGenerator()
    all_qa_pairs = []
    
    for i, para in enumerate(paragraphs):
        print(f"\n  处理段落 {i+1}/{len(paragraphs)}:")
        print(f"    关键词: {para.get('keywords', [])}")
        
        qa_pairs = qa_generator.generate_qa_pairs(para, max_pairs=3)
        if qa_pairs:
            all_qa_pairs.append(qa_pairs)
            print(f"    生成 {len(qa_pairs)} 对QA")
            for qa in qa_pairs:
                print(f"      Q: {qa['question'][:50]}...")
        else:
            print(f"    ⚠️  未能生成QA对（关键词不足）")
    
    print(f"\n✅ QA生成完成: {len(all_qa_pairs)} 个对话组")
    
    # ==============================
    # Phase 3: SFT格式转换
    # ==============================
    print("\n" + "=" * 60)
    print("🔄 Phase 3: SFT格式转换")
    print("=" * 60)
    
    formatter = SFTFormatter(
        system_prompt="你是一个精通人工智能和深度学习的学术专家，请基于论文内容，用专业且清晰的语言回答问题。"
    )
    
    converted_data = []
    for i, qa_pairs in enumerate(all_qa_pairs):
        result = formatter.to_sft_format(qa_pairs)
        if result:
            converted_data.append(result)
            print(f"  ✅ 第{i+1}组转换成功")
    
    formatter.print_stats()
    
    # ==============================
    # Phase 4: 数据质量审计
    # ==============================
    print("\n" + "=" * 60)
    print("🔍 Phase 4: 数据质量量化质检")
    print("=" * 60)
    
    auditor = DataAuditor()
    clean_data = auditor.audit_batch(converted_data)
    
    # ==============================
    # Phase 5: JSONL物理写入
    # ==============================
    print("=" * 60)
    print("💾 Phase 5: JSONL物理级写入")
    print("=" * 60)
    
    writer = JSONLWriter(output_path="sft_dataset.jsonl")
    written_count = writer.write_batch(clean_data)
    
    # ==============================
    # Phase 6: 写入后验证
    # ==============================
    print("\n" + "=" * 60)
    print("🔬 Phase 6: 写入后逐行验证")
    print("=" * 60)
    
    try:
        with open("sft_dataset.jsonl", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"  文件行数: {len(lines)}")
            
            # 展示第一条数据
            if lines:
                sample = json.loads(lines[0].strip())
                print(f"\n  📋 样例数据:")
                print(f"    对话轮数: {len(sample['messages'])}")
                for msg in sample['messages']:
                    content_preview = msg['content'][:80].replace('\n', ' ')
                    print(f"    [{msg['role']}] {content_preview}...")
            
            # 验证所有行
            for i, line in enumerate(lines):
                try:
                    json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"  ❌ 第{i+1}行JSON解析失败！")
                    break
            else:
                print(f"\n  ✅ 所有 {len(lines)} 行JSON格式验证通过！")
    except FileNotFoundError:
        print("  ❌ 输出文件未找到！")
    
    # ==============================
    # 最终报告
    # ==============================
    print("\n" + "=" * 60)
    print("🏁 流水线执行完毕")
    print("=" * 60)
    print(f"  论文段落数:      {len(paragraphs)}")
    print(f"  生成对话组:      {len(all_qa_pairs)}")
    print(f"  格式转换通过:    {len(converted_data)}")
    print(f"  审计通过:        {len(clean_data)}")
    print(f"  最终写入JSONL:   {written_count}")
    print(f"  输出文件:        sft_dataset.jsonl")
    print("=" * 60)
    print("🎯 系统已就绪，可直接对接大模型训练框架！")
    print("=" * 60)


if __name__ == "__main__":
    main()