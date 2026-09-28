"""反例集指标：误召回率 / 拒答准确率

反例集回答的是"**系统什么时候不该答**"，这是检索层 P/R/F1 照不到的盲区。
分两类：

1. **越界召回类**：问题明确指向某本书 A，结果召回了书 B 的切片。
   指标是**误召回率** = 命中禁用定位符的切片数 / 召回总数。
2. **库外问题类**：知识库里根本没有答案（例如"今天东莞天气怎么样"）。
   期望系统明确说"没有找到"，指标是**拒答准确率** = 答案包含拒绝表述记 1。

两类都越低/越高越好，混在一起看会互相掩盖，所以分开统计。
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

from .matcher import chunk_matches_any, find_hit_positions

# 常见的"知识库里没有"表述，用于判定系统是否诚实拒答
ABSTAIN_PATTERNS = (
    "没有找到", "未找到", "知识库中没有", "知识库中未",
    "没有相关", "暂无相关", "无法回答", "无法确定",
    "没有检索到", "未检索到", "不存在相关", "没有收录",
    "我不确定您指的是哪本书", "不确定您指的是哪本书",
    "您想了解的是以下书籍吗", "请确认您想了解的是哪本书",
)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def false_recall_rate(
    retrieved: Sequence[Dict[str, Any]],
    forbidden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> Optional[float]:
    """误召回率：召回的切片里有多少命中了"不该出现"的定位符。

    Args:
        retrieved: 有序召回切片列表。
        forbidden_locators: 禁用定位符列表。
        k: 截断位置。

    Returns:
        0.0 ~ 1.0（越低越好）；召回为空时返回 None。
    """
    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]
    if not docs:
        return None

    hits = sum(1 for c in docs if chunk_matches_any(c, forbidden_locators))
    return _safe_div(hits, len(docs))


def abstention_accuracy(answer: str) -> float:
    """拒答准确率：库外问题是否给出了明确的"没有找到"表述。

    Args:
        answer: 生成的答案。

    Returns:
        1.0 表示正确拒答，0.0 表示系统在瞎编。
    """
    text = (answer or "").strip()
    if not text:
        return 0.0
    return 1.0 if any(p in text for p in ABSTAIN_PATTERNS) else 0.0


def evaluate_negative_case(
    qid: str,
    query: str,
    retrieved: Sequence[Dict[str, Any]],
    forbidden_locators: Sequence[Dict[str, Any]],
    answer: str = "",
    k: Optional[int] = None,
    expect_abstain: bool = False,
) -> Dict[str, Any]:
    """评估单条反例。

    Args:
        qid: 用例编号。
        query: 反例问题。
        retrieved: 有序召回切片列表。
        forbidden_locators: 禁用定位符列表。
        answer: 生成的答案。
        k: 截断位置。
        expect_abstain: True 表示这是"库外问题"，期望系统拒答。

    Returns:
        单条反例的指标字典。
    """
    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]

    hit_positions = find_hit_positions(docs, forbidden_locators)

    result: Dict[str, Any] = {
        "qid": qid,
        "query": query,
        "type": "abstain" if expect_abstain else "cross_recall",
        "retrieved_count": len(docs),
        "forbidden_hit_count": len(hit_positions),
        "false_recall_rate": round(false_recall_rate(docs, forbidden_locators, k) or 0.0, 4)
        if docs else None,
    }

    if expect_abstain:
        result["abstention_accuracy"] = abstention_accuracy(answer)
        result["answer_preview"] = (answer or "")[:100]

    return result


def aggregate_negative(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总反例集指标（两类分开统计）。"""
    cases = list(case_results or [])
    if not cases:
        return {"case_count": 0}

    cross = [c for c in cases if c.get("type") == "cross_recall"]
    abstain = [c for c in cases if c.get("type") == "abstain"]

    result: Dict[str, Any] = {"case_count": len(cases)}

    if cross:
        frr = [c.get("false_recall_rate") for c in cross if c.get("false_recall_rate") is not None]
        result["cross_recall_count"] = len(cross)
        result["mean_false_recall_rate"] = round(statistics.fmean(frr), 4) if frr else None
        result["clean_cases"] = sum(1 for c in cross if c.get("forbidden_hit_count", 0) == 0)
        result["contaminated_cases"] = sum(1 for c in cross if c.get("forbidden_hit_count", 0) > 0)

    if abstain:
        result["abstain_count"] = len(abstain)
        result["abstention_accuracy"] = round(
            statistics.fmean([c.get("abstention_accuracy", 0.0) for c in abstain]), 4
        )

    return result
