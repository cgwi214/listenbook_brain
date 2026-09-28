"""检索层指标：P@K / R@K / F1@K / Macro-F1 / HitRate@K / MRR

口径说明（答辩时容易被追问，这里写清楚）
----------------------------------------
* **召回集合的口径**：取 rerank 精排 + 断崖截断**之后**的切片列表，
  即真正拼进 Prompt 的那一批，而不是向量库原始 Top-N。
* **K 的取值**：默认 ``K = None`` 表示"取全部召回结果"；
  指定 K 时截断到前 K 条。
* **精确率 P@K** = 命中的切片数 / min(K, 实际召回数)。
  分母用实际召回数而非 K，避免"库里本来就只有 3 条"被误判成低精确率。
* **召回率 R@K** = 命中的**金标定位符数** / 金标定位符总数。
  用定位符数而非切片数，防止"一条金标被召回 5 个相邻切片"把召回率刷满。
* **F1@K** = 2·P·R / (P + R)，P、R 均为 0 时 F1 记 0。
* **Macro-F1** = 所有 query 的 F1 的**算术平均**（每条 query 等权）。
* **Micro-F1** = 先汇总 TP / FP / FN 再算 F1（每条切片等权）。
  金标条数差异大时，Macro 与 Micro 会明显分叉，两个都要看。
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

from .matcher import covered_locators, find_hit_positions


def _safe_div(numerator: float, denominator: float) -> float:
    """安全除法，分母为 0 时返回 0.0。"""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """由 P、R 计算 F1，两者都为 0 时返回 0。"""
    return _safe_div(2 * precision * recall, precision + recall)


def precision_at_k(
    retrieved: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> float:
    """精确率：召回的切片里有多少是真正相关的。

    Args:
        retrieved: 有序召回切片列表（rerank 之后）。
        golden_locators: 金标定位符列表。
        k: 截断位置，None 表示不截断。

    Returns:
        0.0 ~ 1.0。
    """
    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]
    if not docs:
        return 0.0

    hits = len(find_hit_positions(docs, golden_locators))
    return _safe_div(hits, len(docs))


def recall_at_k(
    retrieved: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> float:
    """召回率：金标切片里有多少被召回了。

    Args:
        retrieved: 有序召回切片列表。
        golden_locators: 金标定位符列表。
        k: 截断位置，None 表示不截断。

    Returns:
        0.0 ~ 1.0。
    """
    locators = list(golden_locators or [])
    if not locators:
        return 0.0

    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]

    return _safe_div(len(covered_locators(docs, locators)), len(locators))


def f1_at_k(
    retrieved: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> float:
    """F1：精确率与召回率的调和平均。"""
    return _f1(
        precision_at_k(retrieved, golden_locators, k),
        recall_at_k(retrieved, golden_locators, k),
    )


def hit_rate_at_k(
    retrieved: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> float:
    """命中率：Top-K 里至少命中一条金标记 1，否则记 0。"""
    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]
    if not golden_locators:
        return 0.0
    return 1.0 if find_hit_positions(docs, golden_locators) else 0.0


def reciprocal_rank(
    retrieved: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> float:
    """倒数排名：第一条命中的金标所在的位次取倒数（1/rank），未命中记 0。"""
    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]
    positions = find_hit_positions(docs, golden_locators)
    if not positions:
        return 0.0
    return 1.0 / (positions[0] + 1)


def strict_accuracy(
    retrieved: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> float:
    """严格准确率：召回结果**全部命中金标**且至少命中一条才记 1。

    这是个非常苛刻的指标——只要混进一条噪声就归零，
    用来观察"检索结果纯净度"的上限，通常数值会明显低于 P@K。
    """
    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]
    if not docs or not golden_locators:
        return 0.0

    hits = find_hit_positions(docs, golden_locators)
    return 1.0 if hits and len(hits) == len(docs) else 0.0


# ------------------------------------------------------------------
# 单条 case 评估
# ------------------------------------------------------------------
def evaluate_case(
    qid: str,
    query: str,
    retrieved: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """评估单条 query 的检索质量。

    Args:
        qid: 用例编号。
        query: 原始问题。
        retrieved: 有序召回切片列表（含 content / book_name / source_file_name 等字段）。
        golden_locators: 金标定位符列表。
        k: 截断位置。
        extra: 附加到结果里的额外信息（如角色、耗时）。

    Returns:
        单条 case 的指标字典。
    """
    docs = list(retrieved or [])
    if k is not None:
        docs = docs[:k]

    p = precision_at_k(docs, golden_locators, k)
    r = recall_at_k(docs, golden_locators, k)
    f1 = _f1(p, r)

    hits = find_hit_positions(docs, golden_locators)

    result = {
        "qid": qid,
        "query": query,
        "retrieved_count": len(docs),
        "golden_count": len(golden_locators or []),
        "hit_count": len(hits),
        "covered_golden": len(covered_locators(docs, golden_locators)),
        "first_hit_rank": (hits[0] + 1) if hits else None,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "hit_rate": round(hit_rate_at_k(docs, golden_locators, k), 4),
        "mrr": round(reciprocal_rank(docs, golden_locators, k), 4),
        "strict_accuracy": round(strict_accuracy(docs, golden_locators, k), 4),
    }
    if extra:
        result.update(extra)
    return result


# ------------------------------------------------------------------
# 汇总
# ------------------------------------------------------------------
def aggregate(
    case_results: Sequence[Dict[str, Any]],
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """把多条 case 汇总成 Micro / Macro 两套指标。

    Args:
        case_results: ``evaluate_case`` 的返回值列表。
        k: 本次评估使用的截断位置，仅用于报告标注。

    Returns:
        汇总指标字典，含 micro_*、macro_*、以及分布统计。
    """
    cases = list(case_results or [])
    if not cases:
        return {
            "case_count": 0,
            "k": k,
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "hit_rate": 0.0, "mrr": 0.0, "strict_accuracy": 0.0,
        }

    # ---- Micro：先累加 TP / FP / FN 再算 ----
    tp = fp = fn = 0
    for c in cases:
        tp += c.get("hit_count", 0)
        fp += max(c.get("retrieved_count", 0) - c.get("hit_count", 0), 0)
        fn += max(c.get("golden_count", 0) - c.get("covered_golden", 0), 0)

    micro_p = _safe_div(tp, tp + fp)
    micro_r = _safe_div(tp, tp + fn)
    micro_f1 = _f1(micro_p, micro_r)

    # ---- Macro：逐条算完再平均（每条 query 等权）----
    macro_p = statistics.fmean([c.get("precision", 0.0) for c in cases])
    macro_r = statistics.fmean([c.get("recall", 0.0) for c in cases])
    macro_f1 = statistics.fmean([c.get("f1", 0.0) for c in cases])

    return {
        "case_count": len(cases),
        "k": k,
        "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "hit_rate": round(statistics.fmean([c.get("hit_rate", 0.0) for c in cases]), 4),
        "mrr": round(statistics.fmean([c.get("mrr", 0.0) for c in cases]), 4),
        "strict_accuracy": round(statistics.fmean([c.get("strict_accuracy", 0.0) for c in cases]), 4),
        "avg_retrieved": round(statistics.fmean([c.get("retrieved_count", 0) for c in cases]), 2),
        "zero_hit_cases": sum(1 for c in cases if c.get("hit_count", 0) == 0),
    }


def list_zero_hit_cases(case_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """列出完全没召回的 case（调参时优先看这批）。"""
    return [
        {"qid": c.get("qid"), "query": c.get("query"),
         "retrieved_count": c.get("retrieved_count", 0)}
        for c in (case_results or [])
        if c.get("hit_count", 0) == 0
    ]


def list_low_precision_cases(
    case_results: Sequence[Dict[str, Any]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """列出精确率低于阈值的 case（噪声重灾区）。"""
    return [
        {"qid": c.get("qid"), "query": c.get("query"),
         "precision": c.get("precision", 0.0), "recall": c.get("recall", 0.0)}
        for c in (case_results or [])
        if c.get("retrieved_count", 0) > 0 and c.get("precision", 1.0) < threshold
    ]
