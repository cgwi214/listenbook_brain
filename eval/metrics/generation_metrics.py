"""生成层指标：引用准确率 / 关键词准确率 / 字符级 F1 / 忠实度 / 答案相关性

为什么生成层不能照搬 P/R/F1
----------------------------
检索层有确定的"相关 / 不相关"二值判定，可以严格算 P、R、F1；
生成层的答案是自由文本，同一句话有无数种正确表达，
**严格 F1 会系统性低估生成质量**。所以这里用四种互补口径：

1. **引用准确率**（零标注成本）：答案末尾【来源】区块里引用的切片，
   有多少真的命中金标。答得好不好难判，但"引得对不对"是客观的。
2. **关键词准确率**：金标里标注 ``must_include`` / ``must_not_include``，
   前者全中且后者全不中记 1。这是最接近"人工判对/判错"的廉价替代。
3. **字符级 F1**：与参考答案做字级别的重叠 F1（中文按字切分，
   等价于 ROUGE-1 的字级版本）。只作为参考趋势，绝对值不宜苛求。
4. **忠实度 + 答案相关性**（可选，需调 LLM）：忠实度看答案有没有脱离上下文瞎编，
   相关性看答案有没有答到问题上。这两项对应 RAGAS 的 faithfulness / answer_relevancy。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .matcher import chunk_matches_any

logger = logging.getLogger(__name__)

# 计算字符级 F1 时忽略的字符（标点、空白）
_IGNORE_PATTERN = re.compile(r"[\s，。！？；：、（）()【】\[\]《》\"'“”‘’\-—…,.\?!;:]")


# ------------------------------------------------------------------
# 1. 引用准确率
# ------------------------------------------------------------------
def citation_accuracy(
    cited_chunks: Sequence[Dict[str, Any]],
    golden_locators: Sequence[Dict[str, Any]],
) -> Optional[float]:
    """引用准确率：被引用的切片里命中金标的比例。

    Args:
        cited_chunks: 答案【来源】区块实际引用的切片列表。
        golden_locators: 金标定位符列表。

    Returns:
        0.0 ~ 1.0；没有引用任何切片时返回 None（不参与平均，避免稀释）。
    """
    cited = list(cited_chunks or [])
    if not cited:
        return None

    hit = sum(1 for c in cited if chunk_matches_any(c, golden_locators))
    return hit / len(cited)


# ------------------------------------------------------------------
# 2. 关键词准确率
# ------------------------------------------------------------------
def keyword_accuracy(
    answer: str,
    must_include: Optional[Sequence[str]] = None,
    must_not_include: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """关键词准确率：必含词全中且禁含词全不中记 1，否则记 0。

    Args:
        answer: 生成的答案文本。
        must_include: 必须出现的关键词列表。
        must_not_include: 禁止出现的关键词列表。

    Returns:
        1.0 / 0.0；若金标未定义任何关键词则返回 None。
    """
    required = [k for k in (must_include or []) if str(k).strip()]
    forbidden = [k for k in (must_not_include or []) if str(k).strip()]

    if not required and not forbidden:
        return None

    text = (answer or "").lower()

    for kw in required:
        if str(kw).strip().lower() not in text:
            return 0.0
    for kw in forbidden:
        if str(kw).strip().lower() in text:
            return 0.0
    return 1.0


# ------------------------------------------------------------------
# 3. 字符级 F1
# ------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """中文按字切分，去掉标点空白。"""
    return [c for c in _IGNORE_PATTERN.sub("", text or "")]


def char_f1(answer: str, reference: str) -> Optional[float]:
    """字符级 F1（多重集重叠），衡量答案与参考答案的字面重合度。

    Args:
        answer: 生成的答案。
        reference: 金标参考答案。

    Returns:
        0.0 ~ 1.0；参考答案为空时返回 None。
    """
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return None

    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    ans_counter = Counter(ans_tokens)

    overlap = sum(min(ans_counter[c], ref_counter[c]) for c in ans_counter)

    precision = overlap / len(ans_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------
# 4. LLM 裁判：忠实度 / 答案相关性
# ------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """你是 RAG 系统的评测裁判。请判断下面的答案是否**忠实于给定的上下文**。

判定标准：
- 答案中的每个事实性陈述，都能在上下文中找到依据 → 忠实
- 出现上下文中没有的内容、凭空编造的细节、与上下文矛盾的表述 → 不忠实

【问题】
{question}

【上下文】
{context}

【答案】
{answer}

请只返回一个 JSON 对象，不要有任何其他文字：
{{"score": <0到1之间的小数，1表示完全忠实>, "reason": "<一句话说明原因>"}}
"""

_RELEVANCY_PROMPT = """你是 RAG 系统的评测裁判。请判断下面的答案是否**切题**。

判定标准：
- 直接回应了问题所问的内容 → 切题
- 答非所问、绕开核心问题、或者只是复述无关背景 → 不切题

【问题】
{question}

【答案】
{answer}

请只返回一个 JSON 对象，不要有任何其他文字：
{{"score": <0到1之间的小数，1表示完全切题>, "reason": "<一句话说明原因>"}}
"""


def _llm_judge(prompt: str, default: Optional[float] = None) -> Dict[str, Any]:
    """调用 LLM 打分，失败时返回默认分。

    Args:
        prompt: 完整提示词。
        default: 失败时的兜底分数，None 表示该项不计入统计。

    Returns:
        ``{"score": float|None, "reason": str}``。
    """
    try:
        from knowledge.utils.llm_client_util import get_llm_client

        client = get_llm_client(temperature=0.0)
        if client is None:
            return {"score": default, "reason": "LLM 客户端不可用"}

        resp = client.invoke(prompt)
        content = (getattr(resp, "content", "") or "").strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)

        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(content[start:end + 1])
            score = parsed.get("score")
            if score is not None:
                return {
                    "score": max(0.0, min(1.0, float(score))),
                    "reason": str(parsed.get("reason", ""))[:200],
                }
        return {"score": default, "reason": "返回结果无法解析"}
    except Exception as e:  # noqa: BLE001 —— 评测不应因为裁判失败而中断
        logger.warning(f"LLM 裁判调用失败: {e}")
        return {"score": default, "reason": f"调用失败: {e}"}


def faithfulness(
    question: str,
    answer: str,
    contexts: Sequence[str],
    max_context_chars: int = 3000,
) -> Dict[str, Any]:
    """忠实度：答案是否脱离上下文编造。需要调用 LLM。"""
    context_text = "\n\n---\n\n".join(c[:600] for c in (contexts or [])[:8])
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + " ...(已截断)"

    return _llm_judge(
        _FAITHFULNESS_PROMPT.format(
            question=question, context=context_text, answer=answer or ""
        )
    )


def answer_relevancy(question: str, answer: str) -> Dict[str, Any]:
    """答案相关性：答案是否切题。需要调用 LLM。"""
    return _llm_judge(
        _RELEVANCY_PROMPT.format(question=question, answer=answer or "")
    )


# ------------------------------------------------------------------
# 汇总
# ------------------------------------------------------------------
def _mean_ignore_none(values: Sequence[Optional[float]]) -> Optional[float]:
    """忽略 None 求均值，全为 None 时返回 None。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate_generation(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总生成层指标。

    Args:
        case_results: 含 ``generation`` 子字典的 case 结果列表。

    Returns:
        汇总字典，某项无数据时对应值为 None。
    """
    gens = [c.get("generation") or {} for c in (case_results or [])]
    if not gens:
        return {}

    citation = _mean_ignore_none([g.get("citation_accuracy") for g in gens])
    keyword = _mean_ignore_none([g.get("keyword_accuracy") for g in gens])
    f1 = _mean_ignore_none([g.get("char_f1") for g in gens])
    faith = _mean_ignore_none([g.get("faithfulness") for g in gens])
    relev = _mean_ignore_none([g.get("answer_relevancy") for g in gens])

    return {
        "citation_accuracy": round(citation, 4) if citation is not None else None,
        "keyword_accuracy": round(keyword, 4) if keyword is not None else None,
        "char_f1": round(f1, 4) if f1 is not None else None,
        "faithfulness": round(faith, 4) if faith is not None else None,
        "answer_relevancy": round(relev, 4) if relev is not None else None,
        "sample_count": {
            "citation": sum(1 for g in gens if g.get("citation_accuracy") is not None),
            "keyword": sum(1 for g in gens if g.get("keyword_accuracy") is not None),
            "char_f1": sum(1 for g in gens if g.get("char_f1") is not None),
            "faithfulness": sum(1 for g in gens if g.get("faithfulness") is not None),
            "relevancy": sum(1 for g in gens if g.get("answer_relevancy") is not None),
        },
    }
