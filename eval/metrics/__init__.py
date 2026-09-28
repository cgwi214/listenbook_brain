"""指标计算子包

分层设计::

    matcher.py            金标定位符 <-> 切片 的匹配规则
    retrieval_metrics.py  检索层：P@K / R@K / F1@K / Macro-F1 / HitRate / MRR
    generation_metrics.py 生成层：引用准确率 / 关键词准确率 / 字符级 F1 / 忠实度 / 相关性
    negative_metrics.py   反例集：误召回率 / 拒答准确率
    report.py             报表渲染（文本 / JSON / Markdown）
"""

from .matcher import chunk_matches_any, chunk_matches_locator, covered_locators
from .retrieval_metrics import (
    evaluate_case,
    aggregate,
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    strict_accuracy,
    list_zero_hit_cases,
    list_low_precision_cases,
)
from .generation_metrics import (
    citation_accuracy,
    keyword_accuracy,
    char_f1,
    faithfulness,
    answer_relevancy,
    aggregate_generation,
)
from .negative_metrics import (
    evaluate_negative_case,
    aggregate_negative,
    abstention_accuracy,
    false_recall_rate,
)
from .report import build_report_dict, render_text_report, render_markdown_report

__all__ = [
    "chunk_matches_any", "chunk_matches_locator", "covered_locators",
    "evaluate_case", "aggregate", "precision_at_k", "recall_at_k", "f1_at_k",
    "hit_rate_at_k", "reciprocal_rank", "strict_accuracy",
    "list_zero_hit_cases", "list_low_precision_cases",
    "citation_accuracy", "keyword_accuracy", "char_f1",
    "faithfulness", "answer_relevancy", "aggregate_generation",
    "evaluate_negative_case", "aggregate_negative",
    "abstention_accuracy", "false_recall_rate",
    "build_report_dict", "render_text_report", "render_markdown_report",
]
