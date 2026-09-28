"""评估报告渲染

输出三种形态：
1. ``render_text_report`` —— 人类可读的纯文本报表（推给前端日志框）
2. ``build_report_dict`` —— 结构化字典（落盘 JSON，便于做趋势对比）
3. ``render_markdown_report`` —— Markdown 报表（落盘存档，可贴进项目报告）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from ..perf.timer import render_perf_report


def _pct(value: Optional[float]) -> str:
    """把 0~1 的小数渲染成百分比字符串，None 渲染成 "-"。"""
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


# ------------------------------------------------------------------
# 结构化报告
# ------------------------------------------------------------------
def build_report_dict(
    retrieval_summary: Dict[str, Any],
    generation_summary: Optional[Dict[str, Any]] = None,
    negative_summary: Optional[Dict[str, Any]] = None,
    cross_recall_summary: Optional[Dict[str, Any]] = None,
    perf_summary: Optional[Dict[str, Any]] = None,
    cases: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """组装完整评估报告字典。"""
    from .retrieval_metrics import list_low_precision_cases, list_zero_hit_cases

    case_list = list(cases or [])
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "retrieval": retrieval_summary,
        "generation": generation_summary or {},
        "negative": negative_summary or {},
        "cross_recall": cross_recall_summary or {},
        "perf": perf_summary or {},
        "diagnostics": {
            "zero_hit_cases": list_zero_hit_cases(case_list),
            "low_precision_cases": list_low_precision_cases(case_list),
        },
        "cases": case_list,
    }


# ------------------------------------------------------------------
# 文本报表
# ------------------------------------------------------------------
def render_text_report(report: Dict[str, Any]) -> str:
    """把报告字典渲染成可读文本。"""
    lines: List[str] = []
    bar = "=" * 84

    retrieval = report.get("retrieval") or {}
    generation = report.get("generation") or {}
    negative = report.get("negative") or {}
    cross_recall = report.get("cross_recall") or {}
    perf = report.get("perf") or {}
    diagnostics = report.get("diagnostics") or {}

    k = retrieval.get("k")
    k_label = "全部召回" if k is None else f"Top-{k}"

    lines.append(bar)
    lines.append("听书知识库 · RAG 评估报告")
    lines.append(bar)
    lines.append(f"生成时间 : {report.get('generated_at', '')}")
    cfg = report.get("config") or {}
    if cfg:
        for key, value in cfg.items():
            lines.append(f"{key:<9} : {value}")
    lines.append(f"用例数   : {retrieval.get('case_count', 0)} 条     截断口径 : {k_label}")

    # ---------------- 检索层 ----------------
    lines.append("")
    lines.append("【一、检索层指标】(口径：rerank 精排 + 断崖截断之后的最终上下文)")
    lines.append("-" * 84)
    lines.append(f"{'指标':<22}{'Micro(按切片加权)':>22}{'Macro(按用例等权)':>22}{'':>18}")
    lines.append("-" * 84)
    lines.append(
        f"{'精确率 Precision':<22}"
        f"{_pct(retrieval.get('micro_precision')):>22}"
        f"{_pct(retrieval.get('macro_precision')):>22}"
    )
    lines.append(
        f"{'召回率 Recall':<22}"
        f"{_pct(retrieval.get('micro_recall')):>22}"
        f"{_pct(retrieval.get('macro_recall')):>22}"
    )
    lines.append(
        f"{'F1 值':<22}"
        f"{_pct(retrieval.get('micro_f1')):>22}"
        f"{_pct(retrieval.get('macro_f1')):>22}"
    )
    lines.append("-" * 84)
    lines.append(f"{'命中率 HitRate':<22}{_pct(retrieval.get('hit_rate')):>22}")
    lines.append(f"{'倒数排名 MRR':<22}{_num(retrieval.get('mrr')):>22}")
    lines.append(f"{'严格准确率':<22}{_pct(retrieval.get('strict_accuracy')):>22}"
                 f"   (召回全对才记 1，衡量纯净度上限)")
    lines.append("-" * 84)
    lines.append(
        f"混淆矩阵汇总：TP={retrieval.get('tp', 0)}  FP={retrieval.get('fp', 0)}  "
        f"FN={retrieval.get('fn', 0)}"
    )
    lines.append(
        f"平均召回条数：{retrieval.get('avg_retrieved', 0)}       "
        f"零召回用例：{retrieval.get('zero_hit_cases', 0)} 条"
    )

    # ---------------- 生成层 ----------------
    if generation:
        lines.append("")
        lines.append("【二、生成层指标】(严格 F1 不适用于自由文本，改用四项互补口径)")
        lines.append("-" * 84)
        rows = [
            ("引用准确率", generation.get("citation_accuracy"),
             generation.get("sample_count", {}).get("citation"),
             "答案【来源】引用的切片命中金标的比例，零标注成本"),
            ("关键词准确率", generation.get("keyword_accuracy"),
             generation.get("sample_count", {}).get("keyword"),
             "金标 must_include 全中且 must_not_include 全不中记 1"),
            ("字符级 F1", generation.get("char_f1"),
             generation.get("sample_count", {}).get("char_f1"),
             "与参考答案的字面重合度，仅看趋势不看绝对值"),
            ("忠实度", generation.get("faithfulness"),
             generation.get("sample_count", {}).get("faithfulness"),
             "LLM 判定：答案是否脱离上下文编造"),
            ("答案相关性", generation.get("answer_relevancy"),
             generation.get("sample_count", {}).get("relevancy"),
             "LLM 判定：答案是否切题"),
        ]
        lines.append(f"{'指标':<16}{'得分':>12}{'样本数':>10}   说明")
        lines.append("-" * 84)
        for name, value, count, desc in rows:
            lines.append(f"{name:<16}{_pct(value):>12}{str(count or 0):>10}   {desc}")

    # ---------------- 越界召回集 ----------------
    if cross_recall and cross_recall.get("case_count"):
        lines.append("")
        lines.append("【三、越界召回集指标】(问 A 书却召回 B 书切片 = 书名过滤失效)")
        lines.append("-" * 84)
        lines.append(f"越界召回用例：{cross_recall.get('case_count')} 条")
        lines.append(
            f"  平均误召回率 {_pct(cross_recall.get('mean_false_recall_rate'))}   "
            f"干净 {cross_recall.get('clean_cases', 0)} 条 / "
            f"污染 {cross_recall.get('contaminated_cases', 0)} 条（越低越好）"
        )

    # ---------------- 反例集（库外拒答） ----------------
    if negative and negative.get("case_count"):
        lines.append("")
        lines.append("【四、反例集指标 · 库外拒答】(知识库真没有，应明确说没找到)")
        lines.append("-" * 84)
        lines.append(f"库外问题用例：{negative.get('case_count')} 条")
        if negative.get("abstain_count"):
            lines.append(
                f"  拒答准确率 {_pct(negative.get('abstention_accuracy'))}   "
                f"（答案包含“没有找到/未检索到”等表述记 1，越高越好）"
            )

    # ---------------- 语料导入 ----------------
    corpus = report.get("corpus") or {}
    if corpus and not corpus.get("skipped_all"):
        lines.append("")
        lines.append("【语料导入明细】")
        lines.append("-" * 84)
        if corpus.get("imported"):
            lines.append("  新增：")
            for it in corpus["imported"]:
                lines.append(
                    f"    · {it.get('file')}（耗时 {it.get('cost_ms', 0) / 1000:.1f}s）"
                )
        if corpus.get("skipped"):
            lines.append("  跳过（已存在）：")
            for it in corpus["skipped"]:
                lines.append(
                    f"    · {it.get('file')}（{it.get('chunks', 0)} 条切片）"
                )
        if corpus.get("failed"):
            lines.append("  失败：")
            for it in corpus["failed"]:
                lines.append(
                    f"    · {it.get('file')} — {it.get('error', '')[:120]}"
                )

    # ---------------- 性能 ----------------
    if perf:
        lines.append("")
        lines.append(render_perf_report(perf))

    # ---------------- 诊断 ----------------
    zero_hits = diagnostics.get("zero_hit_cases") or []
    low_prec = diagnostics.get("low_precision_cases") or []

    if zero_hits:
        lines.append("")
        lines.append("【诊断】完全没召回的用例（优先排查）")
        lines.append("-" * 84)
        for c in zero_hits[:20]:
            lines.append(f"  [{c.get('qid')}] {c.get('query')}  (召回 {c.get('retrieved_count')} 条)")

    if low_prec:
        lines.append("")
        lines.append("【诊断】精确率低于 50% 的用例（噪声重灾区）")
        lines.append("-" * 84)
        for c in low_prec[:20]:
            lines.append(
                f"  [{c.get('qid')}] {c.get('query')}  "
                f"P={_pct(c.get('precision'))} R={_pct(c.get('recall'))}"
            )

    lines.append("")
    lines.append(bar)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Markdown 报表
# ------------------------------------------------------------------
def render_markdown_report(report: Dict[str, Any]) -> str:
    """把报告字典渲染成 Markdown。"""
    retrieval = report.get("retrieval") or {}
    generation = report.get("generation") or {}
    negative = report.get("negative") or {}
    cross_recall = report.get("cross_recall") or {}

    lines: List[str] = []
    lines.append("# 听书知识库 RAG 评估报告")
    lines.append("")
    lines.append(f"- 生成时间：{report.get('generated_at', '')}")
    cfg = report.get("config") or {}
    for key, value in cfg.items():
        lines.append(f"- {key}：{value}")
    lines.append("")

    lines.append("## 一、检索层指标")
    lines.append("")
    lines.append("| 指标 | Micro（按切片加权） | Macro（按用例等权） |")
    lines.append("|---|---|---|")
    lines.append(f"| 精确率 Precision | {_pct(retrieval.get('micro_precision'))} | "
                 f"{_pct(retrieval.get('macro_precision'))} |")
    lines.append(f"| 召回率 Recall | {_pct(retrieval.get('micro_recall'))} | "
                 f"{_pct(retrieval.get('macro_recall'))} |")
    lines.append(f"| F1 值 | {_pct(retrieval.get('micro_f1'))} | "
                 f"{_pct(retrieval.get('macro_f1'))} |")
    lines.append("")
    lines.append(f"- 命中率 HitRate：**{_pct(retrieval.get('hit_rate'))}**")
    lines.append(f"- 倒数排名 MRR：**{_num(retrieval.get('mrr'))}**")
    lines.append(f"- 严格准确率：**{_pct(retrieval.get('strict_accuracy'))}**")
    lines.append(f"- TP / FP / FN：{retrieval.get('tp', 0)} / {retrieval.get('fp', 0)} / "
                 f"{retrieval.get('fn', 0)}")
    lines.append("")

    if generation:
        lines.append("## 二、生成层指标")
        lines.append("")
        lines.append("| 指标 | 得分 | 样本数 |")
        lines.append("|---|---|---|")
        sc = generation.get("sample_count", {})
        lines.append(f"| 引用准确率 | {_pct(generation.get('citation_accuracy'))} | "
                     f"{sc.get('citation', 0)} |")
        lines.append(f"| 关键词准确率 | {_pct(generation.get('keyword_accuracy'))} | "
                     f"{sc.get('keyword', 0)} |")
        lines.append(f"| 字符级 F1 | {_pct(generation.get('char_f1'))} | "
                     f"{sc.get('char_f1', 0)} |")
        lines.append(f"| 忠实度 | {_pct(generation.get('faithfulness'))} | "
                     f"{sc.get('faithfulness', 0)} |")
        lines.append(f"| 答案相关性 | {_pct(generation.get('answer_relevancy'))} | "
                     f"{sc.get('relevancy', 0)} |")
        lines.append("")

    if cross_recall and cross_recall.get("case_count"):
        lines.append("## 三、越界召回集指标")
        lines.append("")
        lines.append(f"- 越界召回用例：{cross_recall.get('case_count')} 条")
        lines.append(f"- 平均误召回率：**{_pct(cross_recall.get('mean_false_recall_rate'))}**"
                     f"（干净 {cross_recall.get('clean_cases', 0)} 条 / "
                     f"污染 {cross_recall.get('contaminated_cases', 0)} 条）")
        lines.append("")

    if negative and negative.get("case_count"):
        lines.append("## 四、反例集指标 · 库外拒答")
        lines.append("")
        lines.append(f"- 库外问题用例：{negative.get('case_count')} 条")
        if negative.get("abstain_count"):
            lines.append(f"- 库外拒答类 {negative['abstain_count']} 条，"
                         f"拒答准确率 **{_pct(negative.get('abstention_accuracy'))}**")
        lines.append("")

    # ---------------- 语料导入 ----------------
    corpus = report.get("corpus") or {}
    if corpus and not corpus.get("skipped_all"):
        lines.append("## 五、语料导入明细")
        lines.append("")
        if corpus.get("imported"):
            lines.append("**新增**：")
            for it in corpus["imported"]:
                lines.append(
                    f"- `{it.get('file')}`（耗时 {it.get('cost_ms', 0) / 1000:.1f}s）"
                )
        if corpus.get("skipped"):
            lines.append("")
            lines.append("**跳过（已存在）**：")
            for it in corpus["skipped"]:
                lines.append(
                    f"- `{it.get('file')}`（{it.get('chunks', 0)} 条切片）"
                )
        if corpus.get("failed"):
            lines.append("")
            lines.append("**失败**：")
            for it in corpus["failed"]:
                lines.append(
                    f"- `{it.get('file')}` — {it.get('error', '')[:200]}"
                )
        lines.append("")

    return "\n".join(lines)
