"""评估主流程

一次完整评估分五步::

    1. 语料准备   检查 eval/item 下的材料是否已入库，缺哪个补导哪个
    2. 金标评估   逐条跑查询图 -> 钩子取 chunk_id -> 反查切片 -> 算 P/R/F1
    3. 反例评估   越界召回类算误召回率，库外问题类算拒答准确率
    4. 汇总       检索层 Micro/Macro、生成层四项、性能报表
    5. 落盘       reports/ 下存 JSON + Markdown

关键设计
--------
* **评测模式（eval_mode）**：跑查询图时置 ``eval_mode=True``，
  联网搜索节点会主动跳过。联网结果每次都不一样，不关掉的话
  同一份金标集跑两遍会得到两个分数，评估就失去意义了。
* **日志回调**：所有过程信息通过 ``log`` 回调往外推，
  命令行里打到 stdout，界面上推给 SSE，同一套代码两处复用。
* **失败不中断**：单条用例报错只记一条 error，继续跑下一条，
  避免一条脏数据让整轮评估白跑。
* **导入清单（import manifest）**：每次成功导入一个文件都会把
  ``{source_file_name, book_name, imported_at}`` 写入
  ``reports/_import_manifest.json``。这一份清单是 reset 的**权威来源**——
  即使 ``eval/item/`` 下的文件名被改了，旧导入的切片也能被
  ``reports/_import_manifest.json`` 找到并清理掉，不会变成"幽灵切片"
  把分数虚高。
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()

from knowledge.processor.import_process.main_graph import kb_import__graph_app
from knowledge.processor.query_process.main_graph import query_app
from knowledge.processor.query_process.config import get_config
from knowledge.utils.milvus_util import fetch_chunks_by_chunk_ids, get_milvus_client

from .hooks import attach_eval_trace, extract_chunk_ids
from .metrics import (
    aggregate,
    aggregate_generation,
    aggregate_negative,
    build_report_dict,
    char_f1,
    citation_accuracy,
    evaluate_case,
    evaluate_negative_case,
    faithfulness,
    answer_relevancy,
    keyword_accuracy,
    render_markdown_report,
    render_text_report,
)
from .perf import PerfTracer, register_tracer, unregister_tracer

# eval 包所在目录（knowledge/eval）
EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "dataset"
ITEM_DIR = EVAL_DIR / "item"
REPORT_DIR = EVAL_DIR / "reports"

# 导入清单：所有曾经被评估流程导入过的 source_file_name / book_name。
# reset 必须读这个文件才能正确清理被改名/被删除的文件残留。
IMPORT_MANIFEST_PATH = REPORT_DIR / "_import_manifest.json"

GOLDEN_SET_PATH = DATASET_DIR / "golden_set.jsonl"
NEGATIVE_SET_PATH = DATASET_DIR / "negative_set.jsonl"
CROSS_RECALL_SET_PATH = DATASET_DIR / "cross_recall_set.jsonl"

# 导入中间产物的落盘目录（与源文件目录分离，见 _import_one_file 的说明）
TEMP_IMPORT_DIR = Path(
    os.path.join(
        os.getenv("EVAL_TEMP_DIR", "")
        or os.path.abspath(os.path.join(EVAL_DIR, "..", "temp_data", "eval_import"))
    )
)

# 反查切片时需要取回的字段（必须与 Milvus 集合 schema 对齐）
CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "content", "title", "book_name", "author_name",
    "content_type", "category", "audio_duration", "entry_name",
    "source_file_name", "file_title",
]


# ------------------------------------------------------------------
# 错误归一化（让报表里的 error 字段短而清晰）
# ------------------------------------------------------------------
def distill_error(raw: str) -> str:
    """把 LangChain 报上来的超长异常文本压缩成一行可读字符串。

    典型输入::

        "[book_name_confirm_node] Error code: 403 - {'error': {'message': 'Free quota
        exhausted. To continue accessing the model on a paid basis, please add funds
        or disable the \"use free tier only\" mode ...', 'id': '...', 'type':
        'insufficient_quota', 'code': 'insufficient_quota'}} (原因: Error code: 403 - {...})"

    输出::

        "[book_name_confirm_node] 403 insufficient_quota — Free quota exhausted"
    """
    s = str(raw or "").strip()
    if not s:
        return ""

    # 1. 摘出节点名前缀 "[xxxx_node]"
    import re as _re
    node_match = _re.match(r"^\s*\[([^\]]+)\]", s)
    node = f"[{node_match.group(1)}] " if node_match else ""

    # 2. 摘 HTTP code
    code_match = _re.search(r"\b(\d{3})\b", s)
    http_code = code_match.group(1) if code_match else ""

    # 3. 摘 code 字段（"code": 'xxx' 或 "type": 'xxx'）
    code_field = ""
    for pat in (r"'code':\s*'([^']+)'", r'"code":\s*"([^"]+)"',
                r"'type':\s*'([^']+)'", r'"type":\s*"([^"]+)"'):
        m = _re.search(pat, s)
        if m:
            code_field = m.group(1)
            break

    # 4. 摘 message 字段（截断到句号，避免复读 Free quota 完整提示）
    msg = ""
    for pat in (r"'message':\s*'([^']+)'", r'"message":\s*"([^"]+)"'):
        m = _re.search(pat, s)
        if m:
            raw_msg = m.group(1).strip()
            # 取第一个句号前的内容
            period = raw_msg.find(". ")
            if period > 0:
                msg = raw_msg[: period + 1].strip()
            else:
                # 没有句号时硬截断 80 字
                msg = raw_msg[:80].strip()
                if len(raw_msg) > 80:
                    msg += "..."
            break

    # 5. 兜底：截前 80 字
    if not msg:
        msg = s[:80]

    # 6. 拼成一行
    head = node
    if http_code:
        head = f"{node}{http_code} "
    if code_field:
        head += f"{code_field} — "
    return (head + msg).strip()
def scan_item_files() -> List[Path]:
    """扫描 eval/item 下的全部可导入材料（按文件名排序）。"""
    if not ITEM_DIR.exists():
        return []
    suffixes = {".pdf", ".md", ".txt", ".html", ".htm", ".mp3"}
    return sorted(
        p for p in ITEM_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，自动跳过空行。"""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[eval] 跳过 {path.name} 第 {line_no} 行（JSON 解析失败）: {e}")
    return records


# ------------------------------------------------------------------
# 导入清单（manifest）
# ------------------------------------------------------------------
def load_import_manifest() -> List[Dict[str, Any]]:
    """读导入清单。文件不存在或损坏时返回空列表，绝不抛错。

    每条记录形如::

        {
          "source_file_name": "三国演义- 罗贯中_部分1",
          "file_name": "三国演义- 罗贯中_部分1.pdf",
          "book_name": "三国演义",
          "chunks": 16,
          "imported_at": "2026-09-05T14:35:43"
        }
    """
    if not IMPORT_MANIFEST_PATH.exists():
        return []
    try:
        with open(IMPORT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError) as e:
        print(f"[eval] 读取导入清单失败（{IMPORT_MANIFEST_PATH}）: {e}")
        return []


def save_import_manifest(records: List[Dict[str, Any]]) -> None:
    """把导入清单写回磁盘。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(IMPORT_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[eval] 写入导入清单失败: {e}")


def append_import_record(record: Dict[str, Any]) -> None:
    """追加一条导入记录，并按 ``source_file_name`` 去重（同文件再导只更新时间）。"""
    records = load_import_manifest()
    records = [r for r in records if r.get("source_file_name") != record.get("source_file_name")]
    records.append(record)
    # 按导入时间倒序
    records.sort(key=lambda r: r.get("imported_at") or "", reverse=True)
    save_import_manifest(records)


def manifest_source_names() -> List[str]:
    """从清单里提取全部 source_file_name（按字母排序、去重）。"""
    return sorted({r["source_file_name"] for r in load_import_manifest() if r.get("source_file_name")})


def manifest_book_names() -> List[str]:
    """从清单里提取全部 book_name（按字母排序、去重）。"""
    return sorted({r["book_name"] for r in load_import_manifest() if r.get("book_name")})


# ------------------------------------------------------------------
# Milvus 辅助
# ------------------------------------------------------------------
def _chunks_collection() -> str:
    return os.getenv("CHUNKS_COLLECTION", "kb_tingbook_chunks_v1")


def count_chunks_by_source(source_keyword: str) -> int:
    """统计某个来源文件已入库的切片数（用于判断是否需要补导）。"""
    client = get_milvus_client()
    if client is None:
        return -1

    collection = _chunks_collection()
    try:
        if not client.has_collection(collection):
            return 0
        # Milvus 的字符串过滤不支持 LIKE，这里用 == 精确匹配来源文件名
        expr = f'source_file_name == "{_escape_expr(source_keyword)}"'
        res = client.query(
            collection_name=collection,
            filter=expr,
            output_fields=["count(*)"],
        )
        if res and isinstance(res, list):
            return int(res[0].get("count(*)", 0))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[eval] 查询切片数量失败（{source_keyword}）: {e}")
        return -1


def _escape_expr(value: str) -> str:
    """转义 Milvus 过滤表达式里的双引号与反斜杠。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
class EvalRunner:
    """评估执行器。

    Args:
        log: 日志回调 ``f(message: str, level: str)``，level 取 info/warn/error/success。
    """

    def __init__(self, log: Optional[Callable[[str, str], None]] = None):
        self.log = log or (lambda msg, level="info": print(msg))
        self._tracer: Optional[PerfTracer] = None

    # ---------------- 日志 ----------------
    def _info(self, msg: str) -> None:
        self.log(msg, "info")

    def _ok(self, msg: str) -> None:
        self.log(msg, "success")

    def _warn(self, msg: str) -> None:
        self.log(msg, "warn")

    def _err(self, msg: str) -> None:
        self.log(msg, "error")

    # ---------------- 第一步：语料准备 ----------------
    def scan_item_files(self) -> List[Path]:
        """扫描 eval/item 下的全部可导入材料（委托给模块级同名函数）。"""
        return scan_item_files()

    def ensure_corpus(self, skip_if_exists: bool = True) -> Dict[str, Any]:
        """确保评测语料已入库，缺的文件自动补导。

        Args:
            skip_if_exists: 已存在同名来源切片时跳过该文件。

        Returns:
            ``{"total": n, "imported": [...], "skipped": [...], "failed": [...]}``。
        """
        files = self.scan_item_files()
        self._info(f"扫描到评测材料 {len(files)} 个（目录：{ITEM_DIR}）")

        result: Dict[str, Any] = {
            "total": len(files), "imported": [], "skipped": [], "failed": [],
        }

        for path in files:
            stem = path.stem
            if skip_if_exists:
                existing = count_chunks_by_source(stem)
                if existing > 0:
                    self._info(f"  · {stem} 已入库（{existing} 条切片），跳过导入")
                    result["skipped"].append({"file": path.name, "chunks": existing})
                    continue
                if existing == -1:
                    self._warn(f"  · {stem} 无法确认入库状态（Milvus 不可用），仍尝试导入")

            self._info(f"  · 开始导入 {path.name} ...")
            started = time.perf_counter()
            try:
                chunks = self._import_one_file(path)
                cost = (time.perf_counter() - started) * 1000
                self._ok(f"    √ {path.name} 导入完成，产生 {chunks} 条切片，耗时 {cost / 1000:.1f}s")
                result["imported"].append({"file": path.name, "cost_ms": round(cost, 1)})
                if self._tracer:
                    self._tracer.count("import_files", 1)
                    self._tracer.count("import_chunks", chunks)
            except Exception as e:  # noqa: BLE001
                self._err(f"    × {path.name} 导入失败：{e}")
                self.log(traceback.format_exc(), "error")
                result["failed"].append({"file": path.name, "error": str(e)})

        return result

    def _import_one_file(self, path: Path) -> int:
        """导入单个文件，返回切片数量。

        ``file_dir`` 是整个导入流水线的**输出目录**：PDF 解析、HTML 转 md、
        MP3 转写、切片备份（chunks.json）、图片抽取全部往这里写。
        这里刻意指向 ``temp_data/eval_import/`` 而不是源文件所在的 ``eval/item/``——
        否则下一次扫描语料时，转换出来的 .md 会和原始 .html/.mp3 一起被当成
        新语料重复导入。

        成功导入后会把 ``{source_file_name, file_name, book_name, chunks,
        imported_at}`` 追加到 ``reports/_import_manifest.json``，作为 reset 的
        权威清理依据（即使 ``eval/item/`` 下的文件被改名，旧记录也能被清掉）。
        """
        task_id = f"eval_import_{uuid.uuid4().hex[:12]}"
        file_dir = TEMP_IMPORT_DIR / path.stem
        file_dir.mkdir(parents=True, exist_ok=True)

        init_state = {
            "task_id": task_id,
            "file_dir": str(file_dir),
            "import_file_path": str(path),
            # 元数据留空，交给 book_info_recognition_node 的 LLM 识别
            "book_name": "", "author_name": "", "content_type": "",
            "category": "", "audio_duration": "", "entry_name": "", "source_url": "",
        }

        final_state: Dict[str, Any] = {}
        for event in kb_import__graph_app.stream(init_state):
            for _node_name, node_state in event.items():
                if isinstance(node_state, dict):
                    final_state.update(node_state)

        chunks = final_state.get("chunks") or []
        chunk_count = len(chunks)

        # ---- 写入导入清单（reset 的权威来源）----
        # 取第一条 chunk 的 book_name 作为整本书的归属；空时回退为 file_stem
        sample_book = ""
        if chunks and isinstance(chunks[0], dict):
            sample_book = (chunks[0].get("book_name") or "").strip()
        append_import_record({
            "source_file_name": path.stem,
            "file_name": path.name,
            "book_name": sample_book or path.stem,
            "chunks": chunk_count,
            "imported_at": datetime.now().isoformat(timespec="seconds"),
        })

        return chunk_count

    # ---------------- 查询执行 ----------------
    def run_one_query(
        self,
        query: str,
        role: str = "listener",
        with_generation: bool = True,
        enable_llm_judge: bool = False,
    ) -> Dict[str, Any]:
        """跑一条 query，返回召回切片 / 答案 / 引用来源 / 耗时。

        Args:
            query: 问题文本。
            role: 提问角色。
            with_generation: 是否调用答案生成（False 可只测检索层，省时间）。
            enable_llm_judge: 是否调用 LLM 判忠实度与相关性。

        Returns:
            ``{"retrieved": [...], "answer": str, "cited": [...], "cost_ms": float}``。
        """
        task_id = f"eval_q_{uuid.uuid4().hex[:12]}"
        session_id = f"eval_sess_{uuid.uuid4().hex[:12]}"

        state = {
            "original_query": query,
            "session_id": session_id,
            "task_id": task_id,
            "is_stream": False,
            "role": role,
            "eval_mode": True,   # 关闭联网召回，保证结果可复现
        }

        started = time.perf_counter()
        result = query_app.invoke(state)
        cost_ms = (time.perf_counter() - started) * 1000

        # 兜底：若 rerank 节点未写入（例如书名确认阶段就短路返回了），这里再抽一次
        chunk_ids = result.get("eval_chunk_ids")
        if chunk_ids is None:
            chunk_ids = extract_chunk_ids(result.get("reranked_docs") or [])

        retrieved = fetch_chunks_by_chunk_ids(
            collection_name=_chunks_collection(),
            chunk_ids=chunk_ids,
            output_fields=CHUNK_OUTPUT_FIELDS,
        ) if chunk_ids else []

        # 保持精排顺序（Milvus 的 get 不保证按传入顺序返回）
        order = {cid: i for i, cid in enumerate(chunk_ids)}
        retrieved.sort(key=lambda c: order.get(c.get("chunk_id"), 10 ** 6))

        # 答案【来源】区块引用的切片 = 精排后进入上下文的全部切片
        cited = retrieved

        out: Dict[str, Any] = {
            "retrieved": retrieved,
            "cited": cited,
            "answer": result.get("answer", "") or "",
            "cost_ms": round(cost_ms, 2),
            "reranked_count": len(result.get("reranked_docs") or []),
            "book_names": result.get("book_names", []),
            "rewritten_query": result.get("rewritten_query", ""),
        }

        if with_generation and enable_llm_judge and out["answer"]:
            contexts = [c.get("content", "") for c in cited]
            out["faithfulness"] = faithfulness(query, out["answer"], contexts)
            out["relevancy"] = answer_relevancy(query, out["answer"])

        return out

    # ---------------- 第二步：金标集评估 ----------------
    def run_golden_set(
        self,
        top_k: Optional[int] = None,
        roles: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        enable_llm_judge: bool = False,
    ) -> Dict[str, Any]:
        """跑完整个金标集。"""
        cases = load_jsonl(GOLDEN_SET_PATH)
        if roles:
            cases = [c for c in cases if c.get("role", "listener") in roles]
        if limit:
            cases = cases[:limit]

        self._info(f"金标集共 {len(cases)} 条用例，开始逐条评估（截断口径："
                   f"{'Top-' + str(top_k) if top_k else '全部召回'}）")

        case_results: List[Dict[str, Any]] = []

        for index, case in enumerate(cases, start=1):
            qid = case.get("qid", f"G{index:03d}")
            query = case.get("query", "")
            role = case.get("role", "listener")
            golden = case.get("golden") or []

            self._info(f"[{index}/{len(cases)}] ({qid}) [{role}] {query}")

            try:
                run = self.run_one_query(
                    query, role=role, enable_llm_judge=enable_llm_judge
                )
            except Exception as e:  # noqa: BLE001
                self._err(f"      × 查询执行失败：{e}")
                case_results.append({
                    "qid": qid, "query": query, "role": role, "error": distill_error(str(e)),
                    "retrieved_count": 0, "golden_count": len(golden),
                    "hit_count": 0, "covered_golden": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "hit_rate": 0.0, "mrr": 0.0, "strict_accuracy": 0.0,
                })
                continue

            metrics = evaluate_case(qid, query, run["retrieved"], golden, k=top_k)
            metrics["role"] = role
            metrics["cost_ms"] = run["cost_ms"]

            # ---- 生成层 ----
            gen: Dict[str, Any] = {}
            gen["citation_accuracy"] = citation_accuracy(run["cited"], golden)
            gen["keyword_accuracy"] = keyword_accuracy(
                run["answer"],
                case.get("must_include"),
                case.get("must_not_include"),
            )
            gen["char_f1"] = char_f1(run["answer"], case.get("reference_answer", ""))
            if enable_llm_judge:
                gen["faithfulness"] = (run.get("faithfulness") or {}).get("score")
                gen["answer_relevancy"] = (run.get("relevancy") or {}).get("score")
            metrics["generation"] = gen
            metrics["answer_preview"] = run["answer"][:150]

            case_results.append(metrics)

            self._info(
                f"      召回 {metrics['retrieved_count']} 条 / 命中 {metrics['hit_count']} 条 | "
                f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                f"F1={metrics['f1']:.2f} | {run['cost_ms'] / 1000:.1f}s"
            )

        summary = aggregate(case_results, k=top_k)
        return {"cases": case_results, "summary": summary}

    # ---------------- 第三步：反例集评估 ----------------
    def run_negative_set(
        self,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """跑完整个反例集。"""
        cases = load_jsonl(NEGATIVE_SET_PATH)
        if limit:
            cases = cases[:limit]

        self._info(f"反例集共 {len(cases)} 条用例，开始评估")

        results: List[Dict[str, Any]] = []

        for index, case in enumerate(cases, start=1):
            qid = case.get("qid", f"N{index:03d}")
            query = case.get("query", "")
            forbidden = case.get("forbidden") or []
            expect_abstain = bool(case.get("expect_abstain", False))

            self._info(f"[{index}/{len(cases)}] ({qid}) "
                       f"{'[库外拒答]' if expect_abstain else '[越界召回]'} {query}")

            try:
                run = self.run_one_query(query, role=case.get("role", "listener"))
            except Exception as e:  # noqa: BLE001
                self._err(f"      × 查询执行失败：{e}")
                results.append({
                    "qid": qid, "query": query, "error": distill_error(str(e)),
                    "type": "abstain" if expect_abstain else "cross_recall",
                    "retrieved_count": 0, "forbidden_hit_count": 0,
                })
                continue

            metrics = evaluate_negative_case(
                qid, query, run["retrieved"], forbidden,
                answer=run["answer"], k=top_k, expect_abstain=expect_abstain,
            )
            results.append(metrics)

            if expect_abstain:
                self._info(f"      拒答准确率={metrics.get('abstention_accuracy', 0):.2f} "
                           f"| 答案：{(run['answer'] or '')[:60]}")
            else:
                self._info(f"      召回 {metrics['retrieved_count']} 条 / "
                           f"越界 {metrics['forbidden_hit_count']} 条 | "
                           f"误召回率={metrics.get('false_recall_rate') or 0:.2f}")

        return {"cases": results, "summary": aggregate_negative(results)}

    # ---------------- 越界召回集评估 ----------------
    def run_cross_recall_set(
        self,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """跑完越界召回集（问 A 书，检验是否误召回 B 书切片）。"""
        cases = load_jsonl(CROSS_RECALL_SET_PATH)
        if limit:
            cases = cases[:limit]

        self._info(f"越界召回集共 {len(cases)} 条用例，开始评估")

        results: List[Dict[str, Any]] = []

        for index, case in enumerate(cases, start=1):
            qid = case.get("qid", f"C{index:03d}")
            query = case.get("query", "")
            forbidden = case.get("forbidden") or []

            self._info(f"[{index}/{len(cases)}] ({qid}) [越界召回] {query}")

            try:
                run = self.run_one_query(query, role=case.get("role", "listener"))
            except Exception as e:  # noqa: BLE001
                self._err(f"      × 查询执行失败：{e}")
                results.append({
                    "qid": qid, "query": query, "error": distill_error(str(e)),
                    "type": "cross_recall",
                    "retrieved_count": 0, "forbidden_hit_count": 0,
                })
                continue

            metrics = evaluate_negative_case(
                qid, query, run["retrieved"], forbidden,
                answer=run["answer"], k=top_k, expect_abstain=False,
            )
            results.append(metrics)

            self._info(f"      召回 {metrics['retrieved_count']} 条 / "
                       f"越界 {metrics['forbidden_hit_count']} 条 | "
                       f"误召回率={metrics.get('false_recall_rate') or 0:.2f}")

        return {"cases": results, "summary": aggregate_negative(results)}

    # ---------------- 主入口 ----------------
    def run(
        self,
        top_k: Optional[int] = None,
        roles: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        enable_llm_judge: bool = False,
        skip_import: bool = False,
        with_generation: bool = True,
        hyde_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """执行一轮完整评估。

        Returns:
            完整报告字典（同时已落盘）。
        """
        started = time.perf_counter()
        self._tracer = register_tracer(f"eval_{uuid.uuid4().hex[:12]}")

        self._info("=" * 72)
        self._info("听书知识库 · RAG 评估开始")
        self._info("=" * 72)

        # ---- 1. 语料准备 ----
        with self._tracer.stage("1.语料准备"):
            if skip_import:
                self._warn("已指定跳过语料导入，直接开始评估（知识库为空时分数会很难看）")
                corpus_info: Dict[str, Any] = {"skipped_all": True}
            else:
                corpus_info = self.ensure_corpus()

        # ---- 2. 金标集 ----
        with self._tracer.stage("2.金标集评估"):
            golden = self.run_golden_set(
                top_k=top_k, roles=roles, limit=limit,
                enable_llm_judge=enable_llm_judge,
            )

        # ---- 3. 越界召回集 ----
        cross_recall = {"cases": [], "summary": {}}
        if with_generation:
            with self._tracer.stage("3.越界召回集评估"):
                cross_recall = self.run_cross_recall_set(top_k=top_k, limit=limit)

        # ---- 4. 反例集（库外拒答） ----
        negative = {"cases": [], "summary": {}}
        if with_generation:
            with self._tracer.stage("4.反例集评估"):
                negative = self.run_negative_set(top_k=top_k, limit=limit)

        # ---- 5. 汇总 ----
        self._tracer.finish()
        perf_summary = self._tracer.summary()
        unregister_tracer(self._tracer.task_id)

        retrieval_summary = golden["summary"]
        generation_summary = aggregate_generation(golden["cases"]) if with_generation else {}
        negative_summary = negative.get("summary", {})
        cross_recall_summary = cross_recall.get("summary", {})

        report = build_report_dict(
            retrieval_summary=retrieval_summary,
            generation_summary=generation_summary,
            negative_summary=negative_summary,
            cross_recall_summary=cross_recall_summary,
            perf_summary=perf_summary,
            cases=golden["cases"],
            config={
                "截断口径": "Top-" + str(top_k) if top_k else "全部召回",
                "角色筛选": "/".join(roles) if roles else "全部",
                "LLM 裁判": "开启" if enable_llm_judge else "关闭",
                "联网召回": "已关闭（评测模式）",
                "HyDE 检索": "开启" if get_config().hyde_enabled else "关闭（A/B 对照）",
                "语料导入": f"新增 {len(corpus_info.get('imported', []))} / "
                            f"跳过 {len(corpus_info.get('skipped', []))} / "
                            f"失败 {len(corpus_info.get('failed', []))}",
            },
        )
        report["corpus"] = corpus_info
        report["negative_cases"] = negative.get("cases", [])
        report["cross_recall_cases"] = cross_recall.get("cases", [])

        # ---- 5. 落盘 ----
        total_cost = (time.perf_counter() - started) * 1000
        report["total_cost_ms"] = round(total_cost, 2)
        self._save_reports(report)

        # ---- 输出报表 ----
        self._info("")
        self._info(render_text_report(report))
        self._ok(f"评估完成，总耗时 {total_cost / 1000:.1f}s，报告已保存到 {REPORT_DIR}")

        return report

    # ---------------- 落盘 ----------------
    def _save_reports(self, report: Dict[str, Any]) -> None:
        """把报告写成 JSON 与 Markdown 两个文件。"""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = REPORT_DIR / f"eval_report_{stamp}.json"
        md_path = REPORT_DIR / f"eval_report_{stamp}.md"
        latest_md = REPORT_DIR / "latest.md"

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:  # noqa: BLE001
            self._warn(f"JSON 报告落盘失败：{e}")

        try:
            md_text = render_markdown_report(report)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            with open(latest_md, "w", encoding="utf-8") as f:
                f.write(md_text)
        except Exception as e:  # noqa: BLE001
            self._warn(f"Markdown 报告落盘失败：{e}")

        self._info(f"报告文件：{json_path.name} / {md_path.name}")
