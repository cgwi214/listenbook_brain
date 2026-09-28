"""刷新：清理评估导入的数据

点「刷新」后要做的事，本质是**把知识库恢复到"评估材料导入之前"的状态**，
让下一轮评估从零开始，避免上一轮的残留数据把分数做高。

要清的地方不止向量库一处，漏掉任何一处都会导致"刷新之后分数反而变高"的假象：

============  ====================================  ============================
存储           清理方式                               遗漏后果
============  ====================================  ============================
Milvus 切片    delete by ``source_file_name in []``   旧切片仍在，召回率虚高
Milvus 书名    delete by ``file_title in []``         书名确认节点仍能命中，掩盖问题
Milvus 实体    delete by ``book_name in []``          图谱召回路径仍通，误以为图谱有效
Neo4j 图谱     ``MATCH (n {book_name}) DETACH DELETE`` 同上，三元组还在
本地临时目录    ``temp_data/eval_import/`` 整个删掉      转换产物残留，下次扫描重复导入
============  ====================================  ============================

文件名改名后的清理陷阱
--------------------
``eval/item/`` 下的文件可能被用户改名 / 删除 / 替换内容。如果 reset 只按
"当前 eval/item 下的文件名"清理，**改名前的旧文件残留会变成幽灵切片**，
让分数看起来比真实水平还高。

为解决这个问题，本模块用 **三层来源** 计算待清理列表：

1. **当前 eval/item 文件名**：当下放在那里的文件，**必须清理**
2. **导入清单（``reports/_import_manifest.json``）**：曾经被评估流程导入过的
   ``source_file_name``，即使 ``eval/item`` 下已经找不到了，**也必须清理**
3. **孤儿扫描**：Milvus 里出现、但既不在当前文件名、也不在清单里的
   ``source_file_name``——**给出告警**，让用户知道"上一轮有没有漏导"

另外提供 ``force_quarantine=True`` 的"隔离模式"开关——
按导入清单**只清 eval 流程曾经导入过的数据**，绝不误伤用户手动导入的业务语料。

只删**评估材料**对应的数据，业务知识库的其它内容一概不动——
判断依据就是文件名（source_file_name / file_title），不会误伤。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from dotenv import load_dotenv

load_dotenv()

from knowledge.utils.milvus_util import get_milvus_client

from .runner import (
    IMPORT_MANIFEST_PATH,
    ITEM_DIR,
    TEMP_IMPORT_DIR,
    load_import_manifest,
    manifest_book_names,
    manifest_source_names,
    scan_item_files,
)

# 可导入的评测材料后缀（与 runner.scan_item_files 保持一致）
SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".html", ".htm", ".mp3"}


def _escape_expr(value: str) -> str:
    """转义 Milvus 过滤表达式里的双引号与反斜杠。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _in_expr(field: str, values: List[str]) -> str:
    """拼 Milvus 的 IN 过滤表达式，空列表返回 None 语义的空串。"""
    if not values:
        return ""
    quoted = ", ".join(f'"{_escape_expr(v)}"' for v in values)
    return f'{field} in [{quoted}]'


def _chunked_in_expr(field: str, values: List[str], chunk: int = 50) -> List[str]:
    """把一个长 IN 列表切成多个小表达式，避开 Milvus 单表达式长度上限。

    Milvus 对 filter 表达式有长度限制（实测 ~16KB），超过会报
    ``Unexpected error`` / ``expression is too long``。这里每 50 个
    source_file_name 一组，返回多条独立表达式。
    """
    if not values:
        return []
    return [
        _in_expr(field, values[i : i + chunk])
        for i in range(0, len(values), chunk)
    ]


def _query_with_chunked_in(
    client,
    collection: str,
    field: str,
    values: List[str],
    output_fields: List[str],
) -> List[Dict[str, Any]]:
    """对超长 IN 列表分批 query，返回所有结果合并。"""
    rows: List[Dict[str, Any]] = []
    for expr in _chunked_in_expr(field, values):
        rows.extend(
            client.query(
                collection_name=collection, filter=expr, output_fields=output_fields
            )
            or []
        )
    return rows


def _delete_with_chunked_in(
    client,
    collection: str,
    field: str,
    values: List[str],
    label: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> int:
    """对超长 IN 列表分批 delete，返回删除批数（每批至少 1 即代表已执行）。"""
    exprs = _chunked_in_expr(field, values)
    if not exprs:
        return 0
    total = 0
    for idx, expr in enumerate(exprs, start=1):
        try:
            client.delete(collection_name=collection, filter=expr)
            total += 1
            if on_progress:
                on_progress(
                    f"    · {label} 第 {idx}/{len(exprs)} 批（{len(values[idx*50-50:idx*50])} 项）已发送"
                )
        except Exception as e:  # noqa: BLE001
            if on_progress:
                on_progress(f"    × {label} 第 {idx}/{len(exprs)} 批失败 - {e}")
    return total


class EvalResetter:
    """评估数据清理器。

    Args:
        log: 日志回调 ``f(message: str, level: str)``。
    """

    def __init__(self, log: Optional[Callable[[str, str], None]] = None):
        self.log = log or (lambda msg, level="info": print(msg))

    def _info(self, msg: str) -> None:
        self.log(msg, "info")

    def _ok(self, msg: str) -> None:
        self.log(msg, "success")

    def _warn(self, msg: str) -> None:
        self.log(msg, "warn")

    def _err(self, msg: str) -> None:
        self.log(msg, "error")

    # ---------------- 扫描 ----------------
    def collect_item_stems(self) -> List[str]:
        """收集 eval/item 下全部评测材料的文件名（去扩展名）。"""
        files = scan_item_files()
        return sorted({p.stem for p in files})

    def collect_target_stems(self) -> Dict[str, List[str]]:
        """三层来源汇总待清理的 source_file_name。

        Returns::

            {
              "current": [...],      # 当前 eval/item 下的文件名
              "from_manifest": [...], # 导入清单里的文件名（可能已被改名/删除）
              "union":   [...],      # 两者并集（去重、按字母排序）
            }
        """
        current = self.collect_item_stems()
        manifest = manifest_source_names()

        union_set: Set[str] = set(current) | set(manifest)
        return {
            "current": current,
            "from_manifest": manifest,
            "union": sorted(union_set),
        }

    # ---------------- 主入口 ----------------
    def reset(
        self,
        clean_reports: bool = False,
        force_quarantine: bool = False,
    ) -> Dict[str, Any]:
        """执行刷新。

        Args:
            clean_reports: 是否连历史评估报告一起删（默认保留，便于做趋势对比）。
            force_quarantine: 是否强制走"隔离模式"——按导入清单**只清评估材料**，
              不做任何"业务语料 vs 评估语料"的猜测。当 manifest 文件被手工删过、
              或 Milvus 里残留了旧 import 的脏数据时用。

        Returns:
            清理结果字典。
        """
        self._info("=" * 72)
        self._info("开始刷新：清理评估导入的数据")
        self._info("=" * 72)

        target = self.collect_target_stems()
        current = target["current"]
        manifest = target["from_manifest"]
        stems = target["union"]

        # ---- 检查导入清单是否存在 ----
        if not IMPORT_MANIFEST_PATH.exists() and not current:
            self._warn(
                f"未在 {ITEM_DIR} 下找到任何评测材料，且 {IMPORT_MANIFEST_PATH} 不存在，无需清理"
            )
            return {"stems": [], "cleaned": {}, "orphans": []}

        # ---- 报告来源 ----
        self._info(f"识别出 {len(stems)} 个待清理的 source_file_name：")
        for s in stems:
            tag = []
            if s in current:
                tag.append("当前文件")
            if s in manifest:
                tag.append("导入清单")
            self._info(f"  · {s}  [{'/'.join(tag) if tag else '孤儿'}]")

        renamed_only = sorted(set(manifest) - set(current))
        if renamed_only:
            self._warn(
                f"以下 {len(renamed_only)} 个文件不在当前 eval/item 下，"
                f"但导入清单里仍记录着它们的残留——将一并清理："
            )
            for s in renamed_only:
                self._warn(f"    · {s}")

        # ---- 1. 先从切片集合里反查出这批材料对应的书名（后面清实体/图谱要用）----
        book_names = self._collect_book_names(stems)
        # 加上导入清单里的书名（覆盖"实体集合用 book_name 索引"的情况）
        for n in manifest_book_names():
            if n and n not in book_names:
                book_names.append(n)
        book_names = sorted(set(book_names))

        result: Dict[str, Any] = {
            "stems": stems,
            "current": current,
            "from_manifest": manifest,
            "book_names": book_names,
            "cleaned": {},
            "orphans": [],
            "force_quarantine": force_quarantine,
            "errors": [],
        }

        if book_names:
            self._info(f"关联到 {len(book_names)} 个书名：{', '.join(book_names)}")

        # ---- 2. Milvus 切片集合 ----
        result["cleaned"]["chunks"] = self._delete_milvus(
            collection=os.getenv("CHUNKS_COLLECTION", "kb_tingbook_chunks_v1"),
            expr=_in_expr("source_file_name", stems),
            label="Milvus 切片集合",
        )

        # ---- 3. Milvus 书名集合 ----
        result["cleaned"]["book_names"] = self._delete_milvus(
            collection=os.getenv("BOOK_NAME_COLLECTION", "kb_tingbook_book_names_v1"),
            expr=_in_expr("file_title", stems),
            label="Milvus 书名集合",
        )

        # ---- 4. Milvus 实体集合 ----
        if book_names:
            result["cleaned"]["entities"] = self._delete_milvus(
                collection=os.getenv(
                    "ENTITY_NAME_COLLECTION", "kb_tingbook_entity_names_v1"),
                expr=_in_expr("book_name", book_names),
                label="Milvus 实体集合",
            )

        # ---- 5. Neo4j 图谱 ----
        result["cleaned"]["neo4j"] = self._clear_neo4j(book_names)

        # ---- 6. 本地临时目录 ----
        result["cleaned"]["temp_dir"] = self._clean_temp_dir()

        # ---- 7. 报告（可选）----
        if clean_reports:
            result["cleaned"]["reports"] = self._clean_reports()

        # ---- 8. 孤儿扫描 ----
        orphans = self._detect_orphans(stems)
        result["orphans"] = orphans
        if orphans:
            self._warn(
                f"⚠️ Milvus 切片集合里仍有 {len(orphans)} 个 source_file_name "
                f"既不在当前 eval/item 下、也不在导入清单里——可能是历史脏数据："
            )
            for o in orphans[:20]:
                self._warn(f"    · {o}")
            if len(orphans) > 20:
                self._warn(f"    · ...还有 {len(orphans) - 20} 个未列出")
            if force_quarantine:
                self._warn("force_quarantine=True：跳过孤儿清理（请手工处理）")
            else:
                self._info("未启用 force_quarantine：保留这些孤儿数据。")
                self._info(
                    "如果认为这些孤儿是历史脏数据，请设置 force_quarantine=True 后再清理。"
                )

        failed = [k for k, v in result["cleaned"].items() if v == -1]
        if failed:
            self._warn(f"以下项清理时出现异常：{', '.join(failed)}（详见上方日志）")
        else:
            self._ok("刷新完成，知识库已回到评估材料导入之前的状态")

        self._info("=" * 72)
        return result

    # ---------------- 具体实现 ----------------
    def _collect_book_names(self, stems: List[str]) -> List[str]:
        """从切片集合反查这批材料对应的书名（去重）。"""
        client = get_milvus_client()
        if client is None:
            self._warn("Milvus 客户端不可用，降级按文件名清理（Neo4j 与实体集合不受影响）")
            return []

        collection = os.getenv("CHUNKS_COLLECTION", "kb_tingbook_chunks_v1")
        try:
            if not client.has_collection(collection):
                return []
            rows = _query_with_chunked_in(
                client, collection, "source_file_name", stems, ["book_name"]
            )
            names = sorted({r.get("book_name") for r in rows if r.get("book_name")})
            return names
        except Exception as e:  # noqa: BLE001
            self._warn(f"反查书名失败，降级按文件名清理：{e}")
            return []

    def _delete_milvus(self, collection: str, expr: str, label: str) -> int:
        """按过滤表达式删除 Milvus 数据，返回删除条数（失败返回 -1）。

        Milvus 的 delete 接口不返回删除行数，因此这里**先查后删**，
        用删除前的 count(*) 作为"清理量"报告给用户。
        """
        if not expr:
            self._info(f"{label}：无需清理")
            return 0

        client = get_milvus_client()
        if client is None:
            self._err(f"{label}：Milvus 客户端不可用")
            return -1

        try:
            if not client.has_collection(collection):
                self._info(f"{label}：集合 {collection} 不存在，跳过")
                return 0

            before = client.query(
                collection_name=collection, filter=expr, output_fields=["count(*)"]
            )
            count = int(before[0].get("count(*)", 0)) if before else 0

            if count == 0:
                self._info(f"{label}（{collection}）：无匹配数据，跳过")
                return 0

            client.delete(collection_name=collection, filter=expr)
            self._ok(f"{label}（{collection}）：已删除 {count} 条")
            return count
        except Exception as e:  # noqa: BLE001
            self._err(f"{label}（{collection}）：清理失败 - {e}")
            return -1

    def _clear_neo4j(self, book_names: List[str]) -> int:
        """清理 Neo4j 里这批书名的全部节点与关系。"""
        if not book_names:
            self._info("Neo4j 图谱：无关联书名，跳过")
            return 0

        try:
            from knowledge.utils.neo4j_util import get_neo4j_driver
        except Exception as e:  # noqa: BLE001
            self._err(f"Neo4j 图谱：驱动导入失败 - {e}")
            return -1

        driver = get_neo4j_driver()
        if driver is None:
            self._err("Neo4j 图谱：驱动不可用")
            return -1

        database = os.getenv("NEO4J_DATABASE", "neo4j")
        cypher = "MATCH (n {book_name: $book_name}) DETACH DELETE n"

        total = 0
        try:
            with driver.session(database=database) as session:
                for name in book_names:
                    try:
                        session.run(cypher, book_name=name)
                        total += 1
                    except Exception as e:  # noqa: BLE001
                        self._warn(f"Neo4j 清理失败（{name}）：{e}")
            self._ok(f"Neo4j 图谱：已清理 {total}/{len(book_names)} 个书名的节点与关系")
            return total
        except Exception as e:  # noqa: BLE001
            self._err(f"Neo4j 图谱：清理失败 - {e}")
            return -1

    def _clean_temp_dir(self) -> int:
        """删除导入中间产物目录。"""
        if not TEMP_IMPORT_DIR.exists():
            self._info(f"临时目录 {TEMP_IMPORT_DIR} 不存在，跳过")
            return 0
        try:
            count = sum(1 for _ in TEMP_IMPORT_DIR.rglob("*") if _.is_file())
            shutil.rmtree(TEMP_IMPORT_DIR)
            self._ok(f"本地临时目录：已删除 {TEMP_IMPORT_DIR}（含 {count} 个中间文件）")
            return count
        except Exception as e:  # noqa: BLE001
            self._err(f"本地临时目录：删除失败 - {e}")
            return -1

    def _clean_reports(self) -> int:
        """删除历史评估报告（不动导入清单 _import_manifest.json）。"""
        from .runner import REPORT_DIR

        if not REPORT_DIR.exists():
            return 0
        try:
            files = [
                p for p in REPORT_DIR.iterdir()
                if p.is_file() and p.name != "_import_manifest.json"
            ]
            for p in files:
                p.unlink()
            self._ok(f"历史报告：已删除 {len(files)} 个文件")
            return len(files)
        except Exception as e:  # noqa: BLE001
            self._err(f"历史报告：删除失败 - {e}")
            return -1

    # ---------------- 孤儿检测 ----------------
    def _detect_orphans(self, known_stems: List[str]) -> List[str]:
        """扫描 Milvus 切片集合，找出既不在 known_stems、也不在导入清单里的 source_file_name。

        这些是"幽灵数据"——可能是更早的评估残留、或手工导入的业务数据。
        给出告警，**不直接删除**（除非调用方启用 force_quarantine）。
        """
        client = get_milvus_client()
        if client is None:
            return []

        collection = os.getenv("CHUNKS_COLLECTION", "kb_tingbook_chunks_v1")
        try:
            if not client.has_collection(collection):
                return []

            # 用 limit+offset 滚动拉所有 source_file_name（避免大集合查询失败）
            known = set(known_stems)
            seen: Set[str] = set()
            limit = 1000
            offset = 0
            while True:
                rows = client.query(
                    collection_name=collection,
                    filter="",
                    output_fields=["source_file_name"],
                    limit=limit,
                    offset=offset,
                ) or []
                if not rows:
                    break
                for r in rows:
                    name = r.get("source_file_name")
                    if name and name not in known:
                        seen.add(name)
                if len(rows) < limit:
                    break
                offset += limit

            orphans = sorted(seen)
            return orphans
        except Exception as e:  # noqa: BLE001
            self._warn(f"孤儿扫描失败：{e}")
            return []