"""rerank 之后抽取 chunk_id 的钩子

为什么挂在这里
--------------
评测的"召回"口径必须对齐**真正喂给 LLM 的那批切片**。整条链路里切片集合变化四次：

    向量/HyDE/图谱/联网  →  RRF 融合  →  Rerank 精排  →  断崖截断  →  答案

只有 rerank 且截断之后的结果才是最终上下文。挂在 ``rerank_node.process()`` 末尾，
能保证评测口径与线上真实行为严格一致。

抽取结果写回 State 的两个键（见 ``EVAL_CHUNK_IDS_KEY`` / ``EVAL_TRACE_KEY``），
下游 runner 直接读取，不需要再回查图内部状态。

chunk_id 为什么不写死在金标集里
-------------------------------
``chunk_id`` 是 Milvus 自动生成的主键，每次重新导入都会变。
金标集因此改用**稳定定位符**（source_file_name / book_name / author_name / 关键词），
由 runner 用 ``fetch_chunks_by_chunk_ids`` 反查出切片字段后再做匹配。
"""

from typing import Any, Dict, List

# State 中存放最终 chunk_id 列表的键
EVAL_CHUNK_IDS_KEY = "eval_chunk_ids"

# State 中存放完整评测轨迹的键（含每篇切片的来源元数据）
EVAL_TRACE_KEY = "eval_trace"

# 本地切片来源标记（联网结果 source == "web"，没有 chunk_id）
LOCAL_SOURCE = "local"


def extract_chunk_ids(reranked_docs: List[Dict[str, Any]]) -> List[Any]:
    """从 rerank 后的文档列表中抽取本地切片的 chunk_id。

    联网结果（``source == "web"``）没有 chunk_id，直接跳过。

    Args:
        reranked_docs: rerank 节点产出的有序文档列表。

    Returns:
        按精排顺序排列的 chunk_id 列表（已去重，保留首次出现顺序）。
    """
    chunk_ids: List[Any] = []
    seen = set()

    for doc in reranked_docs or []:
        if not isinstance(doc, dict):
            continue
        if doc.get("source") and doc.get("source") != LOCAL_SOURCE:
            continue

        chunk_id = doc.get("chunk_id")
        if chunk_id is None:
            continue
        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        chunk_ids.append(chunk_id)

    return chunk_ids


def extract_reranked_trace(
    reranked_docs: List[Dict[str, Any]],
    meta_fields: tuple = (
        "chunk_id", "title", "source", "score",
        "book_name", "author_name", "content_type",
        "category", "source_file_name", "file_title",
    ),
) -> List[Dict[str, Any]]:
    """抽取完整的精排轨迹（含元数据与分数），用于逐条 case 复盘。

    Args:
        reranked_docs: rerank 节点产出的有序文档列表。
        meta_fields: 需要保留的字段白名单。

    Returns:
        精简后的文档字典列表，顺序与精排结果一致。
    """
    trace: List[Dict[str, Any]] = []

    for rank, doc in enumerate(reranked_docs or [], start=1):
        if not isinstance(doc, dict):
            continue
        item = {"rank": rank}
        for field in meta_fields:
            value = doc.get(field)
            if value is not None:
                item[field] = value
        # content 单独截断，避免轨迹文件过大
        content = doc.get("content") or ""
        item["content_preview"] = content[:120]
        trace.append(item)

    return trace


def attach_eval_trace(state: Dict[str, Any]) -> Dict[str, Any]:
    """把钩子产物写回 State（原地修改并返回同一个 state 对象）。

    Args:
        state: rerank 节点处理中的图状态。

    Returns:
        写入 ``eval_chunk_ids`` 与 ``eval_trace`` 后的 state。
    """
    reranked_docs = state.get("reranked_docs") or []

    state[EVAL_CHUNK_IDS_KEY] = extract_chunk_ids(reranked_docs)
    state[EVAL_TRACE_KEY] = extract_reranked_trace(reranked_docs)

    return state
