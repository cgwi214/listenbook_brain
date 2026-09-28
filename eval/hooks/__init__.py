"""评测钩子子包

在检索链路的关键位置挂"探针"，把评测所需的中间产物导出到 State，
再由 runner 收集。核心是 rerank 之后的 chunk_id 抽取。
"""

from .chunk_id_hook import (
    extract_chunk_ids,
    extract_reranked_trace,
    attach_eval_trace,
    EVAL_CHUNK_IDS_KEY,
    EVAL_TRACE_KEY,
)

__all__ = [
    "extract_chunk_ids",
    "extract_reranked_trace",
    "attach_eval_trace",
    "EVAL_CHUNK_IDS_KEY",
    "EVAL_TRACE_KEY",
]
