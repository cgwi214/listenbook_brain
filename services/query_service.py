"""查询业务服务"""

import uuid
import logging
from typing import List, Dict, Any

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import update_task_status, get_task_result, \
    get_task_status, get_done_task_list, get_running_task_list, \
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED
from knowledge.utils.sse_util import create_sse_queue, push_sse_event

logger = logging.getLogger(__name__)


class QueryService:

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def submit_query(self, task_id: str, is_stream: bool):
        """提交查询任务：更新状态 + 流式模式创建 SSE 队列。"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    def run_query_graph(self, task_id: str, session_id: str, user_query: str,
                        is_stream: bool, role: str = "listener"):
        """执行 LangGraph 查询流程。

        Args:
            role: 提问角色，listener(听众)/operator(内容运营)/editor(内容编辑)
        """
        try:
            default_state = {
                "original_query": user_query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
                "role": role,
            }
            query_app.invoke(default_state)
        except Exception as e:
            logger.error(f"查询流程执行失败: {e}", exc_info=True)
        finally:
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            if is_stream:
                push_sse_event(task_id, "progress", {
                    "status": get_task_status(task_id),
                    "done_list": get_done_task_list(task_id),
                    "running_list": get_running_task_list(task_id),
                })

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")

    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """查询任务状态与结果，供 /status/{task_id} 轮询使用（与前端 poll 约定一致）。"""
        return {
            "status": get_task_status(task_id),
            "running_list": get_running_task_list(task_id),
            "done_list": get_done_task_list(task_id),
            "answer": get_task_result(task_id, "answer", ""),
            "error": get_task_result(task_id, "error", ""),
        }

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "book_names": r.get("book_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)