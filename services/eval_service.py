"""评估业务服务

提供两种动作：run / reset。两者都通过 SSE 队列往外推日志，
前端用 ``/eval/stream/{task_id}`` 端点订阅。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from typing import Any, Dict, Optional

from knowledge.eval.reset import EvalResetter
from knowledge.eval.runner import EvalRunner
from knowledge.processor.query_process.config import get_config
from knowledge.utils.sse_util import (
    create_sse_queue,
    push_sse_event,
    remove_sse_queue,
)


class EvalService:
    """封装评估 / 刷新两条业务线的服务类。"""

    # ---------- 日志回调：把日志塞进 SSE 队列 ----------
    @staticmethod
    def _make_log_emitter(task_id: str) -> callable:
        def emit(message: str, level: str = "info") -> None:
            push_sse_event(task_id, "log", {
                "level": level,
                "message": message,
            })
        return emit

    # ---------- 评估 ----------
    def submit_run(
        self,
        top_k: Optional[int] = None,
        roles: Optional[list] = None,
        limit: Optional[int] = None,
        enable_llm_judge: bool = False,
        skip_import: bool = False,
        hyde_enabled: Optional[bool] = None,
    ) -> str:
        """提交一轮评估，返回 task_id。实际执行在后台线程中跑。"""
        task_id = f"eval_run_{uuid.uuid4().hex[:12]}"
        create_sse_queue(task_id)

        def worker() -> None:
            # 临时开关 HyDE（仅本轮生效，结束后复原，避免影响线上问答）
            cfg = get_config()
            prev_hyde = cfg.hyde_enabled
            if hyde_enabled is not None:
                cfg.hyde_enabled = hyde_enabled
            try:
                runner = EvalRunner(log=self._make_log_emitter(task_id))
                report = runner.run(
                    top_k=top_k,
                    roles=roles,
                    limit=limit,
                    enable_llm_judge=enable_llm_judge,
                    skip_import=skip_import,
                    hyde_enabled=hyde_enabled,
                )
            except Exception as e:  # noqa: BLE001
                push_sse_event(task_id, "result", {
                    "task_id": task_id,
                    "retrieval": report.get("retrieval", {}),
                    "generation": report.get("generation", {}),
                    "negative": report.get("negative", {}),
                    "cross_recall": report.get("cross_recall", {}),
                    "total_cost_ms": report.get("total_cost_ms"),
                })
            except Exception as e:  # noqa: BLE001
                push_sse_event(task_id, "log", {
                    "level": "error",
                    "message": f"评估线程异常：{e}\n{traceback.format_exc()}",
                })
            finally:
                cfg.hyde_enabled = prev_hyde
                push_sse_event(task_id, "done", {"task_id": task_id})

        threading.Thread(target=worker, daemon=True).start()
        return task_id

    # ---------- 刷新 ----------
    def submit_reset(
        self,
        clean_reports: bool = False,
        force_quarantine: bool = False,
    ) -> str:
        """提交一次刷新（清理评估材料），返回 task_id。

        Args:
            clean_reports: 是否连历史评估报告一起删。
            force_quarantine: 是否强制走"隔离模式"——
                按导入清单只清评估材料，绝不误伤业务语料。
        """
        task_id = f"eval_reset_{uuid.uuid4().hex[:12]}"
        create_sse_queue(task_id)

        def worker() -> None:
            try:
                resetter = EvalResetter(log=self._make_log_emitter(task_id))
                resetter.reset(
                    clean_reports=clean_reports,
                    force_quarantine=force_quarantine,
                )
            except Exception as e:  # noqa: BLE001
                push_sse_event(task_id, "log", {
                    "level": "error",
                    "message": f"刷新线程异常：{e}\n{traceback.format_exc()}",
                })
            finally:
                push_sse_event(task_id, "done", {"task_id": task_id})

        threading.Thread(target=worker, daemon=True).start()
        return task_id
