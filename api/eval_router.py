"""评估路由

端口 8002。与导入（8000）/ 查询（8001）解耦，互不干扰。

接口清单::

    GET  /eval                 评估页面（eval.html）
    GET  /eval.html            同上，兼容直接访问
    POST /eval/run             启动一轮评估，返回 task_id
    POST /eval/reset           启动一次刷新，返回 task_id
    GET  /eval/stream/{task_id} SSE 订阅日志（与查询侧共用 SSE 工具）
    GET  /eval/config          返回当前可用的配置（数据集规模、最近报告）
"""

from __future__ import annotations

import os.path
from typing import Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from knowledge.core.paths import get_front_page_dir
from knowledge.core.deps import get_eval_service
from knowledge.services.eval_service import EvalService
from knowledge.utils.sse_util import sse_generator
from knowledge.processor.query_process.base import setup_logging


class EvalRunRequest(BaseModel):
    """评估请求体（前端以 JSON 发送；此前用 Query 声明导致前端 JSON 参数被整体忽略）。"""
    top_k: Optional[int] = Field(default=None, ge=1, le=1000)
    role: Optional[Literal["listener", "operator", "editor"]] = None
    limit: Optional[int] = Field(default=None, ge=1, le=100000)
    enable_llm_judge: bool = False
    skip_import: bool = False
    hyde_enabled: Optional[bool] = None


class EvalResetRequest(BaseModel):
    """刷新请求体。"""
    clean_reports: bool = False
    force_quarantine: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="Eval Service", description="听书知识库 RAG 评估服务")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:

    @app.get("/")
    def root():
        return {"service": "eval", "endpoints": [
            "/eval", "/eval/run", "/eval/reset", "/eval/stream/{task_id}"
        ]}

    @app.get("/eval")
    @app.get("/eval.html")
    async def eval_page():
        """评估页面。"""
        return FileResponse(os.path.join(get_front_page_dir(), "eval.html"))

    @app.post("/eval/run")
    async def run_eval(
        body: EvalRunRequest,
        service: EvalService = Depends(get_eval_service),
    ):
        """启动一轮评估，返回 task_id。"""
        roles = [body.role] if body.role else None
        task_id = service.submit_run(
            top_k=body.top_k,
            roles=roles,
            limit=body.limit,
            enable_llm_judge=body.enable_llm_judge,
            skip_import=body.skip_import,
            hyde_enabled=body.hyde_enabled,
        )
        return {"message": "评估已启动", "task_id": task_id}

    @app.post("/eval/reset")
    async def reset_eval(
        body: EvalResetRequest,
        service: EvalService = Depends(get_eval_service),
    ):
        """启动一次刷新，返回 task_id。"""
        task_id = service.submit_reset(
            clean_reports=body.clean_reports,
            force_quarantine=body.force_quarantine,
        )
        return {"message": "刷新已启动", "task_id": task_id}

    @app.get("/eval/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        """SSE 订阅评估/刷新日志。"""
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
        )

    @app.get("/eval/config")
    async def eval_config():
        """返回当前配置信息（数据集规模、报告清单、导入清单）。"""
        from knowledge.eval.runner import (
            DATASET_DIR, GOLDEN_SET_PATH, NEGATIVE_SET_PATH, CROSS_RECALL_SET_PATH,
            REPORT_DIR, IMPORT_MANIFEST_PATH, load_import_manifest, manifest_source_names,
        )

        def _count_lines(path) -> int:
            if not path.exists():
                return 0
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())

        # 扫一份最新报告的元信息（如果有）
        latest = REPORT_DIR / "latest.md" if REPORT_DIR.exists() else None
        manifest = load_import_manifest()

        return {
            "dataset_dir": str(DATASET_DIR),
            "golden_count": _count_lines(GOLDEN_SET_PATH),
            "negative_count": _count_lines(NEGATIVE_SET_PATH),
            "cross_recall_count": _count_lines(CROSS_RECALL_SET_PATH),
            "report_dir": str(REPORT_DIR),
            "latest_report": str(latest) if latest and latest.exists() else None,
            "import_manifest_path": str(IMPORT_MANIFEST_PATH),
            "import_manifest_count": len(manifest),
            "import_manifest_sources": manifest_source_names(),
        }


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8002)
