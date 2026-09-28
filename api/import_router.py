import os.path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, Depends, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.core.deps import get_task_service
from knowledge.core.deps import get_import_file_service
from knowledge.services.import_file_service import ImportFileService
from knowledge.services.task_service import TaskService
from  knowledge.processor.import_process.base import setup_logging


# 内容类型白名单（与前端下拉、导入识别口径保持一致）
ALLOWED_CONTENT_TYPES = (
    "有声书信息", "书籍简介", "作者介绍", "听书笔记",
    "推荐运营资料", "用户评论摘要", "常见问答",
)


def create_app() -> FastAPI:
    """
    负责创建fastapi实例
    Returns:

    """
    # 1. 实例化FastAPI实例
    app = FastAPI(description="知识库导入")

    # 2. 跨域配置（顺便配上:当前项目不会出现）---浏览器会出现
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许任意的源
        allow_credentials=True,  # 允许cookie中携带任意的自定义参数
        allow_methods=["*"],  # 允许任意的请求方式
        allow_headers=["*"],  # 允许请求头中携带任意的我自定义参数
    )

    # 3. 将静态资源的目录挂载到app实例上
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    # 4. 注册路由（接收前端发送的各种方式的请求）
    register_router(app)

    # 返回实例
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    # 1. 处理导入页面访问请求
    @app.get("/import")
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    # 2. 上传请求（支持可选的书籍元数据表单字段，用户显式提供的值优先于LLM识别结果）
    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                                   book_name: str = Form(default=""),
                                   author_name: str = Form(default=""),
                                   content_type: str = Form(default=""),
                                   category: str = Form(default=""),
                                   audio_duration: str = Form(default=""),
                                   entry_name: str = Form(default=""),
                                   source_url: str = Form(default=""),
                                   service: ImportFileService = Depends(get_import_file_service)):
        # 1. 上传文件（本地/minio）
        task_id, file_dir, import_file_path = service.process_upload_file(file)

        # 2. 收集用户显式提供的元数据（空值不入，交由LLM识别兜底）
        metadata = {
            k: v.strip()
            for k, v in {
                "book_name": book_name,
                "author_name": author_name,
                "content_type": content_type,
                "category": category,
                "audio_duration": audio_duration,
                "entry_name": entry_name,
                "source_url": source_url,
            }.items()
            if v and v.strip()
        }

        # 2.1 内容类型白名单强校验（可选字段：填了就必须合法，避免脏数据进入知识库）
        content_type_value = metadata.get("content_type")
        if content_type_value and content_type_value not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"内容类型不受支持：{content_type_value}，仅支持：{' / '.join(ALLOWED_CONTENT_TYPES)}"
            )

        # 3. 运行后台任务（跑graph的整个流程）
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path, metadata)

        # 4. 返回
        return UploadResponse(message="文件上传成功", task_id=task_id)


    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(task_id:str,task_service:TaskService=Depends(get_task_service)):
        """
        根据任务id 查询任务的状态
        Returns:
        """
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)


if __name__ == '__main__':
    """
    启动web服务器  (fastapi实例)
    uvicorn(性能高)
    """
    setup_logging()
    uvicorn.run(app=create_app(), port=8000, host="0.0.0.0")
