from functools import lru_cache
from knowledge.services.import_file_service import ImportFileService
from knowledge.services.task_service import TaskService
from knowledge.services.query_service import QueryService
from knowledge.services.eval_service import EvalService


@lru_cache
def get_task_service() -> TaskService:
    return TaskService()

@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService(get_task_service())

@lru_cache
def get_query_service() -> QueryService:
    return QueryService()

@lru_cache
def get_eval_service() -> EvalService:
    """评估服务单例（端口 8002 的 eval_router 通过它启动评估/刷新后台线程）。"""
    return EvalService()