from functools import lru_cache
from services.import_file_service import ImportFileService
from services.task_service import TaskService
from services.query_service import QueryService


@lru_cache
def get_task_service() -> TaskService:
    return TaskService()

@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService(get_task_service())

@lru_cache
def get_query_service() -> QueryService:
    return QueryService()