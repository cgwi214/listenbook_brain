"""

导入流程状态类型定义



定义完整的状态结构和辅助函数

"""

from typing import TypedDict, List

import copy


class ImportGraphState(TypedDict, total=False):


    """

    导入流程图状态



    包含整个导入流程中传递的所有数据

    """

    # ==================== 任务标识 ====================

    task_id: str  # 任务 ID，用于任务追踪(web交互的时候用到，实时看到节点的处理日志)

    # ==================== 控制标志 ====================

    is_md_read_enabled: bool  # 是否启用 MD 读取

    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    is_html_read_enabled: bool  # 是否启用 HTML 读取

    is_mp3_read_enabled: bool  # 是否启用 MP3 读取（ASR语音转写）

    # ==================== 路径信息 ====================

    import_file_path: str  # 导入文件路径

    file_dir: str  # 导入(出)文件目录

    pdf_path: str  # PDF 文件路径

    md_path: str  # 转换后Markdown 文件路径

    # ==================== 文件信息 ====================

    file_title: str  # 文件标题（不含扩展名，同时作为来源文件名）

    book_name: str  # 识别出的书名(方便程序员用)

    # ==================== 书籍元数据（听书知识库需求） ====================

    author_name: str  # 作者名（上传时用户可显式提供，否则由LLM识别）
    content_type: str  # 内容类型：有声书信息/书籍简介/作者介绍/听书笔记/推荐运营资料/用户评论摘要/常见问答
    category: str  # 类别/标签（科幻/悬疑/儿童/教育等）
    audio_duration: str  # 有声书时长（有声书类型内容，如 "78小时"）
    entry_name: str  # 条目名称（该资料自身的条目标题）
    source_url: str  # 来源路径或资源链接（上传时用户显式提供，可空）

    # ==================== 处理中间数据 ====================

    md_content: str  # Markdown 文档内容

    chunks: List  # 文档切片列表

    # ==================== 默认状态 ====================




GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": False,

    "is_md_read_enabled": False,

    "is_html_read_enabled": False,

    "is_mp3_read_enabled": False,

    "file_dir": "",

    "import_file_path": "",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "book_name": "",
    "author_name": "",
    "content_type": "",
    "category": "",
    "audio_duration": "",
    "entry_name": "",
    "source_url": "",

}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
