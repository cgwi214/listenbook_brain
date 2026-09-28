import json

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from processor.import_process.state import ImportGraphState
from processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from processor.import_process.nodes.html_to_md_node import HtmlToMdNode
from processor.import_process.nodes.mp3_to_md_node import Mp3ToMdNode
from processor.import_process.nodes.entry_node import EntryNode
from processor.import_process.nodes.md_img_node import MarkDownImageNode
from processor.import_process.nodes.document_split_node import DocumentSplitNode
from processor.import_process.nodes.book_info_recognition_node import BookInfoRecognitionNode
from processor.import_process.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from processor.import_process.nodes.import_milvus_node  import ImportMilvusNode
from processor.import_process.nodes.kg_graph_node import KnowLedgeGraphNode
from processor.import_process.nodes.md_img_node import MarkDownImageNode
from processor.import_process.state import create_default_state
from processor.import_process.base import setup_logging


def import_router(state: ImportGraphState):
    if state.get('is_md_read_enabled'):
        return "md_img_node"
    if state.get('is_pdf_read_enabled'):
        return "pdf_to_md_node"
    if state.get('is_html_read_enabled'):
        return "html_to_md_node"
    if state.get('is_mp3_read_enabled'):
        return "mp3_to_md_node"
    return END  # 安全降级


def create_import_graph() -> CompiledStateGraph:
    """
    定义导入业务的graph状态拓扑谱（langgraph构建流水线）整个流水线各个节点要读取或者写入的节点。
    Returns:

    """

    # 1. 定义状态图
    graph_pineline = StateGraph(ImportGraphState)  # type:ignore

    # 2. 定义节点（入口、结束节点、自己需要添加的）
    # 2.1 定义入口节点
    graph_pineline.set_entry_point("entry_node")

    # 2.2 添加剩下的节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "html_to_md_node": HtmlToMdNode(),
        "mp3_to_md_node": Mp3ToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node":DocumentSplitNode(),
        "book_name_rec_node":BookInfoRecognitionNode(),
        "bge_embedding_node":BgeEmbeddingChunksNode(),
        "import_milvus_node":ImportMilvusNode(),
        "kg_node":KnowLedgeGraphNode()
    }
    for key, value in nodes.items():
        graph_pineline.add_node(key, value)

    # 3. 定义边（顺序边、条件边）
    # source:  路由开始节点
    # path:    路由函数
    # path_map 路由函数的映射

    # 3.1 入口的条件边(按文件类型路由：md/txt直达图片处理 pdf/html/mp3先走各自的转换节点)
    #     注意：entry_node 只有这一条条件边 不再追加无条件边 否则所有文件都会被强制送进pdf_to_md_node
    #     (md/txt/html/mp3导入失败的bug根因就是原代码多了一条 entry_node -> pdf_to_md_node 的无条件边)
    graph_pineline.add_conditional_edges("entry_node",
                                         import_router,
                                         {
                                             "md_img_node": "md_img_node",
                                             "pdf_to_md_node": "pdf_to_md_node",
                                             "html_to_md_node": "html_to_md_node",
                                             "mp3_to_md_node": "mp3_to_md_node",
                                             END: END
                                         }
                                         )

    # 3.2 转换节点各自汇聚到 md_img_node 复用后续的图片处理与切片流程
    graph_pineline.add_edge("pdf_to_md_node","md_img_node")
    graph_pineline.add_edge("html_to_md_node","md_img_node")
    graph_pineline.add_edge("mp3_to_md_node","md_img_node")
    graph_pineline.add_edge("md_img_node","document_split_node")
    graph_pineline.add_edge("document_split_node","book_name_rec_node")
    graph_pineline.add_edge("book_name_rec_node","bge_embedding_node")
    graph_pineline.add_edge("bge_embedding_node","import_milvus_node")
    graph_pineline.add_edge("import_milvus_node","kg_node")
    graph_pineline.add_edge("kg_node",END)

    # 4. 编译（编排）
    return graph_pineline.compile()


kb_import__graph_app = create_import_graph()


# 测试使用
def run_import_graph(import_file_path: str, file_dir: str):
    # 1. 构建state
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    init_state = create_default_state(**state)  # 发生了解包

    # 2. 调用stream(用流式获取每一个节点的处理情况：event事件[节点名字 节点处理后的状态])
    final_state = None
    for event in kb_import__graph_app.stream(init_state):
        for node_name, state in event.items():
            print(f"运行节点的:{node_name}")
            final_state = state

    return final_state


if __name__ == '__main__':
    setup_logging()

    import_file_path = r"D:\work\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\三体_书籍简介.md"
    file_dir = r"D:\work\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir"
    # 1. 测试编排流程
    final_state = run_import_graph(import_file_path=import_file_path, file_dir=file_dir)
    print(json.dumps(final_state, indent=2, ensure_ascii=False))

    # 2.打印图结构（ASCII 可视化）# 1. 单独安装：pip install grandalf 2.(单独安装还出错)  【pydantic：定义数据模型 】pip uninstall gradio  3. 单独安装 pip install grandalf 解决冲突
    print("-" * 50)
    print("图结构:")
    kb_import__graph_app.get_graph().print_ascii()
