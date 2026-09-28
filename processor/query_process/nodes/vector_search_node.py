import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode, T ,setup_logging
from processor.query_process.exceptions import StateFieldError
from utils.bge_m3_embedding_util import get_beg_m3_embedding_model,generate_hybrid_embeddings
from utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_query, validate_book_names, validate_author_names, validate_categories = self._validate_query_inputs(state)

        # 2. 获取嵌入模型&milvus客户端
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        if embedding_model is None or milvus_client is None:
            return state

        # 3. 对问题向量化(稀疏向量做了字典的处理) 注意：【generate_hybrid_embeddings】
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_documents=[validated_query])
        if not embedding_result:
            return state

        # 4. 构建多条件过滤表达式（书名 / 作者名 / 类别 三者可组合）
        filter_expr = self._filter_expr(validate_book_names, validate_author_names, validate_categories)

        # 5. 执行混合搜索请求：带条件检索
        reps = self._search(milvus_client, embedding_result, expr=filter_expr)

        # ================================
        # fallback：无结果时逐级放宽检索条件，保证推荐/检索类问题的召回
        # ================================
        if not reps and (validate_author_names or validate_categories):
            # 第一级放宽：取消作者/类别等附加条件，仅保留书名
            book_only_expr = self._filter_expr(validate_book_names, [], [])
            if book_only_expr and book_only_expr != filter_expr:
                logger.warning("作者/类别过滤搜索无结果，尝试仅按书名重新搜索")
                reps = self._search(milvus_client, embedding_result, expr=book_only_expr)

        if not reps and filter_expr:
            # 第二级放宽：取消全部过滤条件（全库兜底）
            logger.warning("按条件检索无结果，尝试取消全部过滤条件重新搜索")
            reps = self._search(milvus_client, embedding_result, expr=None)

        # 多次尝试均无结果
        if not reps:
            return state

        # 6. 更新state的embedding_chunks
        return {"embedding_chunks": reps}

    def _search(self, milvus_client: Any, embedding_result: Dict[str, Any], expr) -> List[Dict[str, Any]]:
        """执行一次混合检索；内部捕获异常，保证单个检索失败不影响整体服务。"""
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=expr,
            limit=5
        )
        try:
            reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_requests,
                norm_score=True,
                output_fields=[
                    "chunk_id",
                    "content",
                    "title",
                    "book_name",
                    "author_name",
                    "content_type",
                    "category",
                    "audio_duration",
                    "entry_name",
                    "source_file_name",
                    "file_title"
                ]
            )
        except Exception as e:
            logger.error(f"混合检索执行异常: {e}")
            return []
        return reps[0] if reps else []

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str], List[str], List[str]]:

        # 1. 获取state的rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取state的book_names
        book_names = state.get('book_names', [])

        # 3. 获取state的检索条件（作者名/类别）
        author_names = state.get('author_names', [])
        categories = state.get('categories', [])

        # 4. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if book_names is None or not isinstance(book_names, list):
            raise StateFieldError(node_name=self.name, field_name="book_names", expected_type=list)

        if author_names is None or not isinstance(author_names, list):
            raise StateFieldError(node_name=self.name, field_name="author_names", expected_type=list)

        if categories is None or not isinstance(categories, list):
            raise StateFieldError(node_name=self.name, field_name="categories", expected_type=list)

        # 5. 返回
        return rewritten_query, book_names, author_names, categories

    def _filter_expr(self, validate_book_names: List[str], validate_author_names: List[str],
                     validate_categories: List[str]) -> str:
        """
        构建检索过滤表达式（Milvus bool expr，标量字段过滤）。
        支持 书名(book_name) / 作者名(author_name) / 类别(category) 的任意组合；
        全部为空时返回空字符串（表示不做过滤）。
        """
        clauses = []

        if validate_book_names:
            quoted = ", ".join(f'"{v}"' for v in validate_book_names)
            clauses.append(f"book_name in [{quoted}]")

        if validate_author_names:
            quoted = ", ".join(f'"{v}"' for v in validate_author_names)
            clauses.append(f"author_name in [{quoted}]")

        if validate_categories:
            quoted = ", ".join(f'"{v}"' for v in validate_categories)
            clauses.append(f"category in [{quoted}]")

        return " and ".join(clauses)


_node_instance = VectorSearchNode()


def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("向量检索节点测试")
    print("=" * 60)

    # 1. 准备测试状态
    test_state = {
        "session_id": "test_001",
        "rewritten_query": "如何使用万用表测量电压？",
        "book_names": ["RS-12 数字万用表"],
        "embedding_chunks": [],
    }

    print(f"\n输入状态:")
    print(f"  rewritten_query: {test_state['rewritten_query']}")
    print(f"  book_names: {test_state['book_names']}")
    print("-" * 60)

    # 2. 执行节点
    try:
        result = node_search_embedding(test_state)
        chunks = result.get("embedding_chunks", [])

        print(f"\n检索到 {len(chunks)} 条结果:")
        print("-" * 60)

        for i, chunk in enumerate(chunks, 1):
            # 兼容不同的返回格式
            entity = chunk.get("entity", chunk) if isinstance(chunk, dict) else {}
            content = entity.get("content", "")
            book_name = entity.get("book_name", "未知")
            chunk_id = entity.get("chunk_id", "N/A")
            score = chunk.get("distance", 0)

            print(f"[{i}] 商品: {book_name}")
            print(f"    ID: {chunk_id}")
            print(f"    分数: {score:.4f}")
            print(f"    内容: {content[:80]}...")
            print()

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback

        traceback.print_exc()

    # 3. 测试无过滤条件的场景
    print("=" * 60)
    print("测试无书名过滤")
    print("=" * 60)

    test_state_no_filter = {
        "session_id": "test_002",
        "rewritten_query": "如何测量电压？",
        "book_names": [],  # 无书名
        "embedding_chunks": [],
    }

    print(f"\n输入状态:")
    print(f"  rewritten_query: {test_state_no_filter['rewritten_query']}")
    print(f"  book_names: {test_state_no_filter['book_names']} (无过滤)")
    print("-" * 60)

    try:
        result2 = node_search_embedding(test_state_no_filter)
        chunks2 = result2.get("embedding_chunks", [])
        print(f"\n检索到 {len(chunks2)} 条结果（无过滤）")

        # 打印前 3 条
        for i, chunk in enumerate(chunks2[:3], 1):
            entity = chunk.get("entity", chunk) if isinstance(chunk, dict) else {}
            print(f"[{i}] {entity.get('book_name', '?')} | score={chunk.get('distance', 0):.4f}")

    except Exception as e:
        print(f"\n执行失败: {e}")
