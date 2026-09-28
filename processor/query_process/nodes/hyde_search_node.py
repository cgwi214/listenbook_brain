import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Tuple, Union,Any,Dict
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError

from knowledge.utils.llm_client_util import get_llm_client
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model


class HyDeSearchNode(BaseNode):
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:

        # 0. 评估开关：关闭 HyDE 时跳过假设性文档生成与混合检索
        #    （省一次 LLM 调用，且不参与 RRF 融合）
        if not self.config.hyde_enabled:
            self.logger.info("HyDE 已禁用（hyde_enabled=False），跳过假设性文档生成与混合检索")
            return {}

        # 1. 参数校验
        validated_query, validate_book_names, validate_author_names, validate_categories = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validate_book_names)

        # 3. 获取嵌入模型 & milvus客户端
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        if not embedding_model or not milvus_client:
            return state

        # 4. 假设性文档嵌入(注入问题+假设性文档)
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_documents=[embedding_document])

        if not embedding_result:
            return state

        # 5. 构建多条件过滤表达式（书名 / 作者名 / 类别 三者可组合）
        filter_expr = self._filter_expr(validate_book_names, validate_author_names, validate_categories)

        # 6-7. 执行混合检索；无结果时逐级放宽条件
        reps = self._search(milvus_client, embedding_result, expr=filter_expr)

        if not reps and (validate_author_names or validate_categories):
            # 放宽一：取消作者/类别附加条件，仅保留书名
            book_only_expr = self._filter_expr(validate_book_names, [], [])
            if book_only_expr and book_only_expr != filter_expr:
                logger.warning("HyDE 作者/类别过滤无结果，尝试仅按书名重新搜索")
                reps = self._search(milvus_client, embedding_result, expr=book_only_expr)

        if not reps and filter_expr:
            # 放宽二：取消全部过滤条件（全库兜底）
            logger.warning("HyDE 按条件检索无结果，尝试取消全部过滤条件重新搜索")
            reps = self._search(milvus_client, embedding_result, expr=None)

        if not reps:
            return state

        # 8. 只更新hyde_embedding_chunks
        return {"hyde_embedding_chunks": reps}

    def _search(self, milvus_client: Any, embedding_result: Dict[str, Any], expr) -> List[Dict[str, Any]]:
        """执行一次混合检索；内部捕获异常，保证单个检索失败不影响整体服务。"""
        hybrid_search_requests = create_hybrid_search_requests(dense_vector=embedding_result['dense'][0],
                                                               sparse_vector=embedding_result['sparse'][0],
                                                               expr=expr)
        try:
            reps = execute_hybrid_search_query(milvus_client,
                                               collection_name=self.config.chunks_collection,
                                               search_requests=hybrid_search_requests,
                                               norm_score=True,
                                               output_fields=["chunk_id", "content", "title", "book_name",
                                                              "author_name", "content_type", "category",
                                                              "audio_duration", "entry_name", "source_file_name",
                                                              "file_title"])
        except Exception as e:
            logger.error(f"HyDE 混合检索执行异常: {e}")
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

        # 空列表合法：推荐类查询（如"有哪些科幻类有声书"）没有指定书名，允许为空
        if book_names is None or not isinstance(book_names, list):
            raise StateFieldError(node_name=self.name, field_name="book_names", expected_type=list)

        if author_names is None or not isinstance(author_names, list):
            raise StateFieldError(node_name=self.name, field_name="author_names", expected_type=list)

        if categories is None or not isinstance(categories, list):
            raise StateFieldError(node_name=self.name, field_name="categories", expected_type=list)

        # 5. 返回
        return rewritten_query, book_names, author_names, categories

    def _generate_hy_document(self, validated_query: str, validate_book_names: List[str]) -> str:

        # 1. 获取LLM客户端
        llm_client = get_llm_client()

        # 2. 判断
        if llm_client is None:
            return ""

        # 3. 获取系统提示词以及用户提示词
        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(book_hint=validate_book_names, rewritten_query=validated_query)
        system_prompt = f"您是一位听书平台的资深编辑和书评专家，擅长撰写书籍简介、有声书推荐语、听书笔记和书评摘要"
        try:
            # 4. 获取AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, 'content', "").strip()

            # 6. 判断是否存在
            if not llm_response_content:
                return ""

            return llm_response_content
        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""

    def _filter_expr(self, validate_book_names: List[str], validate_author_names: List[str],
                     validate_categories: List[str]) -> str:
        """
        构建检索过滤表达式（Milvus bool expr，标量字段过滤）。
        支持 书名(book_name) / 作者名(author_name) / 类别(category) 的任意组合；
        全部为空时返回空表达式（表示不做过滤）。
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


# if __name__ == '__main__':
#
#     state = {
#         "rewritten_query": "万用表如何测量电阻",
#         "book_names": ["RS-12 数字万用表"]  # 对齐字段
#     }
#
#     vector_search = HyDeSearchNode()
#
#     result = vector_search.process(state)
#
#     for r in result.get('hyde_embedding_chunks'):
#         print(json.dumps(r, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    import json

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "book_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['book_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"book_name={entity.get('book_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    hyde_doc = result.get("hyde_doc", "")
    if hyde_doc:
        print(f"\n【假设性文档】:\n{hyde_doc[:200]}...")

    print("-" * 60)
    print("测试完成")
