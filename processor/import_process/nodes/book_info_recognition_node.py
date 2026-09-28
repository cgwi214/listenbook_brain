import json
import re
from json import JSONDecodeError
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType
from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.state import ImportGraphState
from processor.import_process.exceptions import ValidationError, EmbeddingError
from processor.import_process.config import get_config
from utils.llm_client_util import get_llm_client
from utils.milvus_util import get_milvus_client
from utils.bge_m3_embedding_util import get_beg_m3_embedding_model
from prompts.upload.import_prompt import BOOK_INFO_SYSTEM_PROMPT, \
    BOOK_INFO_USER_PROMPT_TEMPLATE, ALLOWED_CONTENT_TYPES

# 元数据的兜底内容类型（LLM 无法判断时）
DEFAULT_CONTENT_TYPE = "有声书信息"


class BookInfoRecognitionNode(BaseNode):
    """
    书籍信息识别节点（听书知识库）

    职责：
    1. 从文档切片中由 LLM 识别书籍元数据（书名/作者/内容类型/类别/有声书时长/条目名称）
    2. 与上传时用户显式提供的元数据合并（用户优先）
    3. 书名向量化并写入 Milvus 书名集合（携带元数据标量字段）
    4. 将元数据回填到 state 与每一个 chunk，供下游节点（嵌入/入库/图谱）使用
    """

    name = "book_info_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        file_title, chunks, config = self._validate_inputs(state)

        # 2. 构建LLM的上下文（提取书籍元数据）
        book_info_context = self._prepare_book_info_context(chunks, config)

        # 3. 调用LLM模型识别元数据
        llm_meta = self._recognition_book_info_by_llm(file_title, book_info_context)

        # 4. 与用户上传时提供的元数据合并（用户提供的优先）
        metadata = self._merge_metadata(state, llm_meta, file_title)

        # 5. 嵌入书名(稠密向量：语义相似性、稀疏向量：关键词相似【bgem3】)
        dense_vector, sparse_vector = self._embedding_book_name(metadata["book_name"])

        # 6. 存储到Milvus数据库（书名集合，携带元数据）
        self._save_to_milvus(file_title, metadata, dense_vector, sparse_vector, config)

        # 7. 回填书籍元数据【state/chunk对象】
        self._fill_book_info(metadata, state, chunks)

        return state

    # ------------------------------------------------------------------
    # 元数据合并
    # ------------------------------------------------------------------
    def _merge_metadata(self, state: ImportGraphState, llm_meta: Dict[str, str],
                        file_title: str) -> Dict[str, str]:
        """
        合并规则：上传时用户显式提供的元数据 > LLM 识别结果 > 兜底值
        """
        self.log_step("step4", "合并元数据（用户上传值优先于LLM识别值）")
        metadata = {
            "book_name": (state.get("book_name") or llm_meta.get("book_name") or file_title).strip(),
            "author_name": (state.get("author_name") or llm_meta.get("author_name") or "").strip(),
            "content_type": (state.get("content_type") or llm_meta.get("content_type") or DEFAULT_CONTENT_TYPE).strip(),
            "category": (state.get("category") or llm_meta.get("category") or "").strip(),
            "audio_duration": (state.get("audio_duration") or llm_meta.get("audio_duration") or "").strip(),
            "entry_name": (state.get("entry_name") or llm_meta.get("entry_name") or "").strip(),
        }

        # 内容类型合法性校验：不在白名单中则回退默认值
        if metadata["content_type"] not in ALLOWED_CONTENT_TYPES:
            self.logger.warning(f"内容类型[{metadata['content_type']}]不在允许列表中，回退为默认值[{DEFAULT_CONTENT_TYPE}]")
            metadata["content_type"] = DEFAULT_CONTENT_TYPE

        self.logger.info(f"最终元数据: {json.dumps(metadata, ensure_ascii=False)}")
        return metadata

    def _fill_book_info(self, metadata: Dict[str, str], state: ImportGraphState,
                        chunks: List[Dict[str, Any]]):
        self.log_step("step7", "回填书籍元数据信息")
        for chunk in chunks:
            # 方便下游模型（嵌入/入库/图谱）能有参考
            chunk['book_name'] = metadata["book_name"]
            chunk['author_name'] = metadata["author_name"]
            chunk['content_type'] = metadata["content_type"]
            chunk['category'] = metadata["category"]
            chunk['audio_duration'] = metadata["audio_duration"]
            chunk['entry_name'] = metadata["entry_name"]
            chunk['source_file_name'] = state.get("file_title", "")
            chunk['source_url'] = state.get("source_url", "")

        # 程序员使用的时候更加方便
        state['book_name'] = metadata["book_name"]
        state['author_name'] = metadata["author_name"]
        state['content_type'] = metadata["content_type"]
        state['category'] = metadata["category"]
        state['audio_duration'] = metadata["audio_duration"]
        state['entry_name'] = metadata["entry_name"]

    def _embedding_book_name(self, book_name: str) -> Optional[Tuple[list, dict[Any, Any]]]:

        self.log_step("step5", "embedding模型嵌入书名")
        try:
            # 1. 获取嵌入模型
            embedding_model = get_beg_m3_embedding_model()

            # 2. 嵌入book_name
            embedding_result = embedding_model.encode_documents([book_name])

            # 3. 获取稠密和稀疏向量
            dense = embedding_result['dense'][0].tolist()
            start_index = embedding_result['sparse'].indptr[0]
            end_index = embedding_result['sparse'].indptr[1]
            weights = embedding_result['sparse'].data[start_index:end_index].tolist()
            tokenIds = embedding_result['sparse'].indices[start_index:end_index].tolist()
            sparse = dict(zip(tokenIds, weights))
            return dense, sparse
        except Exception as e:
            self.log_step(f"嵌入书名:{book_name}失败,原因是：{str(e)}")
            raise EmbeddingError(f"嵌入书名:{book_name}失败,原因是：{str(e)}", self.name)

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step("step1", "检验输入参数")
        config = get_config()

        # 1. 获取state的file_title以及 chunks
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        # 2. 判断提取到的参数
        if not file_title:
            raise ValidationError("文件标题为空", self.name)

        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunk为空或者无效", self.name)

        book_name_chunk_k = config.book_name_chunk_k
        if not book_name_chunk_k or book_name_chunk_k <= 0:
            raise ValidationError("book_name_chunk_k为空或者无效", self.name)

        self.logger.info(f"检测到文件：{file_title},对应的切片长度:{len(chunks)}")
        # 3. 返回
        return file_title, chunks, config

    def _prepare_book_info_context(self, chunks: Optional[List[Dict[str, Any]]], config):

        self.log_step("step2", "构建书籍元数据提取的上下文")
        result = []
        # 从前N块中收集内容，总字符数不能超过阈值
        total = 0
        for index, chunk in enumerate(chunks[:config.book_name_chunk_k]):

            # 1. 判断chunk的类型
            if not isinstance(chunk, dict):
                continue
            # 2. 提取内容（标题+正文）
            content = chunk.get('content')
            spices = f"【切片】- {index + 1} - {content}"

            # 3. 计算长度
            total += len(spices)

            result.append(spices)

            # 4. 判断收集到的长度是否超过阈值设定
            if total > config.book_name_chunk_size:
                break

        return "\n\n".join(result)[:config.book_name_chunk_size]

    def _recognition_book_info_by_llm(self, file_title: str, book_info_context: str) -> Dict[str, str]:
        """
        调用LLM识别书籍元数据，返回 dict；任何失败都做安全降级（返回空元数据，由上层兜底）
        """
        self.log_step("step3", "LLM识别书籍元数据")
        empty_meta = {"book_name": "", "author_name": "", "content_type": "",
                      "category": "", "audio_duration": "", "entry_name": ""}

        # 0. 若上传时用户已显式提供了全部关键元数据，则跳过 LLM 识别
        #    （在 _merge_metadata 中用户值优先，这里仅节省一次调用）

        # 1. 获取LLM客户端
        llm_client = get_llm_client()
        if llm_client is None:
            self.logger.error(f"LLM初始化失败,元数据识别安全回退")
            return empty_meta

        # 2. 构建LLM提示词(格式化用户提示词模版)
        prompt = BOOK_INFO_USER_PROMPT_TEMPLATE.format(
            file_title=file_title,
            context_header="【上传时提供的元数据】\n（无）",
            context=book_info_context
        )

        # 3. 调用模型
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=BOOK_INFO_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])

            # 3.1 获取模型的输出内容
            content = getattr(llm_response, 'content', '').strip()

            # 3.2 解析JSON（容忍 ```json 代码块包裹）
            meta = self._parse_json_response(content)
            if meta is None:
                self.logger.warning(f"LLM输出无法解析为JSON,安全回退。原始输出: {content[:200]}")
                return empty_meta

            # 3.3 字段规整
            result = {
                "book_name": str(meta.get("book_name", "") or "").strip(),
                "author_name": str(meta.get("author_name", "") or "").strip(),
                "content_type": str(meta.get("content_type", "") or "").strip(),
                "category": str(meta.get("category", "") or "").strip(),
                "audio_duration": str(meta.get("audio_duration", "") or "").strip(),
                "entry_name": str(meta.get("entry_name", "") or "").strip(),
            }
            if not result["book_name"] or result["book_name"].upper() == 'UNKNOWN':
                self.logger.warning(f"LLM无法提取有效的书名,安全回退")
                result["book_name"] = ""

            self.logger.info(f"LLM识别到的元数据: {json.dumps(result, ensure_ascii=False)}")
            return result

        # 4. 降级（安全处理）
        except Exception as e:
            self.logger.error(f"LLM调用失败,元数据识别安全回退: {e}")
            return empty_meta

    @staticmethod
    def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
        """容错解析 LLM 返回的 JSON（可能被 ```json ... ``` 包裹）"""
        if not content:
            return None
        # 1. 去掉 markdown 代码块包裹
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        # 2. 尝试直接解析
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except JSONDecodeError:
            pass
        # 3. 尝试截取第一个 { 到最后一个 } 之间的内容
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except JSONDecodeError:
                return None
        return None

    def _save_to_milvus(self, file_title, metadata: Dict[str, str],
                        dense_vector, sparse_vector, config):

        self.log_step("step6", "保存书名及元数据到向量数据库中")
        # 1. 参数检验
        if not dense_vector or not sparse_vector:
            self.logger.warning(f"[{metadata['book_name']}] 向量生成不完整，跳过入库！")
            return

        # 2. 操作MilVus
        try:
            # 2.1 获取Milvus的客户端
            milvus_client = get_milvus_client()

            # 判断
            if milvus_client is None:
                return

            # 2.2 获取集合的名字
            collection_name = config.book_name_collection

            # 2.3 幂等性校验（不存在则创建新的）
            if not milvus_client.has_collection(collection_name=collection_name):
                self._create_book_name_collection(milvus_client, collection_name)

            # 2.4 构建字典结构数据（书名 + 元数据 + 向量）
            data = {
                "file_title": file_title,  # 来源文件名
                "book_name": metadata["book_name"],  # 书名
                "author_name": metadata["author_name"],  # 作者名
                "content_type": metadata["content_type"],  # 内容类型
                "category": metadata["category"],  # 类别/标签
                "audio_duration": metadata["audio_duration"],  # 有声书时长
                "entry_name": metadata["entry_name"],  # 条目名称
                "dense_vector": dense_vector,  # 稠密向量 （list）
                "sparse_vector": sparse_vector  # 稀疏向量  (dict:{tokenId:weight})
            }

            # 2.5 插入数据到Milvus:{"insert_count":1,"ids":[10001]}
            result = milvus_client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")

        except Exception as e:
            self.logger.error(f"Milvus 数据库保存操作彻底失败: {e}")

    def _create_book_name_collection(self, client, collection_name):
        self.logger.info(f"正在创建集合: {collection_name}")

        # 1. 创建约束
        schema = client.create_schema()
        # 1.1 主键字段的约束
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)

        # 1.2 标量字段的约束（书名 + 元数据）
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="book_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="author_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="audio_duration", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="entry_name", datatype=DataType.VARCHAR, max_length=65535)

        # 1.3 向量字段的约束
        # 1.3.1 稠密向量字段约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 1.3.2 稀疏向量字段约束
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = client.prepare_index_params()
        # 2.1 创建稠密向量字段索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 2.2 创建稀疏向量字段索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")


# ---------------------------------
# 测试
# ---------------------------------
if __name__ == '__main__':
    import json

    setup_logging()

    chunk_json_path = r"D:\work\shopkeeper_brain-1\knowledge\temp_data\chunks_demo.json"

    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    state = {
        "file_title": "三体-有声书介绍",
        "chunks": chunk_content
    }

    book_info_recognition_node = BookInfoRecognitionNode()
    result = book_info_recognition_node.process(state)

    print(json.dumps(result, ensure_ascii=False, indent=4))
