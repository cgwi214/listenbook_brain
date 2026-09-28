"""答案输出节点 —— 骨架版本（第一步）"""

from typing import List, Dict, Any, Tuple
from processor.query_process.base import BaseNode
from processor.query_process.state import QueryGraphState
from prompts.query.query_prompt import ANSWER_PROMPT
from prompts.query.role_profile import get_role_system
from utils.llm_client_util import get_llm_client
from utils.task_util import set_task_result
from utils.sse_util import push_sse_event, SSEEvent
from utils.mongo_history_util import save_chat_message


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        task_id = state.get("task_id")
        is_stream = state.get("is_stream")

        # 1. 已有答案 → 直接返回
        if state.get("answer"):
            self._push_existing_answer(state)

        # 2. 构建提示词 → 调用 LLM 生成答案
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(state, prompt)

        # 3. 写入历史记录（用户问题 + 助手回答）
        self._write_history(state)

        # 4. 流式模式发送结束事件
        if is_stream:
            push_sse_event(task_id, SSEEvent.FINAL,
                           {"answer": state.get("answer", "")})
        return state

    def _push_existing_answer(self, state: QueryGraphState):
        """非流式模式：存入任务结果；流式模式：让 FINAL 统一推送。"""
        if not state.get("is_stream"):
            set_task_result(state["task_id"], "answer", state["answer"])

    def _generate_answer(self, state, prompt):
        self.log_step("generate", "生成答案")
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")

        task_id = state.get("task_id")
        if not task_id:
            raise ValueError("缺少 task_id")

        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(
                llm_client,
                prompt
            )
            set_task_result(task_id, "answer", state["answer"])

    def _build_prompt(self, state: QueryGraphState) -> str:
        char_budget = self.config.max_context_chars

        # 1. 获取问题和书名
        question = state.get("rewritten_query") or state.get("original_query", "")
        book_names = state["book_names"]

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 格式化图谱关系
        graph_str, char_budget = self._format_kg_triples(
            state.get("kg_triples") or [], char_budget
        )

        # 5. 组装提示词（注入当前角色职能定位）
        return ANSWER_PROMPT.format(
            role_instruction=get_role_system(state.get("role")),
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            book_names=", ".join(book_names),
            graph_relation_description=graph_str or "无图谱关系",
            question=question,
        )

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("book_name", "[书名={}]"), ("author_name", "[作者={}]"),
                ("content_type", "[内容类型={}]"), ("category", "[类别={}]"),
                ("audio_duration", "[有声书时长={}]"), ("entry_name", "[条目名={}]"),
                ("source_file_name", "[来源文件名={}]"), ("file_title", "[来源文件={}]"),
                ("source", "[source={}]"), ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"), ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    @staticmethod
    def _format_kg_triples(kg_triples: List, char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0
        for triple in kg_triples:
            triple_text = (str(triple) if triple is not None else "").strip()
            if not triple_text:
                continue
            if used_chars + len(triple_text) > char_budget:
                break
            formatted_lines.append(triple_text)
            used_chars += len(triple_text) + 1
        return "\n".join(formatted_lines), char_budget - used_chars

    def _invoke_generate(self, llm_client, prompt: str):
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    def _stream_generate(self, llm_client, prompt, task_id):
        """流式生成，逐 chunk 推送 delta 事件。
         返回的是一个一个token(不是一个中文字符就是一个token )
        """
        accumulated_answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id, "delta", {"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return accumulated_answer

        # ★ 新增方法

    def _write_history(self, state: QueryGraphState):
        session_id = state["session_id"]
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        book_names = state.get("book_names") or []
        try:
            # 1. 写用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
                rewritten_query=rewritten_query,
                book_names=book_names,
            )
            # 2. AI回复（假的+真的）
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],  # 模型的输出
                    rewritten_query=rewritten_query,
                    book_names=book_names,
                )
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")

_node_instance = AnswerOutputNode()


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)


if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    # 加载环境变量
    load_dotenv()

    # 初始化日志
    from processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("开始测试: 答案生成节点 (AnswerOutputNode)")
    print("=" * 60)

    # 构造模拟状态
    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": False,

        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",

        "book_names": [
            "RS-12数字万用表"
        ],

        "reranked_docs": [
            {
                "content": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            }
        ],

        "history": [
            {
                "role": "user",
                "text": "万用表是什么？"
            },
            {
                "role": "assistant",
                "text": "万用表是一种多功能电子测量仪器..."
            }
        ],

        "kg_triples": [
            "万用表 -[包含]-> 表笔",
            "万用表 -[包含]-> 旋钮",
            "表笔 -[用于]-> 测量电压"
        ]
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  book_names: {mock_state['book_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")
    print(f"  kg_triples: {len(mock_state['kg_triples'])} 条")
    print("-" * 60)

    # 执行答案生成
    result = node_answer_output(mock_state)

    # 打印结果
    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)

    # 打印提示词（调试用）
    if result.get("prompt"):
        print("\n【构建的提示词】:")
        print(result["prompt"][:500] + "...")

    print("\n测试完成")
