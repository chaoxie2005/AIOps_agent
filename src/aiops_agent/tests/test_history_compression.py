import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from aiops_agent.graph_components.graph import main_graph
from aiops_agent.graph_components.nodes import chat_node
from aiops_agent.graph_components.nodes.compress_history_node import compress_history_node


_PROMPT = [SystemMessage(content="intent system"), HumanMessage(content="current question")]


def _turn(query="保留用户原话", reply="旧回复"):
    return {"query": query, "reply": reply, "result": None}


def _client(responses=(), *, historical_tokens=71):
    return SimpleNamespace(
        input_token_limit=100,
        estimate_message_tokens=Mock(side_effect=[historical_tokens] + [20] * 30),
        estimate_text_tokens=len,
        ainvoke=AsyncMock(side_effect=responses),
    )


def test_below_watermark_skips_archive_and_summary_model():
    model = _client(historical_tokens=70)
    ingestion = SimpleNamespace(read_history_tool_events=AsyncMock())

    assessment, history = asyncio.run(compress_history_node([_turn()], _PROMPT, model, ingestion))

    assert assessment is not None and not assessment.needs_compression
    assert history is None
    ingestion.read_history_tool_events.assert_not_awaited()
    model.ainvoke.assert_not_awaited()


def test_contract_retries_oversized_draft_and_retains_verbatim_fields():
    stack = 'Traceback (most recent call last):\r\n  File "app.py", line 42\r\nRuntimeError: test\r\n'
    events = [{
        "_id": "tool_out_42",
        "tool_call_id": "call_42",
        "tool_name": "query_metrics",
        "arguments": {"query": "cpu_total", "minutes": 5},
        "status": "error",
        "content": stack,
    }]
    ingestion = SimpleNamespace(read_history_tool_events=AsyncMock(return_value=events))
    model = _client([AIMessage(content="长" * 201), AIMessage(content="已测试 CPU。")])
    query = "只查看最近 5 分钟"

    assessment, history = asyncio.run(compress_history_node([_turn(query)], _PROMPT, model, ingestion))

    assert assessment is not None
    assert history is not None
    assert model.ainvoke.await_count == 2
    assert "收紧措辞，不准新增事实" in model.ainvoke.await_args.args[0][-1].content
    assert "长" * 201 in [message.content for message in model.ainvoke.await_args.args[0]]
    assert [chunk.source for chunk in assessment.compressible_chunks] == ["turn"]
    assert "已测试 CPU。" in history
    assert query in history
    assert '"query": "cpu_total", "minutes": 5' in history
    assert all(value in history for value in ("query_metrics", "call_42", "tool_out_42", stack))
    assert history.endswith(stack)
    assert all(stack not in message.content for call in model.ainvoke.await_args_list for message in call.args[0])


def test_batches_merge_in_order_using_previous_summary():
    turns = [_turn(f"第 {index} 个请求", f"第 {index} 个回复") for index in range(1, 10)]
    model = _client([AIMessage(content="前八轮摘要"), AIMessage(content="合并后的九轮摘要")])

    _, history = asyncio.run(compress_history_node(turns, _PROMPT, model, None))

    assert model.ainvoke.await_count == 2
    first, second = model.ainvoke.await_args_list
    assert "第 8 个回复" in first.args[0][1].content
    assert "第 9 个回复" not in first.args[0][1].content
    assert "前八轮摘要" in second.args[0][1].content
    assert "第 9 个回复" in second.args[0][1].content
    assert history is not None and "合并后的九轮摘要" in history


def test_stack_only_bypasses_summary_but_can_be_injected():
    stack = "TypeError: test\n    at run (/app/index.js:42:10)"
    model = _client()
    assessment, history = asyncio.run(compress_history_node([_turn(stack)], _PROMPT, model, None))

    assert assessment is not None and not assessment.compressible_chunks
    assert history is not None and stack in history
    model.ainvoke.assert_not_awaited()


def test_required_history_over_budget_skips_model():
    model = _client()
    model.estimate_message_tokens.side_effect = [71, 101]
    evaluation, history = asyncio.run(compress_history_node([_turn()], _PROMPT, model, None))

    assert evaluation is not None and evaluation.needs_compression
    assert history is None
    model.ainvoke.assert_not_awaited()


def test_oversized_summary_exhausts_two_corrections():
    model = _client([AIMessage(content="长" * 201)] * 3)
    _, history = asyncio.run(compress_history_node([_turn()], _PROMPT, model, None))

    assert history is None
    assert model.ainvoke.await_count == 3


@pytest.mark.parametrize("summary_fails", [False, True])
def test_main_graph_passes_history_to_intent_and_chat(monkeypatch, summary_fails):
    async def run():
        def count(messages):
            # 只有待压缩的完整旧问答达到水位线，最终输入仍在预算内。
            if any(message.content == "old answer" for message in messages):
                return 71
            return 20

        async def invoke(messages):
            if "会话历史压缩器" in messages[0].content:
                if summary_fails:
                    raise RuntimeError("synthetic model failure")
                return AIMessage(content="compressed answer")
            return AIMessage(content='{"intent":"chat"}')

        answers = [AIMessage(content="old answer"), AIMessage(content="new answer")]
        chat = SimpleNamespace(ainvoke=AsyncMock(side_effect=lambda *_: {"messages": [answers.pop(0)]}))
        model = SimpleNamespace(
            model=object(), ainvoke=AsyncMock(side_effect=invoke), input_token_limit=100,
            estimate_message_tokens=count, estimate_text_tokens=len,
        )
        nodes = SimpleNamespace(model_client=model, retriever=object())
        monkeypatch.setattr(chat_node, "create_agent", lambda **kwargs: chat)
        monkeypatch.setattr(main_graph, "make_rag_search_tool", lambda _: object())
        store = InMemorySaver()

        for query, reply in (("old question", "old answer"), ("new question", "new answer")):
            graph = main_graph.MainGraph(query, rag_nodes=nodes, thread_id="session", checkpointer=store)
            assert "compress_history" in graph.graph.nodes
            assert await graph.ainvoke() == {"intent": "chat", "reply": reply, "result": None}

        summary_calls = [
            call for call in model.ainvoke.await_args_list if "会话历史压缩器" in call.args[0][0].content
        ]
        assert len(summary_calls) == 1
        intent_history = model.ainvoke.await_args.args[0][1:-1]
        chat_history = chat.ainvoke.await_args.args[0]["messages"][:-1]
        assert [message.content for message in intent_history] == [message.content for message in chat_history]
        if summary_fails:
            assert graph.compressed_history is None
            assert [message.content for message in chat_history] == ["old question", "old answer"]
        else:
            assert graph.compressed_history is not None
            assert "compressed answer" in chat_history[0].content
            assert "old question" in chat_history[0].content

    asyncio.run(run())


def test_final_history_over_budget_is_not_injected():
    model = _client([AIMessage(content="短摘要")])
    model.estimate_message_tokens.side_effect = [71, 95, 50, 101]
    _, history = asyncio.run(compress_history_node([_turn()], _PROMPT, model, None))

    assert history is None
    model.ainvoke.assert_awaited_once()


def test_main_graph_injects_summary_into_diagnosis_stages(monkeypatch):
    async def run():
        def count(messages):
            if any(message.content == "old diagnostic result" for message in messages):
                return 71
            return 20

        async def invoke(messages):
            if "会话历史压缩器" in messages[0].content:
                return AIMessage(content="compressed diagnosis")
            return AIMessage(content='{"intent":"aiops"}')

        result = {
            "summary": "old diagnostic result", "root_cause": None,
            "evidence_ids": [], "knowledge_document_ids": [], "recommendations": [], "limitations": [],
        }
        rag = AsyncMock(return_value={"unique_documents": []})
        plan = AsyncMock(return_value={"steps": []})
        replan = AsyncMock(return_value={"result": result})
        monkeypatch.setattr(main_graph, "rag_subgraph", rag)
        monkeypatch.setattr(main_graph, "plan_node", plan)
        monkeypatch.setattr(main_graph, "replan_node", replan)
        model = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=invoke), input_token_limit=100,
            estimate_message_tokens=count, estimate_text_tokens=len,
        )
        store = InMemorySaver()
        for query in ("old diagnostic request", "new diagnostic request"):
            graph = main_graph.MainGraph(
                query, max_steps=0, rag_nodes=SimpleNamespace(model_client=model),
                thread_id="diagnostic session", checkpointer=store,
            )
            assert (await graph.ainvoke())["result"] == result

        rag_query = rag.await_args.args[0]
        assert "compressed diagnosis" in rag_query and "old diagnostic request" in rag_query
        assert rag_query.endswith("本轮问题：\nnew diagnostic request")
        assert plan.await_args.args[0]["query"] == rag_query
        assert replan.await_args.args[0]["query"] == rag_query

    asyncio.run(run())
