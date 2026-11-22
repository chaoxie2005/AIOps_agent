import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from aiops_agent.agent_components import llm
from aiops_agent.agent_components.memory.context import (
    estimate_words, evaluate_history, tool_result_chunks,
)
from aiops_agent.agent_components.memory.ingestion import TurnIngestion
from aiops_agent.graph_components.graph.main_graph import MainGraph


@pytest.mark.parametrize(("text", "expected"), [
    (" \n\t", 0),
    ("one two", 2),
    ("你好", 1),
    ("hello世界 and日本語", 6),
    ("あア한", 2),
    ("中" * 15, 11),
])
def test_estimate_words(text, expected):
    assert estimate_words(text) == expected


def test_watermark_counts_full_checkpoint_history_in_prompt_order():
    query = "长" * 8000
    turns = [{"query": query, "reply": "done", "result": None}]
    prompt = [SystemMessage(content="system"), HumanMessage(content="current")]
    counter = Mock(side_effect=[69, 70, 71])

    for tokens in (69, 70):
        assessment = evaluate_history(turns, prompt, 100, counter)
        assert assessment.estimated_tokens == tokens
        assert not assessment.needs_compression
        assert assessment.chunks == ()

    assessment = evaluate_history(turns, prompt, 100, counter)
    assert assessment.needs_compression
    assert [message.content for message in counter.call_args.args[0]] == ["system", query, "done", "current"]
    assert [chunk.part_index for chunk in assessment.chunks] == [0, 1]
    assert all(chunk.source == "turn" and chunk.source_id == "1" for chunk in assessment.chunks)
    assert all(len(chunk.text) < 8000 for chunk in assessment.chunks)
    assert "".join(chunk.text for chunk in assessment.chunks) == f"用户：{query}\n助手：done"


def test_tool_slices_preserve_full_payload_and_event_id():
    content = "日志" * 4000 + "\n尾行"
    event = {
        "_id": "tool_out_1",
        "tool_call_id": "call_1",
        "tool_name": "query_metrics",
        "arguments": {"query": "cpu"},
        "status": "success",
        "content": content,
    }
    chunks = tool_result_chunks(event)
    assert all(chunk.source_id == event["_id"] and chunk.source == "tool" for chunk in chunks)
    assert all(0 < len(chunk.text) < 8000 for chunk in chunks)
    assert [chunk.part_index for chunk in chunks] == list(range(len(chunks)))
    call, _, restored = "".join(chunk.text for chunk in chunks).partition("\n")
    assert '"query": "cpu"' in call
    assert restored == content


def test_guard_drops_empty_units_and_preserves_short_user_instructions():
    turns = [
        {"query": " \t", "reply": "OK。", "result": None},
        {"query": "ok", "reply": "", "result": None},
        {"query": "不行", "reply": " ", "result": None},
        {"query": "", "reply": "错误码 E42", "result": None},
    ]
    prompt = [SystemMessage(content="system"), HumanMessage(content="current")]

    assessment = evaluate_history(turns, prompt, 100, lambda _: 71)

    assert [(chunk.source_id, chunk.text) for chunk in assessment.compressible_chunks] == [
        ("2", "用户：ok"),
        ("3", "用户：不行"),
        ("4", "助手：错误码 E42"),
    ]


@pytest.mark.parametrize("body", ["", " \n", "OK。", "收到"])
def test_guard_skips_filler_tool_body_but_keeps_call_fields(body):
    event = {
        "_id": "event",
        "tool_name": "query_metrics",
        "tool_call_id": "call",
        "arguments": {"query": "up"},
        "status": "success",
        "content": body,
    }
    chunk, = tool_result_chunks(event)

    assert chunk.source_id == "event"
    assert "\n" not in chunk.text
    assert '"query_metrics"' in chunk.text and '"query": "up"' in chunk.text
    assert '"call"' in chunk.text
    assert not chunk.preserve_original


def test_guard_preserves_short_tool_errors_and_structured_empty_results():
    failed, = tool_result_chunks({"_id": "error", "status": "error", "content": "E1"})
    empty_data, = tool_result_chunks({"_id": "data", "status": "success", "content": []})

    assert failed.text.partition("\n")[2] == "E1"
    assert empty_data.text.partition("\n")[2] == "[]"


def test_guard_protects_all_parts_of_a_split_traceback():
    query = (
        "日志如下\n" + "x" * 8000
        + '\n```text\nTraceback (most recent call last):\r\n'
        + '  File "app.py", line 42, in process\r\nRuntimeError: test\n```'
    )
    turns = [{"query": query, "reply": "检查第 42 行", "result": None}]
    prompt = [SystemMessage(content="system"), HumanMessage(content="current")]

    assessment = evaluate_history(turns, prompt, 100, lambda _: 71)

    assert len(assessment.chunks) > 1
    assert all(chunk.preserve_original and len(chunk.text) < 8000 for chunk in assessment.chunks)
    assert assessment.compressible_chunks == ()
    assert "".join(chunk.text for chunk in assessment.chunks) == f"用户：{query}\n助手：检查第 42 行"


@pytest.mark.parametrize("body", [
    "2026-01-01 ERROR worker Traceback (most recent call last):\n2026-01-01 ERROR RuntimeError: test",
    "Error: test\n    at run (/app/index.js:42:10)",
    "java.lang.RuntimeException: test\n\tat example.App.main(App.java:42)",
    '{"stack":"Error: test\\n    at run (node:internal/test:42:10)"}',
])
def test_guard_marks_tool_stack_traces_as_original(body):
    chunks = tool_result_chunks({"_id": "event", "status": "error", "content": body})

    assert all(chunk.preserve_original for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).partition("\n")[2] == body


def test_archive_read_is_scoped_and_excludes_current_turn():
    collection = Mock()
    collection.find.return_value.sort.return_value = [{"_id": "tool_out_old"}]
    ingestion = TurnIngestion(collection, "session-1")

    assert asyncio.run(ingestion.read_history_tool_events()) == [{"_id": "tool_out_old"}]
    assert collection.find.call_args.args[0] == {
        "session_id": "session-1", "kind": "tool", "turn_id": {"$ne": ingestion.turn_id},
    }
    collection.find.return_value.sort.assert_called_once_with([("created_at", 1), ("_id", 1)])


def test_graph_prepares_slices_only_above_watermark(monkeypatch):
    async def run():
        model = SimpleNamespace(
            input_token_limit=100,
            # 第二轮超线，必保的长堆栈也超出输入预算，继续使用近期历史。
            estimate_message_tokens=Mock(side_effect=[70, 71, 101]),
            ainvoke=AsyncMock(return_value=AIMessage(content='{"intent":"chat"}')),
        )
        traceback = 'Traceback (most recent call last):\n  File "app.py", line 42\n' + "log" * 4000
        reader = AsyncMock(return_value=[{"_id": "tool_out_1", "content": traceback}])
        ingestion = SimpleNamespace(read_history_tool_events=reader, record=AsyncMock())
        store = InMemorySaver()
        nodes = SimpleNamespace(model_client=model)
        for query in ("first", "second"):
            graph = MainGraph(query, rag_nodes=nodes, checkpointer=store, thread_id="session", ingestion=ingestion)
            assert await graph.ainvoke() == {"intent": "chat", "reply": "answer", "result": None}
            assert graph.context_evaluation is not None
            if query == "first":
                assert graph.context_evaluation.chunks == ()
                reader.assert_not_awaited()

        assert [message.content for message in model.ainvoke.call_args.args[0][1:]] == [
            "first", "answer", "second",
        ]
        assert [chunk.source for chunk in graph.context_evaluation.chunks] == ["turn", "tool", "tool"]
        assert [chunk.source for chunk in graph.context_evaluation.compressible_chunks] == ["turn"]
        reader.assert_awaited_once()
        checkpoint = await store.aget_tuple({"configurable": {"thread_id": "session"}})
        assert "context_evaluation" not in checkpoint.checkpoint["channel_values"]

    async def reply(self, state):
        return {"reply": "answer", "result": None}

    monkeypatch.setattr(MainGraph, "_chat", reply)
    asyncio.run(run())


def test_model_budget_uses_explicit_window_and_falls_back_to_local_tokenizer(monkeypatch):
    model = SimpleNamespace(
        profile={"max_input_tokens": 8000},
        get_num_tokens_from_messages=Mock(side_effect=NotImplementedError),
        get_num_tokens=len,
    )
    monkeypatch.setattr(llm, "load_dotenv", Mock())
    monkeypatch.setattr(llm, "init_chat_model", Mock(return_value=model))

    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "4000")
    explicit = llm.ModelClient("test", "test", base_url="http://localhost/v1", max_tokens=500)
    assert explicit.input_token_limit == 3500
    assert explicit.estimate_message_tokens([HumanMessage(content="test")]) == 11

    monkeypatch.delenv("LLM_CONTEXT_WINDOW")
    profiled = llm.ModelClient("test", "test", base_url="http://localhost/v1", max_tokens=500)
    assert profiled.input_token_limit == 7500

    model.profile = None
    unknown = llm.ModelClient("test", "test", base_url="http://localhost/v1", max_tokens=500)
    assert unknown.input_token_limit is None
