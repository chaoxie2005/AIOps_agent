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
            estimate_message_tokens=Mock(side_effect=[70, 71]),
            ainvoke=AsyncMock(return_value=AIMessage(content='{"intent":"chat"}')),
        )
        reader = AsyncMock(return_value=[{"_id": "tool_out_1", "content": "log" * 4000}])
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
