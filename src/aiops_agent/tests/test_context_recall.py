import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph.checkpoint.memory import InMemorySaver

from aiops_agent.agent_components.memory.ingestion import TurnIngestion
from aiops_agent.agent_components.toolkits.recall_history import HistoryRecall
from aiops_agent.graph_components.graph import main_graph
from aiops_agent.graph_components.nodes.compress_history_node import compress_history_node


_EVENT_ID = "tool_out_previous_query"
_SYSTEM = SystemMessage(content="system")
_QUESTION = HumanMessage(content="继续查看 CPU")
_TURN = {"query": "只检查最近 5 分钟", "reply": "已发起检查", "result": None}


def _counter(messages):
    return sum(len(message.content) + 1 for message in messages)


def _summary(text, *refs):
    return AIMessage(content=json.dumps([{"text": text, "refs": refs}], ensure_ascii=False))


def _call(call_id, source_id=_EVENT_ID, offset=0):
    return {
        "name": "recall_history", "id": call_id,
        "args": {"source_id": source_id, "offset": offset},
    }


def _lookup(content="CPU=92%"):
    event = {
        "_id": _EVENT_ID, "kind": "tool", "tool_name": "query_metrics",
        "arguments": {"query": "cpu"}, "content": content,
    }
    collection = Mock()
    collection.find_one.side_effect = lambda query, _: (
        event if query["_id"] == event["_id"] and query["session_id"] == "owner_session" else None
    )
    return TurnIngestion(collection, "owner_session"), event


def test_invalid_summary_reference_retries_and_program_renders_canonical_ids():
    async def run():
        ingestion, event = _lookup()
        ingestion.read_history_tool_events = AsyncMock(return_value=[event])
        model = SimpleNamespace(
            input_token_limit=100,
            estimate_message_tokens=Mock(side_effect=[71] + [20] * 10),
            estimate_text_tokens=len,
            ainvoke=AsyncMock(side_effect=[
                _summary("无来源的结论", 99),
                _summary("CPU 达到 92%", 1, 2),
            ]),
        )

        _, history = await compress_history_node([_TURN], [_SYSTEM, _QUESTION], model, ingestion)

        assert model.ainvoke.await_count == 2
        assert "可见的来源编号" in model.ainvoke.await_args.args[0][-1].content
        assert history is not None
        assert f"CPU 达到 92% [溯源: turn:1, {_EVENT_ID}]" in history
        assert "无来源的结论" not in history

    asyncio.run(run())


def test_recalled_pages_reassemble_archived_data_and_respect_session():
    async def run():
        ingestion, event = _lookup("日志正文\n" * 2000)
        owner = HistoryRecall(ingestion, [_TURN], SimpleNamespace(
            input_token_limit=30_000, estimate_message_tokens=_counter,
        ))
        tool = owner.make_tool(_SYSTEM)
        assert set(tool.tool_call_schema.model_json_schema()["properties"]) == {"source_id", "offset"}

        messages = [_QUESTION]
        pages = []
        offset = 0
        while offset is not None:
            assert len(pages) < 4
            runtime = SimpleNamespace(state={"messages": messages}, tool_call_id=f"call_{len(pages)}")
            reply = await tool.coroutine(source_id=_EVENT_ID, offset=offset, runtime=runtime)
            payload = json.loads(reply.content)
            assert payload["status"] == "ok" and payload["offset"] == offset
            assert 0 < len(payload["content"]) < 8000
            pages.append(payload["content"])
            messages.append(reply)
            offset = payload["next_offset"]

        assert len(pages) > 1
        assert json.loads("".join(pages)) == event
        assert ingestion.collection.find_one.call_args.args[0] == {
            "_id": _EVENT_ID, "session_id": "owner_session", "turn_id": {"$ne": ingestion.turn_id},
        }
        other = HistoryRecall(
            TurnIngestion(ingestion.collection, "different_session"), [], owner.model_client,
        )
        runtime = SimpleNamespace(state={"messages": []}, tool_call_id="unrelated")
        denied = await other.make_tool(_SYSTEM).coroutine(source_id=_EVENT_ID, runtime=runtime)
        assert json.loads(denied.content)["status"] == "not_found"

    asyncio.run(run())


def test_turn_reference_returns_full_checkpoint_turn():
    async def run():
        ingestion, _ = _lookup()
        turn = {
            "query": "原始输入", "reply": None,
            "result": {"summary": "摘要", "root_cause": "原始根因", "recommendations": ["原始建议"]},
        }
        recall = HistoryRecall(
            ingestion, [turn], SimpleNamespace(input_token_limit=1000, estimate_message_tokens=_counter),
        )
        runtime = SimpleNamespace(state={"messages": []}, tool_call_id="turn_call")
        tool = recall.make_tool(_SYSTEM)
        reply = await tool.coroutine(source_id="turn:1", runtime=runtime)
        assert json.loads(json.loads(reply.content)["content"]) == turn
        bad = await tool.coroutine(source_id="turn:2", runtime=runtime)
        assert json.loads(bad.content)["status"] == "not_found"
        ingestion.collection.find_one.assert_not_called()

    asyncio.run(run())


def test_parallel_recalls_share_remaining_input_budget():
    async def run():
        ingestion, _ = _lookup("log" * 1000)
        model = SimpleNamespace(input_token_limit=300, estimate_message_tokens=_counter)
        tool = HistoryRecall(ingestion, [], model).make_tool(_SYSTEM)
        calls = [_call("parallel_1"), _call("parallel_2")]
        history = [_QUESTION, AIMessage(content="", tool_calls=calls)]
        responses = await asyncio.gather(*(
            tool.coroutine(source_id=_EVENT_ID, runtime=SimpleNamespace(
                state={"messages": history}, tool_call_id=call["id"],
            ))
            for call in calls
        ))

        assert _counter([_SYSTEM, *history, *responses]) <= model.input_token_limit
        statuses = {json.loads(response.content)["status"] for response in responses}
        assert statuses == {"ok", "budget_exhausted"}

    asyncio.run(run())


def test_summary_reference_round_trips_through_real_langchain_tool_call():
    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    async def run():
        ingestion, event = _lookup()
        ingestion.read_history_tool_events = AsyncMock(return_value=[event])
        compressor = SimpleNamespace(
            input_token_limit=100,
            estimate_message_tokens=Mock(side_effect=[71] + [20] * 10),
            estimate_text_tokens=len,
            ainvoke=AsyncMock(return_value=_summary("已查询 CPU", 2)),
        )
        _, history = await compress_history_node([_TURN], [_SYSTEM, _QUESTION], compressor, ingestion)
        assert history is not None
        ref = history.split("[溯源: ", 1)[1].split("]", 1)[0]
        assert ref == _EVENT_ID

        recall = HistoryRecall(
            ingestion, [_TURN], SimpleNamespace(input_token_limit=5000, estimate_message_tokens=_counter),
        )
        agent = create_agent(
            model=ToolModel(responses=[
                AIMessage(content="", tool_calls=[_call("recall-1", ref)]),
                AIMessage(content="CPU 使用率为 92%。"),
            ]),
            tools=[recall.make_tool(_SYSTEM)],
            system_prompt=_SYSTEM,
        )
        output = await agent.ainvoke({"messages": [HumanMessage(content=history), _QUESTION]})
        reply = next(message for message in output["messages"] if isinstance(message, ToolMessage))

        assert reply.tool_call_id == "recall-1"
        payload = json.loads(reply.content)
        assert payload["source_id"] == ref
        assert json.loads(payload["content"])["content"] == "CPU=92%"
        assert output["messages"][-1].content == "CPU 使用率为 92%。"

    asyncio.run(run())


@pytest.mark.parametrize("intent", ["chat", "aiops"])
def test_main_graph_mounts_and_executes_recall_in_both_agents(monkeypatch, intent):
    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    @tool
    def search_docs(query: str) -> str:
        """Return synthetic knowledge for the test."""
        return query

    async def run():
        ingestion, event = _lookup()
        ingestion.read_history_tool_events = AsyncMock(return_value=[event])
        ingestion.record = AsyncMock()

        def count(messages):
            if any(message.content == _TURN["reply"] for message in messages):
                return 4000
            return _counter(messages)

        async def invoke(messages):
            if "会话历史压缩器" in messages[0].content:
                return _summary("已查询 CPU", 2)
            return AIMessage(content=json.dumps({"intent": intent}))

        final = "CPU 92%" if intent == "chat" else '{"status":"completed","summary":"CPU 92%"}'
        model = SimpleNamespace(
            model=ToolModel(responses=[
                AIMessage(content="", tool_calls=[_call("recall-from-main")]),
                AIMessage(content=final),
            ]),
            input_token_limit=5000, estimate_message_tokens=count, estimate_text_tokens=len,
            ainvoke=AsyncMock(side_effect=invoke),
        )
        monkeypatch.setattr(main_graph, "make_rag_search_tool", lambda _: search_docs)
        monkeypatch.setattr(main_graph, "rag_subgraph", AsyncMock(return_value={"unique_documents": []}))
        monkeypatch.setattr(main_graph, "plan_node", AsyncMock(return_value={
            "steps": [{"id": "s1", "goal": "检查 CPU", "status": "pending"}],
        }))
        diagnosis = {
            "summary": "CPU 92%", "root_cause": None, "evidence_ids": [], "knowledge_document_ids": [],
            "recommendations": [], "limitations": [],
        }
        monkeypatch.setattr(main_graph, "replan_node", AsyncMock(return_value={"result": diagnosis}))

        graph = main_graph.MainGraph(
            "查看具体 CPU 数值", rag_nodes=SimpleNamespace(model_client=model, retriever=object()),
            checkpointer=InMemorySaver(), thread_id="owner_session", ingestion=ingestion,
        )
        await graph.graph.aupdate_state(
            {"configurable": {"thread_id": "owner_session"}}, {"turns": [_TURN]}, as_node="record_turn",
        )
        output = await graph.ainvoke()

        assert output["intent"] == intent
        assert f"已查询 CPU [溯源: {_EVENT_ID}]" in graph.compressed_history
        call = next(call for call in ingestion.record.await_args_list if call.args[0] == "tool")
        assert call.kwargs["tool_name"] == "recall_history"
        payload = json.loads(call.args[1])
        assert json.loads(payload["content"])["content"] == "CPU=92%"

    asyncio.run(run())
