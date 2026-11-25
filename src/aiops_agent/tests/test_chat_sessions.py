import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from aiops_agent.api.app import app
from aiops_agent.graph_components.graph import main_graph
from aiops_agent.graph_components.nodes import chat_node
from aiops_agent.agent_components.memory.chat_sessions import ChatSessions
from aiops_agent.agent_components.memory.ingestion import TurnIngestion


class SessionDirectory:
    def __init__(self):
        """初始化内存会话目录，替代测试中的 MongoDB 集合。"""
        self.rows = {}

    def insert_one(self, row):
        """以会话 ID 为键保存一份记录副本。

        Args:
            row: 包含会话主键和归属信息的目录记录。
        """
        self.rows[row["_id"]] = dict(row)

    def find_one(self, match, projection):
        """同时按会话 ID 和用户 ID 模拟 MongoDB 的单条查询。

        Args:
            match: 包含 _id 和 user_id 的查询条件。
            projection: 为兼容 MongoDB 接口而接收的字段投影；替身不裁剪字段。

        Returns:
            匹配的目录记录；会话不存在或用户不匹配时返回 None。
        """
        row = self.rows.get(match["_id"])
        return row if row is not None and row["user_id"] == match["user_id"] else None

    def find(self, match):
        """模拟按用户查询 MongoDB 会话目录。

        Args:
            match: 包含 user_id 的筛选条件。

        Returns:
            带 sort 方法的游标替身，调用后返回按创建时间倒序排列的记录。
        """
        return SimpleNamespace(sort=lambda _: sorted(
            (row for row in self.rows.values() if row["user_id"] == match["user_id"]),
            key=lambda row: row["created_at"],
            reverse=True,
        ))


class ContextEvents:
    """模拟 MongoDB 的会话事件插入语义。"""

    def __init__(self):
        """初始化空的内存事件集合。"""
        self.rows = {}

    def update_one(self, match, update, *, upsert):
        """模拟按事件 ID 进行幂等插入。

        Args:
            match: 包含事件 ID 和会话 ID 的过滤条件。
            update: 包含待插入事件的 MongoDB 更新文档。
            upsert: 允许在事件缺失时插入。

        Raises:
            AssertionError: 未启用 upsert，或事件与过滤条件的 ID 不一致。
        """
        assert upsert
        event = update["$setOnInsert"]
        assert event["_id"] == match["_id"]
        assert event["session_id"] == match["session_id"]
        self.rows.setdefault(match["_id"], event)


@pytest.fixture
def storage(monkeypatch):
    """用内存会话目录、事件集合和 LangGraph checkpointer 替换 MongoDB。

    Args:
        monkeypatch: pytest 提供的临时替换工具。

    Returns:
        供同一测试内多个应用生命周期共用的存储对象。
    """
    backend = SimpleNamespace(
        client=Mock(),
        sessions=SessionDirectory(),
        checkpointer=InMemorySaver(),
        context_events=ContextEvents(),
    )
    monkeypatch.setattr(ChatSessions, "_connect", staticmethod(lambda: backend))
    return backend


@pytest.fixture
def chat_model(monkeypatch):
    """替换模型和检索依赖，让测试可以直接检查传给模型的会话上下文。

    Args:
        monkeypatch: pytest 提供的临时替换工具。

    Returns:
        可记录调用参数的意图分类器和聊天 Agent 替身。
    """
    classifier = AsyncMock(return_value=AIMessage(content='{"intent":"chat"}'))

    async def answer(state):
        """根据最新用户消息生成固定格式的测试回复。

        Args:
            state: 传给聊天 Agent 的消息列表及其他状态。

        Returns:
            包含一条助手回复的状态。

        Raises:
            ValueError: 收到专门用于模拟模型失败的测试问题。
        """
        query = state["messages"][-1].content
        if isinstance(query, str) and query.startswith("fail this turn"):
            raise ValueError("model failed")
        return {"messages": [AIMessage(content=f"Reply: {query}")]}

    async def classify(messages):
        """按输入内容分类；失败用例保留原有语义，让 Agent 抛错。"""
        query = messages[-1].content
        if isinstance(query, str) and query.startswith("fail this turn"):
            raise ValueError("model failed")
        return await classifier(messages)

    agent = SimpleNamespace(ainvoke=AsyncMock(side_effect=answer))
    model = SimpleNamespace(model=object(), ainvoke=classify)
    monkeypatch.setattr(main_graph, "ModelClient", lambda: model)
    monkeypatch.setattr(main_graph.RagNodes, "from_env", lambda: SimpleNamespace(retriever=object()))
    monkeypatch.setattr(main_graph, "make_rag_search_tool", lambda _: object())
    monkeypatch.setattr(chat_node, "create_agent", lambda **kwargs: agent)
    return classifier, agent


def _create(client, owner="alice"):
    """通过 API 创建会话，并检查接口成功返回。

    Args:
        client: FastAPI 测试客户端。
        owner: 要创建会话的用户 ID。

    Returns:
        后端生成的会话 ID。
    """
    response = client.post("/sessions", json={"user_id": owner})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _message(session_id, query, user_id="alice", **extra):
    """构造普通聊天和流式聊天共用的 JSON 请求体。

    Args:
        session_id: 本轮聊天使用的会话 ID。
        query: 本轮用户问题。
        user_id: 发起请求的用户 ID。
        **extra: 要加入请求体的额外选项，例如 max_steps。

    Returns:
        可直接传给测试客户端的请求字段字典。
    """
    return {"user_id": user_id, "session_id": session_id, "query": query, **extra}


def _history(client, session_id, owner="alice"):
    """请求指定会话的历史消息。

    Args:
        client: FastAPI 测试客户端。
        session_id: 要读取的会话 ID。
        owner: 请求历史时传入的用户 ID。

    Returns:
        原始 HTTP 响应，供测试检查状态码和消息内容。
    """
    return client.get(f"/sessions/{session_id}/messages", params={"user_id": owner})


def test_sessions_are_listed_before_first_turn_and_lookup_checks_owner(storage):
    """验证空会话可以列出，读历史和发消息都要通过归属检查。

    Args:
        storage: 替换真实 MongoDB 的测试存储。
    """
    with TestClient(app) as client:
        first = _create(client)
        second = _create(client)
        other = _create(client, "bob")
        response = client.get("/sessions", params={"user_id": "alice"})
        assert [item["session_id"] for item in response.json()] == [second, first]
        assert _history(client, first).json() == []
        assert _history(client, other).status_code == 404
        assert client.post("/chat", json=_message(other, "hidden")).status_code == 404
        assert client.post("/chat/stream", json=_message("unknown", "hidden")).status_code == 404


def test_sync_and_stream_share_history_and_sessions_do_not_mix(storage, chat_model):
    """验证普通和流式聊天共享历史与归档，并按会话隔离。

    Args:
        storage: 跨应用生命周期共用的测试存储。
        chat_model: 可检查输入消息的模型和 Agent 替身。
    """
    classifier, agent = chat_model
    with TestClient(app) as client:
        first, second = _create(client), _create(client)
        initial = client.post("/chat", json=_message(first, "CPU is high"))
        assert initial.json() == {"intent": "chat", "reply": "Reply: CPU is high", "result": None}
        stream = client.post("/chat/stream", json=_message(first, "What next?"))
        assert stream.status_code == 200 and "event: result" in stream.text
        assert "event: progress" in stream.text and "record_turn" not in stream.text
        assert [m.content for m in agent.ainvoke.await_args.args[0]["messages"]] == [
            "CPU is high", "Reply: CPU is high", "What next?",
        ]
        assert [m.content for m in classifier.await_args.args[0][1:]] == [
            "CPU is high", "Reply: CPU is high", "What next?",
        ]
        assert client.post("/chat", json=_message(second, "Separate question")).status_code == 200
        assert [m.content for m in agent.ainvoke.await_args.args[0]["messages"]] == ["Separate question"]
        first_events = [event for event in storage.context_events.rows.values() if event["session_id"] == first]
        assert [(event["kind"], event["content"]) for event in first_events] == [
            ("user", "CPU is high"), ("assistant", "Reply: CPU is high"),
            ("user", "What next?"), ("assistant", "Reply: What next?"),
        ]
        assert first_events[0]["turn_id"] == first_events[1]["turn_id"]
        assert first_events[0]["turn_id"] != first_events[2]["turn_id"]
        assert all(event["created_at"].utcoffset().total_seconds() == 0 for event in first_events)

    # 重新启动应用生命周期后，仍从同一存储读取历史，而非从路由的内存变量读取。
    with TestClient(app) as client:
        history = _history(client, first).json()
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", "CPU is high"), ("assistant", "Reply: CPU is high"),
            ("user", "What next?"), ("assistant", "Reply: What next?"),
        ]
        followup = client.post("/chat", json=_message(first, "Summarize"))
        assert followup.status_code == 200
        assert [m.content for m in agent.ainvoke.await_args.args[0]["messages"]] == [
            "CPU is high", "Reply: CPU is high", "What next?", "Reply: What next?", "Summarize",
        ]


def test_failed_turn_does_not_appear_in_history(storage, chat_model):
    """验证失败的流式请求不进入历史，也不影响后续提问。

    Args:
        storage: 保存已完成对话的测试存储。
        chat_model: 能在指定问题上模拟失败的模型替身。
    """
    _, agent = chat_model
    with TestClient(app) as client:
        session = _create(client)
        assert client.post("/chat", json=_message(session, "First")).status_code == 200
        failed = client.post("/chat/stream", json=_message(session, "fail this turn"))
        assert "暂时无法生成可靠回复" in failed.text
        assert [message["content"] for message in _history(client, session).json()] == [
            "First", "Reply: First", "fail this turn", "当前服务暂时无法生成可靠回复，请稍后重试。",
        ]
        assert client.post("/chat", json=_message(session, "Retry")).status_code == 200
        # 降级回复也作为一轮历史保留，后续 Agent 会看到原始问题、降级回复和新问题。
        assert [m.content for m in agent.ainvoke.await_args.args[0]["messages"]] == [
            "First", "Reply: First", "fail this turn",
            "当前服务暂时无法生成可靠回复，请稍后重试。", "Retry",
        ]



def test_tool_return_is_archived_before_later_model_failure(storage, chat_model, monkeypatch):
    """验证模型在工具返回后失败时，仍能找到完整的工具事件。

    Args:
        storage: 保存会话和归档事件的测试存储。
        chat_model: 初始化普通聊天测试所需的模型和检索替身。
        monkeypatch: 用于接入真实 Agent 工厂和测试工具的临时替换工具。
    """
    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            """模拟模型绑定工具。

            Args:
                tools: Agent 提供的工具列表。
                **kwargs: Agent 提供的其他工具绑定选项。

            Returns:
                继续返回预设消息的测试模型。
            """
            return self

    @tool
    def get_logs(query: str) -> str:
        """Return the log excerpt for a query.

        Args:
            query: Log query supplied by the test agent.

        Returns:
            The synthetic traceback used to verify that archiving preserves the full response.
        """
        return original

    original = "Traceback (most recent call last):\n  File \"app.py\", line 42\n" + "x" * 9000
    model = main_graph.ModelClient()
    model.model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "logs-1", "name": "get_logs", "args": {"query": "crash"}},
        ]),
        AIMessage(content=""),
    ])
    monkeypatch.setattr(chat_node, "create_agent", create_agent)
    monkeypatch.setattr(main_graph, "make_rag_search_tool", lambda _: get_logs)

    with TestClient(app) as client:
        session = _create(client)
        response = client.post("/chat/stream", json=_message(session, "Inspect logs"))
        assert "暂时无法生成可靠回复" in response.text
        history = _history(client, session).json()
        # 业务降级仍完成一轮，因此有用户消息和降级回复；工具调用前的失败语义由归档事件验证。
        assert [item["content"] for item in history] == [
            "Inspect logs", "当前服务暂时无法生成可靠回复，请稍后重试。",
        ]

    events = list(storage.context_events.rows.values())
    user = next(item for item in events if item["kind"] == "user")
    result = next(item for item in events if item["_id"].startswith("tool_out_"))
    assert user["session_id"] == result["session_id"] == session
    assert user["turn_id"] == result["turn_id"]
    assert result["tool_call_id"] == "logs-1"
    assert result["tool_name"] == "get_logs"
    assert result["arguments"] == {"query": "crash"}
    assert result["content"] == original
    assert result["tag"] == "core"


def test_archive_write_is_idempotent_and_redacts_credentials(storage):
    """验证归档会保留首次写入并遮蔽可识别的凭据值。

    Args:
        storage: 保存测试归档事件的内存存储。
    """
    async def record_events():
        """写入成功、重复和含凭据的工具事件。"""
        ingestion = TurnIngestion(storage.context_events, "session")
        await ingestion.record("tool", '{"status": 200}', source_id="call-1", status="success")
        await ingestion.record("tool", "duplicate", source_id="call-1", status="error")
        await ingestion.record(
            "tool", "set MODEL_API_KEY=secret-example\nkeep this line",
            source_id="call-2", status="error",
            arguments={"authorization": "Bearer example", "query": "keep this"},
        )

    asyncio.run(record_events())
    first, second = storage.context_events.rows.values()
    assert first["content"] == '{"status": 200}'
    assert first["tag"] == "volatile"
    assert second["tag"] == "core"
    assert second["content"] == "set MODEL_API_KEY=[REDACTED]\nkeep this line"
    assert second["arguments"] == {"authorization": "[REDACTED]", "query": "keep this"}


def test_running_stream_keeps_session_reserved_until_it_finishes(storage, chat_model):
    """验证流式回复未结束时同会话返回 409，结束后可以继续提问。

    Args:
        storage: 测试存储；会话占用由真实服务逻辑处理。
        chat_model: 用于暂停首个请求的聊天 Agent 替身。
    """
    _, agent = chat_model
    entered, release = Event(), Event()

    async def wait_for_release(state):
        """通知测试回复已开始，等另一条请求发出后再完成。

        Args:
            state: 聊天 Agent 的输入；本测试只关心执行时机。

        Returns:
            解除等待后产生的助手回复状态。
        """
        entered.set()
        assert await asyncio.to_thread(release.wait, 5)
        return {"messages": [AIMessage(content="Finished")]}

    agent.ainvoke.side_effect = wait_for_release
    with TestClient(app) as client, ThreadPoolExecutor(1) as executor:
        session = _create(client)
        running = executor.submit(client.post, "/chat/stream", json=_message(session, "First"))
        try:
            assert entered.wait(5)
            concurrent = client.post("/chat", json=_message(session, "Second"))
            assert concurrent.status_code == 409
        finally:
            release.set()
        assert "event: result" in running.result(timeout=5).text
        assert client.post("/chat", json=_message(session, "After stream")).status_code == 200


def test_new_diagnosis_reuses_dialogue_but_not_previous_execution_state(storage, monkeypatch):
    """验证新排障轮保留历史问答、重置执行状态，并归档本轮检索资料。

    Args:
        storage: 保存会话问答和归档事件的测试存储。
        monkeypatch: 用于替换模型、检索和排障节点的临时工具。
    """
    seen = []
    monkeypatch.setattr(main_graph, "ModelClient", object)
    monkeypatch.setattr(main_graph.MainGraph, "_intent", AsyncMock(return_value={"intent": "aiops"}))

    monkeypatch.setattr(main_graph.RagNodes, "from_env", lambda: SimpleNamespace())
    monkeypatch.setattr(
        main_graph, "rag_subgraph",
        AsyncMock(return_value={"unique_documents": ["Retrieved material"]}),
    )

    async def plan(self, state):
        """检查本轮排障数据已清空，记录主图保留的对话历史。

        Args:
            state: 本轮检索完成后的主图状态。

        Returns:
            含一个待执行步骤的排障计划。
        """
        assert state["step_results"] == []
        assert state["evidence"] == []
        assert [doc["content"] for doc in state["knowledge_documents"]] == ["Retrieved material"]
        seen.append([turn["query"] for turn in state.get("turns", [])])
        return {"steps": [{"id": "s1", "goal": "Inspect", "status": "pending"}]}

    async def execute(self, state):
        """模拟执行一个排障步骤，并写入结果和证据。

        Args:
            state: 含本轮计划的主图状态；替身使用固定执行结果。

        Returns:
            一条步骤结果和一条证据的节点更新。
        """
        return {"step_results": [{"step_id": "s1"}], "evidence": [{"id": "e1"}]}

    async def replan(self, state):
        """直接生成当前轮次的诊断结果，让排障流程进入收尾。

        Args:
            state: 已包含本轮问题和执行结果的主图状态。

        Returns:
            包含诊断摘要和证据引用的结构化结果更新。
        """
        return {"result": {
            "summary": f"Diagnosis: {state['query']}",
            "root_cause": None,
            "evidence_ids": ["e1"],
            "knowledge_document_ids": ["k1"],
            "recommendations": [],
            "limitations": [],
        }}

    for name, fn in {"_plan": plan, "_execute_step": execute, "_replan": replan}.items():
        monkeypatch.setattr(main_graph.MainGraph, name, fn)
    with TestClient(app) as client:
        session = _create(client)
        for query in ("First diagnosis", "Second diagnosis"):
            response = client.post("/chat", json=_message(session, query, max_steps=1))
            assert response.status_code == 200, response.text
        messages = _history(client, session).json()
    assert seen == [[], ["First diagnosis"]]
    assert messages[1]["result"]["summary"] == "Diagnosis: First diagnosis"
    assert messages[3]["content"] == "Diagnosis: Second diagnosis"
    events = list(storage.context_events.rows.values())
    assert [event["kind"] for event in events] == ["user", "rag", "assistant"] * 2
    assert events[1]["content"] == events[4]["content"] == "Retrieved material"
    assert events[2]["content"]["summary"] == "Diagnosis: First diagnosis"
