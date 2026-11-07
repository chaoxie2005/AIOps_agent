import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from aiops_agent.api.app import app
from aiops_agent.graph_components.graph import main_graph
from aiops_agent.graph_components.nodes import chat_node
from aiops_agent.service.chat_sessions import ChatSessions


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


@pytest.fixture
def storage(monkeypatch):
    """用内存目录和真实的 LangGraph 内存 checkpointer 替换 MongoDB。

    Args:
        monkeypatch: pytest 提供的临时替换工具。

    Returns:
        供同一测试内多个应用生命周期共用的存储对象。
    """
    backend = SimpleNamespace(
        client=Mock(),
        sessions=SessionDirectory(),
        checkpointer=InMemorySaver(),
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
        if query == "fail this turn":
            raise ValueError("model failed")
        return {"messages": [AIMessage(content=f"Reply: {query}")]}

    agent = SimpleNamespace(ainvoke=AsyncMock(side_effect=answer))
    model = SimpleNamespace(model=object(), ainvoke=classifier)
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
    """验证普通和流式聊天共用历史，并按会话隔离和恢复上下文。

    Args:
        storage: 跨两个应用生命周期共用的测试存储。
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
        assert "event: error" in failed.text
        assert [message["content"] for message in _history(client, session).json()] == [
            "First", "Reply: First",
        ]
        assert client.post("/chat", json=_message(session, "Retry")).status_code == 200
        assert [m.content for m in agent.ainvoke.await_args.args[0]["messages"]] == [
            "First", "Reply: First", "Retry",
        ]


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
    """验证新一轮排障保留历史问答，同时清空上轮的步骤、证据和知识。

    Args:
        storage: 保存同一会话两轮排障结果的测试存储。
        monkeypatch: 用固定结果替换模型和排障节点的工具。
    """
    seen = []
    monkeypatch.setattr(main_graph, "ModelClient", object)
    monkeypatch.setattr(main_graph.MainGraph, "_intent", AsyncMock(return_value={"intent": "aiops"}))

    async def rag(self, state):
        """以本轮问题构造一条可辨认的知识记录。

        Args:
            state: 当前主图状态，包含本轮问题。

        Returns:
            仅包含本轮知识记录的节点更新。
        """
        return {"knowledge_documents": [{"id": "k1", "content": state["query"]}]}

    async def plan(self, state):
        """检查本轮排障数据已清空，记录主图保留的对话历史。

        Args:
            state: 本轮检索完成后的主图状态。

        Returns:
            含一个待执行步骤的排障计划。
        """
        assert state["step_results"] == []
        assert state["evidence"] == []
        assert [doc["content"] for doc in state["knowledge_documents"]] == [state["query"]]
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

    for name, fn in {"_rag": rag, "_plan": plan, "_execute_step": execute, "_replan": replan}.items():
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
