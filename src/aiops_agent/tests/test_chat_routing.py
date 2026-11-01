"""意图识别、聊天和主图分流的离线测试。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from aiops_agent.agent_components.toolkits.rag_search import make_rag_search_tool
from aiops_agent.graph_components.graph import main_graph as main
from aiops_agent.graph_components.nodes import chat_node as chat_module
from aiops_agent.graph_components.nodes import execute_node as execution
from aiops_agent.graph_components.nodes.chat_node import chat_node, intent_node
from aiops_agent.graph_components.router.intent_router import intent_router


class ToolCallingModel(FakeMessagesListChatModel):
    """以预设回复驱动真实 Agent 工具调用循环。"""

    def bind_tools(self, tools, **kwargs):
        """接受工具绑定并返回模型替身。"""
        return self


class KnowledgeRetriever(BaseRetriever):
    """记录查询并返回固定知识文档的检索器替身。"""

    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(self, query, *, run_manager):
        """记录实际检索并返回知识文档。"""
        self.queries.append(query)
        return [Document(page_content="CPU throttling 表示 CPU 配额限制。")]

    async def _aget_relevant_documents(self, query, *, run_manager):
        """提供异步知识检索。"""
        return self._get_relevant_documents(query, run_manager=run_manager)


def fake_model(contents):
    """创建按顺序返回指定消息的模型替身。

    Args:
        contents: 模型回复内容列表。

    Returns:
        支持异步调用的模型替身。
    """
    return SimpleNamespace(model=object(), ainvoke=AsyncMock(side_effect=[
        SimpleNamespace(content=content) for content in contents
    ]))


@pytest.mark.parametrize("intent,target", [("chat", "chat"), ("aiops", "rag")])
def test_intent_and_route(intent, target):
    """验证合法意图的解析、路由及用户消息传递。"""
    model = fake_model(['```json\n' + json.dumps({"intent": intent}) + '\n```'])
    update = asyncio.run(intent_node({"query": "测试 {literal}"}, model))
    assert update == {"intent": intent}
    assert intent_router(update) == target
    assert model.ainvoke.call_args.args[0][1].content == "测试 {literal}"


@pytest.mark.parametrize("content", [
    "没有 JSON", "{invalid}", "{}", '{"intent":"other"}',
    '{"intent":["chat"]}', '{"intent":"chat","extra":true}', [],
])
def test_invalid_intent_output(content):
    """非法模型输出必须报错。"""
    with pytest.raises(ValueError):
        asyncio.run(intent_node({"query": "你好"}, fake_model([content])))


@pytest.mark.parametrize("intent", [None, "other", ""])
def test_invalid_route(intent):
    """无效意图不能进入任意业务分支。"""
    with pytest.raises(ValueError):
        intent_router({"intent": intent})


@pytest.mark.parametrize("content", ["", "   ", []])
def test_empty_chat_reply(content):
    """通过真实 Agent 验证空文本和非文本回答被拒绝。"""
    model = SimpleNamespace(model=ToolCallingModel(responses=[AIMessage(content=content)]))
    with pytest.raises(ValueError):
        asyncio.run(chat_node({"query": "你好"}, model, make_rag_search_tool(KnowledgeRetriever())))


def test_chat_direct_answer(monkeypatch):
    """聊天注册检索工具，直接回答时不实际查询知识库。"""
    model = fake_model(['{"intent":"chat"}', '{"intent":"chat"}'])
    model.model = ToolCallingModel(responses=[AIMessage(content="  你好！  "), AIMessage(content="第二次回答")])
    retriever = KnowledgeRetriever()
    nodes = SimpleNamespace(model_client=model, retriever=retriever)
    factory = Mock(return_value=nodes)
    monkeypatch.setattr(main, "ModelClient", lambda: model)
    monkeypatch.setattr(main.RagNodes, "from_env", factory)
    graph = main.MainGraph("你好")
    factory.assert_not_called()
    assert asyncio.run(graph.ainvoke()) == {"intent": "chat", "reply": "你好！", "result": None}
    assert asyncio.run(graph.ainvoke()) == {"intent": "chat", "reply": "第二次回答", "result": None}
    factory.assert_called_once_with()
    assert retriever.queries == []
    assert graph.rag_nodes is nodes and graph.execute_nodes is None


def test_chat_retrieves_knowledge():
    """真实 Agent 调用注册的 RAG 工具，再返回最终回答。"""
    retriever = KnowledgeRetriever()
    model = SimpleNamespace(model=ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "id": "rag-call", "name": "rag_search_tool", "args": {"query": "CPU throttling"},
        }]),
        AIMessage(content="根据知识库，CPU throttling 表示 CPU 配额限制。"),
    ]))
    output = asyncio.run(chat_node(
        {"query": "查一下内部手册中的 CPU throttling"}, model, make_rag_search_tool(retriever),
    ))
    assert retriever.queries == ["CPU throttling"]
    assert output == {"reply": "根据知识库，CPU throttling 表示 CPU 配额限制。", "result": None}


@pytest.mark.parametrize("messages", [
    [], [ToolMessage(content="检索结果", tool_call_id="call1")],
    [AIMessage(content="准备检索", tool_calls=[{
        "id": "call1", "name": "rag_search_tool", "args": {"query": "test"},
    }])],
])
def test_chat_requires_final_model_reply(monkeypatch, messages):
    """工具结果或未完成的工具调用不能作为最终聊天回答。"""
    agent = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": messages}))
    creator = Mock(return_value=agent)
    monkeypatch.setattr(chat_module, "create_agent", creator)
    model = fake_model([])
    tool = make_rag_search_tool(KnowledgeRetriever())
    with pytest.raises(ValueError):
        asyncio.run(chat_node({"query": "测试问题"}, model, tool))
    assert creator.call_args.kwargs["model"] is model.model
    assert creator.call_args.kwargs["tools"] == [tool]
    assert creator.call_args.kwargs["system_prompt"].type == "system"
    assert agent.ainvoke.call_args.args[0]["messages"][0].content == "测试问题"


def test_aiops_loop_keeps_internal_state(monkeypatch):
    """排障分支延迟初始化，保留两步历史及证据，并仅输出对外字段。"""
    step = lambda number: {"id": f"s{number}", "goal": "检查 CPU", "status": "pending"}
    result = {
        "summary": "模拟诊断完成", "root_cause": None,
        "evidence_ids": ["s1:call1", "s2:call1"], "knowledge_document_ids": ["k1"],
        "recommendations": [], "limitations": [],
    }
    model = fake_model([json.dumps(item) for item in [
        {"intent": "aiops"}, {"steps": [step(1)]},
        {"steps": [step(2)], "result": None}, {"steps": [], "result": result},
    ]])
    nodes = SimpleNamespace(model_client=model, retriever=object())
    factory = Mock(return_value=nodes)
    tool_factory = Mock(return_value=object())
    monkeypatch.setattr(main, "ModelClient", lambda: model)
    monkeypatch.setattr(main.RagNodes, "from_env", factory)
    monkeypatch.setattr(main, "make_rag_search_tool", tool_factory)
    monkeypatch.setattr(main, "rag_subgraph", AsyncMock(return_value={"unique_documents": ["测试知识"]}))
    agent = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [
        AIMessage(content="", tool_calls=[{"id": "call1", "name": "query_metrics", "args": {"query": "test_cpu"}}]),
        ToolMessage(content=json.dumps({"resultType": "vector", "result": [
            {"metric": {}, "value": [1700000000, "0.95"]},
        ]}), tool_call_id="call1"),
        AIMessage(content=json.dumps({"status": "completed", "summary": "模拟指标查询完成"})),
    ]}))
    monkeypatch.setattr(execution, "create_agent", lambda **kwargs: agent)
    original_replan = main.replan_node
    history = []

    async def inspect_replan(state, client):
        history.append([item["step_id"] for item in state["step_results"]])
        assert len(state["evidence"]) == len(state["step_results"])
        assert state["knowledge_documents"][0]["id"] == "k1"
        return await original_replan(state, client)

    monkeypatch.setattr(main, "replan_node", inspect_replan)
    graph = main.MainGraph("请检查当前 CPU 异常", max_steps=2)
    factory.assert_not_called()
    output = asyncio.run(graph.ainvoke())
    assert output == {"intent": "aiops", "reply": None, "result": result}
    assert history == [["s1"], ["s1", "s2"]]
    assert agent.ainvoke.await_count == 2
    factory.assert_called_once_with()
    tool_factory.assert_called_once_with(nodes.retriever)
