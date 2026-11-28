"""结构化输出解析与业务降级的离线测试。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from aiops_agent import output_validation
from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.graph import main_graph as main
from aiops_agent.graph_components.nodes import chat_node as chat_module
from aiops_agent.graph_components.nodes.rag import subquery_node
from aiops_agent.graph_components.nodes.chat_node import chat_node, intent_node
from aiops_agent.graph_components.nodes.plan_node import plan_node
from aiops_agent.graph_components.nodes.replan_node import replan_node
from aiops_agent.graph_components.router.replan_router import replan_router
from aiops_agent.output_validation import (
    DiagnosisResultModel,
    IntentOutput,
    ReplanOutput,
)


def fake_model(contents):
    """创建返回指定 AIMessage 的模型替身。"""
    responses = [
        AIMessage(content=item) if isinstance(item, str) else item for item in contents
    ]
    if not responses:
        responses = [AIMessage(content="")]
    index = 0

    def next_response(*_args, **_kwargs):
        """顺序返回预设响应，耗尽后重复最后一个，避免测试因重试中断。"""
        nonlocal index
        item = responses[min(index, len(responses) - 1)]
        index += 1
        return item

    return SimpleNamespace(
        model=SimpleNamespace(ainvoke=AsyncMock(side_effect=next_response)),
        ainvoke=AsyncMock(),
    )


def test_parse_structured_response_rejects_example_and_extra():
    """贪婪正则原本会截入示例，严格解析必须拒绝无效对象。"""
    payload = {"intent": "chat"}
    response = AIMessage(
        content=' explanatory text\n{"intent": "aiops"}\n' + json.dumps(payload)
    )
    with pytest.raises(ValueError):
        output_validation.parse_structured_response(response, IntentOutput)

    with pytest.raises(ValueError):
        output_validation.parse_structured_response(
            AIMessage(content='{"intent": "chat", "extra": true}'), IntentOutput,
        )
    assert output_validation.parse_structured_response(
        AIMessage(content='```json\n{"intent": "chat"}\n```'), IntentOutput,
    ).intent == "chat"


def test_parse_structured_response_reads_tool_call_args():
    """原生 tool call 参数作为结构化输出的第一来源。"""
    response = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "IntentOutput", "args": {"intent": "aiops"}}],
    )
    assert output_validation.parse_structured_response(response, IntentOutput).intent == "aiops"


def test_replan_schema_requires_exclusive_terminal_state():
    """steps 和 result 必须二选一。"""
    result = DiagnosisResultModel(
        summary="x", root_cause=None, evidence_ids=[], knowledge_document_ids=[],
        recommendations=[], limitations=[],
    )
    assert ReplanOutput(steps=[], result=result).result == result
    with pytest.raises(ValueError):
        ReplanOutput(steps=[], result=None)
    with pytest.raises(ValueError):
        ReplanOutput(steps=[{"id": "s1", "goal": "g", "status": "pending"}], result=result)


def test_intent_fallback_is_chat():
    """意图输出错误时降级为 chat，不进入现场排障分支。"""
    model = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="没有 JSON")),
        model=object(),
    )
    update = asyncio.run(intent_node({"query": "检查 CPU"}, model))
    assert update["intent"] == "chat"
    assert "degraded" in update


def test_plan_invalid_output_falls_back_and_replan_closes_graph():
    """规划失败不抛 500，最终由诊断结果收尾。"""
    result = {
        "summary": "规划输出无效", "root_cause": None, "evidence_ids": [],
        "knowledge_document_ids": [], "recommendations": [], "limitations": ["测试"],
    }
    answers = iter([
        AIMessage(content='{"intent":"aiops"}'),
        AIMessage(content="invalid"),
        AIMessage(content="invalid"),
        AIMessage(content="invalid"),
        AIMessage(content=json.dumps({"steps": [], "result": result}, ensure_ascii=False)),
    ])

    def next_answer(*_args, **_kwargs):
        """意图返回 aiops，规划失败重试耗尽后由 replan 给出诊断。"""
        return next(answers, AIMessage(content="invalid"))

    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=next_answer), model=object())
    from aiops_agent.graph_components.graph import main_graph as graph_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        graph_module, "ModelClient", lambda: SimpleNamespace(model=object(), ainvoke=model.ainvoke),
    )
    nodes = SimpleNamespace(model_client=model, retriever=object())
    monkeypatch.setattr(graph_module.RagNodes, "from_env", lambda: nodes)
    monkeypatch.setattr(
        graph_module, "rag_subgraph",
        AsyncMock(return_value={"unique_documents": []}),
    )
    monkeypatch.setattr(
        graph_module, "make_rag_search_tool", Mock(return_value=object()),
    )
    try:
        graph = graph_module.MainGraph("检查 CPU", max_steps=2)
        output = asyncio.run(graph.ainvoke())
    finally:
        monkeypatch.undo()
    assert output["intent"] == "aiops"
    assert output["result"]["summary"] == "规划输出无效"


def test_replan_budget_exhausted_returns_result():
    """预算耗尽时 replan 生成诊断而不是抛异常。"""
    model = fake_model([])
    state = {
        "query": "检查 CPU", "steps": [], "step_results": [
            {"step_id": "s1", "status": "completed", "summary": "x", "evidence_ids": [],
             "knowledge_document_ids": [], "tool_call_ids": [], "error": None},
        ],
        "evidence": [], "knowledge_documents": [], "max_steps": 1,
    }
    output = asyncio.run(replan_node(state, model))
    assert output["steps"] == []
    assert output["result"] is not None
    from langgraph.graph import END

    assert replan_router({**state, **output}) == END


def test_replan_truncates_steps_to_remaining_budget():
    """重规划步骤数被硬限制在剩余预算内。"""
    steps = [
        {"id": f"s{i}", "goal": f"检查 {i}", "status": "pending"} for i in range(2)
    ]
    model = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content=json.dumps({"steps": steps, "result": None}))),
        model=object(),
    )
    state = {
        "query": "检查 CPU", "steps": [], "step_results": [], "evidence": [],
        "knowledge_documents": [], "max_steps": 1,
    }
    output = asyncio.run(replan_node(state, model))
    assert len(output["steps"]) == 1


def test_subquery_fallback_to_original_question():
    """子查询解析失败时仍执行一次原始问题检索。"""
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="not json")), model=object())
    output = asyncio.run(subquery_node.subquery_node({"query": "CPU 高", "max_subqueries": 3}, model))
    assert output["unique_subqueries"] == ["CPU 高"]


def test_chat_failure_returns_degraded_reply(monkeypatch):
    """聊天 Agent 异常不向客户端抛异常。"""
    class BrokenAgent:
        async def ainvoke(self, messages):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(chat_module, "create_agent", lambda **kwargs: BrokenAgent())
    model = SimpleNamespace(model=object(), ainvoke=AsyncMock())
    output = asyncio.run(chat_node({"query": "你好"}, model, object()))
    assert "暂" in output["reply"]
    assert output["result"] is None
