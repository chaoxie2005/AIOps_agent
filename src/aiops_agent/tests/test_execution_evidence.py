"""验证现场证据从工具结果经执行子图追加到主状态，无需外部服务。"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from aiops_agent.graph_components.nodes import execute_node as execution
from aiops_agent.graph_components.nodes.execute_step_node import execute_step_node
from aiops_agent.graph_components.states.state import OverviewState


METRICS = {"resultType": "vector", "result": [
    {"metric": {"instance": "test-host"}, "value": [1700000000, "0.95"]},
]}


def make_nodes(monkeypatch, payload, status="success", tool_name="query_metrics"):
    messages = [
        AIMessage(content="", tool_calls=[{
            "id": "call-1", "name": tool_name, "args": {"query": "test_cpu_usage"},
        }]),
        ToolMessage(
            content=payload if isinstance(payload, str) else json.dumps(payload),
            tool_call_id="call-1", status=status,
        ),
        AIMessage(content=json.dumps({
            "status": "completed" if status == "success" else "failed",
            "summary": "模拟查询结果",
        })),
    ]
    monkeypatch.setattr(execution, "create_agent", lambda **kwargs: SimpleNamespace(
        ainvoke=AsyncMock(return_value={"messages": messages}),
    ))
    return execution.ExecuteNodes(SimpleNamespace(model=None), object())


@pytest.mark.parametrize("payload,status,tool_name,expected", [
    (METRICS, "success", "query_metrics", True),
    ({"resultType": "scalar", "result": [1700000000, "0"]}, "success", "query_metrics", True),
    ({"resultType": "matrix", "result": [{"metric": {}, "values": [
        [1700000000, "1"], [1700000060, "2"],
    ]}]}, "success", "query_metrics", True),
    ({"resultType": "vector", "result": []}, "success", "query_metrics", False),
    ("不是 JSON", "success", "query_metrics", False),
    (METRICS, "error", "query_metrics", False),
    (METRICS, "success", "rag_search_tool", False),
])
def test_evidence_from_tool_results(monkeypatch, payload, status, tool_name, expected):
    nodes = make_nodes(monkeypatch, payload, status, tool_name)
    state = asyncio.run(nodes.prepare({
        "steps": [{"id": "s1", "goal": "检查 CPU", "status": "pending"}],
        "current_step_id": "s1",
    }))
    output = asyncio.run(nodes.execute(state))
    evidence = output["evidence_candidates"]
    assert bool(evidence) is expected
    assert output["step_result"]["evidence_ids"] == [item["id"] for item in evidence]
    assert len(output["tool_results"]) == 1
    if expected:
        assert evidence[0]["source"] == "prometheus"
        assert evidence[0]["query"] == "test_cpu_usage"
        assert evidence[0]["time_range"]["start"] == datetime.fromtimestamp(1700000000, timezone.utc)
        assert evidence[0]["time_range"]["end"] == datetime.fromtimestamp(
            1700000060 if payload["resultType"] == "matrix" else 1700000000, timezone.utc,
        )
        assert evidence[0]["observation"] == "Prometheus 实际返回：" + json.dumps(payload, ensure_ascii=False)


def test_main_state_appends_each_steps_evidence_once(monkeypatch):
    nodes = make_nodes(monkeypatch, METRICS)

    async def run_step(state: OverviewState):
        return await execute_step_node(state, nodes)

    builder = StateGraph(OverviewState)
    builder.add_node("first", run_step)
    builder.add_node("second", run_step)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    state: OverviewState = {
        "query": "检查 CPU", "max_steps": 2,
        "steps": [
            {"id": "s1", "goal": "检查主机一 CPU", "status": "pending"},
            {"id": "s2", "goal": "检查主机二 CPU", "status": "pending"},
        ],
        "step_results": [], "evidence": [],
    }
    output = asyncio.run(builder.compile().ainvoke(state))
    assert [item["id"] for item in output["evidence"]] == ["s1:call-1", "s2:call-1"]
    assert [item["step_id"] for item in output["step_results"]] == ["s1", "s2"]
    assert all(item["status"] == "completed" for item in output["steps"])
    assert all(result["evidence_ids"] == [evidence["id"]] for result, evidence in zip(
        output["step_results"], output["evidence"],
    ))
    assert output["current_step_id"] is None
    assert state["step_results"] == state["evidence"] == []
    assert all(item["status"] == "pending" for item in state["steps"])
