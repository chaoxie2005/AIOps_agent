"""执行节点的最小联调测试，依赖真实模型、RAG 检索服务及工具配置。"""

import asyncio

from aiops_agent.graph_components.nodes.execute_node import ExecuteNodes


def test_execute_node():
    """执行一个 CPU 排查步骤，检查结果与工具记录是否对应。"""
    async def run():
        nodes = ExecuteNodes.from_env()
        state = await nodes.prepare({
            "steps": [{"id": "s1", "goal": "检索 CPU 使用率过高的排查方法", "status": "pending"}],
            "current_step_id": "s1",
        })
        return await nodes.execute(state)

    result = asyncio.run(run())
    print(result)
    step_result = result["step_result"]
    assert step_result["step_id"] == "s1"
    assert step_result["status"] in ("completed", "failed", "skipped")
    assert isinstance(step_result["summary"], str) and step_result["summary"]
    call_ids = [call["id"] for call in result["tool_calls"]]
    assert step_result["tool_call_ids"] == call_ids
    assert {item["tool_call_id"] for item in result["tool_results"]} == set(call_ids)
