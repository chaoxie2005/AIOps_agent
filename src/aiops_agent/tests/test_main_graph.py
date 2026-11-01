"""主图最小联调测试，依赖真实模型、检索服务和工具配置。"""

import asyncio

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.graph.main_graph import MainGraph
from aiops_agent.graph_components.nodes.rag_nodes import RagNodes


def test_main_graph():
    """运行完整排障主图，验证诊断输出及步骤预算。"""
    async def run():
        nodes = RagNodes.from_env()
        try:
            nodes.model_client = ModelClient(temperature=0)
            return await MainGraph(
                "CPU突然异常高，导致系统响应缓慢", max_steps=3, rag_nodes=nodes,
            ).ainvoke()
        finally:
            await nodes.retriever.client.close()

    output = asyncio.run(run())
    print(output)
    assert isinstance(output.get("result"), dict)
    assert isinstance(output.get("result")["summary"], str)
    assert output.get("result")["summary"].strip()
    assert output.get("steps") == []
    assert len(output.get("step_results")) <= output.get("max_steps") == 3
