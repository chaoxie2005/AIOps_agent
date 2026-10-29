"""手动集成测试：在项目根目录运行 python -m aiops_agent.tests.test_subquery_agent_manual。"""
import asyncio

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.nodes.subquery_agent import subquery_agent_node
from aiops_agent.graph_components.states.state import RagState


async def main() -> None:
    model = ModelClient(
        model_name="Qwen/Qwen3-0.6B",
        model_provider="openai",
        base_url="http://127.0.0.1:8002/v1",
        api_key="EMPTY",
        max_tokens=512,
    )
    state: RagState = {
        "query": "CPU突然异常高，导致系统响应缓慢",
        "max_subqueries": 3,
    }
    result = await subquery_agent_node(state, model)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
