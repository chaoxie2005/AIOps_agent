"""手动rag检索tool节点集成测试：在项目根目录运行 python -m aiops_agent.tests.test_rag_retriever_manual。"""
import asyncio

from aiops_agent.graph_components.nodes.rag_retrierver import retriever
from aiops_agent.graph_components.states.state import RagState


async def main() -> None:
    state: RagState = {
        "query": "CPU使用率过高如何排查？",
        "unique_subqueries": ["CPU使用率过高如何排查？", "mysql数据库性能问题"],
    }
    result = await retriever(state)
    

if __name__ == "__main__":
    asyncio.run(main())
