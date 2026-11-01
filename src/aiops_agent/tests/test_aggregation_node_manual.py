"""手动调试聚合去重rag检索节点脚本：在项目根目录运行 $env:PYTHONPATH="src"; python -m aiops_agent.tests.test_aggregation_node_manual。"""
import asyncio

from langchain_core.documents import Document

from aiops_agent.graph_components.nodes.aggregation_node import aggregation_node
from aiops_agent.graph_components.states.state import RagState


def make_doc(content: str, parent_id: str, child_id: str, score: float, page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"download_files/aiops-docs/{parent_id}.md",
            "source_url": f"http://127.0.0.1:9000/camel-agent-rag/aiops-docs/{parent_id}.md",
            "file_name": f"{parent_id}.md",
            "page": page,
            "parent_id": parent_id,
            "child_id": child_id,
            "parent_content": content,
            "score": score,
        },
    )


async def main() -> None:
    state: RagState = {
        "query": "CPU突然异常高，导致系统响应缓慢",
        "retrieval_hits": [
            # 子查询 1：命中父文档 5874d4cb
            {
                "subquery": "CPU 使用率过高的原因",
                "document": make_doc(
                    "服务器或容器的CPU使用率持续超过80%时，会导致应用响应变慢、请求超时增加、系统负载过高、可能触发雪崩效应。",
                    parent_id="5874d4cb-d83a-4b88-9d1f-4e6e50ac3db1",
                    child_id="5874d4cb-d83a-4b88-9d1f-4e6e50ac3db1_child_64f101d4-8595-4041-be81-4469f6533d01",
                    score=0.6318189612679032,
                ),
            },
            # 子查询 2：同一父文档的另一个子块（parent_content 相同），应被去重
            {
                "subquery": "系统响应慢怎么排查",
                "document": make_doc(
                    "服务器或容器的CPU使用率持续超过80%时，会导致应用响应变慢、请求超时增加、系统负载过高、可能触发雪崩效应。",
                    parent_id="5874d4cb-d83a-4b88-9d1f-4e6e50ac3db1",
                    child_id="5874d4cb-d83a-4b88-9d1f-4e6e50ac3db1_child_9a2b",
                    score=0.55,
                ),
            },
            # 子查询 2：另一个父文档，应保留
            {
                "subquery": "系统响应慢怎么排查",
                "document": make_doc(
                    "内存泄漏会导致 GC 频繁，CPU 同步升高。",
                    parent_id="a1b2c3d4-0000-1111-2222-333344445555",
                    child_id="a1b2c3d4-0000-1111-2222-333344445555_child_1",
                    score=0.48,
                ),
            },
        ],
    }
    result = await aggregation_node(state)
    print("return:", result)
    print("unique_documents:", result["unique_documents"])
    assert len(result["unique_documents"]) == 2


if __name__ == "__main__":
    asyncio.run(main())
