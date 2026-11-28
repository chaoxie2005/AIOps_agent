"""RAG 检索子图：子问题拆解 -> 调度（optimize/retrieve/aggregate） -> 检索 -> 记录 -> 调度循环 -> 聚合。"""

from langgraph.graph import END, StateGraph

from aiops_agent.graph_components.nodes.rag.rag_container import RagNodes
from aiops_agent.graph_components.router.rag_retriever_router import schedule_router
from aiops_agent.graph_components.states.state import RagInput, RagOutput, RagState


async def rag_subgraph(query: str, nodes: RagNodes | None = None, *, max_subqueries: int = 4):
    """运行一次 RAG 检索子图。

    流程：先拆问题，然后调度节点一次挑一个子问题来检索；检索完由记录
    节点标记完成，再回到调度节点挑下一个，直到全部处理完，最后聚合去重。

    Args:
        query: 用户原始问题。
        nodes: 预先构造好的节点依赖容器，缺省时从环境变量创建。
        max_subqueries: 最多拆出几个子问题。

    Returns:
        包含 unique_documents（去重后的父文档正文）的字典。
    """
    nodes = nodes or RagNodes.from_env()
    builder = StateGraph(RagState, input_schema=RagInput, output_schema=RagOutput)
    builder.add_node("subquery", nodes.subquery)
    builder.add_node("schedule", nodes.schedule)
    builder.add_node("optimize", nodes.optimize)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("record", nodes.record)
    builder.add_node("aggregate", nodes.aggregate)

    builder.set_entry_point("subquery")
    builder.add_edge("subquery", "schedule")
    # schedule 决定下一步：有依赖走 optimize 改写，没依赖直接 retrieve，都处理完走 aggregate。
    builder.add_conditional_edges("schedule", schedule_router, {
        "optimize": "optimize",
        "retrieve": "retrieve",
        "aggregate": "aggregate",
    })
    # SDOM 只改写查询，不改状态里的其他内容，改写完就去检索。
    builder.add_edge("optimize", "retrieve")
    builder.add_edge("retrieve", "record")
    # 记录完成后回到调度，形成循环；全部完成时 schedule 会走向 aggregate。
    builder.add_edge("record", "schedule")
    builder.add_edge("aggregate", END)

    rag_agent = builder.compile()
    # 循环次数与子问题数量成正比，给足递归额度，避免被默认上限误判为死循环。
    return await rag_agent.ainvoke(
        {"query": query, "max_subqueries": max_subqueries},
        config={"recursion_limit": max_subqueries * 4 + 10},
    )
