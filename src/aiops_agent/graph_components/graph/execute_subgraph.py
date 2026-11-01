"""执行子图：准备当前步骤，再交给 Agent 调用工具完成排查。"""

from langgraph.graph import END, START, StateGraph

from aiops_agent.graph_components.nodes.execute_node import ExecuteNodes
from aiops_agent.graph_components.states.state import ExecutorInput, ExecutorOutput, ExecutorState


async def execute_subgraph(state: ExecutorInput, nodes: ExecuteNodes | None = None) -> ExecutorOutput:
    """构建并运行单个排障步骤的执行子图。

    Args:
        state: 包含步骤列表 steps 和当前步骤 ID current_step_id 的输入。
        nodes: 节点依赖容器；未传入时使用现有环境配置创建。

    Returns:
        当前步骤的 step_result、tool_calls、tool_results 和 evidence_candidates。
    """
    nodes = nodes or ExecuteNodes.from_env()
    builder = StateGraph(ExecutorState, input_schema=ExecutorInput, output_schema=ExecutorOutput)
    builder.add_node(nodes.prepare)
    builder.add_node(nodes.execute)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "execute")
    builder.add_edge("execute", END)
    graph = builder.compile()
    return await graph.ainvoke(state)
