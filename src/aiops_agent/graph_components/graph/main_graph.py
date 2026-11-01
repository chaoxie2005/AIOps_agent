"""串联知识检索、规划、单步执行与重新规划的主图。"""

from langgraph.graph import END, START, StateGraph

from aiops_agent.agent_components.toolkits.rag_search import make_rag_search_tool
from aiops_agent.graph_components.graph.rag_subgraph import rag_subgraph
from aiops_agent.graph_components.nodes.execute_node import ExecuteNodes
from aiops_agent.graph_components.nodes.execute_step_node import execute_step_node
from aiops_agent.graph_components.nodes.plan_node import plan_node
from aiops_agent.graph_components.nodes.rag_nodes import RagNodes
from aiops_agent.graph_components.nodes.replan_node import replan_node
from aiops_agent.graph_components.router.plan_router import plan_router
from aiops_agent.graph_components.router.replan_router import replan_router
from aiops_agent.graph_components.states.state import OverviewOutput, OverviewState


class MainGraph:
    """管理排障主图及其共享的模型、检索器和执行依赖。"""

    def __init__(self, query: str, max_steps: int = 5, rag_nodes: RagNodes | None = None):
        """保存输入并构建主图。

        Args:
            query: 用户提出的故障问题。
            max_steps: 允许执行的步骤上限，零表示直接进入诊断收尾。
            rag_nodes: 已初始化的依赖；未提供时通过现有环境配置创建。

        Raises:
            ValueError: max_steps 不是非负整数。
        """
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps 必须为非负整数")
        self.query = query
        self.max_steps = max_steps
        self.rag_nodes = rag_nodes or RagNodes.from_env()
        self.execute_nodes = ExecuteNodes(
            self.rag_nodes.model_client, make_rag_search_tool(self.rag_nodes.retriever),
        )
        self.builder = StateGraph(OverviewState, output_schema=OverviewOutput)
        self.builder.add_node("rag", self._rag) # 检索知识并转换为主图文档
        self.builder.add_node("plan", self._plan) 
        self.builder.add_node("execute_step", self._execute_step)
        self.builder.add_node("replan", self._replan)
        self.builder.add_edge(START, "rag")
        self.builder.add_edge("rag", "plan")
        self.builder.add_conditional_edges("plan", plan_router, {
            "execute_step": "execute_step", "replan": "replan",
        }) # 这里加 "replan" 是处理无法开始执行的情况：planner 返回了空步骤列表。步骤预算为 0。避免直接调用执行节点会报“没有待执行的步骤”或“预算已耗尽”，所以交给 replanner 根据现有信息生成诊断结果，说明为什么无法继续。
        self.builder.add_edge("execute_step", "replan")
        self.builder.add_conditional_edges("replan", replan_router, {
            "execute_step": "execute_step", END: END,
        })
        self.graph = self.builder.compile()

    async def _rag(self, state: OverviewState) -> dict:
        """检索知识并转换为主图文档。

        Args:
            state: 包含用户问题的主状态。

        Returns:
            本轮检索得到的知识文档状态更新。
        """
        output = await rag_subgraph(state["query"], nodes=self.rag_nodes)
        return {"knowledge_documents": [
            {
                "id": f"k{index}", "title": f"检索资料 {index}", "content": content,
                "source_uri": None, "score": None,
            }
            for index, content in enumerate(output["unique_documents"], start=1)
        ]}

    async def _plan(self, state: OverviewState) -> dict:
        """调用规划节点。

        Args:
            state: 已包含检索知识的主状态。

        Returns:
            初始诊断步骤的状态更新。
        """
        return await plan_node(state, self.rag_nodes.model_client)

    async def _execute_step(self, state: OverviewState) -> dict:
        """调用单步执行包装节点。

        Args:
            state: 包含待执行步骤和历史结果的主状态。

        Returns:
            步骤状态以及本步新增的执行结果和证据。
        """
        return await execute_step_node(state, self.execute_nodes)

    async def _replan(self, state: OverviewState) -> dict:
        """调用重新规划节点。

        Args:
            state: 包含最新诊断进展的主状态。

        Returns:
            后续计划或最终诊断结果的状态更新。
        """
        return await replan_node(state, self.rag_nodes.model_client)

    async def ainvoke(self) -> OverviewOutput:
        """使用独立初始状态运行本次诊断。

        Returns:
            仅包含 result 字段的对外输出，字段值为最终结构化诊断结果。
        """
        state: OverviewState = {
            "query": self.query, 
            "max_steps": self.max_steps,
            "steps": [], 
            "current_step_id": None, 
            "step_results": [],
            "evidence": [], 
            "knowledge_documents": [], 
            "result": None,
        }
        return await self.graph.ainvoke(
            state,
            config={"recursion_limit": 2 * self.max_steps + 10},
        )  # 执行一步 → 重新规划 每个排查步骤大约需要两轮，再额外预留 10 轮给检索、初始规划和收尾。
