"""根据用户意图选择普通聊天或故障排查的主图。"""

from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime, timezone
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.nodes.chat_node import chat_node, contextual_question, intent_node
from aiops_agent.graph_components.router.intent_router import intent_router

from aiops_agent.agent_components.toolkits.rag_search import make_rag_search_tool
from aiops_agent.graph_components.graph.rag_subgraph import rag_subgraph
from aiops_agent.graph_components.nodes.execute_node import ExecuteNodes
from aiops_agent.graph_components.nodes.execute_step_node import execute_step_node
from aiops_agent.graph_components.nodes.plan_node import plan_node
from aiops_agent.graph_components.nodes.rag_nodes import RagNodes
from aiops_agent.graph_components.nodes.replan_node import replan_node
from aiops_agent.graph_components.router.plan_router import plan_router
from aiops_agent.graph_components.router.replan_router import replan_router
from aiops_agent.graph_components.states.state import ChatTurn, OverviewOutput, OverviewState


class MainGraph:
    """管理排障主图及其共享的模型、检索器和执行依赖。"""

    def __init__(
        self,
        query: str,
        max_steps: int = 5,
        rag_nodes: RagNodes | None = None,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        thread_id: str | None = None,
    ):
        """保存输入并构建主图。

        Args:
            query: 用户提出的故障问题。
            max_steps: 允许执行的步骤上限，零表示直接进入诊断收尾。
            rag_nodes: 已初始化的排障依赖；未提供时在进入排障分支后创建。
            checkpointer: 保存会话状态的 LangGraph 后端。
            thread_id: 使用 checkpoint 时对应的会话 ID。

        Raises:
            ValueError: max_steps 不是非负整数。
        """
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps 必须为非负整数")
        if (checkpointer is None) != (thread_id is None):
            raise ValueError("checkpointer 和 thread_id 必须同时提供")
        self.query = query
        self.max_steps = max_steps
        self.thread_id = thread_id
        self.rag_nodes = rag_nodes
        self.model_client = rag_nodes.model_client if rag_nodes is not None else ModelClient()
        self.execute_nodes = None
        self.builder = StateGraph(OverviewState, output_schema=OverviewOutput)
        self.builder.add_node("rag", self._rag) # 检索知识并转换为主图文档
        self.builder.add_node("plan", self._plan) 
        self.builder.add_node("execute_step", self._execute_step)
        self.builder.add_node("replan", self._replan)
        self.builder.add_node("intent", self._intent)
        self.builder.add_node("chat", self._chat)
        self.builder.add_edge(START, "intent")
        self.builder.add_conditional_edges("intent", intent_router, {"chat": "chat", "rag": "rag"})
        terminal = END
        if checkpointer is not None:
            terminal = "record_turn"
            self.builder.add_node(terminal, self._record_turn)
            self.builder.add_edge(terminal, END)
        self.builder.add_edge("chat", terminal)
        self.builder.add_edge("rag", "plan")
        self.builder.add_conditional_edges("plan", plan_router, {
            "execute_step": "execute_step", "replan": "replan",
        }) # 这里加 "replan" 是处理无法开始执行的情况：planner 返回了空步骤列表。步骤预算为 0。避免直接调用执行节点会报“没有待执行的步骤”或“预算已耗尽”，所以交给 replanner 根据现有信息生成诊断结果，说明为什么无法继续。
        self.builder.add_edge("execute_step", "replan")
        self.builder.add_conditional_edges("replan", replan_router, {
            "execute_step": "execute_step", END: terminal,
        })
        self.graph = self.builder.compile(checkpointer=checkpointer)

    async def _intent(self, state: OverviewState) -> dict:
        """调用意图识别节点。

        Args:
            state: 包含用户问题的主状态。

        Returns:
            经过校验的意图状态更新。
        """
        return await intent_node(state, self.model_client)

    async def _chat(self, state: OverviewState) -> dict:
        """调用普通聊天节点。

        Args:
            state: 包含用户问题的主状态。

        Returns:
            普通聊天回答的状态更新。
        """
        if self.rag_nodes is None:
            self.rag_nodes = RagNodes.from_env()
            self.rag_nodes.model_client = self.model_client
        rag_tool = make_rag_search_tool(self.rag_nodes.retriever)
        return await chat_node(state, self.model_client, rag_tool)

    async def _rag(self, state: OverviewState) -> dict:
        """检索知识并转换为主图文档。

        Args:
            state: 包含用户问题的主状态。

        Returns:
            本轮检索得到的知识文档状态更新。
        """
        if self.rag_nodes is None:
            self.rag_nodes = RagNodes.from_env()
            self.rag_nodes.model_client = self.model_client
        output = await rag_subgraph(contextual_question(state), nodes=self.rag_nodes)
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
        return await plan_node({**state, "query": contextual_question(state)}, self.model_client)

    async def _execute_step(self, state: OverviewState) -> dict:
        """调用单步执行包装节点。

        Args:
            state: 包含待执行步骤和历史结果的主状态。

        Returns:
            步骤状态以及本步新增的执行结果和证据。
        """
        if self.execute_nodes is None:
            self.execute_nodes = ExecuteNodes(
                self.model_client, make_rag_search_tool(self.rag_nodes.retriever),
            )
        return await execute_step_node(state, self.execute_nodes)

    async def _replan(self, state: OverviewState) -> dict:
        """调用重新规划节点。

        Args:
            state: 包含最新诊断进展的主状态。

        Returns:
            后续计划或最终诊断结果的状态更新。
        """
        return await replan_node({**state, "query": contextual_question(state)}, self.model_client)

    async def _record_turn(self, state: OverviewState) -> dict:
        """成功完成普通聊天或排障后，将本轮问答写入会话状态。"""
        intent = state.get("intent")
        reply = state.get("reply")
        result = state.get("result")
        if not ((intent == "chat" and reply) or (intent == "aiops" and result is not None)):
            raise ValueError("本轮没有有效的聊天或排障结果")
        turn: ChatTurn = {
            "query": state["query"],
            "intent": intent,
            "reply": reply,
            "result": result,
            "created_at": datetime.now(timezone.utc),
        }
        return {"turns": [turn]}

    async def ainvoke(self) -> OverviewOutput:
        """使用独立初始状态运行本次诊断。

        Returns:
            包含 intent、reply 和 result 的对外输出。
        """
        return await self.graph.ainvoke(
            self._initial_state(),
            config=self._run_config(),
            durability="sync" if self.thread_id is not None else None,
        )

    def _initial_state(self) -> dict:
        # checkpoint 会保留带 add reducer 的列表；新一轮显式覆盖排障临时数据。
        empty = Overwrite([]) if self.thread_id is not None else []
        return {
            "query": self.query, 
            "max_steps": self.max_steps,
            "steps": [], 
            "current_step_id": None, 
            "step_results": empty,
            "evidence": empty,
            "knowledge_documents": empty,
            "result": None,
            "intent": None,
            "reply": None,
        }

    def _run_config(self) -> RunnableConfig:
        config: RunnableConfig = {"recursion_limit": 2 * self.max_steps + 10}
        if self.thread_id is not None:
            config["configurable"] = {"thread_id": self.thread_id}
        return config

    async def astream(
        self,
    ) -> AsyncIterator[tuple[Literal["progress"], str] | tuple[Literal["result"], OverviewOutput]]:
        """流式处理完成的节点和最终的公开输出。"""

        final_state = {}
        stream = self.graph.astream(
            self._initial_state(),
            config=self._run_config(),
            stream_mode=["updates", "values"],
            version="v2",
            durability="sync" if self.thread_id is not None else None,
        )
        async with aclosing(stream):
            async for part in stream:
                if part["type"] == "updates":
                    for node_name in part["data"]:
                        if node_name != "record_turn":
                            yield "progress", node_name
                elif part["type"] == "values":
                    final_state = part["data"]

        yield "result", {
            "intent": final_state["intent"],
            "reply": final_state["reply"],
            "result": final_state["result"],
        }
