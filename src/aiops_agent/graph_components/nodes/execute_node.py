import json

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from loguru import logger

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.agent_components.toolkits.query_metrics import query_metrics
from aiops_agent.agent_components.toolkits.rag_search import make_rag_search_tool
from aiops_agent.graph_components.states.state import ExecutorInput, ExecutorPrivateState
from aiops_agent.prompts.load_texts.load_execute_prompt import execute_agent_prompt


async def prepare_step(state: ExecutorInput, max_tool_calls: int = 3) -> dict:
    """根据步骤 ID 找到本次要执行的步骤，标记为执行中，并清空本步骤的工具记录和结果。"""
    current_step = next((step for step in state["steps"] if step["id"] == state["current_step_id"]), None)
    if current_step is None:
        logger.error("准备步骤失败，未找到步骤 ID: {}", state["current_step_id"])
        raise ValueError(f"未找到 ID 为 {state['current_step_id']} 的步骤")
    logger.info("准备步骤 {}，目标: {}，工具调用预算: {}", current_step["id"], current_step["goal"], max_tool_calls)
    return {
        "current_step": {**current_step, "status": "running"},
        "tool_calls": [],
        "tool_results": [],
        "evidence_candidates": [],
        "knowledge_candidates": [],
        "step_result": None,
        "max_tool_calls": max_tool_calls,
    }


async def execute_node(state: ExecutorPrivateState, model_client: ModelClient, rag_tool: BaseTool) -> dict:
    """让 Agent 自行选择 RAG 或监控工具，完成当前排障步骤。

    Args:
        state: prepare_step 初始化的步骤状态。
        model_client: 模型客户端，使用其底层模型创建 Agent。
        rag_tool: 使用 make_rag_search_tool 创建并绑定检索器的 RAG 工具。

    Returns:
        包含 tool_calls、tool_results 和 step_result 的状态更新。
        Agent 内部完成工具选择、执行和结束判断。
    """
    if state.get("step_result") is not None:
        logger.info("步骤 {} 已有执行结果，跳过执行", state["step_result"]["step_id"])
        return {}
    step_id = state["current_step"]["id"]
    calls = list(state.get("tool_calls", []))
    results = list(state.get("tool_results", []))
    remaining = state.get("max_tool_calls", 3) - len(calls)
    logger.info("开始执行步骤 {}，目标: {}，剩余工具调用次数: {}", step_id, state["current_step"]["goal"], remaining)

    @wrap_tool_call
    async def handle_tool_error(request, handler):
        """把工具异常作为失败结果交还模型，让它决定是否继续。"""
        logger.info("步骤 {} 调用工具 {}，调用 ID: {}", step_id, request.tool_call["name"], request.tool_call["id"])
        try:
            return await handler(request)
        except Exception as exc:
            logger.warning("步骤 {} 工具调用失败，调用 ID: {}，原因: {}", step_id, request.tool_call["id"], exc)
            return ToolMessage(content=str(exc), tool_call_id=request.tool_call["id"], status="error")

    decision = {"status": "failed", "summary": "工具调用次数已达到上限"}
    if remaining > 0:
        messages = execute_agent_prompt.format_messages(
            remaining_tool_calls=remaining,
            execution_context=json.dumps(state, ensure_ascii=False, default=str),
        )
        agent = create_agent(
            model=model_client.model,
            tools=[rag_tool, query_metrics],
            system_prompt=messages[0],
            middleware=[ToolCallLimitMiddleware(run_limit=remaining, exit_behavior="end"), handle_tool_error],
        )
        try:
            output = await agent.ainvoke({"messages": messages[1:]})
        except Exception as exc:
            logger.error("步骤 {} Agent 请求失败，异常类型: {}，原因: {}", step_id, type(exc).__name__, exc)
            raise
        for message in output["messages"]:
            if isinstance(message, AIMessage):
                calls.extend({"id": call["id"], "tool_name": call["name"], "arguments": call["args"]}
                             for call in message.tool_calls)
            elif isinstance(message, ToolMessage):
                logger.info("步骤 {} 收到工具结果，调用 ID: {}，状态: {}", step_id, message.tool_call_id, message.status)
                content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
                results.append({
                    "tool_call_id": message.tool_call_id,
                    "success": message.status != "error",
                    "summary": content,
                    "raw_result_ref": None,
                    "error": content if message.status == "error" else None,
                })
        if len(calls) <= state.get("max_tool_calls", 3):
            try:
                decision = json.loads(output["messages"][-1].content)
                if (not isinstance(decision, dict)
                        or decision.get("status") not in ("completed", "failed", "skipped")
                        or not isinstance(decision.get("summary"), str)):
                    raise ValueError("步骤结果格式错误")
            except (ValueError, TypeError) as exc:
                logger.warning("步骤 {} 模型结果格式错误: {}", step_id, exc)
                decision = {"status": "failed", "summary": "模型未返回有效的步骤结果 JSON"}
        else:
            logger.warning("步骤 {} 工具调用超出预算，Agent 已停止执行", step_id)
    else:
        logger.warning("步骤 {} 工具调用预算已耗尽", step_id)

    logger.info("步骤 {} 执行结束，状态: {}，累计工具调用数: {}", step_id, decision["status"], len(calls))
    return {"tool_calls": calls, "tool_results": results, "step_result": {
        "step_id": state["current_step"]["id"],
        "status": decision["status"],
        "summary": decision["summary"],
        "evidence_ids": [item["id"] for item in state.get("evidence_candidates", [])],
        "knowledge_document_ids": [item["id"] for item in state.get("knowledge_candidates", [])],
        "tool_call_ids": [call["id"] for call in calls],
        "error": decision["summary"] if decision["status"] == "failed" else None,
    }}


class ExecuteNodes:
    """保存执行子图的依赖，通过方法转发到已有节点函数。"""

    def __init__(self, model_client: ModelClient, rag_tool: BaseTool, max_tool_calls: int = 3) -> None:
        """保存模型、RAG 工具和每个步骤的工具调用预算。

        Args:
            model_client: 支持工具调用的模型客户端。
            rag_tool: 使用 make_rag_search_tool 创建的知识检索工具。
            max_tool_calls: 单个步骤最多允许的工具调用次数，默认 3 次。
        """
        self.model_client = model_client
        self.rag_tool = rag_tool
        self.max_tool_calls = max_tool_calls

    @classmethod
    def from_env(cls) -> "ExecuteNodes":
        """复用 RAG 节点的模型和检索器配置，创建执行节点依赖。"""
        from aiops_agent.graph_components.nodes.rag_nodes import RagNodes

        nodes = RagNodes.from_env()
        return cls(nodes.model_client, make_rag_search_tool(nodes.retriever))

    async def prepare(self, state: ExecutorInput) -> dict:
        """找到当前步骤，并按配置初始化工具预算和执行记录。"""
        return await prepare_step(state, self.max_tool_calls)

    async def execute(self, state: ExecutorPrivateState) -> dict:
        """使用保存的模型和 RAG 工具执行当前步骤。"""
        return await execute_node(state, self.model_client, self.rag_tool)
