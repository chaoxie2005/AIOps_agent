import json
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from loguru import logger

from aiops_agent.agent_components.llm import ModelClient, client_for_node
from aiops_agent.agent_components.memory.ingestion import TurnIngestion
from aiops_agent.agent_components.toolkits.query_metrics import query_metrics
from aiops_agent.agent_components.toolkits.rag_search import make_rag_search_tool
from aiops_agent.agent_components.toolkits.recall_history import HistoryRecall
from aiops_agent.graph_components.states.state import (
    Evidence, ExecutorInput, ExecutorPrivateState, ToolCallRecord, ToolResultRecord,
)
from aiops_agent.prompts.load_texts.load_execute_prompt import execute_agent_prompt


def _collect_metric_evidence(
    step_id: str, calls: list[ToolCallRecord], results: list[ToolResultRecord],
) -> list[Evidence]:
    """从成功的监控返回提取证据；时间和观察均取自实际样本。"""
    calls_by_id = {call["id"]: call for call in calls}
    evidence = {}
    for result in results:
        call = calls_by_id.get(result["tool_call_id"])
        if not result["success"] or call is None or call["tool_name"] != "query_metrics":
            continue
        try:
            data = json.loads(result["summary"])
            values = data["result"]
            if data["resultType"] == "vector":
                samples = [item["value"] for item in values]
            elif data["resultType"] == "matrix":
                samples = [sample for item in values for sample in item["values"]]
            elif data["resultType"] in ("scalar", "string"):
                samples = [values] if values else []
            else:
                continue
            if not samples:
                continue
            times = [datetime.fromtimestamp(float(sample[0]), timezone.utc) for sample in samples]
            evidence_id = f"{step_id}:{call['id']}"
            evidence[evidence_id] = {
                "id": evidence_id,
                "step_id": step_id,
                "source": "prometheus",
                "query": call["arguments"]["query"],
                "time_range": {"start": min(times), "end": max(times)},
                "observation": "Prometheus 实际返回：" + json.dumps(data, ensure_ascii=False),
                "raw_result_ref": None,
            }
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, OSError) as exc:
            logger.warning("工具调用 {} 的返回无法生成监控证据: {}", call["id"], exc)
    return list(evidence.values())


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


async def execute_node(
    state: ExecutorPrivateState, model_client: ModelClient, rag_tool: BaseTool,
    *, ingestion: TurnIngestion | None = None, history_recall: HistoryRecall | None = None,
    compressed_history: str | None = None,
) -> dict:
    """让 Agent 自行选择 RAG 或监控工具来执行当前排障步骤。

    Args:
        state: prepare_step 初始化的步骤状态。
        model_client: 提供 Agent 所使用模型的客户端。
        rag_tool: 使用 make_rag_search_tool 创建并绑定检索器的 RAG 工具。
        ingestion: 本轮会话事件采集器。缺省时跳过工具事件归档。
        history_recall: 当前会话的原文召回能力。
        compressed_history: 含溯源标签的本轮压缩历史。

    Returns:
        包含工具调用、工具结果、候选证据和步骤结果的状态更新。
        当前步骤已有结果时返回空字典。
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
        """将工具异常转换为可交还给模型的失败消息。

        Args:
            request: 包含工具名称和调用 ID 的请求。
            handler: 执行工具调用或转交给下一层中间件的异步函数。

        Returns:
            原始工具响应，或包含异常文本的失败工具消息。
        """
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
        middleware = [ToolCallLimitMiddleware(run_limit=remaining, exit_behavior="end")]
        if ingestion is not None:
            middleware.append(ingestion.tool_middleware(step_id))
        middleware.append(handle_tool_error)
        tools = [rag_tool, query_metrics]
        if history_recall is not None:
            tools.append(history_recall.make_tool(messages[0]))
        if compressed_history is not None:
            history_message = HumanMessage(content=compressed_history)
            with_history = [messages[0], history_message, *messages[1:]]
            limit = getattr(model_client, "input_token_limit", None)
            if limit is not None and model_client.estimate_message_tokens(with_history) <= limit:
                messages = with_history
        agent = create_agent(
            model=client_for_node(model_client, "execute").model,
            tools=tools,
            system_prompt=messages[0],
            middleware=middleware,
        )
        try:
            output = await agent.ainvoke({"messages": messages[1:]})
        except Exception as exc:
            logger.error("步骤 {} Agent 请求失败，异常类型: {}，原因: {}", step_id, type(exc).__name__, exc)
            raise
        for message in output["messages"]:
            if isinstance(message, AIMessage):
                calls.extend({"id": call["id"], "tool_name": call["name"], "arguments": call["args"]}  for call in message.tool_calls)
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

    evidence = {item["id"]: item for item in state.get("evidence_candidates", [])}
    evidence.update({item["id"]: item for item in _collect_metric_evidence(step_id, calls, results)})
    logger.info("步骤 {} 执行结束，状态: {}，累计工具调用数: {}", step_id, decision["status"], len(calls))
    return {
        "tool_calls": calls, 
        "tool_results": results, 
        "evidence_candidates": list(evidence.values()),
        "step_result": {
        "step_id": state["current_step"]["id"],
        "status": decision["status"],
        "summary": decision["summary"],
        "evidence_ids": list(evidence),
        "knowledge_document_ids": [item["id"] for item in state.get("knowledge_candidates", [])],
        "tool_call_ids": [call["id"] for call in calls],
        "error": decision["summary"] if decision["status"] == "failed" else None,
    }}


class ExecuteNodes:
    """保存执行子图的依赖，通过方法转发到已有节点函数。"""

    def __init__(
        self, model_client: ModelClient, rag_tool: BaseTool, max_tool_calls: int = 3,
        *, ingestion: TurnIngestion | None = None, history_recall: HistoryRecall | None = None,
        compressed_history: str | None = None,
    ) -> None:
        """保存执行子图所需的依赖。

        Args:
            model_client: 支持工具调用的模型客户端。
            rag_tool: 使用 make_rag_search_tool 创建的知识检索工具。
            max_tool_calls: 单个步骤允许的最大工具调用次数。
            ingestion: 本轮会话事件采集器。缺省时跳过工具事件归档。
            history_recall: 当前会话的原文召回能力。
            compressed_history: 含溯源标签的本轮压缩历史。
        """
        self.model_client = model_client
        self.rag_tool = rag_tool
        self.max_tool_calls = max_tool_calls
        self.ingestion = ingestion
        self.history_recall = history_recall
        self.compressed_history = compressed_history

    @classmethod
    def from_env(cls) -> "ExecuteNodes":
        """复用 RAG 节点的模型和检索器配置，创建执行节点依赖。"""
        from aiops_agent.graph_components.nodes.rag.rag_container import RagNodes

        nodes = RagNodes.from_env()
        return cls(nodes.model_client, make_rag_search_tool(nodes.retriever))

    async def prepare(self, state: ExecutorInput) -> dict:
        """找到当前步骤，并按配置初始化工具预算和执行记录。"""
        return await prepare_step(state, self.max_tool_calls)

    async def execute(self, state: ExecutorPrivateState) -> dict:
        """使用保存的依赖执行当前排障步骤。

        Args:
            state: 执行子图为当前步骤准备的状态。

        Returns:
            当前步骤的状态更新。已有步骤结果时返回空字典。
        """
        return await execute_node(
            state, self.model_client, self.rag_tool, ingestion=self.ingestion,
            history_recall=self.history_recall, compressed_history=self.compressed_history,
        )
