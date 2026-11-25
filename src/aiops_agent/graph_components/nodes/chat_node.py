"""意图识别与普通聊天节点。"""

import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.agent_components.memory.context import recent_messages
from aiops_agent.agent_components.memory.ingestion import TurnIngestion
from aiops_agent.agent_components.toolkits.recall_history import HistoryRecall
from aiops_agent.graph_components.states.state import OverviewState
from aiops_agent.output_validation import (
    IntentOutput,
    ainvoke_structured,
    dump_model,
    fallback_chat,
    fallback_intent,
)
from aiops_agent.prompts.load_texts.load_chat_prompt import chat_agent_prompt, intent_agent_prompt

logger = logging.getLogger(__name__)


def intent_prompt_messages(question: str) -> list[BaseMessage]:
    """构造意图识别和历史预算评估共用的提示消息。

    Args:
        question: 本轮用户输入。

    Returns:
        包含系统提示和本轮输入的消息列表。
    """
    return intent_agent_prompt.format_messages(question=question)


async def intent_node(
    state: OverviewState, model_client: ModelClient, *, prompt: list[BaseMessage] | None = None,
    compressed_history: str | None = None,
) -> dict:
    """识别用户消息属于普通聊天还是故障排查。

    模型输出格式异常时降级为 chat，不触发现场查询。

    Args:
        state: 包含本轮用户问题 query 的主状态。
        model_client: 由主图注入的模型客户端。
        prompt: 已构造的意图识别提示，省略时按本轮问题构造。
        compressed_history: 通过预算校验的本轮压缩历史。

    Returns:
        包含 chat 或 aiops 意图、以及可选降级说明的状态更新。
    """
    if prompt is None:
        prompt = intent_prompt_messages(state["query"])
    history = recent_messages(state, compressed_history=compressed_history)
    parsed, error = await ainvoke_structured(
        model_client, [prompt[0], *history, *prompt[1:]], IntentOutput,
    )
    if error is not None:
        logger.warning("意图识别失败，降级为 chat: %s", error)
        return {"intent": fallback_intent(), "degraded": error}
    assert parsed is not None
    return dump_model(parsed)


async def chat_node(
    state: OverviewState, model_client: ModelClient, rag_tool: BaseTool,
    *, ingestion: TurnIngestion | None = None, compressed_history: str | None = None,
    history_recall: HistoryRecall | None = None,
) -> dict:
    """通过 Agent 按需检索知识库并回答本轮聊天消息。

    Agent 异常或无有效文本时返回明确的降级回复，不向 API 层抛未处理异常。

    Args:
        state: 包含本轮用户问题和近期对话的主图状态。
        model_client: 由主图注入的模型客户端。
        rag_tool: 使用 make_rag_search_tool 创建的知识库检索工具。
        ingestion: 本轮会话事件采集器，缺省时跳过归档。
        compressed_history: 通过预算校验的本轮压缩历史。
        history_recall: 当前会话的原文召回能力，可选。

    Returns:
        包含聊天回复和空诊断结果的状态更新。
    """
    try:
        messages = chat_agent_prompt.format_messages(question=state["query"])
        tools = [rag_tool]
        if history_recall is not None:
            tools.append(history_recall.make_tool(messages[0]))
        agent = create_agent(
            model=model_client.model,
            tools=tools,
            system_prompt=messages[0],
            middleware=[ingestion.tool_middleware("chat")] if ingestion is not None else [],
        )
        history = recent_messages(state, compressed_history=compressed_history)
        output = await agent.ainvoke({"messages": [*history, *messages[1:]]})
        output_messages = output.get("messages", [])
        response = output_messages[-1] if output_messages else None
        if (not isinstance(response, AIMessage) or response.tool_calls
                or not isinstance(response.content, str) or not response.content.strip()):
            raise ValueError("聊天节点未返回有效的最终文本回答")
        return {"reply": response.content.strip(), "result": None}
    except Exception as exc:
        logger.exception("聊天节点执行失败")
        return fallback_chat(str(exc))
