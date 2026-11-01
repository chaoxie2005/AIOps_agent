"""意图识别与普通聊天节点。"""

import json
import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import OverviewState
from aiops_agent.prompts.load_texts.load_chat_prompt import chat_agent_prompt, intent_agent_prompt


async def intent_node(state: OverviewState, model_client: ModelClient) -> dict:
    """识别用户消息属于普通聊天还是故障排查。

    Args:
        state: 包含本轮用户问题 query 的主状态。
        model_client: 由主图注入的模型客户端。

    Returns:
        包含 chat 或 aiops 意图的状态更新。

    Raises:
        ValueError: 模型未返回合法的意图 JSON。
    """
    messages = intent_agent_prompt.format_messages(question=state["query"])
    response = await model_client.ainvoke(messages)
    content = response.content
    match = re.search(r"\{.*\}", content, re.DOTALL) if isinstance(content, str) else None
    if match is None:
        raise ValueError("意图识别节点未获取到 JSON 输出")
    try:
        decision = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError("意图识别节点输出不是有效的 JSON") from exc
    if (not isinstance(decision, dict) or set(decision) != {"intent"}
            or decision["intent"] not in ("chat", "aiops")):
        raise ValueError("意图识别节点必须返回 chat 或 aiops")
    return {"intent": decision["intent"]}


async def chat_node(state: OverviewState, model_client: ModelClient, rag_tool: BaseTool) -> dict:
    """通过 Agent 按需检索知识库并回答本轮聊天消息。

    Args:
        state: 包含本轮用户问题 query 的主状态。
        model_client: 由主图注入的模型客户端。
        rag_tool: 使用 make_rag_search_tool 创建的知识库检索工具。

    Returns:
        包含最终聊天回复 reply 和空诊断结果 result 的状态更新。

    Raises:
        ValueError: Agent 未返回包含非空文本的最终模型回答。
    """
    messages = chat_agent_prompt.format_messages(question=state["query"])
    agent = create_agent(
        model=model_client.model,
        tools=[rag_tool],
        system_prompt=messages[0],
    )
    output = await agent.ainvoke({"messages": messages[1:]})
    output_messages = output.get("messages", [])
    response = output_messages[-1] if output_messages else None
    if (not isinstance(response, AIMessage) or response.tool_calls
            or not isinstance(response.content, str) or not response.content.strip()):
        raise ValueError("聊天节点未返回有效的最终文本回答")
    return {"reply": response.content.strip(), "result": None}
