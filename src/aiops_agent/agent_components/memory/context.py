"""构建聊天和排障使用的近期对话上下文。"""

from langchain_core.messages import AIMessage, HumanMessage

from aiops_agent.graph_components.states.state import OverviewState

_CONTEXT_TURNS = 5
_CONTEXT_MESSAGE_CHARS = 2000


def recent_messages(state: OverviewState) -> list[HumanMessage | AIMessage]:
    """将最近几轮已完成的问答转换为有角色的模型上下文。"""
    messages = []
    for turn in state.get("turns", [])[-_CONTEXT_TURNS:]:
        answer = turn["reply"]
        if turn["result"] is not None:
            result = turn["result"]
            answer = "\n".join(filter(None, [
                result["summary"],
                result["root_cause"],
                *result["recommendations"],
            ]))
        messages.extend([
            HumanMessage(content=turn["query"][:_CONTEXT_MESSAGE_CHARS]),
            AIMessage(content=(answer or "")[:_CONTEXT_MESSAGE_CHARS]),
        ])
    return messages


def contextual_question(state: OverviewState) -> str:
    """为接收字符串问题的排障节点保留近期对话背景。"""
    history = recent_messages(state)
    if not history:
        return state["query"]
    context = "\n".join(
        f"{'用户' if isinstance(message, HumanMessage) else '助手'}: {message.content}"
        for message in history
    )
    return f"近期对话：\n{context}\n\n本轮问题：\n{state['query']}"
