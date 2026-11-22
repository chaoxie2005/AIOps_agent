"""构建聊天和排障使用的近期对话上下文。"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aiops_agent.graph_components.states.state import ChatTurn, OverviewState

_CONTEXT_TURNS = 5 # 上下文轮次数
_CONTEXT_MESSAGE_CHARS = 2000 # 每个轮次的最大字符数
_MAX_CHUNK_CHARS = 7999 # 每个历史单元（用户的输入和模型的回答、单次工具调用结果）的最大字符数
_CJK = re.compile(r"[\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]") # 中日韩字符正则表达式


@dataclass(frozen=True)
class HistoryChunk:
    """保留一段历史正文及其来源和顺序。"""

    source: Literal["turn", "tool"] # 历史单元的来源类型
    source_id: str # 历史单元的来源 ID，如轮次序号或归档事件 ID
    part_index: int # 历史单元在轮次中的索引
    text: str # 原始历史单元文本
    estimated_words: int # 估算的词数，包含中日韩字符的加权字数


@dataclass(frozen=True)
class ContextEvaluation:
    """记录当前输入预算和供后续阶段处理的切片。"""

    estimated_tokens: int # 估算的 token 数
    input_token_limit: int # 输入 token 限制，单位为 Token
    chunks: tuple[HistoryChunk, ...] = () # 历史单元切片

    @property
    def needs_compression(self) -> bool:
        """判断估算输入量是否严格超过 70% 水位线。"""
        return self.estimated_tokens * 10 > self.input_token_limit * 7


def estimate_words(text: str) -> int:
    """估算空白分隔的词数和中日韩字符的加权字数。

    Args:
        text: 待统计的文本。

    Returns:
        非 CJK 词数加上按 0.7 加权并四舍五入的 CJK 字符数。
    """
    without_cjk, cjk_count = _CJK.subn(" ", text)
    return len(without_cjk.split()) + (7 * cjk_count + 5) // 10 # 英文单词数 + 中日韩字符数的加权值


def _turn_answer(turn: ChatTurn) -> str:
    """提取与现有模型上下文一致的回答正文。

    Args:
        turn: 已完成的一轮问答。

    Returns:
        普通聊天回复或排障摘要、根因和建议。
    """
    result = turn["result"]
    if result is None:
        return turn["reply"] or ""
    return "\n".join(filter(None, [result["summary"], result["root_cause"], *result["recommendations"]]))


def _history_chunks(
    source: Literal["turn", "tool"], source_id: str, text: str,
) -> tuple[HistoryChunk, ...]:
    """优先在换行处拆分大的历史单元，保留原始正文。

    Args:
        source: 历史单元的来源类型。
        source_id: 轮次序号或归档事件 ID。
        text: 完整的历史单元文本。

    Returns:
        可按序拼回原文的切片。每片严格少于 8,000 个字符。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + _MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            line_end = text.rfind("\n", start + _MAX_CHUNK_CHARS // 2, end)
            if line_end != -1:
                end = line_end + 1
        part = text[start:end]
        chunks.append(HistoryChunk(source, source_id, len(chunks), part, estimate_words(part)))
        start = end
    return tuple(chunks)


def evaluate_history(
    turns: Sequence[ChatTurn], prompt: Sequence[BaseMessage],
    input_token_limit: int, count_tokens: Callable[[Sequence[BaseMessage]], int],
) -> ContextEvaluation:
    """评估完整问答历史的预计输入量，超线时生成轮次切片。

    Args:
        turns: 按时间排序的已完成问答。
        prompt: 包含系统提示和本轮用户输入的意图识别提示。
        input_token_limit: 为输出预留空间后可用的输入 Token 上限。
        count_tokens: 模型客户端提供的本地 Token 计数函数。

    Returns:
        输入估算以及超过水位线时的历史切片。
    """
    answers = [_turn_answer(turn) for turn in turns]
    history = []
    for turn, answer in zip(turns, answers, strict=True):
        history.extend((HumanMessage(content=turn["query"]), AIMessage(content=answer)))
    estimated_tokens = count_tokens([prompt[0], *history, *prompt[1:]])
    evaluation = ContextEvaluation(estimated_tokens, input_token_limit)
    if not evaluation.needs_compression:
        return evaluation
    chunks = tuple(
        chunk
        for index, (turn, answer) in enumerate(zip(turns, answers, strict=True), start=1)
        for chunk in _history_chunks("turn", str(index), f"用户：{turn['query']}\n助手：{answer}")
    )
    return replace(evaluation, chunks=chunks)


def tool_result_chunks(event: Mapping[str, Any]) -> tuple[HistoryChunk, ...]:
    """将一条归档工具结果单独切片。

    Args:
        event: 包含事件 ID、工具调用信息和工具返回的归档记录。

    Returns:
        带有归档事件 ID 的有序切片。
    """
    call = {key: event[key] for key in ("tool_name", "tool_call_id", "arguments", "status") if key in event}
    content = event["content"]
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    text = f"{json.dumps(call, ensure_ascii=False, default=str)}\n{content}"
    return _history_chunks("tool", event["_id"], text)


def recent_messages(state: OverviewState) -> list[HumanMessage | AIMessage]:
    """将最近几轮已完成的问答转换为有角色的模型上下文。"""
    messages = []
    for turn in state.get("turns", [])[-_CONTEXT_TURNS:]:
        answer = _turn_answer(turn)
        messages.extend([
            HumanMessage(content=turn["query"][:_CONTEXT_MESSAGE_CHARS]),
            AIMessage(content=answer[:_CONTEXT_MESSAGE_CHARS]),
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
