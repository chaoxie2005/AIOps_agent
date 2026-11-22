"""在意图识别前生成符合预算和保留要求的会话摘要。"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import groupby
from logging import getLogger
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.agent_components.memory.context import (
    ContextEvaluation, evaluate_history, tool_result_chunks,
)
from aiops_agent.agent_components.memory.ingestion import TurnIngestion
from aiops_agent.graph_components.states.state import ChatTurn
from aiops_agent.utils.prompt_loader import load_prompt


_SUMMARY_TOKEN_LIMIT = 200 # 压缩后的摘要最大token数
_MAX_CORRECTIONS = 2 # 最大允许的错误修正次数 
_MAX_BATCH_CHUNKS = 8 # 最大批量处理的chunk数
_TIGHTEN = "收紧措辞，不准新增事实。只输出完整、可用的摘要。"
_COMPRESS_PROMPT = load_prompt("compress_history_prompt")
_TOOL_FIELDS = ("_id", "tool_call_id", "tool_name", "arguments", "status")
logger = getLogger(__name__)


def _protected_history(
    turns: Sequence[ChatTurn], events: Sequence[Mapping[str, Any]], evaluation: ContextEvaluation,
) -> str:
    """从原始记录组装必须保留的历史内容。

    Args:
        turns: 按时间排序的已完成问答。
        events: 对应会话中独立归档的工具事件。
        evaluation: 包含原文保留标记的历史评估结果。

    Returns:
        含用户原话、工具调用字段和完整堆栈的历史文本。
    """
    sections = []
    user_inputs = [f"第 {index} 轮用户原话：{turn['query']}" for index, turn in enumerate(turns, start=1)]
    if user_inputs:
        sections.append("\n".join(user_inputs))
    if events:
        tool_calls = [
            {key: event[key] for key in _TOOL_FIELDS if key in event}
            for event in events
        ]
        sections.append("历史工具调用：\n" + json.dumps(tool_calls, ensure_ascii=False, default=str))
    protected = (chunk for chunk in evaluation.chunks if chunk.preserve_original)
    for (source, source_id), parts in groupby(protected, key=lambda chunk: (chunk.source, chunk.source_id)):
        sections.append(f"原始记录 {source} {source_id}：\n" + "".join(part.text for part in parts))
    return "\n\n".join(sections)


def _budget_messages(prompt: Sequence[BaseMessage], history: str) -> list[BaseMessage]:
    """构造意图识别节点会收到的消息，用于检查注入预算。

    Args:
        prompt: 意图识别节点的系统提示和当前输入。
        history: 准备注入的压缩历史。

    Returns:
        模型实际接收顺序下的完整消息。
    """
    return [prompt[0], HumanMessage(content=history), *prompt[1:]]


async def _summarize(
    evaluation: ContextEvaluation, model_client: ModelClient,
) -> str | None:
    """按输入预算分批合并历史，并验证每次生成的摘要。

    Args:
        evaluation: 包含可摘要切片和模型输入上限的历史评估。
        model_client: 提供模型调用和 Token 估算的客户端。

    Returns:
        通过长度校验的最终摘要。批次放不下或校验耗尽时返回 None。
    """
    system = SystemMessage(content=f"{_COMPRESS_PROMPT}\n摘要必须在 {_SUMMARY_TOKEN_LIMIT} Token 以内。")
    correction = HumanMessage(content=_TIGHTEN)
    chunks = evaluation.compressible_chunks
    start = 0
    summary = ""
    while start < len(chunks):
        batch = []
        messages: list[BaseMessage] = []
        for chunk in chunks[start:start + _MAX_BATCH_CHUNKS]:
            entry = f"来源 {chunk.source} {chunk.source_id}，片 {chunk.part_index}：\n{chunk.text}"
            content = "\n\n".join([f"已有摘要：\n{summary}", "本批历史：", *batch, entry])
            candidate = [system, HumanMessage(content=content)]
            if model_client.estimate_message_tokens([*candidate, correction]) > evaluation.input_token_limit:
                break
            batch.append(entry)
            messages = candidate
        if not batch:
            return None

        request = messages
        for attempt in range(_MAX_CORRECTIONS + 1):
            response = await model_client.ainvoke(request)
            draft = response.content
            if isinstance(draft, str) and draft.strip():
                draft = draft.strip()
                if model_client.estimate_text_tokens(draft) <= _SUMMARY_TOKEN_LIMIT:
                    summary = draft
                    break
            if attempt == _MAX_CORRECTIONS:
                return None
            retry = [*messages, correction]
            if isinstance(draft, str) and draft.strip():
                with_draft = [*messages, AIMessage(content=draft), correction]
                if model_client.estimate_message_tokens(with_draft) <= evaluation.input_token_limit:
                    retry = with_draft
            request = retry
        start += len(batch)

    return summary


async def compress_history_node(
    turns: Sequence[ChatTurn], prompt: Sequence[BaseMessage],
    model_client: ModelClient, ingestion: TurnIngestion | None,
) -> tuple[ContextEvaluation | None, str | None]:
    """评估历史并在超线时调用模型生成可信长度的摘要候选。

    Args:
        turns: 当前会话的已完成问答。
        prompt: 即将用于意图识别的系统提示和用户输入。
        model_client: 当前主图使用的模型客户端。
        ingestion: 绑定到当前会话的事件读取器。

    Returns:
        历史评估和可以注入本轮的压缩文本。沿用近期历史时文本为 None。
    """
    input_limit = getattr(model_client, "input_token_limit", None)
    if input_limit is None:
        return None, None
    evaluation = None
    try:
        evaluation = evaluate_history(turns, prompt, input_limit, model_client.estimate_message_tokens)
        if not evaluation.needs_compression:
            return evaluation, None
        events = await ingestion.read_history_tool_events() if ingestion is not None else []
        tool_chunks = tuple(chunk for event in events for chunk in tool_result_chunks(event))
        evaluation = replace(evaluation, chunks=evaluation.chunks + tool_chunks)
        if not evaluation.chunks:
            return evaluation, None

        protected = _protected_history(turns, events, evaluation)
        if model_client.estimate_message_tokens(_budget_messages(prompt, protected)) > input_limit:
            return evaluation, None
        if not evaluation.compressible_chunks:
            return evaluation, protected
        summary = await _summarize(evaluation, model_client)
        if summary is None:
            return evaluation, None
        combined = f"历史摘要：\n{summary}"
        if protected:
            combined += f"\n\n{protected}"
        if model_client.estimate_message_tokens(_budget_messages(prompt, combined)) <= input_limit:
            return evaluation, combined
    except Exception as exc:
        logger.warning("历史压缩失败，沿用近期会话历史（%s）", type(exc).__name__)
    return evaluation, None
