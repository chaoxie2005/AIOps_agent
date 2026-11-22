"""按来源 ID 召回当前会话的历史原文。"""

import asyncio
import json
from collections.abc import Sequence
from uuid import uuid4

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.agent_components.memory.ingestion import TurnIngestion, _redact
from aiops_agent.graph_components.states.state import ChatTurn


_MAX_PAGE_CHARS = 7999


class HistoryRecall:
    """绑定一次会话请求的原文来源，并协调并发的召回预算。"""

    def __init__(
        self, ingestion: TurnIngestion, turns: Sequence[ChatTurn], model_client: ModelClient,
    ) -> None:
        """保存已校验会话的归档读取器和历史问答。

        Args:
            ingestion: 已绑定到当前会话的归档读取器。
            turns: 当前会话的已完成问答。
            model_client: 提供本地模型预算和计数的客户端。
        """
        self.ingestion = ingestion
        self.turns = tuple(turns)
        self.model_client = model_client
        self._budget_lock = asyncio.Lock()
        self._allocated: dict[str, ToolMessage] = {}

    async def _read(self, source_id: str) -> str | None:
        """按本轮会话权限读取一条完整历史正文。

        Args:
            source_id: 历史轮次引用或归档事件 ID。

        Returns:
            可分页的原文文本。不存在或来源格式无效时返回 None。
        """
        if source_id.startswith("turn:"):
            index = source_id.removeprefix("turn:")
            if not index.isascii() or not index.isdecimal() or len(index) > len(str(len(self.turns))):
                return None
            number = int(index)
            if not 1 <= number <= len(self.turns):
                return None
            turn = self.turns[number - 1]
            original = {key: turn[key] for key in ("query", "reply", "result")}
            return json.dumps(_redact(original), ensure_ascii=False, default=str)
        if source_id.startswith(("msg_", "tool_out_", "rag_")):
            event = await self.ingestion.read_history_event(source_id)
            if event is not None:
                return json.dumps(event, ensure_ascii=False, default=str)
        return None

    async def _page(
        self, source_id: str, offset: int, original: str,
        system_prompt: BaseMessage, runtime: ToolRuntime,
    ) -> ToolMessage:
        """生成可放进模型剩余输入预算的最长一页。

        Args:
            source_id: 原文来源。
            offset: 起始字符位置。
            original: 完整序列化原文。
            system_prompt: 当前 Agent 的系统消息。
            runtime: 当前工具调用的运行状态。

        Returns:
            带原文片段、下一页位置或预算不足状态的工具消息。
        """
        call_id = runtime.tool_call_id
        response_id = f"history-recall-{uuid4().hex}"
        available = getattr(self.model_client, "input_token_limit", None)
        if available is None:
            return ToolMessage(content='{"status":"budget_unknown"}', tool_call_id=call_id, name="recall_history")

        async with self._budget_lock:
            history = runtime.state.get("messages", [])
            visible_ids = {message.id for message in history}
            reserved = [message for key, message in self._allocated.items() if key not in visible_ids]
            # 同一批工具调用可能并行执行，为其他召回的最短回复留出空间。
            siblings = []
            if history and isinstance(history[-1], AIMessage):
                siblings = [
                    ToolMessage(content='{"status":"budget_exhausted"}', tool_call_id=call["id"], name="recall_history")
                    for call in history[-1].tool_calls
                    if call["name"] == "recall_history" and call["id"] != call_id
                    and all(message.tool_call_id != call["id"] for message in reserved)
                ]
            prefix = [system_prompt, *history, *reserved, *siblings]
            low, high = 1, min(_MAX_PAGE_CHARS, len(original) - offset)
            best = None
            while low <= high:
                size = (low + high) // 2
                end = offset + size
                body = json.dumps({
                    "status": "ok",
                    "source_id": source_id,
                    "offset": offset,
                    "next_offset": end if end < len(original) else None,
                    "content": original[offset:end],
                }, ensure_ascii=False, separators=(",", ":"))
                candidate = ToolMessage(
                    content=body, id=response_id, tool_call_id=call_id, name="recall_history",
                )
                if self.model_client.estimate_message_tokens([*prefix, candidate]) <= available:
                    best = candidate
                    low = size + 1
                else:
                    high = size - 1
            if best is not None:
                self._allocated[response_id] = best
                return best
            return ToolMessage(
                content='{"status":"budget_exhausted"}', tool_call_id=call_id, name="recall_history",
            )

    def make_tool(self, system_prompt: BaseMessage) -> BaseTool:
        """为使用指定系统提示的 Agent 创建召回工具。

        Args:
            system_prompt: 已绑定到 Agent 的系统消息，用于预算评估。

        Returns:
            仅向模型暴露来源 ID 和分页位置的工具。
        """
        @tool
        async def recall_history(source_id: str, runtime: ToolRuntime, offset: int = 0) -> ToolMessage:
            """当历史摘要缺少细节时，按溯源标记中的来源 ID 读取原文。offset 缺省为 0，后续使用返回的 next_offset。"""
            original = await self._read(source_id) if offset >= 0 else None
            if original is None or offset >= len(original):
                return ToolMessage(
                    content='{"status":"not_found"}', tool_call_id=runtime.tool_call_id, name="recall_history",
                )
            return await self._page(source_id, offset, original, system_prompt, runtime)

        return recall_history
