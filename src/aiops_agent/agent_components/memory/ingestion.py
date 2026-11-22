"""为会话原始输入分配溯源 ID，并记录供后续压缩使用的标签。"""

import asyncio
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from pymongo.collection import Collection


EventKind = Literal["user", "assistant", "tool", "rag"]

_SECRET_NAME = r"[A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?token|authorization|password|secret(?:[_-]?key)?|token)" # 密钥名模式
_SECRET_KEY = re.compile(_SECRET_NAME, re.IGNORECASE) # 密钥名正则表达式
_SECRET_MENTION = re.compile(rf"\b{_SECRET_NAME}\b", re.IGNORECASE) # 密钥名提及正则表达式
_SECRET_ASSIGNMENT = re.compile(
    rf"(?P<key>\b{_SECRET_NAME}\b['\"]?[ \t]*[:=][ \t]*)"
    r"""(?P<value>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|(?:Bearer[ \t]+)?[^\s"',;}\]]+)""",
    re.IGNORECASE,
) # 密钥赋值正则表达式
_ERROR_MARKER = re.compile(
    r"\b(?:traceback|error(?:[_ -]?code)?|exception|errno|HTTP[ \t]+[45]\d\d)\b|报错|错误码|堆栈",
    re.IGNORECASE,
) # 错误标记正则表达式
_ID_PREFIX = {"user": "msg", "assistant": "msg", "tool": "tool_out", "rag": "rag"} # 事件 ID 前缀


def _redact(value: Any) -> Any:
    """递归遮蔽可识别的凭据值。

    Args:
        value: 待归档的正文或元数据。

    Returns:
        脱敏后的值。映射保留键名，序列保留元素顺序。
    """
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            """替换文本中匹配的凭据值。

            Args:
                match: 包含凭据名、赋值符号和值的正则匹配。

            Returns:
                保留凭据名、分隔符和引号的脱敏片段。
            """
            raw = match["value"]
            replacement = f"{raw[0]}[REDACTED]{raw[0]}" if raw[0] in "\"'" else "[REDACTED]"
            return match["key"] + replacement

        return _SECRET_ASSIGNMENT.sub(replace, value)
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if _SECRET_KEY.fullmatch(key) and item else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _tag(kind: EventKind, content: Any, status: str | None) -> str:
    """为归档事件分配压缩标签。

    Args:
        kind: 事件来源类型。
        content: 待打标的事件正文。
        status: 工具执行状态。缺省时按内容判断。

    Returns:
        `core` 表示工具执行失败（`status` 为 `error`），或正文命中报错、
        堆栈、凭据变量名等标记，例如 `Traceback` 或 `MODEL_API_KEY`。
        `volatile` 表示工具未报告失败，且正文只有 `{"status": 200}`。
        `unclassified` 表示以上规则均未命中，例如普通聊天内容，或含有
        实际指标数据的成功工具返回。这些事件仍会照常归档。
    """
    if status == "error":
        return "core"
    if kind == "tool" and isinstance(content, str) and len(content) <= 128:
        try:
            content = json.loads(content)
        except ValueError:
            pass
    if kind == "tool" and content == {"status": 200}:
        return "volatile"
    text = content if isinstance(content, str) else str(content)
    if _ERROR_MARKER.search(text) or _SECRET_MENTION.search(text):
        return "core"
    return "unclassified"


class TurnIngestion:
    """写入单次请求内的输入事件，使用稳定 ID 保证重复写入幂等。"""

    def __init__(self, collection: Collection, session_id: str) -> None:
        """为指定会话初始化采集器，并生成本轮 ID。

        Args:
            collection: 保存会话事件的 MongoDB 集合。
            session_id: 本轮请求所属的会话 ID。
        """
        self.collection = collection
        self.session_id = session_id
        self.turn_id = uuid4().hex

    async def record(
        self, kind: EventKind, content: Any, *, source_id: str = "", **metadata: Any,
    ) -> None:
        """打标签并脱敏事件，通过稳定 ID 幂等写入归档。

        Args:
            kind: 事件来源类型。
            content: 文本正文或结构化内容。
            source_id: 同一轮次内区分同类事件的来源标识。
            **metadata: 工具调用 ID、函数名、参数、状态等附加字段。

        Raises:
            PyMongoError: MongoDB 写入失败。
        """
        event_id = f"{_ID_PREFIX[kind]}_{self.turn_id}_{kind}"
        if source_id:
            event_id = f"{event_id}_{source_id}"
        event = {
            "_id": event_id,  # 事件 ID
            "session_id": self.session_id,  # 会话 ID
            "turn_id": self.turn_id,  # 本轮次 ID
            "kind": kind,  # 事件类型，取值为 user、`assistant`、`tool`、`rag`
            "created_at": datetime.now(timezone.utc),  # UTC 时间戳
            "tag": _tag(
                kind, content, metadata.get("status")
            ),  # 压缩标签，取值为 core、`volatile`、`unclassified`
            "content": _redact(
                content
            ),  # 脱敏后的正文，例如用户原文、工具输出或结构化诊断结果
            **_redact(metadata),  # 事件元数据 ，包含工具调用 ID、函数名、参数、状态等
        }
        await asyncio.to_thread(
            self.collection.update_one,
            {"_id": event_id, "session_id": self.session_id},
            {"$setOnInsert": event},
            upsert=True,
        )

    async def read_history_tool_events(self) -> list[dict[str, Any]]:
        """读取同一会话中先前请求归档的工具结果。

        Returns:
            按时间和事件 ID 排序的工具记录，排除本轮请求。
        """
        return await asyncio.to_thread(
            lambda: list(self.collection.find(
                {"session_id": self.session_id, "kind": "tool", "turn_id": {"$ne": self.turn_id}},
                {"content": 1, "tool_name": 1, "tool_call_id": 1, "arguments": 1, "status": 1},
            ).sort([("created_at", 1), ("_id", 1)]))
        )

    def tool_middleware(self, scope: str):
        """创建工具结果采集的异步中间件。

        Args:
            scope: 工具调用的作用域，例如普通聊天或排障步骤 ID。

        Returns:
            在工具返回或抛出异常时采集事件的 LangChain 中间件。
        """

        @wrap_tool_call
        async def capture_tool(request, handler):
            """归档工具结果并转交原有响应。

            Args:
                request: 包含工具调用 ID、名称和参数的请求。
                handler: 执行工具调用或转交给下一层中间件的异步函数。

            Returns:
                下游返回的原始工具响应。

            Raises:
                Exception: 下游抛出异常时，记录失败事件后重新抛出。
            """
            call = request.tool_call
            details = {
                # 将调用 ID 限定在聊天或排障步骤内，避免不同步骤的归档 ID 冲突。
                "source_id": f"{scope}:{call['id']}",
                "tool_call_id": call["id"],
                "tool_name": call["name"],
                "arguments": call["args"],
            }
            try:
                result = await handler(request)
            except Exception as exc:
                await self.record("tool", str(exc), status="error", **details)
                # 归档后重抛原异常，让上层沿用现有异常处理。
                raise
            if isinstance(result, ToolMessage):
                await self.record("tool", result.content, status=result.status, **details)
            return result  # 将原响应交回 LangChain。

        return capture_tool
