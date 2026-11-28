"""大模型结构化输出工具：优先使用原生 schema 输出，兼容纯文本 JSON。

解析失败会以返回值形式交给调用方做业务降级，不直接把格式错误抛到 HTTP 层。
一些需要模型输出schema的节点使用，调用模型
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _message_text(response: Any) -> str:
    """从模型响应中提取文本内容。

    Args:
        response: 模型响应对象或原始内容。

    Returns:
        拼接后的文本；没有文本时返回空字符串。
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _tool_call_args(response: Any, schema: type[BaseModel]) -> dict[str, Any] | None:
    """从模型响应的工具调用中提取结构化参数。

    Args:
        response: 模型响应对象。
        schema: 用于匹配工具名称的 Pydantic schema。

    Returns:
        找到工具参数时返回字典，否则返回 None。
    """
    if not hasattr(response, "tool_calls"):
        return None
    calls = response.tool_calls or []
    if not calls:
        return None
    expected = schema.model_config.get("title") or schema.__name__
    for call in calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        if name == expected and isinstance(args, dict):
            return args
    first = calls[0]
    args = first.get("args") if isinstance(first, dict) else getattr(first, "args", None)
    return args if isinstance(args, dict) else None


def parse_structured_response(response: Any, schema: type[T]) -> T:
    """把模型响应解析为校验通过的 Pydantic 对象。

    Args:
        response: 模型响应对象或原始内容。
        schema: 用于校验输出的 Pydantic schema。

    Returns:
        校验通过的 schema 实例。

    Raises:
        ValueError: 响应为空、不是合法 JSON，或字段不符合 schema。
    """
    args = _tool_call_args(response, schema)
    if args is not None:
        try:
            return schema.model_validate(args)
        except ValidationError as exc:
            raise ValueError(f"{schema.__name__} 工具调用参数校验失败: {exc}") from exc

    text = _message_text(response).strip()
    if not text:
        raise ValueError(f"{schema.__name__} 响应为空")
    try:
        data = parse_json_markdown(text) # 解析 JSON 格式文本
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{schema.__name__} 响应不是有效 JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{schema.__name__} 响应 JSON 必须是对象")
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{schema.__name__} 响应字段校验失败: {exc}") from exc


async def ainvoke_structured(
    client: Any,
    messages: Sequence[BaseMessage],
    schema: type[T],
    *,
    retries: int = 2,
    correction: str = "输出必须严格符合指定 JSON schema，不要附加解释。",
) -> tuple[T | None, str | None]:
    """调用模型并按 schema 校验输出。

    Args:
        client: 模型对象，或持有 ``.model`` 属性的 ModelClient 包装器。
        messages: 发送给模型的消息列表。
        schema: 用于校验输出的 Pydantic schema。
        retries: 校验失败后的纠错重试次数。
        correction: 重试时追加到对话末尾的纠错指令。

    Returns:
        ``(校验结果, 错误信息)``；两者只有一个非空。
    """
    error: str | None = None
    structured = None
    invoke = getattr(client, "ainvoke", None) # 获取 client 自身的 ainvoke 方法，或 None
    if not callable(invoke):
        # client 没有 ainvoke 时，取 client.model（裸 LangChain 模型）或 client 本身，
        model = getattr(client, "model", client)
        try:
            structured = model.with_structured_output(schema) # 尝试让模型以原生结构化输出方式返回
        # 失败则退化为文本 JSON 解析。
        except (AttributeError, NotImplementedError, ValueError, TypeError) as exc:
            logger.debug("模型不支持原生结构化输出，改用文本 JSON 解析: %s", exc)
        invoke = getattr(structured or model, "ainvoke", None)
    if not callable(invoke):
        return None, "模型不支持 ainvoke"

    request: Any = list(messages)
    for attempt in range(retries + 1):
        try:
            # 真正的模型调用：invoke 是 ModelClient.ainvoke 或模型的 ainvoke，
            # 最终经由 LangChain 的 ChatOpenAI 发送请求到 LLM_BASE_URL。
            response = await invoke(request)
            # 原生结构化输出直接返回 schema 实例；否则从响应文本中解析 JSON 并校验。
            parsed = (
                response
                if isinstance(response, schema)
                else parse_structured_response(response, schema)
            )
            return parsed, None
        except Exception as exc:
            # 模型服务异常也视为输出不可用，由调用方统一降级；
            # 不在节点中重复抛给 HTTP 层。
            error = f"第 {attempt + 1} 次结构化输出失败（{type(exc).__name__}）: {exc}"
            logger.warning("%s", error)
            if attempt == retries:
                break
            request = [*request, HumanMessage(content=correction)]
    return None, error


def dump_model(model: BaseModel) -> dict[str, Any]:
    """把 Pydantic 对象转换为图状态使用的字典。

    Args:
        model: 校验通过的 Pydantic 实例。

    Returns:
        JSON 兼容的字典表示。
    """
    return model.model_dump(mode="json")
