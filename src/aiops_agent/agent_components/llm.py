"""从统一环境配置创建对话模型客户端。"""

import os
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage


class ModelClient:
    """提供统一配置的同步和异步模型调用。"""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        model_provider: str = "openai",
        base_url: str | None = None,
        temperature: float = 0.5,
        max_retries: int = 3,
        max_tokens: int | None = None,
    ):
        """创建模型，显式参数优先于环境变量，环境变量优先于项目 .env。

        Args:
            model_name: 模型名称；未传入时读取 LLM_MODEL_NAME。
            api_key: 模型密钥；未传入时读取 LLM_API_KEY。
            model_provider: 模型提供方，默认使用 openai 兼容接口。
            base_url: 模型地址；未传入时读取 LLM_BASE_URL。
            temperature: 模型采样温度。
            max_retries: 模型请求的最大重试次数。
            max_tokens: 单次输出的 token 上限；未传入时读取 LLM_MAX_TOKENS。
                参数和环境配置均未提供时，采用模型默认值。

        Raises:
            ValueError: 必需配置缺失、模型地址不合法，或 token 上限不是正整数。
        """
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        config = {
            "LLM_MODEL_NAME": model_name if model_name is not None else os.getenv("LLM_MODEL_NAME"),
            "LLM_API_KEY": api_key if api_key is not None else os.getenv("LLM_API_KEY"),
            "LLM_BASE_URL": base_url if base_url is not None else os.getenv("LLM_BASE_URL"),
        }
        for name, value in config.items():
            if value is None or not value.strip():
                raise ValueError(f"缺少必需的模型配置: {name}")
            config[name] = value.strip()
        base_url = config["LLM_BASE_URL"].rstrip("/")
        address = urlsplit(base_url)
        if address.scheme not in ("http", "https") or not address.hostname:
            raise ValueError("LLM_BASE_URL 必须是包含 http 或 https 的模型服务地址")
        kwargs = {
            "model": config["LLM_MODEL_NAME"],
            "model_provider": model_provider,
            "api_key": config["LLM_API_KEY"],
            "base_url": base_url,
            "temperature": temperature,
            "max_retries": max_retries,
        }
        token_config_name = "max_tokens"
        if max_tokens is None:
            token_config_name = "LLM_MAX_TOKENS"
            configured_tokens = os.getenv("LLM_MAX_TOKENS")
            if configured_tokens is not None:
                try:
                    max_tokens = int(configured_tokens)
                except ValueError as exc:
                    raise ValueError("LLM_MAX_TOKENS 必须为正整数") from exc
        if max_tokens is not None:
            if type(max_tokens) is not int or max_tokens <= 0:
                raise ValueError(f"{token_config_name} 必须为正整数")
            kwargs["max_tokens"] = max_tokens
        self.model = init_chat_model(**kwargs)
        self.input_token_limit = self._resolve_input_token_limit(max_tokens or 1024)

    def _resolve_input_token_limit(self, reserved_output_tokens: int) -> int | None:
        """读取模型窗口并扣除为输出预留的预算。

        Args:
            reserved_output_tokens: 为模型生成预留的 Token 数。

        Returns:
            可用的输入预算。窗口未知或模型元数据不足时返回 None。

        Raises:
            ValueError: LLM_CONTEXT_WINDOW 不是大于输出预留的整数。
        """
        configured = os.getenv("LLM_CONTEXT_WINDOW")
        if configured is not None:
            try:
                window = int(configured)
            except ValueError as exc:
                raise ValueError("LLM_CONTEXT_WINDOW 必须是大于输出预留的整数") from exc
            if window <= reserved_output_tokens:
                raise ValueError("LLM_CONTEXT_WINDOW 必须是大于输出预留的整数")
            return window - reserved_output_tokens

        profile = getattr(self.model, "profile", None) or {}
        window = profile.get("max_input_tokens")
        if type(window) is int and window > reserved_output_tokens:
            return window - reserved_output_tokens
        return None

    def estimate_message_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """本地估算文本消息的 Token 数。

        Args:
            messages: 参与历史上下文水位线评估的文本消息。

        Returns:
            模型计数结果，或使用本地文本 tokenizer 加消息开销得到的估算值。
        """
        try:
            return self.model.get_num_tokens_from_messages(messages) # 使用模型计数结果
        except NotImplementedError:
            return 3 + sum(self.model.get_num_tokens(str(message.content)) + 4 for message in messages)

    def estimate_text_tokens(self, text: str) -> int:
        """用本地 tokenizer 估算纯文本的 Token 数。

        Args:
            text: 待校验的摘要正文。

        Returns:
            文本 Token 数，不包含消息包装开销。
        """
        return self.model.get_num_tokens(text)

    def invoke(self, messages: list):
        """同步调用模型。

        Args:
            messages: 发送给模型的消息列表。

        Returns:
            模型返回的消息。
        """
        return self.model.invoke(messages)

    async def ainvoke(self, messages: list):
        """异步调用模型。

        Args:
            messages: 发送给模型的消息列表。

        Returns:
            模型返回的消息。
        """
        return await self.model.ainvoke(messages)
