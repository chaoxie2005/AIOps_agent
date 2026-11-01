"""从统一环境配置创建对话模型客户端。"""

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


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
