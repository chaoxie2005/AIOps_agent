from langchain.chat_models import init_chat_model


class ModelClient:
    """模型客户端"""
    def __init__(
        self, 
        model_name: str, 
        api_key: str | None = None,
        model_provider: str = "openai",
        base_url: str | None = None,
        temperature: float = 0.5,
        max_retries: int = 3,
        ):
        kwargs = {
            "model": model_name,
            "model_provider": model_provider,
            "api_key": api_key,
            "temperature": temperature,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self.model = init_chat_model(**kwargs)

    