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
        max_tokens: int | None = None,
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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        kwargs["max_retries"] = max_retries
        self.model = init_chat_model(**kwargs)



    def invoke(self, messages: list):
        """同步调用模型"""
        return self.model.invoke(messages)

    async def ainvoke(self, messages: list):
        """异步调用模型"""
        return await self.model.ainvoke(messages)
    
    
    