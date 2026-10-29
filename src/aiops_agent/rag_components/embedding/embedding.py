from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingModel(Embeddings):
    """
    嵌入模型
    """
    def __init__(self, model_name: str , base_url: str, key: str) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.key = key
        self.embedding = OpenAIEmbeddings(
            model=model_name,
            base_url=self.base_url,
            api_key=self.key,  # type: ignore
            check_embedding_ctx_length=False,
        )

    def embed_query(self, query: str) -> list[float]:  # type: ignore
        """
        对查询进行嵌入
        """
        return self.embedding.embed_query(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        对多段文本进行嵌入
        分批调用，避免超过服务商 batch size 上限
        """
        batch_size = 20
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self.embedding.embed_documents(batch))
        return all_embeddings

    async def aembed_query(self, query: str) -> list[float]:  # type: ignore
        """
        对查询进行异步嵌入
        """
        return await self.embedding.aembed_query(query)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        对多段文本进行异步嵌入（分批，同 embed_documents）
        """
        batch_size = 20
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(await self.embedding.aembed_documents(batch))
        return all_embeddings
