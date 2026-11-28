import asyncio

import requests


class RerankModel:
    """
    重排模型
    """

    def __init__(
        self, model_name: str, base_url: str, key: str, endpoint: str | None = None
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.endpoint = endpoint.rstrip("/") if endpoint else None

    async def arerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """在线程中执行同步 HTTP 请求，避免阻塞 Agent 事件循环。"""
        return await asyncio.to_thread(self.rerank, query, documents, top_n)

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """
        对候选文档按与查询的相关性重排

        Args:
            query: 查询文本
            documents: 候选文档文本列表
            top_n: 返回条数上限

        Returns:
            [(文档原始下标, 相关性分数)]，按分数降序
        """
        if not documents:
            return []
        url = self.endpoint or f"{self.base_url}/rerank"
        if "/services/rerank/" in url:
            body = {
                "model": self.model_name,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": top_n},
            }
        else:
            body = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        # 兼容两种响应格式:
        #   A: {"results": [{"index": 0, "relevance_score": 0.9}]}
        #   B: {"output": {"results": [{"text": "...", "relevance_score": 0.9}]}}
        results = payload.get("results")
        if results is None:
            results = (payload.get("output") or {}).get("results") or []
        ranked: list[tuple[int, float]] = []
        used: set[int] = set()
        for item in results:
            if "index" in item:
                index = int(item["index"])
            else:
                text = item.get("text", "")
                index = next(
                    (
                        i
                        for i, doc in enumerate(documents)
                        if i not in used and doc == text
                    ),
                    None,
                )
                if index is None:
                    continue
            used.add(index)
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            ranked.append((index, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
