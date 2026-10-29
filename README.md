# AIops Agent

## RAG 组件

RAG 组件迁自 `chaoxie2005/camel-agent` 的 `develop` 分支，固定提交
`ccddf400dca13da3a8f90d85dd12b0ab04af6c6a`。`src/aiops_agent/rag_components`
保留源仓库的目录、文件名和公开类，内部包名改为 `aiops_agent`，检索工具使用
LangChain `StructuredTool`，无需安装 CAMEL。

### 安装与配置

在项目根目录执行 `uv sync`。在 `.env` 中按实际服务填写以下配置：

```dotenv
MODEL_EMBEDDING_NAME=your-embedding-model
BASE_URL=https://your-model-service/v1
MODEL_API_KEY=your-api-key
RERANK_MODEL_NAME=your-rerank-model
# 可选：覆盖重排完整 endpoint，默认 BASE_URL/rerank
# RERANK_BASE_URL=https://your-model-service/v1/rerank
# MinerU 解析时需要；普通 TXT/Markdown/PDF 本地解析不需要
# MINERU_API_KEY=your-mineru-api-key
# MINERU_URL=https://mineru.net/api/v4
# MINERU_API_TIMEOUT=30
# MINERU_TRANSFER_TIMEOUT=300
```

`MODEL_NAME` 是 `MODEL_EMBEDDING_NAME` 的兼容别名，新变量优先。
调用配置工厂前应执行 `load_dotenv()`。模型服务需要提供兼容 OpenAI 的
embedding 接口及受支持的 rerank 接口。MinIO 的 endpoint、access_key、secret_key、
secure 通过 `MinioClient` 构造参数传入。

### 解析、切分与入库

`Parser` 支持 `.txt`、`.md` 和 `.pdf`，`MineruClient` 提供云端解析接口。
`MarkdownChunkSplitter` 和 `ParentChildChunker` 返回包含 `parent` 和 `children`
的字典列表；普通 `ParentChildChunker` 要求源文档 metadata 含 `source` 和零基 `page`。
以下示例保存为项目根目录中的本地脚本后，可用 `uv run python <脚本名>` 执行：

```python
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from aiops_agent.rag_components.parser import Parser
from aiops_agent.rag_components.chunk import MarkdownChunkSplitter
from aiops_agent.rag_components.milvus import AMilvusClient
from aiops_agent.rag_components.retriever.factory import build_embedding_from_env

async def main():
    load_dotenv()
    embedding = build_embedding_from_env()
    chunker = MarkdownChunkSplitter(embedding)
    groups = [
        group
        for document in Parser("knowledge.md").parse()
        for group in chunker.split_text(document)
    ]
    children = []
    for group in groups:
        for child in group["children"]:
            child.metadata["parent_content"] = group["parent"].page_content
            children.append(child)
    if not children:
        return
    vectors = await embedding.aembed_documents([doc.page_content for doc in children])
    rows = [
        {
            "child_chunk_id": doc.metadata["child_id"],
            "parent_chunk_id": doc.metadata["parent_id"],
            "language": doc.metadata.get("language"),
            "child_content": doc.page_content,
            "child_metadata": doc.metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file_type": Path(doc.metadata["file_name"]).suffix.lstrip(".") or None,
            "child_minio_url": doc.metadata.get("source_url"),
            "dense_vector": vector,
        }
        for doc, vector in zip(children, vectors)
    ]
    client = AMilvusClient(url="http://localhost:19530")
    try:
        if not await client.has_collection("aiops_agent"):
            await client.create_collection("aiops_agent", dim=len(vectors[0]))
        await client.insert_entity("aiops_agent", rows)
    finally:
        await client.close()

asyncio.run(main())
```

可先通过 `MinioClient.download_files(bucket_name, download_dir)` 下载对象，
再把得到的路径交给 `Parser`。如需来源链接，在切分前将
`MinioClient.get_minio_url(bucket_name, object_name)` 写入文档的 `source_url` 元数据。

### 检索与 LangChain 工具

```python
import asyncio
from dotenv import load_dotenv
from aiops_agent.rag_components.retriever import build_retriever_from_env
from aiops_agent.agent_components.toolkits import make_rag_search_tool

async def main():
    load_dotenv()
    retriever = build_retriever_from_env(
        milvus_url="http://localhost:19530",
        collection_name="aiops_agent",
        score_threshold=0.3,
    )
    try:
        docs = await retriever.ainvoke("如何排查服务故障？")
        tool = make_rag_search_tool(retriever)
        answer = await tool.ainvoke({"query": "如何排查服务故障？"})
        print(docs, answer)
        # 可将 tools=[tool] 传入 LangChain create_agent。
        # 同步调用使用 retriever.invoke(query) 或 tool.invoke({"query": query})。
    finally:
        await retriever.client.close()

asyncio.run(main())
```

默认检索流程为稠密向量 + BM25 双路召回、RRF 融合、可选重排和阈值过滤。
返回 LangChain `Document`，分数在 `metadata["score"]` 中；父块内容仅在入库时
写入 `parent_content` 后保留，检索返回的是子块。

默认集合名已改为 `aiops_agent`；使用旧数据时显式指定 `collection_name="camel_agent"`。
沿用旧集合前请核对 embedding 模型、向量维度、字段和分析器配置。
Milvus 服务端须支持本 schema 使用的 BM25、多语言分析器和 TIMESTAMPTZ；
仅 SDK 导入成功不代表服务端支持所有特性。此迁移不包含源仓库的部署配置和入库脚本。

### 离线验证

```bash
uv run pytest src/aiops_agent/tests/test_rag_components.py src/aiops_agent/tests/test_load_prompt.py -q
```

回归测试使用替身覆盖同步/异步检索、重排阈值、连接释放、配置校验、父子块关联、
文本解析、SDK schema 构造及 LangChain 工具调用，无需真实数据库或模型凭证。
