# AIops Agent

基于 FastAPI 和 LangGraph 的运维助手，支持会话聊天、知识库问答、文件增量入库和指标排查。

## 本地启动

### 依赖服务

| 使用场景 | 服务 |
| --- | --- |
| 创建会话、查看历史 | AIops Agent API、MongoDB |
| 聊天、知识库问答和排障 | MongoDB、OpenAI 兼容对话模型、Milvus、Embedding 和 Rerank 接口 |
| 上传 Markdown 入库 | MinIO、Milvus、Embedding 接口 |
| 通过上传接口处理 PDF | MinIO、Milvus、Embedding 接口、MinerU |
| 在排障中查询当前指标 | Prometheus |

普通聊天也会初始化知识库检索器，因此要配好 Embedding、Rerank 和 Milvus。
对话模型、Embedding 和 Rerank 可以接入已有服务地址。

### 环境变量

使用 Python 3.11–3.13 和 `uv`。将配置放在 `src/aiops_agent/.env`，按实际部署替换地址、模型名和凭据。

```dotenv
# MongoDB 和对话模型
MONGO_URI=mongodb://127.0.0.1:27018
MONGO_DB_NAME=aiops_agent
LLM_BASE_URL=https://your-chat-service.example/v1
LLM_MODEL_NAME=your-chat-model
LLM_API_KEY=your-chat-api-key
# 单次调用生成的 Token 上限
LLM_MAX_TOKENS=2048

# 知识库检索和入库使用相同的 Milvus 集合与 Embedding 模型
MILVUS_URL=http://127.0.0.1:19530
MILVUS_COLLECTION_NAME=camel_agent
BASE_URL=https://your-rag-model-service.example/v1
MODEL_EMBEDDING_NAME=your-embedding-model
MODEL_API_KEY=your-rag-api-key
RERANK_MODEL_NAME=your-rerank-model
# Rerank 默认使用 BASE_URL/rerank，可指定完整的重排地址
# RERANK_BASE_URL=https://your-rag-model-service.example/v1/rerank

# 上传 Markdown 或 PDF 时使用
# MINIO_ENDPOINT=127.0.0.1:9000
# MINIO_ACCESS_KEY=your-minio-access-key
# MINIO_SECRET_KEY=your-minio-secret-key

# 上传 PDF 时使用，MinerU 还需要能够访问该 PDF 的 MinIO URL
# MINERU_API_KEY=your-mineru-api-key
# MINERU_URL=https://mineru.net/api/v4

# 排障执行步骤需要查询当前指标时使用
# PROMETHEUS_URL=http://127.0.0.1:9090
```

如果使用容器或 WSL，服务地址和端口以 **API 进程所在环境可以访问的地址** 为准。

### 启动 API

在项目根目录运行：

```bash
uv sync
uv run uvicorn aiops_agent.api.app:app --host 127.0.0.1 --port 8000 --reload
```

打开 [Swagger API 文档](http://127.0.0.1:8000/docs)，或先检查 API 是否已启动。

```bash
curl -fsS http://127.0.0.1:8000/health
```

### 创建会话并发消息

首次调用会话接口时，应用会在 `MONGO_DB_NAME` 指定的数据库中初始化索引。缺失的 `chat_sessions`、`chat_checkpoints` 和 `chat_checkpoint_writes` 会随索引自动创建。MongoDB 账号需要有读写和创建索引的权限。

下面的命令创建一个会话，再用返回的 `session_id` 发送问题。后续对话使用相同的 `user_id` 和 `session_id`。

```bash
SESSION_ID=$(curl -fsS http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"local-test"}' | python3 -c 'import json, sys; print(json.load(sys.stdin)["session_id"])')

curl -fsS http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"local-test\",\"session_id\":\"$SESSION_ID\",\"query\":\"你好，请介绍一下 Prometheus\"}"
```

| 接口 | 用途 |
| --- | --- |
| `POST /sessions` | 创建会话，返回 `session_id` |
| `GET /sessions?user_id=local-test` | 列出该用户的会话 |
| `GET /sessions/{session_id}/messages?user_id=local-test` | 读取已完成的会话问答 |
| `POST /chat` | 返回完整回复或诊断结果 |
| `POST /chat/stream` | 使用同样的请求体，以 SSE 返回进度和结果 |
| `POST /load_milvus` | 使用 multipart 表单上传文件入库，字段为 `file` 和可选的 `collection_name`、`bucket_name` |

### 验证知识库检索

先用 Swagger 的 `POST /load_milvus` 上传 Markdown，或使用已入库的知识库。上传和聊天应指向同一个 Milvus 集合。
提问时带上真实的文档名或内部术语，例如「请查阅知识库里的《告警处理手册》，说明 HighCPUUsage 告警触发后应先检查哪些指标」。

如果 `/chat` 在 `intent` 节点报 `OpenAIConnectionError`，先从运行 API 的环境检查 `LLM_BASE_URL` 的地址和端口。此时请求还没进入 RAG 检索。

## RAG 组件

RAG 组件迁自 `chaoxie2005/camel-agent` 的 `develop` 分支，固定提交
`ccddf400dca13da3a8f90d85dd12b0ab04af6c6a`。`src/aiops_agent/rag_components`
保留源仓库的目录、文件名和公开类，内部包名改为 `aiops_agent`，检索工具使用
LangChain `StructuredTool`，无需安装 CAMEL。

### SDK 配置

直接调用 RAG 组件时，沿用上方的 Milvus 和 RAG 模型配置。`MODEL_NAME` 是 `MODEL_EMBEDDING_NAME` 的兼容别名，新变量优先。需要在脚本里主动加载环境变量时，可以用 `load_dotenv("src/aiops_agent/.env")`。

使用 SDK 的 `Parser` 时可本地解析 TXT、Markdown 和 PDF。HTTP 上传接口中的 PDF 走 MinerU 解析流程。

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
from aiops_agent.rag_components.retriever.factory import (
    build_embedding_from_env,
    build_milvus_client_from_env,
    get_milvus_collection_name,
)

async def main():
    load_dotenv("src/aiops_agent/.env")
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
    client = build_milvus_client_from_env()
    collection_name = get_milvus_collection_name()
    try:
        if not await client.has_collection(collection_name):
            await client.create_collection(collection_name, dim=len(vectors[0]))
        await client.insert_entity(collection_name, rows)
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
    load_dotenv("src/aiops_agent/.env")
    retriever = build_retriever_from_env(
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

服务默认检索流程为稠密向量与 BM25 双路召回、RRF 融合、重排和阈值过滤。
返回 LangChain `Document`，分数在 `metadata["score"]` 中；父块内容仅在入库时
写入 `parent_content` 后保留，检索返回的是子块。

默认集合名为 `camel_agent`。可以配置 `MILVUS_COLLECTION_NAME`，也可以在 SDK 调用时显式传入其他集合。
沿用旧集合前请核对 embedding 模型、向量维度、字段和分析器配置。
Milvus 服务端须支持本 schema 使用的 BM25、多语言分析器和 TIMESTAMPTZ；
仅 SDK 导入成功不代表服务端支持所有特性。此迁移不包含源仓库的部署配置和入库脚本。

### 离线验证

```bash
uv run pytest -q \
  src/aiops_agent/tests/test_load_prompt.py \
  src/aiops_agent/tests/test_chat_sessions.py \
  src/aiops_agent/tests/test_chat_routing.py \
  src/aiops_agent/tests/test_incremental_ingestion.py \
  src/aiops_agent/tests/test_upload_chat_integration.py
```

这些测试用替身验证提示词、会话、聊天路由、增量入库和入库后检索，不需要真实数据库或模型凭证。
