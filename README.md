# AIops Agent

基于 FastAPI 和 LangGraph 的运维助手。支持：

- 会话聊天、查看历史
- 知识库问答（RAG）
- Markdown / PDF 增量入库
- Prometheus 指标排查

## 快速开始

### 1. 环境要求

Python 3.11–3.13、`uv`。需要运行 MongoDB 和 Milvus；聊天/排障还需要对话模型、Embedding 和 Rerank 接口。

### 2. 配置 `.env`

把配置写到 `src/aiops_agent/.env`，按实际部署修改地址和凭据：

```dotenv
# MongoDB 和对话模型
MONGO_URI=mongodb://127.0.0.1:27018
MONGO_DB_NAME=aiops_agent
LLM_BASE_URL=https://your-chat-service.example/v1
LLM_MODEL_NAME=your-chat-model
LLM_API_KEY=your-chat-api-key

# 知识库（上传和聊天必须指向同一个集合）
MILVUS_URL=http://127.0.0.1:19530
MILVUS_COLLECTION_NAME=camel_agent
BASE_URL=https://your-rag-model-service.example/v1
MODEL_EMBEDDING_NAME=your-embedding-model
MODEL_API_KEY=your-rag-api-key
RERANK_MODEL_NAME=your-rerank-model

# 上传文件到 MinIO 时填写
# MINIO_ENDPOINT=127.0.0.1:9000
# MINIO_ACCESS_KEY=your-minio-access-key
# MINIO_SECRET_KEY=your-minio-secret-key

# 通过接口上传 PDF 时还需要 MinerU
# MINERU_API_KEY=your-mineru-api-key

# 排障步骤查询指标时填写
# PROMETHEUS_URL=http://127.0.0.1:9090
```

容器或 WSL 环境里，地址要写 **API 进程能访问到** 的地址。

### 3. 启动服务

在根目录执行：

```bash
uv sync
uv run uvicorn aiops_agent.api.app:app --host 127.0.0.1 --port 8000 --reload
```

打开 http://127.0.0.1:8000/docs 查看接口文档，或执行：

```bash
curl -fsS http://127.0.0.1:8000/health
```

### 4. 创建会话并提问

```bash
SESSION_ID=$(curl -fsS http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"local-test"}' | python3 -c 'import json, sys; print(json.load(sys.stdin)["session_id"])')

curl -fsS http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"local-test\",\"session_id\":\"$SESSION_ID\",\"query\":\"你好，请介绍一下 Prometheus\"}"
```

主要接口：

| 接口 | 用途 |
| --- | --- |
| `POST /sessions` | 创建会话 |
| `GET /sessions?user_id=local-test` | 查看用户会话 |
| `GET /sessions/{session_id}/messages?user_id=local-test` | 查看会话历史 |
| `POST /chat` | 获取完整回答 |
| `POST /chat/stream` | 以 SSE 流式返回进度和结果 |
| `POST /load_milvus` | 上传 Markdown/PDF 入库 |

### 5. 验证知识库

通过 `POST /load_milvus` 上传 Markdown，提问时带上文档里的真实术语。例如：

> 请查阅知识库里的《告警处理手册》，说明 HighCPUUsage 告警触发后应先检查哪些指标。

上传和聊天必须使用同一个 Milvus 集合。若 `intent` 阶段报 `OpenAIConnectionError`，先检查 `LLM_BASE_URL`，此时还没有进入 RAG。

## 项目流程

### 上下文压缩

每轮请求进入主图后，先检查历史会话会不会超过模型的输入预算。流程：

```
历史不超预算 → 直接使用近期历史
历史超预算   → 拆分历史 → 保留必须原文的内容 → 其余内容分批摘要 → 再次检查预算
                                                              ├─ 放得下 → 用摘要
                                                              └─ 放不下/失败 → 沿用近期历史
```

必须保留原文的内容包括用户原话、工具调用字段和完整堆栈，避免压缩后丢掉关键线索。其余历史会由模型分批摘要，并要求带来源编号；摘要太长或格式不对时会收紧重试。压缩失败不会中断请求，系统会自动回退到近期历史。

### 聊天与排障

各节点使用独立的模型温度：意图识别、子问题拆解、依赖改写和历史摘要为 `0.0`；规划、重规划和执行为 `0.1`；普通聊天为 `0.5`。配置集中在 `src/aiops_agent/agent_components/llm.py` 的 `NODE_TEMPERATURES`，可根据模型与评测结果调整。

1. 上下文检查完成后，`intent` 判断是普通聊天还是排障；聊天分支直接生成回复，排障分支进入规划。
2. 排障时 `plan` 生成步骤，`execute` 调用日志、指标、RAG 等工具收集证据。
3. `replan` 判断继续排查还是结束；结束时给出结构化诊断结果。
4. 会话历史、checkpoint 和执行过程会保存，支持流式返回和后续查看。

### RAG 检索

RAG 不是一次性把所有子问题扔去检索，而是按顺序处理：

```
拆解问题 → 校验依赖 → 调度一个子问题
                      ├─ 无依赖 → 直接检索
                      └─ 有依赖 → 用上游检索结果改写问题 → 检索
                                      ↓
                          记录完成 → 处理下一个 → 全部完成后去重聚合
```

拆解时，依赖问题会用 `#q1#` 指代上游答案。系统先检查引用是否存在、有没有环、占位符是否和依赖一致；不合法就清空依赖、删掉整个占位符后并行检索。SDOM 改写失败时也用同样的兜底方式，保证不会卡住。

## 测试

```bash
uv run pytest -q \
  src/aiops_agent/tests \
  --ignore-glob='*_manual.py' \
  --ignore=src/aiops_agent/tests/test_rag_subgraph.py \
  --ignore=src/aiops_agent/tests/test_planer_node.py
```

离线测试用替身代替真实模型和数据库。文件名带 `_manual` 的测试、`test_rag_subgraph.py` 和 `test_planer_node.py` 需要真实模型或 Milvus，按部署环境决定是否运行。
