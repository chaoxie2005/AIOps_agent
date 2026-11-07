import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from pymongo.collection import Collection

from aiops_agent.graph_components.states.state import ChatTurn


class SessionError(Exception):
    """会话无法使用。"""


class SessionNotFound(SessionError):
    """会话不存在或不属于请求用户。"""


class SessionBusy(SessionError):
    """会话中已有未完成的请求。"""


class SessionStorageUnavailable(SessionError):
    """会话存储未配置。"""


@dataclass(frozen=True)
class _Storage:
    client: MongoClient
    sessions: Collection
    checkpointer: BaseCheckpointSaver


class ChatSessions:
    """保存会话目录，消息正文只读取 LangGraph checkpoint。"""

    def __init__(self) -> None:
        """初始化会话服务，将 MongoDB 连接推迟到首次请求。

        初始化锁避免多个请求同时创建连接；运行中集合记录当前进程
        正在生成回复的会话 ID。
        """
        self._storage: _Storage | None = None
        self._init_lock = asyncio.Lock()
        self._running: set[str] = set()

    @staticmethod
    def _connect() -> _Storage:
        """连接 MongoDB，并准备会话目录和 checkpoint 存储。

        会话目录只存 ID、所属用户和创建时间，所以空会话也能列出来。
        对话内容由 LangGraph 写入 checkpoint，不在目录中重复保存。

        Returns:
            MongoDB 客户端、会话目录集合及 LangGraph checkpointer。

        Raises:
            SessionStorageUnavailable: 未配置 MONGO_URI。
            PyMongoError: MongoDB 连接失败或无法创建索引。
        """
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        uri = os.getenv("MONGO_URI", "").strip()
        if not uri:
            raise SessionStorageUnavailable("会话存储未配置")
        database = os.getenv("MONGO_DB_NAME", "").strip() or "aiops_agent"
        client = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=3000)
        try:
            sessions = client[database]["chat_sessions"]
            sessions.create_index([("user_id", 1), ("created_at", -1), ("_id", -1)])
            checkpointer = MongoDBSaver(
                client,
                db_name=database,
                checkpoint_collection_name="chat_checkpoints",
                writes_collection_name="chat_checkpoint_writes",
            )
            return _Storage(client, sessions, checkpointer)
        except BaseException:
            client.close()
            raise

    async def _get_storage(self) -> _Storage:
        """首次调用时连接 MongoDB，后续请求复用同一套存储依赖。

        Returns:
            已初始化的会话目录和 checkpoint 存储。
        """
        # 首次调用会创建索引，放到线程中以免阻塞 FastAPI 事件循环。
        async with self._init_lock:
            if self._storage is None:
                self._storage = await asyncio.to_thread(self._connect)
            return self._storage

    async def aclose(self) -> None:
        """应用关闭时释放 MongoDB 连接；尚未连接时直接返回。"""
        if self._storage is not None:
            await asyncio.to_thread(self._storage.client.close)
            self._storage = None

    @staticmethod
    def _summary(row: dict) -> dict:
        """将 MongoDB 的会话目录记录转换成接口字段。

        Args:
            row: 含 MongoDB 主键、用户 ID 和创建时间的记录。

        Returns:
            将 MongoDB 的 _id 映射为 session_id 后的会话摘要。
        """
        return {
            "session_id": row["_id"],
            "user_id": row["user_id"],
            "created_at": row["created_at"],
        }

    async def create(self, user_id: str) -> dict:
        """创建空会话并分配会话 ID；首次运行聊天图时才会生成 checkpoint。

        Args:
            user_id: 新会话所属的用户 ID。

        Returns:
            新会话的 ID、所属用户和创建时间。
        """
        storage = await self._get_storage()
        row = {"_id": uuid4().hex, "user_id": user_id, "created_at": datetime.now(timezone.utc)}
        await asyncio.to_thread(storage.sessions.insert_one, row)
        return self._summary(row)

    async def list_for_user(self, user_id: str) -> list[dict]:
        """读取指定用户的会话目录，无须读取对话内容。

        Args:
            user_id: 要查询的用户 ID。

        Returns:
            按创建时间倒序排列的会话摘要，包含尚未发送消息的会话。
        """
        storage = await self._get_storage()

        def read() -> list[dict]:
            """在线程中执行同步 MongoDB 查询，避免阻塞事件循环。

            Returns:
                当前用户的会话摘要列表。
            """
            rows = storage.sessions.find({"user_id": user_id}).sort([("created_at", -1), ("_id", -1)])
            return [self._summary(row) for row in rows]

        return await asyncio.to_thread(read)

    async def _require(self, session_id: str, user_id: str) -> _Storage:
        """核对会话存在且目录中记录的用户 ID 与请求一致。

        Args:
            session_id: 要访问的会话 ID。
            user_id: 请求提供的用户 ID。

        Returns:
            校验通过后可使用的存储依赖。

        Raises:
            SessionNotFound: 会话不存在，或记录的用户 ID 不匹配。
        """
        storage = await self._get_storage()
        row = await asyncio.to_thread(
            storage.sessions.find_one, {"_id": session_id, "user_id": user_id}, {"_id": 1}
        )
        if row is None:
            raise SessionNotFound("会话不存在")
        return storage

    async def read_turns(self, session_id: str, user_id: str) -> list[ChatTurn]:
        """读取当前会话最近一次 checkpoint 中已完成的全部问答。

        会话 ID 用作 LangGraph 的 thread_id，避免从其他会话读到历史。

        Args:
            session_id: 要读取历史的会话 ID。
            user_id: 用于核对会话归属的用户 ID。

        Returns:
            按对话顺序排列的已完成轮次；空会话返回空列表。

        Raises:
            SessionNotFound: 会话不存在，或记录的用户 ID 不匹配。
        """
        storage = await self._require(session_id, user_id)
        checkpoint = await storage.checkpointer.aget_tuple(
            {"configurable": {"thread_id": session_id}}
        )
        if checkpoint is None:
            return []
        return checkpoint.checkpoint["channel_values"].get("turns", [])

    @asynccontextmanager
    async def claim(
        self, session_id: str, user_id: str
    ) -> AsyncIterator[BaseCheckpointSaver]:
        """占用会话，保证同一进程内一次只有一轮请求写它的 checkpoint。

        用 async with 包裹整轮聊天，正常结束或异常退出时都会解除占用。
        对流式请求，这个范围覆盖整段响应流。

        Args:
            session_id: 本轮聊天要使用的会话 ID。
            user_id: 用于核对会话归属的用户 ID。

        Yields:
            本轮 LangGraph 执行使用的 checkpointer。

        Raises:
            SessionNotFound: 会话不存在，或记录的用户 ID 不匹配。
            SessionBusy: 同一进程中已有请求正在使用该会话。
        """
        storage = await self._require(session_id, user_id)
        if session_id in self._running:
            raise SessionBusy("该会话正在生成回复")
        self._running.add(session_id)
        try:
            yield storage.checkpointer
        finally:
            self._running.remove(session_id)
