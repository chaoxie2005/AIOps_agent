import asyncio
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from aiops_agent.rag_components.ingestion import DocumentIngestor, EmptyDocumentError


class MemoryMinio:
    def __init__(self):
        self.manifests = {}
        self.objects = {}
        self.uploads = []

    def get_minio_url(self, bucket, name):
        return f"https://minio.example/{bucket}/{name}"

    def read_json(self, bucket, name):
        return deepcopy(self.manifests.get((bucket, name)))

    def write_json(self, bucket, name, data):
        self.manifests[bucket, name] = deepcopy(data)

    def upload_file(self, bucket, path, name):
        self.uploads.append(name)
        self.objects[bucket, name] = Path(path).read_bytes()

    def publish_object(self, bucket, stage, name):
        self.objects[bucket, name] = self.objects[bucket, stage]

    def remove_object(self, bucket, name):
        self.objects.pop((bucket, name), None)


class MemoryMilvus:
    def __init__(self):
        self.rows = defaultdict(dict)
        self.queries = 0
        self.writes = []
        self.deletes = []
        self.fail_delete = False

    async def query_ingestion_rows(self, collection, document_id, legacy_url):
        self.queries += 1
        return deepcopy([
            row for row in self.rows[collection].values()
            if row["child_metadata"].get("document_id") == document_id
            or row.get("child_minio_url") == legacy_url
        ])

    async def update_entity(self, *, collection_name, data):
        self.writes.append(deepcopy(data))
        for row in data:
            self.rows[collection_name][row["child_chunk_id"]] = deepcopy(row)

    async def delete_ingestion_rows(self, collection, ids):
        if self.fail_delete:
            raise OSError("delete interrupted")
        self.deletes.extend(ids)
        for key in ids:
            self.rows[collection].pop(key)


def _chunks(*sections):
    return [
        {
            "parent": Document(page_content=parent),
            "children": [
                Document(page_content=child, metadata={"heading": parent})
                for child in children
            ],
        }
        for parent, children in sections
    ]


@pytest.fixture
def storage(tmp_path):
    minio, milvus = MemoryMinio(), MemoryMilvus()

    async def embed(texts):
        return [[float(len(text))] for text in texts]

    embed_mock = AsyncMock(side_effect=embed)
    ingestor = DocumentIngestor(minio, milvus, embed_mock, embedding_id="embedding-v1")

    async def submit(body, chunks=None, *, loader=None, name="guide.md", pipeline="v1"):
        local = tmp_path / name
        local.write_bytes(body)
        return await ingestor.ingest(
            collection="kb", bucket="docs", file_name=name, file_path=local,
            file_sha256=sha256(body).hexdigest(), pipeline_id=pipeline,
            load_chunks=loader if loader is not None else AsyncMock(return_value=chunks),
        )

    return minio, milvus, embed_mock, submit


def test_unchanged_upload_skips_storage_parse_embedding_and_vector_writes(storage):
    minio, milvus, embed, submit = storage
    parse = AsyncMock(return_value=_chunks(("Section", ["Alpha"])))

    async def run():
        first = await submit(b"version one", loader=parse)
        repeat = await submit(b"version one", loader=parse)
        assert first.status == "created" and repeat.status == "unchanged"
        assert repeat.object_name == first.object_name
        assert repeat.url == first.url
        assert (repeat.parent_count, repeat.child_count) == (1, 1)
        assert minio.objects == {("docs", first.object_name): b"version one"}
        assert "/staging/" in parse.await_args.args[0]

    asyncio.run(run())
    assert len(minio.uploads) == milvus.queries == len(milvus.writes) == 1
    parse.assert_awaited_once()
    embed.assert_awaited_once_with(["Alpha"])


def test_changed_document_embeds_and_writes_only_changed_chunks(storage):
    minio, milvus, embed, submit = storage

    async def run():
        first = await submit(b"v1", _chunks(("Stable", ["A"]), ("Old", ["B", "C"])))
        old = {row["child_content"]: row for row in milvus.rows["kb"].values()}
        update = await submit(b"v2", _chunks(("Stable", ["A"]), ("New", ["B new", "D"])))
        rows = {row["child_content"]: row for row in milvus.rows["kb"].values()}
        assert update.status == "updated" and update.url == first.url
        assert rows.keys() == {"A", "B new", "D"}
        assert rows["A"] == old["A"]
        assert set(milvus.deletes) == {old["B"]["child_chunk_id"], old["C"]["child_chunk_id"]}
        assert [row["child_content"] for row in milvus.writes[-1]] == ["B new", "D"]
        assert {row["child_minio_url"] for row in rows.values()} == {first.url}
        assert minio.objects == {("docs", first.object_name): b"v2"}

    asyncio.run(run())
    embed.assert_awaited_with(["B new", "D"])


def test_moved_and_repeated_text_keeps_occurrences_and_reuses_embeddings(storage):
    _, milvus, embed, submit = storage

    async def run():
        await submit(b"v1", _chunks(("Original", ["Repeat"])))
        original = next(iter(milvus.rows["kb"].values()))
        await submit(b"v2", _chunks(("Moved", ["Repeat", "Repeat"])))
        rows = list(milvus.rows["kb"].values())
        assert len(rows) == 2 and rows[0]["child_chunk_id"] != rows[1]["child_chunk_id"]
        assert rows[0]["child_chunk_id"] == original["child_chunk_id"]
        assert rows[0]["parent_chunk_id"] != original["parent_chunk_id"]
        assert {tuple(row["dense_vector"]) for row in rows} == {tuple(original["dense_vector"])}

    asyncio.run(run())
    embed.assert_awaited_once_with(["Repeat"])


def test_legacy_cleanup_is_limited_to_the_matching_url_and_unowned_rows(storage):
    minio, milvus, _, submit = storage
    url = minio.get_minio_url("docs", "guide.md")
    milvus.rows["kb"].update({
        key: {"child_chunk_id": key, "child_metadata": metadata, "child_minio_url": source}
        for key, metadata, source in [
            ("legacy-a", {}, url),
            ("legacy-b", {}, url),
            ("unknown", {}, None),
            ("other-url", {}, url + ".old"),
            ("owned-by-other-doc", {"document_id": "other"}, url),
        ]
    })
    milvus.rows["other"]["legacy-a"] = deepcopy(milvus.rows["kb"]["legacy-a"])

    result = asyncio.run(submit(b"document", _chunks(("Section", ["Text"]))))
    assert result.status == "updated"
    assert milvus.deletes == ["legacy-a", "legacy-b"]
    assert {"unknown", "other-url", "owned-by-other-doc"} <= milvus.rows["kb"].keys()
    assert "legacy-a" in milvus.rows["other"]


@pytest.mark.parametrize("retry_version", ["old", "new"])
def test_partial_write_reconciles_on_retry_even_for_previous_bytes(storage, retry_version):
    minio, milvus, embed, submit = storage
    versions = {
        "old": (b"v1", _chunks(("Section", ["Original"]))),
        "new": (b"v2", _chunks(("Section", ["Changed"]))),
    }

    async def run():
        first = await submit(*versions["old"])
        milvus.fail_delete = True
        with pytest.raises(OSError, match="delete interrupted"):
            await submit(*versions["new"])
        assert {row["child_content"] for row in milvus.rows["kb"].values()} == {"Original", "Changed"}
        assert minio.objects == {("docs", first.object_name): b"v1"}

        milvus.fail_delete = False
        result = await submit(*versions[retry_version])
        assert result.status == "updated"
        assert minio.objects == {("docs", first.object_name): versions[retry_version][0]}
        assert {row["child_content"] for row in milvus.rows["kb"].values()} == {
            "Original" if retry_version == "old" else "Changed"
        }
        assert (await submit(*versions[retry_version])).status == "unchanged"

    asyncio.run(run())
    assert embed.await_count == 2


def test_parse_failure_and_empty_file_leave_previous_version_unchanged(storage):
    minio, milvus, _, submit = storage

    async def run():
        first = await submit(b"v1", _chunks(("Section", ["Original"])))
        with pytest.raises(RuntimeError, match="parse failed"):
            await submit(b"invalid", loader=AsyncMock(side_effect=RuntimeError("parse failed")))
        with pytest.raises(EmptyDocumentError):
            await submit(b"", [])
        assert minio.objects == {("docs", first.object_name): b"v1"}
        assert len(milvus.writes) == 1 and not milvus.deletes
        assert (await submit(b"v1", [])).status == "unchanged"

    asyncio.run(run())


def test_concurrent_uploads_of_the_same_document_only_write_once(storage):
    minio, milvus, embed, submit = storage

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def parse(_url):
            entered.set()
            await release.wait()
            return _chunks(("Section", ["Text"]))

        first = asyncio.create_task(submit(b"same", loader=parse))
        await entered.wait()
        second = asyncio.create_task(submit(b"same", loader=parse))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second)
        assert [result.status for result in results] == ["created", "unchanged"]

    asyncio.run(run())
    assert len(minio.uploads) == len(milvus.writes) == embed.await_count == 1
