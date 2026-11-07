import os
import tempfile
from logging import getLogger

from fastapi import APIRouter, Form, UploadFile

from aiops_agent.rag_components.minio.minio import MinioClient
from aiops_agent.schema.response.upload_file import UploadFileResponse

router = APIRouter()
logger = getLogger(__name__)

minio_client = MinioClient(
    endpoint=os.environ.get("MINIO_ENDPOINT", "127.0.0.1:9000"),
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
)


@router.post("/upload", response_model=UploadFileResponse)
async def upload_file(
    file: UploadFile,
    bucket_name: str = Form(default="aiops"),
) -> UploadFileResponse:
    """将上传的文件存入 MinIO，返回对象名称和访问 URL。

    Args:
        file: 上传的文件。
        bucket_name: 目标存储桶名称，默认为 aiops。

    Returns:
        包含对象名称和 MinIO 访问 URL 的响应。
    """
    object_name = file.filename or "unknown"

    suffix = os.path.splitext(object_name)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        minio_client.upload_file(
            bucket_name=bucket_name,
            file_path=tmp.name,
            object_name=object_name,
        )
    except Exception:
        logger.exception("文件上传 MinIO 失败: %s", object_name)
        raise
    finally:
        os.unlink(tmp.name)

    url = minio_client.get_minio_url(bucket_name=bucket_name, object_name=object_name)
    return UploadFileResponse(object_name=object_name, url=url)
