from typing import Annotated

from fastapi import File, Form, UploadFile


class LoadMilvusRequest:
    """向量入库接口的 multipart 表单。"""

    def __init__(
        self,
        file: Annotated[UploadFile, File(description="要上传的 Markdown 或 PDF 文件")],
        collection_name: Annotated[
            str | None, Form(description="目标 Milvus 集合，留空时使用服务配置")
        ] = None,
        bucket_name: Annotated[str, Form(description="目标 MinIO 存储桶")] = "aiops",
    ) -> None:
        self.file = file
        self.collection_name = collection_name
        self.bucket_name = bucket_name
