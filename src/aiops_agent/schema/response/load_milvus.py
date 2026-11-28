from typing import Literal

from pydantic import BaseModel, Field


class LoadMilvusResponse(BaseModel):
    file_name: str = Field(description="用户上传时的原始文件名。")
    object_name: str = Field(description="原文件在 MinIO 桶中的对象路径。")
    url: str = Field(description="原文件在 MinIO 中的访问地址。")
    parent_count: int = Field(description="当前文档版本的父块总数。")
    child_count: int = Field(description="当前文档版本的子块总数，而非本次新增数。")
    status: Literal["created", "updated", "unchanged"] = Field(
        description="入库结果：created 表示首次入库，updated 表示已更新，"
        "unchanged 表示文件和处理配置均未变化。"
    )
