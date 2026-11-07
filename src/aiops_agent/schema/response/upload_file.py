from pydantic import BaseModel


class UploadFileResponse(BaseModel):
    object_name: str
    url: str
