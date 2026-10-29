from pathlib import Path
from urllib.parse import quote

from minio import Minio

class MinioClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.endpoint = endpoint
        self.secure = secure
        self.client = Minio(
            self.endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def upload_file(self, bucket_name: str, file_path: str, object_name: str):
        """
        上传文件到 MinIO 存储桶

        Args:
            bucket_name: 存储桶名称
            file_path: 文件路径
            object_name: 存储桶中的对象名称

        Returns:
            ObjectWriteResult: MinIO 文件上传结果
        Raises:
            MinioError: MinIO 错误
        """

        result = self.client.fput_object(bucket_name, object_name, file_path)
        print(f"文件 {file_path} 上传到 {bucket_name}/{object_name}")
        return result

    def upload_files(self, bucket_name: str, upload_dir: Path) -> list[str]:
        """
        批量上传本地目录中的所有文件到存储桶

        Args:
            bucket_name: 存储桶名称
            upload_dir: 本地待上传目录

        Returns:
            list[str]: 上传后的对象名称列表

        Raises:
            FileNotFoundError: 本地目录不存在或不是目录时抛出
        """

        if not upload_dir.is_dir():
            raise FileNotFoundError(f"本地目录不存在: {upload_dir}")

        uploaded_objects: list[str] = []

        for file_path in upload_dir.rglob("*"):
            if not file_path.is_file():
                continue

            object_name = file_path.relative_to(upload_dir).as_posix()

            self.upload_file(
                bucket_name=bucket_name,
                file_path=str(file_path),
                object_name=object_name,
            )

            uploaded_objects.append(object_name)

        return uploaded_objects

    def download_file(self, bucket_name: str, object_name: str, file_path: Path):
        """
        从 MinIO 存储桶下载文件

        Args:
            bucket_name: 存储桶名称
            object_name: 存储桶中的对象名称
            file_path: 下载文件到本地的路径

        Returns:
            Path: 下载完成后的本地文件路径
        Raises:
            MinioError: MinIO 操作失败时抛出的异常
        """

        result = self.client.fget_object(bucket_name, object_name, str(file_path))
        print(f"文件 {object_name} 从 {bucket_name} 下载到 {file_path}")
        return file_path

    def download_files(self,bucket_name: str,download_dir: Path) -> list[Path]:
        """
        批量下载存储桶中的所有文件

        Args:
            bucket_name: 存储桶名称
            download_dir: 本地下载目录

        Returns:
            list[Path]: 下载后的本地文件路径列表
        """

        download_dir.mkdir(parents=True, exist_ok=True)

        objects = self.client.list_objects(
            bucket_name,
            recursive=True,
        )

        downloaded_files: list[Path] = []

        for obj in objects:
            object_name = obj.object_name

            if object_name is None:
                continue

            file_path = download_dir / object_name

            # 创建文件的父目录结构
            file_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            self.download_file(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
            )

            downloaded_files.append(file_path)

        return downloaded_files

    
    def get_minio_url(self, bucket_name: str, object_name: str) -> str:
        """
        获取 MinIO 存储桶中的对象 URL

        Args:
            bucket_name: 存储桶名称
            object_name: 存储桶中的对象名称

        Returns:
            str: MinIO 对象 URL，如 http://127.0.0.1:9000/bucket/object.md
        """
        scheme = "https" if self.secure else "http"
        encoded_object_name = quote(object_name, safe="/")
        return f"{scheme}://{self.endpoint}/{bucket_name}/{encoded_object_name}"
