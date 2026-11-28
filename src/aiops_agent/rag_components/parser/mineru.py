import io
import os
import tempfile
import uuid
import zipfile
from pathlib import Path

import dotenv
import requests

from aiops_agent.rag_components.minio import MinioClient

dotenv.load_dotenv()

_DEFAULT_BASE_URL = "https://mineru.net/api/v4"

# 网络超时配置（连接超时, 读取超时），单位秒，可用环境变量覆盖
_API_TIMEOUT = (10, float(os.getenv("MINERU_API_TIMEOUT", "30")))
_TRANSFER_TIMEOUT = (10, float(os.getenv("MINERU_TRANSFER_TIMEOUT", "300")))


def _resolve_base_url() -> str:
    """
    从环境变量 MINERU_URL 解析 API 基础地址

    Returns:
        str: 如 https://mineru.net/api/v4
    """
    url = os.getenv("MINERU_URL") or _DEFAULT_BASE_URL
    for suffix in ("/extract/task/batch", "/extract/task", "/file-urls/batch"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url.rstrip("/")


class MineruClient:
    """
    MinerU 文档解析客户端
    """

    def __init__(self, api_key: str | None = None, model_version: str = "vlm"):
        """
        初始化 MinerU 客户端

        Args:
            api_key: MinerU API Token，默认读取环境变量 MINERU_API_KEY
            model_version: 模型版本，可选 pipeline / vlm / MinerU-HTML

        Raises:
            ValueError: api_key 缺失时抛出
        """
        api_key = api_key or os.getenv("MINERU_API_KEY")
        if not api_key:
            raise ValueError("缺少 MinerU API Key，请设置 MINERU_API_KEY 环境变量或传入 api_key")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.base_url = _resolve_base_url()
        self.model_version = model_version

    def upload_files(self, file_paths: list[str], is_ocr: bool = True) -> str:
        """
        批量上传本地文件并提交解析任务

        流程：申请上传链接 -> 逐个上传文件 -> 系统自动提交解析任务

        Args:
            file_paths: 本地文件路径列表，单次不超过 50 个
            is_ocr: 是否启动 OCR，仅对 pipeline / vlm 模型有效

        Returns:
            str: 批量提取任务 id（batch_id），可用于后续查询解析结果

        Raises:
            FileNotFoundError: 本地文件不存在时抛出
            ValueError: 文件数量超过 50 时抛出
            RuntimeError: 申请上传链接失败或文件上传失败时抛出
        """
        if not file_paths:
            raise ValueError("file_paths 列表不能为空")

        if len(file_paths) > 50:
            raise ValueError(f"单次最多上传 50 个文件，当前 {len(file_paths)} 个")

        for file_path in file_paths:
            if not Path(file_path).is_file():
                raise FileNotFoundError(f"本地文件不存在: {file_path}")

        payload = {
            "files": [
                {
                    "name": Path(file_path).name,
                    "data_id": str(uuid.uuid4()),
                    "is_ocr": is_ocr,
                }
                for file_path in file_paths
            ],
            "model_version": self.model_version,
        }

        response = requests.post(
            f"{self.base_url}/file-urls/batch",
            headers=self.headers,
            json=payload,
            timeout=_API_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"申请上传链接失败: status={response.status_code}, body={response.text}"
            )

        result = response.json()
        print(result)

        if result.get("code") != 0:
            raise RuntimeError(
                f"申请上传链接失败: code={result.get('code')}, msg={result.get('msg')}"
            )

        data = result["data"]
        batch_id = data["batch_id"] # 批量提取任务 id
        file_urls = data["file_urls"]
        if len(file_urls) != len(file_paths):
            raise RuntimeError(
                f"上传链接数量不匹配: 返回 {len(file_urls)} 个，期望 {len(file_paths)} 个"
            )

        for file_path, upload_url in zip(file_paths, file_urls):
            with open(file_path, "rb") as f:
                upload_response = requests.put(upload_url, data=f, timeout=_TRANSFER_TIMEOUT)
            if upload_response.status_code != 200:
                raise RuntimeError(
                    f"文件上传失败或超时: {file_path}, status={upload_response.status_code}"
                )

        return batch_id

    def upload_urls(self, file_urls: list[str], is_ocr: bool = True) -> str:
        """
        URL 批量提交解析任务（适用于 MinIO 等对象存储生成的公网 URL）

        Args:
            urls: 文件 URL 列表，单次不超过 50 个
            is_ocr: 是否启动 OCR，仅对 pipeline / vlm 模型有效

        Returns:
            str: 批量提取任务 id（batch_id），可用于后续查询解析结果

        Raises:
            ValueError: url 列表为空或数量超过 50 时抛出
            RuntimeError: 提交任务失败时抛出
        """
        if not file_urls:
            raise ValueError("file_urls 列表不能为空")

        if len(file_urls) > 50:
            raise ValueError(f"单次最多提交 50 个文件，当前 {len(file_urls)} 个")

        payload = {
            "files": [
                {
                    "url": url,
                    "data_id": str(uuid.uuid4()),
                    "is_ocr": is_ocr,
                }
                for url in file_urls
            ],
            "model_version": self.model_version,
        }

        response = requests.post(
            f"{self.base_url}/extract/task/batch",
            headers=self.headers,
            json=payload,
            timeout=_API_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"提交解析任务失败: status={response.status_code}, body={response.text}"
            )

        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(
                f"提交解析任务失败: code={result.get('code')}, msg={result.get('msg')}"
            )

        return result["data"]["batch_id"]

    def get_parse_results(self, batch_id: str) -> dict:
        """
        查询批量解析结果
        
        Args:
            batch_id: 批量提取任务 id(batch_id)
        
        Returns:
            dict: 解析结果字典，包含解析状态、文档列表等信息
        """
        results = requests.get(
            f"{self.base_url}/extract-results/batch/{batch_id}",
            headers=self.headers,
            timeout=_API_TIMEOUT,
        )
        if results.status_code != 200:
            raise RuntimeError(
                f"查询解析结果失败或超时: status={results.status_code}, body={results.text}"
            )
        return results.json()

    def _extract_full_md(self, full_zip_url: str) -> bytes:
        """
        下载 MinerU 解析结果压缩包，读出其中的 full.md 内容

        Args:
            full_zip_url: 解析结果压缩包地址

        Returns:
            bytes: full.md 文件内容

        Raises:
            RuntimeError: 压缩包下载失败或包内没有 full.md 时抛出
        """
        response = requests.get(full_zip_url, timeout=_TRANSFER_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(
                f"下载解析结果失败或超时: status={response.status_code}, url={full_zip_url}"
            )

        with io.BytesIO(response.content) as buffer:
            with zipfile.ZipFile(buffer) as zf:
                md_name = next(
                    (name for name in zf.namelist() if Path(name).name == "full.md"),
                    None,
                )
                if md_name is None:
                    raise RuntimeError(f"压缩包内没有 full.md: {full_zip_url}")
                return zf.read(md_name)

    def download_md_to_local(self, full_zip_url: str, save_dir: str | Path, save_file_name: str = "full.md") -> Path:
        """
        下载 MinerU 解析结果压缩包，将其中的 full.md 保存到本地

        Args:
            full_zip_url: 解析结果压缩包地址
            save_dir: Markdown 文件保存目录
            save_file_name: 保存的文件名（默认 "full.md"）

        Returns:
            Path: 保存后的本地 .md 文件路径

        Raises:
            RuntimeError: 压缩包下载失败或包内没有 full.md 时抛出
        """
        md_bytes = self._extract_full_md(full_zip_url)

        save_path = Path(save_dir) / save_file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(md_bytes)

        return save_path

    def upload_md_to_minio(
        self,
        full_zip_url: str,
        minio_client: MinioClient,
        bucket_name: str,
        object_name: str = "full.md",
    ) -> str:
        """
        从 MinerU 解析结果压缩包中取出 full.md，直接上传到 MinIO（不落地本地）

        Args:
            full_zip_url: 解析结果压缩包地址（get_parse_results 返回的 full_zip_url）
            minio_client: 已初始化的 MinioClient 实例
            bucket_name: 存储桶名称
            object_name: MinIO 对象名，默认 "full.md"

        Returns:
            str: MinIO 中的对象名（object_name）

        Raises:
            RuntimeError: 压缩包下载失败或包内没有 full.md 时抛出
            MinioError: MinIO 上传失败时透传
        """
        md_bytes = self._extract_full_md(full_zip_url)

        # MinioClient.upload_file 基于文件路径，故经临时文件中转，用完即删
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp.write(md_bytes)
            tmp_path = Path(tmp.name)
        try:
            minio_client.upload_file(bucket_name, str(tmp_path), object_name)
        finally:
            tmp_path.unlink(missing_ok=True)

        return object_name


