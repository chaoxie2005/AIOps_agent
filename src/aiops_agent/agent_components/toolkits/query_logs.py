import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from tencentcloud.cls.v20201016 import cls_client, models
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile


@tool
def query_logs(query: str, start: int, end: int, limit: int = 20) -> dict:
    """查询腾讯云 CLS 指定时间范围内的日志。

    凭据、地域和日志主题从环境变量或项目的 .env 文件读取。

    Args:
        query: CLS 检索分析语句，例如 "ERROR"，使用 "*" 查询所有日志。
        start: 开始时间，Unix 毫秒时间戳，包含该时刻。
        end: 结束时间，Unix 毫秒时间戳，不包含该时刻。
        limit: 单次返回的原始日志条数，范围为 1 到 1000，默认 20。
            使用 SQL 分析时，需在 query 中通过 SQL LIMIT 限制结果条数。

    Returns:
        CLS 查询响应字典，包含日志或分析结果及请求 ID；不自动获取后续分页。

    Raises:
        ValueError: 缺少必要配置、时间范围错误或返回条数超出范围。
        TencentCloudSDKException: 认证失败、权限不足或查询请求失败。
    """
    if start < 0 or start >= end:
        raise ValueError("时间范围需满足 0 <= start < end，单位为毫秒")
    if not 1 <= limit <= 1000:
        raise ValueError("limit 必须在 1 到 1000 之间")

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    keys = ("TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY", "CLS_REGION", "CLS_TOPIC_ID")
    config = {key: os.getenv(key, "").strip() for key in keys}
    missing = [key for key, value in config.items() if not value]
    if missing:
        raise ValueError(f"请先在 .env 中填写: {', '.join(missing)}")

    credentials = credential.Credential(
        config["TENCENTCLOUD_SECRET_ID"], config["TENCENTCLOUD_SECRET_KEY"]
    )
    profile = ClientProfile()
    profile.httpProfile = HttpProfile()
    profile.httpProfile.reqTimeout = 10
    client = cls_client.ClsClient(credentials, config["CLS_REGION"], profile)
    request = models.SearchLogRequest()
    request.from_json_string(json.dumps({
        "TopicId": config["CLS_TOPIC_ID"],
        "QueryString": query,
        "From": start,
        "To": end,
        "Limit": limit,
        "UseNewAnalysis": True,
    }))
    return json.loads(client.SearchLog(request).to_json_string())
