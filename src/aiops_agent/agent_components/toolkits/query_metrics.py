import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool


@tool
def query_metrics(query: str) -> dict:
    """用 PromQL 查询 Prometheus 当前时刻的指标数据。

    服务地址从环境变量 PROMETHEUS_URL 读取，也支持从项目的 .env 文件加载。

    Args:
        query: 要执行的 PromQL 表达式，例如 "up"。

    Returns:
        Prometheus 响应中的 data 字典，包含结果类型 resultType 和查询结果 result。
        未查到匹配的时间序列时，result 可以是空列表。

    Raises:
        ValueError: 未填写服务地址、Prometheus 返回查询错误，或响应不是有效 JSON。
        requests.RequestException: 请求超时、连接失败，或 HTTP 响应状态表示失败。
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    base_url = os.getenv("PROMETHEUS_URL", "").strip().rstrip("/")
    if not base_url:
        raise ValueError("请先在 .env 中填写 PROMETHEUS_URL（Prometheus 服务地址）")

    response = requests.get(
        f"{base_url}/api/v1/query",
        params={"query": query},
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "success":
        raise ValueError(f"Prometheus 查询失败: {result.get('error', '未知错误')}")
    return result["data"]
