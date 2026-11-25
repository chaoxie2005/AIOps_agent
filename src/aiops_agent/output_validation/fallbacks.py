"""模型输出不可用时的业务降级结果。

降级结果不会伪造诊断事实。面向用户的文案保持中文。
"""

from __future__ import annotations

from aiops_agent.graph_components.states.state import DiagnosisResult


def fallback_intent() -> str:
    """意图识别失败时降级到最保守的 chat 分支。

    Returns:
        chat 意图，不会触发现场查询工具。
    """
    return "chat"


def fallback_plan() -> dict:
    """规划输出不可用时返回空计划。

    Returns:
        只包含空 steps 的状态更新，由 replan 节点负责收尾。
    """
    return {"steps": []}


def fallback_replan(error: str) -> dict:
    """重规划失败或步骤预算耗尽时生成带限制说明的诊断结果。

    Args:
        error: 结构化输出失败原因。

    Returns:
        包含空 steps 和降级诊断结果的状态更新。
    """
    result: DiagnosisResult = {
        "summary": "诊断流程因规划输出不可用或步骤预算耗尽而结束。",
        "root_cause": None,
        "evidence_ids": [],
        "knowledge_document_ids": [],
        "recommendations": [
            "检查模型结构化输出配置与服务可用性后重新发起诊断。",
            "根据已收集的步骤结果和证据人工继续排查。",
        ],
        "limitations": [
            f"结构化输出失败：{error}" if error else "规划输出不可用。",
            "未能确认根因，已有步骤结果仅供参考。",
        ],
    }
    return {"steps": [], "result": result}


def fallback_chat(error: str) -> dict:
    """聊天生成失败时返回明确提示，不伪造答案。

    Args:
        error: 聊天生成失败原因。

    Returns:
        包含降级回复的状态更新。
    """
    return {
        "reply": "当前服务暂时无法生成可靠回复，请稍后重试。",
        "result": None,
        "degraded": f"聊天生成失败：{error}" if error else "聊天生成失败。",
    }
