"""重新规划节点：继续排查或基于现状结束诊断。"""

import logging

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import OverviewState
from aiops_agent.output_validation import ReplanOutput, ainvoke_structured, dump_model, fallback_replan
from aiops_agent.prompts.load_texts.load_replan_prompt import replan_agent_prompt

logger = logging.getLogger(__name__)


async def replan_node(state: OverviewState, model_client: ModelClient):
    """决定继续排查还是基于现状结束诊断。

    预算耗尽、输出格式错误或字段关系不合法时，都会转成带限制的诊断结果，
    保证主图可以正常结束。

    Args:
        state: 包含步骤、结果、证据和预算的主图状态。
        model_client: 由主图注入的模型客户端。

    Returns:
        包含校验后后续步骤或最终诊断结果的状态更新。
    """
    remaining = state.get("max_steps", 8) - len(state.get("step_results", []))
    messages = replan_agent_prompt.format_messages(
        question=state["query"],
        knowledge_documents=state.get("knowledge_documents", []),
        existing_plan=state.get("steps", []),
        step_results=state.get("step_results", []),
        evidence=state.get("evidence", []),
        remaining_steps=remaining,
    )
    parsed, error = await ainvoke_structured(
        model_client, messages, ReplanOutput,
        correction="只输出符合 ReplanOutput schema 的 JSON 对象；结束诊断时 steps 必须为空且 result 非 null。",
    )
    if error is not None:
        logger.warning("重规划节点结构化输出失败: %s", error)
        return fallback_replan(error)
    assert parsed is not None
    output = dump_model(parsed)
    if remaining <= 0 and output["result"] is None:
        logger.warning("步骤预算已耗尽，模型未给出诊断，改用降级结果")
        return fallback_replan("步骤预算已耗尽")
    if len(output["steps"]) > remaining:
        logger.warning("重规划步骤数超过剩余预算，截断为 %d 步", remaining)
        output["steps"] = output["steps"][:remaining]
    return output
