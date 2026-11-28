"""规划节点：生成结构化的排障步骤。"""

import logging

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import OverviewState
from aiops_agent.output_validation import PlanOutput, ainvoke_structured, dump_model, fallback_plan
from aiops_agent.prompts.load_texts.load_plan_prompt import plan_agent_prompt

logger = logging.getLogger(__name__)


async def plan_node(state: OverviewState, model_client: ModelClient):
    """根据用户输入生成结构化的排障计划。

    模型输出无效时返回空计划，由 replan 节点输出带限制的诊断，
    不把格式错误暴露给调用方。

    Args:
        state: 包含问题、知识和步骤预算的主图状态。
        model_client: 由主图注入的模型客户端。

    Returns:
        包含校验后计划步骤的状态更新。
    """
    step_results = state.get("step_results", [])
    plan_prompt = plan_agent_prompt.format_messages(
        question=state["query"],
        knowledge_documents=state.get("knowledge_documents", []),
        existing_plan=state.get("steps", []),
        step_results=step_results,
        remaining_steps=state.get("max_steps", 8) - len(step_results),
    )
    parsed, error = await ainvoke_structured(
        model_client, plan_prompt, PlanOutput,
        correction="只输出符合 PlanOutput schema 的 JSON 对象，不要输出解释。",
    )
    if error is not None:
        logger.warning("规划节点结构化输出失败: %s", error)
        return fallback_plan()
    assert parsed is not None
    return dump_model(parsed)
