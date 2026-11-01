import json
import re

from loguru import logger

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import OverviewState
from aiops_agent.prompts.load_texts.load_plan_prompt import plan_agent_prompt


async def plan_node(state: OverviewState, model_client: ModelClient):
    """规划节点：根据用户问题，规划诊断流程。"""
    # 调用规划提示词模板，生成诊断流程
    step_results = state.get("step_results", [])
    plan_prompt = plan_agent_prompt.format_messages(
        question=state["query"],
        knowledge_documents=state.get("knowledge_documents", []),
        existing_plan=state.get("steps", []),
        step_results=step_results,
        remaining_steps=state.get("max_steps", 5) - len(step_results),
    )
    # 调用模型，生成诊断流程
    plans = await model_client.ainvoke(plan_prompt)
    match = re.search(r"\{.*\}", plans.content, re.DOTALL)
    if match is None:
        raise ValueError(f"规划节点未获取到 JSON 输出: {plans.content!r}")
    raw_content = json.loads(match.group())
    if isinstance(raw_content, dict) and "steps" in raw_content:
        steps = raw_content["steps"]
    elif isinstance(raw_content, list):
        steps = raw_content
    elif isinstance(raw_content, dict) and {"id", "goal"} <= raw_content.keys():
        steps = [raw_content]
    else:
        raise ValueError(f"规划节点输出缺少 steps 字段: {plans.content!r}")
    logger.info(f"规划节点生成的诊断步骤: {steps}")
    return {"steps": steps}
