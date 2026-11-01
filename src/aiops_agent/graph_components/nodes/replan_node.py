import json
import re

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import OverviewState
from aiops_agent.prompts.load_texts.load_replan_prompt import replan_agent_prompt


async def replan_node(state: OverviewState, model_client: ModelClient):
    """评估当前诊断流程是否需要重新规划，如果需要，返回新的步骤列表。"""
    messages = replan_agent_prompt.format_messages(
        question=state["query"],
        knowledge_documents=state.get("knowledge_documents", []),
        existing_plan=state.get("steps", []),
        step_results=state.get("step_results", []),
        evidence=state.get("evidence", []),
        remaining_steps=state.get("max_steps", 8) - len(state.get("step_results", [])),
    )
    response = await model_client.ainvoke(messages)
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if match is None:
        raise ValueError("重新规划节点未获取到 JSON 输出")
    try:
        raw_content = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError("重新规划节点输出不是有效的 JSON") from exc
    steps = raw_content["steps"]
    result = raw_content["result"]
    if isinstance(result, dict) and result is not  None:
        return {"steps": [], "result": result}
    return {"steps": steps, "result": None}
