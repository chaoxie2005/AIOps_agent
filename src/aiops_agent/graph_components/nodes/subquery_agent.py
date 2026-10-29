"""子问题拆解节点。"""
import json
import re
from loguru import logger

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import RagInput
from aiops_agent.prompts.load_texts.load_subquery_prompt import subquery_agent_prompt


async def subquery_agent_node(state: RagInput, model_client: ModelClient):
    """生成子问题，并返回供图合并的状态更新。"""
    messages = subquery_agent_prompt.format_messages(
        max_sub_questions=state.get("max_subqueries", 4),
        question=state["query"],
    )
    response = await model_client.ainvoke(messages)
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    raw_content = json.loads(match.group())
    logger.info(f"子问题拆解agent生成的原始内容: {raw_content}")
    sub_questions = [item["question"] for item in raw_content["sub_questions"]]
    logger.info(f"子问题拆解agent生成的子问题: {sub_questions}")
    unique_subqueries = list(dict.fromkeys(sub_questions))
    logger.info(f"子问题拆解agent生成的去重子问题: {unique_subqueries}")
    return {"unique_subqueries": unique_subqueries}
