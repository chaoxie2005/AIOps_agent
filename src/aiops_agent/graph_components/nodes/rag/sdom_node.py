"""RAG 子图的 SDOM 节点：用上游信息把带占位符的子问题补全成完整问题。"""

import logging

from aiops_agent.agent_components.llm import ModelClient, client_for_node
from aiops_agent.graph_components.utils.rag_utils import _PLACEHOLDER_RE, strip_placeholders
from aiops_agent.graph_components.states.state import RagState
from aiops_agent.output_validation import SubqueryRewriteOutput, ainvoke_structured
from aiops_agent.prompts.load_texts.load_sdom_prompt import sdom_prompt

analysis_logger = logging.getLogger(__name__)


async def dependency_optimize_node(state: RagState, model_client: ModelClient) -> dict:
    """把带占位符的子问题改写为可独立检索的完整问题（SDOM 节点）。

    工作方式：先根据当前子问题的 depends_on 找到上游子问题，
    再从检索记录（RetrievalHit 类）里找出上游检索到的文档，把这些材料一起交给模型，
    让模型把 ``#qid#`` 占位符替换成具体实体。

    如果模型调用失败或改写结果仍含占位符，就退化为“直接删掉占位符”，
    保证流程还能继续检索。

    Args:
        state: RAG 子图当前状态，至少包含 current_subquery、sub_queries 和 retrieval_hits。
        model_client: 用于改写问题的模型客户端。

    Returns:
        状态更新字典，包含改写后的 current_subquery 和改写记录 rewrites。
    """
    current = state.get("current_subquery")
    if not current:
        raise ValueError("dependency_optimize_node 需要 current_subquery，请先经过 schedule_node")

    upstream_blocks: list[str] = []
    questions = {item["id"]: item for item in state.get("sub_queries", [])}
    hits = state.get("retrieval_hits", [])
    for dep_id in current["depends_on"]:
        upstream = questions.get(dep_id)
        if upstream is None:
            continue
        # 检索记录里的 subquery 是当时实际用的检索词（上游问题的 effective_query）。
        passages = [
            hit["document"].page_content
            for hit in hits
            if hit["subquery"] == upstream["effective_query"]
        ]
        evidence = "\n".join(passages[:3]) or "（未找到检索证据）"
        upstream_blocks.append(
            f"上游问题 {dep_id}: {upstream['question']}\n"
            f"{dep_id} 的检索证据:\n{evidence}"
        )

    upstream_context = "\n\n".join(upstream_blocks) or "（没有上游信息）"
    messages = sdom_prompt.format_messages(
        question=current["question"], upstream_context=upstream_context,
    )
    parsed, error = await ainvoke_structured(
        client_for_node(model_client, "sdom"), messages, SubqueryRewriteOutput,
        correction="只输出符合 SubqueryRewriteOutput schema 的 JSON 对象，不要输出解释。",
    )

    if error is None and parsed is not None:
        rewritten = parsed.rewritten_question.strip()
        # 模型偶尔会忘记规则，这里再检查一遍：不能残留占位符，也不能是空串。
        if not rewritten or _PLACEHOLDER_RE.search(rewritten):
            error = f"改写结果仍包含占位符或为空: {rewritten!r}"
        else:
            analysis_logger.info("SDOM 改写成功: %s -> %s", current["question"], rewritten)

    if error is not None:
        # 降级：直接删掉占位符，虽然语义不如模型补全，但至少能拿去检索。
        analysis_logger.warning("SDOM 改写失败，降级删除占位符: %s", error)
        rewritten = strip_placeholders(current["question"]) or current["question"]

    # 只改 effective_query，保留原始 question 方便后续核对改写前后差异。
    updated = dict(current)
    updated["effective_query"] = rewritten
    rewrites = dict(state.get("rewrites", {}))
    rewrites[current["id"]] = rewritten
    return {"current_subquery": updated, "rewrites": rewrites}
