"""RAG 子图的拆解节点：把原始问题拆成带依赖关系的子问题。"""

import logging

from aiops_agent.agent_components.llm import ModelClient, client_for_node
from aiops_agent.graph_components.utils.rag_utils import (
    _PLACEHOLDER_RE,
    default_subquery,
    strip_placeholders,
)
from aiops_agent.graph_components.states.state import RagInput, SubQueryItem
from aiops_agent.output_validation import SubqueryOutput, ainvoke_structured
from aiops_agent.prompts.load_texts.load_subquery_prompt import subquery_agent_prompt

analysis_logger = logging.getLogger(__name__)


def to_parallel_items(parsed: SubqueryOutput, limit: int) -> list[SubQueryItem]:
    """把拆解结果降级为无依赖、可并行检索的子问题列表。

    保留每个子问题的语义文本，清空 ``depends_on``，并把问题中的整个
    ``#qid#`` 指代标记删除。模型完全失败（无 parsed 结果）时调用方
    改用 ``default_subquery`` 兜底。

    Args:
        parsed: 模型输出的结构化拆解结果。
        limit: 保留的子问题数量上限，必须大于等于 1。

    Returns:
        无依赖、占位符已清洗的子问题项列表。

    Raises:
        ValueError: ``limit`` 小于 1，或清洗后的某个问题为空。
    """
    if limit < 1:
        raise ValueError("max_subqueries 必须大于 0")
    items: list[SubQueryItem] = []
    for item in parsed.sub_questions[:limit]:
        cleaned = strip_placeholders(item.question)
        if not cleaned:
            raise ValueError(f"{item.id} 清洗占位符后问题为空")
        items.append({
            "id": item.id,
            "question": cleaned,
            "depends_on": [],
            "purpose": item.purpose,
            "status": "pending",
            "effective_query": cleaned,
        })
    return items


def to_subquery_items(parsed: SubqueryOutput) -> list[SubQueryItem]:
    """把校验通过的模型输出转换为图状态中的子问题项。

    同时检查每个问题中的 ``#qid#`` 指代标记是否与 ``depends_on`` 完全一致。

    Args:
        parsed: 已通过校验的结构化拆解结果。

    Returns:
        携带依赖信息、状态为 pending 的子问题项列表。

    Raises:
        ValueError: 问题中的占位符与依赖关系不一致。
    """
    items: list[SubQueryItem] = []
    for item in parsed.sub_questions:
        placeholders = set(_PLACEHOLDER_RE.findall(item.question))
        declared = set(item.depends_on)
        if placeholders != declared:
            raise ValueError(
                f"{item.id} 的占位符 {sorted(placeholders)} 与依赖 {sorted(declared)} 不一致"
            )
        items.append({
            "id": item.id,
            "question": item.question,
            "depends_on": list(dict.fromkeys(item.depends_on)),
            "purpose": item.purpose,
            "status": "pending",
            "effective_query": item.question,
        })
    return items


def truncate_with_dependencies(items: list[SubQueryItem], limit: int) -> list[SubQueryItem]:
    """子问题超过上限时截断，并清掉因此产生的悬空依赖, 只能有max_subqueries个子问题。

    保留原有 ID 顺序；任何指向已被截断子问题的依赖都会被删除，以保证剩余依赖图仍然有效。

    Args:
        items: 原始顺序的子问题项列表。
        limit: 保留的子问题数量上限，必须大于等于 1。

    Returns:
        清理依赖后保留的子问题项列表。

    Raises:
        ValueError: ``limit`` 小于 1。
    """
    if limit < 1:
        raise ValueError("max_subqueries 必须大于 0")
    kept = items[:limit]
    kept_ids = {item["id"] for item in kept}
    for item in kept:
        item["depends_on"] = [dep for dep in item["depends_on"] if dep in kept_ids]
        item["effective_query"] = item["question"]
    return kept


async def subquery_node(state: RagInput, model_client: ModelClient) -> dict:
    """拆解用户问题，保留可执行的依赖关系并写入 RAG 子图状态。

    依赖校验失败时降级为“保留子问题、清空依赖、清洗占位符”的并行检索模式；
    模型完全失败时降级为单个原始问题。

    Args:
        state: RAG 输入，包含原始问题和可选的 ``max_subqueries`` 上限。
        model_client: 用于生成问题拆解的模型客户端。

    Returns:
        包含结构化 ``sub_queries`` 和兼容旧字段 ``unique_subqueries`` 的状态更新。
    """
    question = state["query"]
    max_subqueries = state.get("max_subqueries", 4)
    messages = subquery_agent_prompt.format_messages(
        max_sub_questions=max_subqueries,
        question=question,
    )
    parsed, error = await ainvoke_structured(
        client_for_node(model_client, "subquery"), messages, SubqueryOutput,
        correction="只输出符合 SubqueryOutput schema 的 JSON 对象，不要输出解释。",
    )

    if error is not None or parsed is None:
        # 模型完全不可用：没有子问题可保留，只能用原始问题兜底。
        analysis_logger.warning("子问题拆解失败，回退到原始问题: %s", error)
        items = [default_subquery(question)]
    else:
        try:
            items = to_subquery_items(parsed)
            if len(items) > max_subqueries:
                analysis_logger.warning(
                    "子查询数量 %d 超过上限，截断为 %d", len(items), max_subqueries,
                )
                items = truncate_with_dependencies(items, max_subqueries)
        except ValueError as exc:
            # 依赖图非法或占位符不一致：保留子问题语义，清空依赖后并行检索。
            analysis_logger.warning("子问题依赖校验失败，降级为并行检索: %s", exc)
            try:
                items = to_parallel_items(parsed, max_subqueries)
            except ValueError as parallel_exc:
                analysis_logger.warning("并行检索降级也失败，回退到原始问题: %s", parallel_exc)
                items = [default_subquery(question)]

    analysis_logger.info("子问题拆解结果: %s", [item["question"] for item in items])
    return {
        "sub_queries": items,
        "unique_subqueries": [item["question"] for item in items],
    }
