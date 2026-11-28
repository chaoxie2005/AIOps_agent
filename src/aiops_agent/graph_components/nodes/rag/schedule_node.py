"""RAG 子图的调度节点：挑选下一个子问题，并记录完成状态。"""

from aiops_agent.graph_components.states.state import RagState, SubQueryItem


def schedule_node(state: RagState) -> dict:
    """挑出下一轮要处理的子问题，并告诉图下一步该走哪个节点。

    遍历 sub_queries，找到第一个还没开始
    （status=pending）且依赖都已完成的子问题，把它标记为 ready，
    放进 current_subquery 交给下一个节点处理。

    - 有依赖的子问题 → 下一步走 optimize（SDOM 改写节点）
    - 没依赖的子问题 → 下一步走 retrieve（直接检索）
    - 没有可处理的子问题 → 下一步走 aggregate（收尾聚合）

    如果某个子问题等的上游（depends_on 里列出的子问题）已经失败被跳过，
    它永远等不到依赖，这里会直接把它标记为 skipped，避免流程卡住。

    Args:
        state: RAG 子图当前状态，至少包含 sub_queries 和 completed_ids。

    Returns:
        状态更新字典，包含新的 sub_queries、current_subquery、next_action 和 completed_ids。
    """
    # 复制一份再改，避免直接修改 LangGraph 传进来的状态对象。
    items: list[SubQueryItem] = [dict(item) for item in state.get("sub_queries", [])]
    # 用集合查依赖是否完成，比每次遍历列表快，也更直观。
    completed: set[str] = set(state.get("completed_ids", []))
    skipped_now: list[str] = []  # 这轮被跳过的子问题 ID

    # 先找出被跳过的子问题 ID：它们虽然在 completed_ids 里，但没有检索结果，
    # 依赖它们的子问题不能当“依赖已满足”来处理。
    skipped_ids = {other["id"] for other in items if other["status"] == "skipped"}

    for item in items:
        if item["status"] != "pending":
            continue

        # 只要依赖里有“未完成”或“已跳过”的，就还不能执行。
        waiting_for = [
            dep for dep in item["depends_on"]
            if dep not in completed or dep in skipped_ids
        ]
        if waiting_for:
            # 如果等的上游里有已经跳过的，这个子问题也永远等不到结果，直接跳过它。
            if any(dep in skipped_ids for dep in waiting_for):
                # 按 state.py 的约定：跳过时清空依赖和查询，避免留下无效引用。
                item["status"] = "skipped"
                item["depends_on"] = []
                item["effective_query"] = ""
                skipped_now.append(item["id"])
            continue

        # 找到一个可以执行的子问题：标记 ready 并交给下一节点。
        item["status"] = "ready"
        action = "optimize" if item["depends_on"] else "retrieve"
        return {
            "sub_queries": items,
            "current_subquery": dict(item),
            "next_action": action,
            "completed_ids": [*state.get("completed_ids", []), *skipped_now],
        }

    # 没有可执行的了：可能是全部完成，也可能是剩下的都被跳过了，统一去聚合。
    return {
        "sub_queries": items,
        "current_subquery": None,
        "next_action": "aggregate",
        "completed_ids": [*state.get("completed_ids", []), *skipped_now],
    }


def record_node(state: RagState) -> dict:
    """把当前子问题标记为已完成，让调度器可以继续处理下一个。

    检索节点跑完后调用它：把 current_subquery 的状态从 ready 改成 done，
    并把它的 ID 加入 completed_ids。这样依赖它的后续子问题在下一次
    schedule 时才会被认为“依赖已就绪”。

    Args:
        state: RAG 子图当前状态，至少包含 sub_queries 和 current_subquery。

    Returns:
        状态更新字典，包含更新后的 sub_queries、completed_ids 和 rewrites。

    Raises:
        ValueError: current_subquery 缺失，说明有节点跳过了调度直接调用本节点。
    """
    current = state.get("current_subquery")
    if not current:
        raise ValueError("record_node 需要 current_subquery，请先经过 schedule_node")

    current_id = current["id"]
    items: list[SubQueryItem] = [dict(item) for item in state.get("sub_queries", [])]
    for item in items:
        if item["id"] == current_id:
            item["status"] = "done"
            break

    completed = list(state.get("completed_ids", []))
    if current_id not in completed:
        completed.append(current_id)

    # 记录实际用的检索词，方便后续核对 SDOM 改写是否生效。
    rewrites = dict(state.get("rewrites", {}))
    rewrites[current_id] = current["effective_query"]

    return {
        "sub_queries": items,
        "completed_ids": completed,
        "rewrites": rewrites,
    }
