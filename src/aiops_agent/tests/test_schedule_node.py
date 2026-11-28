"""调度节点的单元测试：不依赖模型和 Milvus。"""

from aiops_agent.graph_components.nodes.rag.schedule_node import record_node, schedule_node


def _state(sub_queries, completed_ids=None):
    """构造测试用的最小 RAG 状态。"""
    return {"sub_queries": sub_queries, "completed_ids": completed_ids or []}


def _item(id, depends_on=None, status="pending"):
    """构造测试用的子问题项。"""
    return {
        "id": id,
        "question": f"问题 {id}",
        "depends_on": depends_on or [],
        "purpose": "",
        "status": status,
        "effective_query": f"问题 {id}",
    }


def test_schedule_picks_independent_first():
    """第一轮应该挑出无依赖的子问题，并告诉图直接检索。"""
    result = schedule_node(_state([_item("q1", ["q2"]), _item("q2")]))
    assert result["current_subquery"]["id"] == "q2"
    assert result["next_action"] == "retrieve"


def test_schedule_enables_dependent_after_completion():
    """上游完成后，依赖它的子问题应该被选中并交给 optimize。"""
    result = schedule_node(_state([_item("q1", ["q2"]), _item("q2", status="done")], ["q2"]))
    assert result["current_subquery"]["id"] == "q1"
    assert result["next_action"] == "optimize"


def test_schedule_skips_orphan_and_aggregates():
    """上游被跳过后，下游也应跳过，最后走向聚合。"""
    result = schedule_node(_state([
        _item("q1", ["q2"]),
        _item("q2", status="skipped"),
    ], ["q2"]))
    assert result["current_subquery"] is None
    assert result["next_action"] == "aggregate"
    assert result["sub_queries"][0]["status"] == "skipped"
    assert result["sub_queries"][0]["depends_on"] == []
    assert "q1" in result["completed_ids"]


def test_record_marks_done_and_tracks_query():
    """记录节点应该把当前子问题标为完成，并保存实际检索词。"""
    item = _item("q1")
    item["status"] = "ready"
    item["effective_query"] = "改写后的查询"
    result = record_node({"sub_queries": [item], "current_subquery": item, "completed_ids": []})
    assert result["sub_queries"][0]["status"] == "done"
    assert result["completed_ids"] == ["q1"]
    assert result["effective_queries"]["q1"] == "改写后的查询"
