"""SDOM 依赖优化节点单元测试：用假模型验证改写和降级，不依赖真实服务。"""

import asyncio

from langchain_core.documents import Document

from aiops_agent.graph_components.nodes.rag_nodes import dependency_optimize_node
from aiops_agent.graph_components.states.state import RagState


class FakeModel:
    """返回固定改写结果的假模型。"""

    def __init__(self, content=None, fail=False):
        self.content = content
        self.fail = fail

    async def ainvoke(self, messages):
        if self.fail:
            raise RuntimeError("模型服务不可用")
        self.messages = messages
        return self.content


def _state():
    """构造带上游检索记录的测试状态。"""
    state: RagState = {
        "query": "原始问题",
        "current_subquery": {
            "id": "q2",
            "question": "#q1# 的孩子是谁？",
            "depends_on": ["q1"],
            "purpose": "",
            "status": "ready",
            "effective_query": "#q1# 的孩子是谁？",
        },
        "sub_queries": [
            {
                "id": "q1",
                "question": "电影《Mukhyamantri》的导演是谁？",
                "depends_on": [],
                "purpose": "",
                "status": "done",
                "effective_query": "电影《Mukhyamantri》的导演是谁？",
            },
            {
                "id": "q2",
                "question": "#q1# 的孩子是谁？",
                "depends_on": ["q1"],
                "purpose": "",
                "status": "ready",
                "effective_query": "#q1# 的孩子是谁？",
            },
        ],
        "retrieval_hits": [
            {
                "subquery": "电影《Mukhyamantri》的导演是谁？",
                "document": Document(page_content="该片由 Anjan Choudhury 导演。"),
            }
        ],
    }
    return state


def test_sdom_rewrite_success():
    """模型输出合法 JSON 时，占位符应被具体实体替换。"""
    fake = FakeModel('{"rewritten_question": "Anjan Choudhury 的孩子是谁？"}')
    result = asyncio.run(dependency_optimize_node(_state(), fake))
    assert result["current_subquery"]["effective_query"] == "Anjan Choudhury 的孩子是谁？"
    assert result["rewrites"]["q2"] == "Anjan Choudhury 的孩子是谁？"
    # 确认上游证据真的进了提示词。
    prompt = fake.messages[-1].content
    assert "Anjan Choudhury" in prompt


def test_sdom_fallback_on_model_failure():
    """模型失败时，降级为直接删掉占位符。"""
    fake = FakeModel(fail=True)
    result = asyncio.run(dependency_optimize_node(_state(), fake))
    assert result["current_subquery"]["effective_query"] == "孩子是谁？"


def test_sdom_fallback_on_placeholder_left():
    """模型改写后仍残留占位符时，也要降级。"""
    fake = FakeModel('{"rewritten_question": "#q1# 是谁？"}')
    result = asyncio.run(dependency_optimize_node(_state(), fake))
    assert "#q1#" not in result["current_subquery"]["effective_query"]
