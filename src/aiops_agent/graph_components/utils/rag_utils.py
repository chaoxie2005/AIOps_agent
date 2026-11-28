"""RAG 节点共用的小工具：占位符清洗和兜底子问题构造。

使用者：
- subquery_node.py：用 default_subquery 做拆解失败兜底；用 strip_placeholders
  清洗依赖图非法时的子问题；用 _PLACEHOLDER_RE 检查占位符与 depends_on 一致性。
- sdom_node.py：用 _PLACEHOLDER_RE 检查模型改写结果是否残留占位符；
  用 strip_placeholders 在改写失败时降级删除占位符。
"""

import re

from aiops_agent.graph_components.states.state import SubQueryItem

# 子问题占位符正则，例如 #q1#。
_PLACEHOLDER_RE = re.compile(r"#(q\d+)#")


def default_subquery(question: str) -> SubQueryItem:
    """构造一个无依赖的兜底子问题，子问题拆解节点失败时调用。

    Args:
        question: 用户的原始问题。

    Returns:
        一个 ID 为 ``q1``、无依赖且状态为 pending 的子问题项。
    """
    return {
        "id": "q1",
        "question": question,
        "depends_on": [],
        "purpose": "原始问题",
        "status": "pending",
        "effective_query": question,
    }


def strip_placeholders(text: str) -> str:
    """删除问题文本中的整个 ``#qid#`` 指代标记并规整文本。

    降级为并行检索时无法解析指代指向的答案，因此直接删除整个 ``#qid#``
    标记（如 ``#q1#``），再压缩空白、清理标点前多余空格和句首悬空助词。

    Args:
        text: 可能包含 ``#qid#`` 指代标记的问题文本。

    Returns:
        指代标记已整体删除、空白与标点已规整的问题文本。
    """
    cleaned = _PLACEHOLDER_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([?!.,;:？！。，；：])", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[的之]+\s*", "", cleaned)
    return cleaned.strip()
