"""用于校验大模型输出的 Pydantic schema。

图状态继续使用 TypedDict。节点先用这些 schema 校验模型输出，
再把校验后的对象转换成字典写回图状态。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """拒绝未声明字段的基础 schema。"""

    model_config = ConfigDict(extra="forbid")


class IntentOutput(StrictModel):
    """意图识别结果。"""

    intent: Literal["chat", "aiops"]


class PlanStepModel(StrictModel):
    """排障计划中的单个步骤。"""

    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    status: Literal["pending", "running", "completed", "failed", "skipped"]


class PlanOutput(StrictModel):
    """初始规划输出。"""

    steps: list[PlanStepModel]


class DiagnosisResultModel(StrictModel):
    """最终诊断结果。"""

    summary: str = Field(min_length=1)
    root_cause: str | None  # 根原因
    evidence_ids: list[str] # 支据 ID 列表
    knowledge_document_ids: list[str] # 知识库文档 ID 列表
    recommendations: list[str] # 推荐的解决方案
    limitations: list[str]  # 结论中的不确定性与证据缺口


class ReplanOutput(StrictModel):
    """重新规划输出：继续排查或结束诊断，二者只能选一。"""

    steps: list[PlanStepModel] = Field(default_factory=list) # 排障计划中的步骤
    result: DiagnosisResultModel | None = None # 最终诊断结果

    @model_validator(mode="after")
    def check_exclusive(self) -> "ReplanOutput":
        """校验 steps 和 result 是否互斥。

        Returns:
            校验通过的对象。

        Raises:
            ValueError: 两个字段同时为空或同时非空。
        """
        if self.result is not None and self.steps:
            raise ValueError("steps 和 result 不能同时非空")
        if self.result is None and not self.steps:
            raise ValueError("steps 和 result 不能同时为空")
        return self


class SubQuestionModel(StrictModel):
    """问题拆解产生的单个子查询。"""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1) # 子查询文本
    depends_on: list[str] = Field(default_factory=list) # 依赖的子查询 ID 列表
    purpose: str = "" # 子查询的目的


class SubqueryRewriteOutput(StrictModel):
    """SDOM 对依赖子问题的改写结果：一个补全后的可检索问题。"""

    rewritten_question: str = Field(min_length=1)  # 占位符已替换为具体实体的完整问题


class SubqueryOutput(StrictModel):
    """问题拆解结果。"""

    original_question: str
    need_decomposition: bool
    reason: str
    sub_questions: list[SubQuestionModel] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> "SubqueryOutput":
        """校验子查询 ID 唯一性以及依赖关系。

        Returns:
            校验通过的对象。

        Raises:
            ValueError: 子查询 ID 重复，或依赖了不存在的子查询。
        """
        ids = {item.id for item in self.sub_questions} # 子查询 ID 集合
        if len(ids) != len(self.sub_questions):
            raise ValueError("sub_questions.id 必须唯一")
        for item in self.sub_questions:
            if item.id in item.depends_on:
                raise ValueError(f"{item.id} 不能依赖自身")
            if len(set(item.depends_on)) != len(item.depends_on):
                raise ValueError(f"{item.id} 的 depends_on 存在重复 ID")
            if any(dep not in ids for dep in item.depends_on):
                raise ValueError(f"{item.id} 依赖了不存在的子查询")

        # 三色 DFS：发现指向 visiting 节点的边即为环，依赖图必须是 DAG。
        # 0: 白色代表未访问；1: 灰色代表正在访问、(所依赖的前置条件)尚未处理完；2: 黑色代表已处理完。
        colors: dict[str, int] = {}
        deps = {item.id: item.depends_on for item in self.sub_questions}
        visiting: list[str] = []

        def visit(node: str) -> None:
            # 已访问，且正在处理中，说明存在环
            if colors.get(node) == 1:
                cycle = " -> ".join([*visiting[visiting.index(node):], node])
                raise ValueError(f"子查询依赖存在环: {cycle}")
            if colors.get(node) == 2:
                return
            colors[node] = 1
            visiting.append(node)
            for dep in deps[node]:
                visit(dep)
            visiting.pop()
            colors[node] = 2

        for item in self.sub_questions:
            visit(item.id)
        return self
