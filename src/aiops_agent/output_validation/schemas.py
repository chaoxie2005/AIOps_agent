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
        ids = {item.id for item in self.sub_questions}
        if len(ids) != len(self.sub_questions):
            raise ValueError("sub_questions.id 必须唯一")
        for item in self.sub_questions:
            if any(dep not in ids for dep in item.depends_on):
                raise ValueError(f"{item.id} 依赖了不存在的子查询")
        return self
