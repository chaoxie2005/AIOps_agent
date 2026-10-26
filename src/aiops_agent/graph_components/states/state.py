"""LangGraph 排障流程使用的状态及结构化数据类型。"""

from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal

from typing_extensions import Required, TypedDict


StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
EvidenceSource = Literal["prometheus", "cls"]


class TimeRange(TypedDict):
    """查询覆盖的时间范围；时间应带时区。"""

    start: datetime  # 查询开始时间
    end: datetime  # 查询结束时间


class PlanStep(TypedDict):
    """计划中的一个排障步骤。"""

    id: str  # 稳定的步骤 ID
    goal: str  # 本步骤的排障目标
    status: StepStatus  # 当前步骤状态


class StepResult(TypedDict):
    """一个排障步骤的执行结果。"""

    step_id: str  # 对应计划步骤的 ID
    status: Literal["completed", "failed", "skipped"]  # 执行结束状态
    summary: str  # 本步骤的执行结论
    evidence_ids: list[str]  # 本步骤产生的现场证据 ID
    knowledge_document_ids: list[str]  # 本步骤参考的知识文档 ID
    tool_call_ids: list[str]  # 本步骤发起的工具调用 ID
    error: str | None  # 失败原因；未失败时为 None


class Evidence(TypedDict):
    """Prometheus 或 CLS 提供的本次故障现场证据。"""

    id: str  # 证据的唯一 ID
    step_id: str  # 产生此证据的步骤 ID
    source: EvidenceSource  # 现场数据来源
    query: str  # 获取证据时执行的查询
    time_range: TimeRange  # 查询覆盖的时间范围
    observation: str  # 从结果提取的客观观察
    raw_result_ref: str | None  # 原始结果的存储引用


class KnowledgeDocument(TypedDict):
    """RAG 找到的手册或历史经验，不属于现场证据。"""

    id: str  # 文档的唯一 ID
    title: str  # 文档标题
    content: str  # 供诊断参考的文档内容
    source_uri: str | None  # 文档来源地址
    score: float | None  # 重排相关性分数


class RetrievalHit(TypedDict):
    """某个子查询召回的一篇文档。"""

    subquery: str  # 触发召回的子查询
    document: KnowledgeDocument  # 召回的文档
    retrieval_score: float | None  # 初次检索相关性分数


class ToolCallRecord(TypedDict):
    """执行器的一次工具调用。"""

    id: str  # 工具调用的唯一 ID
    tool_name: str  # 被调用的工具名称
    arguments: dict[str, Any]  # 提交给工具的参数


class ToolResultRecord(TypedDict):
    """工具调用的结果；大体积原始内容只保存引用。"""

    tool_call_id: str  # 对应工具调用的 ID
    success: bool  # 工具调用是否成功
    summary: str  # 工具返回内容的简要概括
    raw_result_ref: str | None  # 原始返回内容的存储引用
    error: str | None  # 调用失败时的错误信息


class DiagnosisResult(TypedDict):
    """最终结构化诊断结果。"""

    summary: str  # 故障诊断摘要
    root_cause: str | None  # 推断的根因；不确定时为 None
    evidence_ids: list[str]  # 支持诊断的现场证据 ID
    knowledge_document_ids: list[str]  # 参考的排障知识文档 ID
    recommendations: list[str]  # 建议采取的排障或修复措施
    limitations: list[str]  # 结论中的不确定性与证据缺口


class OverviewState(TypedDict, total=False):
    """主图共享状态；除 query 外，其余字段由节点逐步写入。"""

    query: Required[str]  # 用户提出的故障问题
    steps: list[PlanStep]  # 当前有效的有序步骤表
    current_step_id: str | None  # 当前待执行步骤的 ID
    step_results: Annotated[list[StepResult], add]  # 各步骤的执行结果
    evidence: Annotated[list[Evidence], add]  # 各步骤故障的现场证据列表
    knowledge_documents: Annotated[list[KnowledgeDocument], add]  # RAG 提供的排障知识 RAG检索结果
    result: DiagnosisResult | None  # 最终结构化诊断结果
    max_steps: int  # 本次诊断允许执行的步骤上限


class RagState(TypedDict, total=False):
    """RAG 子图状态；检索结果可以由并行节点追加。"""

    query: Required[str]  # 本次 RAG 检索要解决的问题
    subqueries: list[str]  # 拆解后的子问题列表
    unique_subqueries: list[str]  # 去重后用于检索的子问题列表
    retrieval_hits: Annotated[list[RetrievalHit], add]  # 并行检索的原始召回结果
    unique_documents: list[KnowledgeDocument]  # 聚合去重后的文档列表
    selected_documents: list[KnowledgeDocument]  # 重排筛选后的最终文档 超过max_rag_window_tokens就需进行二次重排筛选最相关或者进行压缩
    max_subqueries: int  # 允许生成的子查询上限数量
    max_rag_window_tokens: int  # 最终文档内容的 token 预算 不能超过该值
    # max_rerank_documents: int  # 最终重排保留的文档数量上限 不能超过该值 先固定为topk3 后续在动态


class ExecutorPrivateState(TypedDict, total=False):
    """执行单个计划步骤时使用的私有状态。"""

    current_step: Required[PlanStep]  # 当前执行的计划步骤
    tool_calls: list[ToolCallRecord]  # 本步骤发起的工具调用
    tool_results: list[ToolResultRecord]  # 本步骤收到的工具结果
    evidence_candidates: list[Evidence]  # 从现场工具结果提取的证据
    knowledge_candidates: list[KnowledgeDocument]  # 本步骤检索到的排障知识
    step_result: StepResult | None  # 本步骤的最终执行结果
    max_tool_calls: int  # 本步骤允许的工具调用上限
