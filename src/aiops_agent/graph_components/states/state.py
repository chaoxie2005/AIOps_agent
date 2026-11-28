"""LangGraph 排障流程使用的状态及结构化数据类型。"""

from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from typing_extensions import Required, TypedDict


StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
EvidenceSource = Literal["prometheus", "cls"]
Intent = Literal["chat", "aiops"]

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
    document: Document  # 原始检索文档，检索器输出分数保存在 metadata["score"]


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


class OverviewOutput(TypedDict):
    """主图对外输出，包含处理意图、聊天回答或诊断结果。"""

    intent: Intent  # 本轮处理意图
    reply: str | None  # 普通聊天回复；排障分支为 None
    result: DiagnosisResult | None  # 排障结果；聊天分支为 None


class ChatTurn(OverviewOutput):
    """已完成并保存在 checkpoint 中的一轮问答。"""

    query: str
    created_at: datetime


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
    intent: Intent | None  # 用户问题的意图分类；由意图识别节点写入
    reply: str | None  # 普通聊天节点生成的回答
    turns: Annotated[list[ChatTurn], add]  # 只追加已完成的会话轮次

class SubQueryItem(TypedDict):
    """拆解后的单个子问题，携带依赖优化需要的字段。"""

    id: str  # 子问题唯一标识，如 q1
    question: str  # 原始子问题文本，可能包含 #qid# 占位符
    depends_on: list[str]  # 依赖的子问题 ID 列表
    purpose: str  # 该子问题需要获取的信息
    status: Literal["pending", "ready", "done", "skipped"]  # 调度状态 ，默认 pending
    # pending: 未开始；ready: 已开始；done: 已完成；skipped: 跳过(依赖未完成)
    # 说明：skipped 子问题的 depends_on 列表为空，且 status 为 skipped 时，该子问题的 effective_query 为空
    effective_query: (
        str  # 实际送进检索器的查询文本，有依赖时由 SDOM(子问题依赖优化模块) 改写
    )


class RagInput(TypedDict, total=False):
    """RAG 子图输入；包含要检索的问题和子问题列表。"""

    query: Required[str]  # 本次 RAG 检索要解决的问题
    unique_subqueries: list[str]  # 去重后用于检索的子问题列表
    max_subqueries: int  # 允许生成的子查询上限数量


class RagState(TypedDict, total=False):
    """RAG 子图状态；检索结果可以由并行节点追加。"""

    query: Required[str]  # 本次 RAG 检索要解决的问题
    unique_subqueries: list[str]  # 兼容旧路由的子查询文本列表
    sub_queries: list[SubQueryItem]  # 携带依赖关系的子问题，供调度与 SDOM 使用
    # 调度器每轮只挑一个子问题来处理，current_subquery 保存这一轮选中的项。
    current_subquery: SubQueryItem  # 当前正在处理的子问题，retrieve/optimize 都读它
    # 已完成或已跳过的子问题 ID；调度器用它判断某个子问题的依赖是否就绪。
    completed_ids: list[str]
    # 调度器告诉图下一步走哪个节点：optimize=做 SDOM 改写，retrieve=直接检索，aggregate=收尾。
    next_action: Literal["optimize", "retrieve", "aggregate"]
    # 供人工核对：SDOM 改写后的查询文本，键是子问题 ID。
    rewrites: dict[str, str]  # 子问题 ID 到改写后查询的映射
    retrieval_hits: Annotated[list[RetrievalHit], add]  # 并行检索的原始召回结果
    unique_documents: list[str]  # 聚合去重后的父文档内容列表
    max_subqueries: int  # 允许生成的子查询上限数量

class RagOutput(TypedDict, total=False):
    """RAG 子图输出；包含检索结果和重排结果。"""
    unique_documents: list[str]  # 聚合去重后的父文档内容列表


class ExecutorInput(TypedDict, total=False):
    """执行子图输入；字段名与 OverviewState 对齐，父图直接传入。"""

    steps: Required[list[PlanStep]]  # 当前有效的有序步骤表
    current_step_id: Required[str]  # 本次要执行的步骤 ID


class ExecutorPrivateState(TypedDict, total=False):
    """执行单个计划步骤时使用的私有状态。"""

    current_step: Required[PlanStep]  # 当前执行的计划步骤
    tool_calls: list[ToolCallRecord]  # 本步骤发起的工具调用
    tool_results: list[ToolResultRecord]  # 本步骤收到的工具结果
    evidence_candidates: list[Evidence]  # 从现场工具结果提取的证据
    knowledge_candidates: list[KnowledgeDocument]  # 本步骤检索到的排障知识
    step_result: StepResult | None  # 本步骤的最终执行结果
    max_tool_calls: int  # 本步骤允许的工具调用上限


class ExecutorState(ExecutorInput, ExecutorPrivateState, total=False):
    """执行子图内部状态，包含入口参数和节点执行期间使用的字段。"""


class ExecutorOutput(TypedDict, total=False):
    """执行子图输出，返回步骤结果、工具调用记录和现场证据。"""

    step_result: StepResult | None
    tool_calls: list[ToolCallRecord]
    tool_results: list[ToolResultRecord]
    evidence_candidates: list[Evidence]
