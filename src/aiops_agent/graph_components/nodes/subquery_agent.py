"""子问题拆解节点"""
from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import RagState


async def subquery_agent(
    state: RagState,
    model_client: ModelClient,
):
    """子问题拆解节点"""
    pass