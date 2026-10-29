from langgraph.types import Send
from aiops_agent.graph_components.states.state import RagState


def rag_retriever_router(state: RagState):
    """RAG 检索路由函数"""
    return [Send("retriever_node", {"query": q}) for q in state["unique_subqueries"]]