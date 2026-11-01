"""聚合去重rag检索节点"""
from aiops_agent.graph_components.states.state import RagState, RetrievalHit


async def aggregation_node(state: RagState):
    """聚合去重rag检索节点"""
    retrieval_hits = state.get("retrieval_hits", [])
    unique_documents = {
        hit["document"].metadata["parent_id"]: hit["document"].metadata["parent_content"]
        for hit in retrieval_hits
    }
    return {"unique_documents": list(unique_documents.values())}
