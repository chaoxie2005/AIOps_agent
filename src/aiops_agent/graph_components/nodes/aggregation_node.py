"""聚合去重rag检索节点"""
from aiops_agent.graph_components.states.state import RagState, RetrievalHit


async def aggregation_node(state: RagState):
    """聚合去重rag检索节点"""
    retrieval_hits = state.get("retrieval_hits", [])
    print(retrieval_hits)
    # if not retrieval_hits:
    #     return {"retrieval_hits": []}
    # unique_documents = list(dict.fromkeys(retrieval_hits))
    # return {"unique_documents": unique_documents}

