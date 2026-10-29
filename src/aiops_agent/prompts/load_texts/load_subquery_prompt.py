"""子问题拆解agent提示词模板"""

from langchain_core.prompts import ChatPromptTemplate

from aiops_agent.utils.prompt_loader import load_prompt


subquery_agent_prompt = ChatPromptTemplate.from_messages(
    [("system", load_prompt("subquery_prompt")), ("human", "{question}")]
)
