"""规划节点提示词模板"""

from langchain_core.prompts import ChatPromptTemplate

from aiops_agent.utils.prompt_loader import load_prompt

plan_agent_prompt = ChatPromptTemplate.from_messages(
    [("system", load_prompt("plan_prompt")), ("human", "{question}")]
)
