"""重新规划节点提示词模板。"""

from langchain_core.prompts import ChatPromptTemplate

from aiops_agent.utils.prompt_loader import load_prompt


replan_agent_prompt = ChatPromptTemplate.from_messages(
    [("system", load_prompt("replan_prompt")), ("human", "{question}")]
)
