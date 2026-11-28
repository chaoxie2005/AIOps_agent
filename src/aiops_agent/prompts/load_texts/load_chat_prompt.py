"""意图识别和普通聊天的提示词模板。"""

from langchain_core.prompts import ChatPromptTemplate

from aiops_agent.utils.prompt_loader import load_prompt


intent_agent_prompt = ChatPromptTemplate.from_messages(
    [("system", load_prompt("intent_prompt")), ("human", "{question}")]
)
chat_agent_prompt = ChatPromptTemplate.from_messages(
    [("system", load_prompt("chat_prompt")), ("human", "{question}")]
)
