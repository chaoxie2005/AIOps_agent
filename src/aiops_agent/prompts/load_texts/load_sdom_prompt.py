"""子问题依赖优化（SDOM）提示词模板。"""

from langchain_core.prompts import ChatPromptTemplate

from aiops_agent.utils.prompt_loader import load_prompt


sdom_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", load_prompt("sdom_prompt")),
        (
            "human",
            "当前子问题：{question}\n\n"
            "上游信息：\n{upstream_context}\n\n"
            "请把当前子问题改写成可独立检索的完整问题。",
        ),
    ]
)
