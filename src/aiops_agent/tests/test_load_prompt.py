from aiops_agent.utils.prompt_loader import load_prompt


def test_load_prompt():
    """测试 load_prompt 函数。"""
    prompt = load_prompt("subquery_prompt")
    assert prompt
    assert isinstance(prompt, str)

def test_subquery_prompt_formats_messages():
    from aiops_agent.prompts.load_texts.load_subquery_prompt import subquery_agent_prompt

    messages = subquery_agent_prompt.format_messages(
        max_sub_questions=3, question="如何排查服务故障？"
    )
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert "2～3" in messages[0].content
    assert '"original_question"' in messages[0].content
    assert messages[1].type == "human"
    assert messages[1].content == "如何排查服务故障？"
