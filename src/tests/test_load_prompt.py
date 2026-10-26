from aiops_agent.utils.prompt_loader import load_prompt


def test_load_prompt():
    """测试 load_prompt 函数。"""
    prompt = load_prompt("subquery_prompt")
    assert prompt
    assert isinstance(prompt, str)