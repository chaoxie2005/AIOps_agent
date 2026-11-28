"""验证节点温度不会修改共享模型或丢失模型配置。"""

import pytest

from aiops_agent.agent_components.llm import ModelClient, NODE_TEMPERATURES, client_for_node


@pytest.mark.parametrize("node,temperature", [
    ("intent", 0.0), ("subquery", 0.0), ("sdom", 0.0), ("summary", 0.0),
    ("plan", 0.1), ("replan", 0.1), ("execute", 0.1), ("chat", 0.5),
])
def test_node_model_preserves_configuration_and_isolation(node, temperature):
    original = ModelClient(
        model_name="test-model", api_key="test-key", base_url="http://localhost/v1",
        temperature=0.7, max_tokens=100,
    )
    node_client = client_for_node(original, node)
    params = node_client.model._get_invocation_params()

    assert NODE_TEMPERATURES[node] == temperature
    assert params["temperature"] == temperature
    assert params["model"] == "test-model"
    assert params["max_completion_tokens"] == 100
    assert original.model.temperature == 0.7
    assert node_client.model is not original.model
    assert node_client.input_token_limit == original.input_token_limit


def test_independent_clients_can_be_used_concurrently():
    original = ModelClient(
        model_name="test-model", api_key="test-key", base_url="http://localhost/v1"
    )
    intent = client_for_node(original, "intent")
    chat = client_for_node(original, "chat")
    assert intent.model.temperature == 0.0
    assert chat.model.temperature == 0.5
    assert original.model.temperature == 0.5
