"""从 prompts 目录读取 Markdown 提示词。"""

from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "texts"


def load_prompt(name: str) -> str:
    """按名称读取 Markdown 提示词。

    Args:
        name: 不带 .md 后缀的提示词文件名。

    Returns:
        提示词文件的文本内容。

    Raises:
        ValueError: 名称不是有效的简单标识符。
        FileNotFoundError: 对应的提示词文件不存在。
    """
    if not name.isidentifier():
        raise ValueError(f"无效的提示词名称: {name!r}")

    path = _PROMPTS_DIR / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"提示词文件不存在: {path}") from exc
