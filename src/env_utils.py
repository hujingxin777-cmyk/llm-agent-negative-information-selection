from __future__ import annotations

"""环境变量与 API Key 获取工具。"""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env_file(env_path: Path | str) -> bool:
    """加载 .env 文件；存在返回 True，否则 False。"""
    env_path = Path(env_path)
    if env_path.exists():
        load_dotenv(env_path)
        return True
    return False


def get_api_key(api_style: str, key_name: str | None = None) -> str:
    """按供应商风格解析 API Key，并在缺失时抛出明确错误。"""
    if key_name:
        api_key = os.getenv(key_name)
        if not api_key and api_style == "zhipu":
            api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    elif api_style == "zhipu":
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    elif api_style == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv(f"{api_style.upper()}_API_KEY")

    if not api_key:
        raise ValueError(
            f"未找到 {api_style} 的 API Key，请设置环境变量 "
            f"{key_name or api_style.upper() + '_API_KEY'}"
        )
    return api_key
