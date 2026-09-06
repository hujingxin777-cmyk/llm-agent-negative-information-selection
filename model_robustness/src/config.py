from __future__ import annotations

"""实验配置加载与合并。"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """加载单个 YAML 或目录下多 YAML（按文件名排序深合并）。"""
    if path.is_dir():
        return _load_config_dir(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    extends = data.pop("extends", None)
    if not extends:
        return data
    base_path = (path.parent / str(extends)).resolve()
    if base_path == path.resolve():
        raise ValueError(f"配置不能继承自身: {path}")
    return _deep_merge(load_config(base_path), data)


def _load_config_dir(path: Path) -> Dict[str, Any]:
    """读取目录中所有 YAML，并按层级递归覆盖。"""
    config: Dict[str, Any] = {}
    for file_path in sorted(path.glob("*.yaml")):
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = _deep_merge(config, data)
    return config


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """递归深合并：字典对字典递归，其它类型以后者覆盖前者。"""
    merged = dict(base)
    for key, value in incoming.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
