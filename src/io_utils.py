from __future__ import annotations

"""通用文件 I/O 工具：JSON/JSONL/CSV 读写辅助。"""

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


def ensure_dir(path: Path) -> None:
    """确保目录存在（幂等）。"""
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """覆盖写入 JSONL。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    """覆盖写入 CSV；空数据时仅跳过，不创建空表头。"""
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    """向 JSONL 追加一行。"""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def append_csv_row(path: Path, row: Mapping[str, object]) -> None:
    """向 CSV 追加一行；若文件为空则先写表头。"""
    ensure_dir(path.parent)
    fieldnames = list(row.keys())
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(row))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """覆盖写入格式化 JSON。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)
