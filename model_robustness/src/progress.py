from __future__ import annotations

"""实验进度追踪：终端进度条 + 磁盘快照。"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from tqdm import tqdm

from .io_utils import append_csv_row, append_jsonl, write_json


def _utc_now() -> str:
    """返回 UTC ISO 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


class ProgressTracker:
    """终端进度条 + 实时进度文件的轻量封装。"""

    def __init__(
        self,
        output_dir: Path,
        run_name: str,
        total_units: int,
        unit_label: str,
    ) -> None:
        self.progress_dir = output_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.total_units = total_units
        self.unit_label = unit_label
        self.completed_units = 0
        self.completed_rows = 0
        self.started_at = _utc_now()
        self.last_cell: dict[str, Any] | None = None
        self.progress_path = self.progress_dir / f"{run_name}_progress.json"
        self.observation_jsonl = self.progress_dir / f"{run_name}_observations.jsonl"
        self.observation_csv = self.progress_dir / f"{run_name}_observations.csv"
        self._lock = Lock()
        for path in [self.progress_path, self.observation_jsonl, self.observation_csv]:
            if path.exists():
                path.unlink()
        self.bar = tqdm(total=total_units, desc=run_name, unit=unit_label, dynamic_ncols=True)
        self._flush_progress(status="running")

    def set_current_cell(self, cell: Any) -> None:
        """更新当前执行单元（用于恢复与可观测性）。"""
        with self._lock:
            if is_dataclass(cell):
                self.last_cell = asdict(cell)
            elif isinstance(cell, Mapping):
                self.last_cell = dict(cell)
            else:
                self.last_cell = {"value": str(cell)}
            self._flush_progress(status="running")

    def record_observation(self, row: Mapping[str, object]) -> None:
        """记录单条观测并刷新进度快照。"""
        with self._lock:
            append_jsonl(self.observation_jsonl, row)
            append_csv_row(self.observation_csv, row)
            self.completed_rows += 1
            self.bar.set_postfix(rows=self.completed_rows, refresh=False)
            self._flush_progress(status="running")

    def advance(self, n: int = 1) -> None:
        """推进已完成单元计数。"""
        with self._lock:
            self.completed_units += n
            self.bar.update(n)
            self._flush_progress(status="running")

    def close(self, status: str = "completed") -> None:
        """关闭追踪器并落盘最终状态。"""
        with self._lock:
            self._flush_progress(status=status)
            self.bar.close()

    def _flush_progress(self, status: str) -> None:
        """将当前运行状态写入 JSON 快照。"""
        payload = {
            "run_name": self.run_name,
            "status": status,
            "unit_label": self.unit_label,
            "total_units": self.total_units,
            "completed_units": self.completed_units,
            "completed_rows": self.completed_rows,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "current_cell": self.last_cell,
        }
        write_json(self.progress_path, payload)
