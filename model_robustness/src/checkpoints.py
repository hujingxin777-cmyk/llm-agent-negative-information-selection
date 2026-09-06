from __future__ import annotations

"""Cell-level checkpoints for resumable static and P3 experiment runs."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from .io_utils import ensure_dir


_WRITE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"checkpoint value must be a dataclass or mapping: {type(value)!r}")


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash the effective config so incompatible checkpoints are never reused."""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class CellCheckpointStore:
    """Persist completed cells atomically and load only matching config snapshots."""

    def __init__(self, output_dir: Path, run_name: str, config: Mapping[str, Any]) -> None:
        self.config_hash = config_fingerprint(config)
        self.root = output_dir / "checkpoints" / f"{run_name}_{self.config_hash}"
        ensure_dir(self.root)

    def load(self, cell: Any) -> dict[str, Any] | None:
        path = self._cell_path(cell)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("config_hash") != self.config_hash:
            return None
        if payload.get("cell") != _as_mapping(cell):
            return None
        return payload

    def save(self, cell: Any, artifacts: Mapping[str, Any]) -> Path:
        path = self._cell_path(cell)
        payload = {
            "checkpoint_version": 1,
            "config_hash": self.config_hash,
            "completed_at": _utc_now(),
            "cell": _as_mapping(cell),
            "artifacts": dict(artifacts),
        }
        ensure_dir(path.parent)
        tmp_path = path.with_suffix(".tmp")
        with _WRITE_LOCK:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp_path.replace(path)
        return path

    def _cell_path(self, cell: Any) -> Path:
        payload = json.dumps(_as_mapping(cell), ensure_ascii=False, sort_keys=True, default=str)
        cell_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return self.root / f"{cell_hash}.json"
