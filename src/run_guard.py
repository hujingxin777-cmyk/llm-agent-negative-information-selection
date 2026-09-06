from __future__ import annotations

"""Fail-fast protection for long-running model experiments."""

import time
from threading import Lock
from typing import Any, Mapping


class RunAbortError(RuntimeError):
    """Raised when a configured experiment safety condition is hit."""


class RunGuard:
    """Stop a run after exhausted model failures or prolonged lack of progress."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
        protection_cfg = (
            runtime_cfg.get("run_protection", {})
            if isinstance(runtime_cfg.get("run_protection", {}), Mapping)
            else {}
        )
        self.enabled = bool(protection_cfg.get("enabled", True))
        self.failed_row_limit = max(1, int(protection_cfg.get("failed_row_limit", 1)))
        self.no_progress_timeout_seconds = max(
            0.0,
            float(protection_cfg.get("no_progress_timeout_seconds", 300)),
        )
        self._lock = Lock()
        self._failed_rows = 0
        self._last_progress_at = time.monotonic()
        self._abort_reason: str | None = None

    def check(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._raise_if_aborted_unlocked()
            if (
                self.no_progress_timeout_seconds > 0
                and time.monotonic() - self._last_progress_at > self.no_progress_timeout_seconds
            ):
                self._abort_reason = (
                    "run protection stopped the experiment: no completed model row for "
                    f"{self.no_progress_timeout_seconds:.0f}s"
                )
                self._raise_if_aborted_unlocked()

    def record_result(self, success: bool, error_message: str | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._raise_if_aborted_unlocked()
            if success:
                self._failed_rows = 0
                self._last_progress_at = time.monotonic()
                return
            self._failed_rows += 1
            message = (error_message or "unknown model error").strip()
            if self._failed_rows >= self.failed_row_limit:
                self._abort_reason = (
                    "run protection stopped the experiment after "
                    f"{self._failed_rows} exhausted model failure(s): {message}"
                )
                self._raise_if_aborted_unlocked()

    def _raise_if_aborted_unlocked(self) -> None:
        if self._abort_reason:
            raise RunAbortError(self._abort_reason)
