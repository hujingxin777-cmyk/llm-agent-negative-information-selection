from __future__ import annotations

"""模型调用与输出解析：OpenAI 兼容 API 封装。"""

import json
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Dict

import certifi
import httpx
from openai import OpenAI
from openai import RateLimitError

from .env_utils import get_api_key
from .io_utils import append_jsonl, ensure_dir
from .schemas import DecisionOutput, P3DecisionOutput


_NON_RETRYABLE_MODEL_ERROR_MARKERS = (
    "余额不足",
    "insufficient balance",
    "insufficient quota",
    "insufficient_quota",
    "billing",
)


def is_non_retryable_model_error(exc: Exception) -> bool:
    """Return true for billing failures that retries cannot resolve."""
    message = str(exc).lower()
    return any(marker in message for marker in _NON_RETRYABLE_MODEL_ERROR_MARKERS)


class LLMClient:
    """参考旧项目的 API 接法，统一封装 OpenAI 兼容接口。"""

    _semaphores: dict[str, BoundedSemaphore] = {}
    _semaphore_lock = Lock()
    _request_log_lock = Lock()

    def __init__(self, config: Dict[str, Any]) -> None:
        model_cfg = config["model"]
        self.model_cfg = model_cfg
        self.output_dir = Path(config.get("project", {}).get("output_dir", "demo/code/outputs"))
        self._request_lock = Lock()
        self._usage_lock = Lock()
        self._next_request_at = 0.0
        protection_cfg = config.get("runtime", {}).get("run_protection", {})
        self.max_total_tokens = int(protection_cfg.get("max_total_tokens", 0) or 0)
        self.price_per_1k_tokens = float(protection_cfg.get("price_per_1k_tokens", 0.003))
        self._cumulative_total_tokens = 0
        api_concurrency = int(
            model_cfg.get(
                "api_concurrency",
                config.get("runtime", {}).get("api_concurrency", config.get("runtime", {}).get("max_concurrency", 1)),
            )
        )
        semaphore_key = f"{model_cfg.get('api_style','default')}::{model_cfg.get('model_name','model')}::{api_concurrency}"
        with self._semaphore_lock:
            if semaphore_key not in self._semaphores:
                self._semaphores[semaphore_key] = BoundedSemaphore(max(1, api_concurrency))
            self._api_semaphore = self._semaphores[semaphore_key]
        self.api_key = get_api_key(
            api_style=model_cfg.get("api_style", model_cfg.get("provider", "openai")),
            key_name=model_cfg.get("api_key_env"),
        )
        verify = self._build_verify_option()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=model_cfg.get("api_base"),
            timeout=float(model_cfg.get("timeout_seconds", 120)),
            http_client=httpx.Client(
                verify=verify,
                timeout=httpx.Timeout(
                    connect=float(model_cfg.get("connect_timeout_seconds", 15)),
                    read=float(model_cfg.get("timeout_seconds", 120)),
                    write=float(model_cfg.get("timeout_seconds", 120)),
                    pool=float(model_cfg.get("pool_timeout_seconds", 30)),
                ),
            ),
        )

    def _build_verify_option(self) -> str | bool:
        """构造 SSL verify 参数（关闭/指定路径/certifi）。"""
        verify_mode = str(self.model_cfg.get("ssl_verify_mode", "certifi")).strip().lower()
        verify_path = self.model_cfg.get("ssl_verify_path")
        if verify_mode == "disabled":
            return False
        if verify_mode == "path" and verify_path:
            return str(verify_path)
        return certifi.where()

    def generate_text(self, system_prompt: str, user_prompt: str, request_meta: Dict[str, Any] | None = None) -> str:
        """执行一次带重试的模型调用，并记录请求日志。"""
        max_retries = int(self.model_cfg.get("max_retries", 0))
        retry_backoff_seconds = float(self.model_cfg.get("retry_backoff_seconds", 1))
        min_interval = float(self.model_cfg.get("min_request_interval_seconds", 0))
        jitter_seconds = float(self.model_cfg.get("request_jitter_seconds", 0))
        last_error: Exception | None = None
        request_id = uuid.uuid4().hex[:12]
        for attempt in range(max_retries + 1):
            with self._usage_lock:
                if self.max_total_tokens and self._cumulative_total_tokens >= self.max_total_tokens:
                    raise RuntimeError(
                        "token budget reached before request: "
                        f"{self._cumulative_total_tokens}/{self.max_total_tokens}"
                    )
            started_at = datetime.now(timezone.utc)
            started_at_mono = time.monotonic()
            self._log_request_event(
                request_id=request_id,
                phase="start",
                attempt=attempt + 1,
                started_at=started_at,
                elapsed_seconds=0.0,
                request_meta=request_meta,
            )
            try:
                with self._api_semaphore:
                    self._wait_for_slot(min_interval=min_interval, jitter_seconds=jitter_seconds)
                    request_kwargs = self._build_request_kwargs(system_prompt, user_prompt)
                    response = self.client.chat.completions.create(**request_kwargs)
                elapsed = time.monotonic() - started_at_mono
                usage = self._extract_usage(response)
                usage.update(self._extract_response_metadata(response))
                with self._usage_lock:
                    self._cumulative_total_tokens += int(usage["total_tokens"])
                    usage["cumulative_total_tokens"] = self._cumulative_total_tokens
                    usage["estimated_cost"] = round(
                        self._cumulative_total_tokens / 1000 * self.price_per_1k_tokens,
                        6,
                    )
                self._log_request_event(
                    request_id=request_id,
                    phase="success",
                    attempt=attempt + 1,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    request_meta=request_meta,
                    usage=usage,
                )
                return response.choices[0].message.content or ""
            except RateLimitError as exc:
                last_error = exc
                elapsed = time.monotonic() - started_at_mono
                self._log_request_event(
                    request_id=request_id,
                    phase="rate_limit",
                    attempt=attempt + 1,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    request_meta=request_meta,
                    error=exc,
                )
                if is_non_retryable_model_error(exc):
                    break
                if attempt >= max_retries:
                    break
                time.sleep(max(retry_backoff_seconds * (2 ** attempt), 15))
            except Exception as exc:
                last_error = exc
                elapsed = time.monotonic() - started_at_mono
                self._log_request_event(
                    request_id=request_id,
                    phase="error",
                    attempt=attempt + 1,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    request_meta=request_meta,
                    error=exc,
                )
                if attempt >= max_retries:
                    break
                time.sleep(retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(f"模型调用失败: {last_error}") from last_error

    def _build_request_kwargs(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """按模型能力构造兼容 Chat Completions 的请求参数。"""
        request_kwargs: Dict[str, Any] = {
            "model": self.model_cfg["model_name"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not bool(self.model_cfg.get("omit_temperature", False)):
            request_kwargs["temperature"] = float(self.model_cfg.get("temperature", 0.7))

        max_tokens = int(self.model_cfg.get("max_tokens", 2048))
        parameter = str(self.model_cfg.get("max_tokens_parameter", "max_tokens"))
        if parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"不支持的 token 上限参数: {parameter}")
        request_kwargs[parameter] = max_tokens

        if self.model_cfg.get("response_format"):
            request_kwargs["response_format"] = self.model_cfg["response_format"]
        if self.model_cfg.get("reasoning_effort"):
            request_kwargs["reasoning_effort"] = self.model_cfg["reasoning_effort"]
        if self.model_cfg.get("extra_body"):
            request_kwargs["extra_body"] = self.model_cfg["extra_body"]
        return request_kwargs

    def _wait_for_slot(self, min_interval: float, jitter_seconds: float) -> None:
        """按最小请求间隔进行串行节流。"""
        if min_interval <= 0 and jitter_seconds <= 0:
            return
        with self._request_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_request_at - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            jitter = random.uniform(0.0, jitter_seconds) if jitter_seconds > 0 else 0.0
            self._next_request_at = time.monotonic() + min_interval + jitter

    def _log_request_event(
        self,
        request_id: str,
        phase: str,
        attempt: int,
        started_at: datetime,
        elapsed_seconds: float,
        request_meta: Dict[str, Any] | None = None,
        error: Exception | None = None,
        usage: Dict[str, Any] | None = None,
    ) -> None:
        progress_dir = self.output_dir / "progress"
        ensure_dir(progress_dir)
        row: Dict[str, Any] = {
            "request_id": request_id,
            "phase": phase,
            "attempt": attempt,
            "started_at": started_at.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "model_name": self.model_cfg.get("model_name"),
            "api_base": self.model_cfg.get("api_base"),
            "ssl_verify_mode": self.model_cfg.get("ssl_verify_mode", "certifi"),
            "connect_timeout_seconds": float(self.model_cfg.get("connect_timeout_seconds", 15)),
            "timeout_seconds": float(self.model_cfg.get("timeout_seconds", 120)),
            "reasoning_effort": self.model_cfg.get("reasoning_effort"),
            "omit_temperature": bool(self.model_cfg.get("omit_temperature", False)),
            "max_tokens_parameter": self.model_cfg.get("max_tokens_parameter", "max_tokens"),
            "extra_body": self.model_cfg.get("extra_body"),
        }
        if request_meta:
            row.update(request_meta)
        if error is not None:
            row["error_type"] = type(error).__name__
            row["error_message"] = str(error)
        if usage:
            row.update(usage)
        with self._request_log_lock:
            append_jsonl(progress_dir / "model_request_log.jsonl", row)

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    @staticmethod
    def _extract_response_metadata(response: Any) -> Dict[str, Any]:
        """记录服务端实际模型标识，避免滚动别名造成复现歧义。"""
        return {
            "resolved_model_name": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        }


class LLMDecisionParser:
    """把模型原始文本解析并归一化为结构化决策。"""

    @staticmethod
    def parse_decision(
        response_text: str,
        report_mapping: Dict[str, str] | None = None,
    ) -> DecisionOutput:
        data = LLMDecisionParser._parse_json(response_text)
        return DecisionOutput(
            final_choice=LLMDecisionParser._normalize_final_choice(data, report_mapping),
            credible_choice=LLMDecisionParser._normalize_credible(data.get("credible_choice"), report_mapping),
            share_choice=LLMDecisionParser._normalize_ternary(data.get("share_choice"), report_mapping),
            comment_choice=LLMDecisionParser._normalize_comment(data.get("comment_choice"), report_mapping),
            like_choice=LLMDecisionParser._normalize_ternary(data.get("like_choice"), report_mapping),
            reason=LLMDecisionParser._normalize_reason(data.get("reason")),
            reason_signal=LLMDecisionParser._normalize_reason_signal(data.get("reason_signal")),
        )

    @staticmethod
    def parse_p3_decision(
        response_text: str,
        report_mapping: Dict[str, str] | None = None,
    ) -> P3DecisionOutput:
        data = LLMDecisionParser._parse_json(response_text)
        return P3DecisionOutput(
            final_choice=LLMDecisionParser._normalize_final_choice(data, report_mapping),
            credible_choice=LLMDecisionParser._normalize_credible(data.get("credible_choice"), report_mapping),
            share_choice=LLMDecisionParser._normalize_ternary(data.get("share_choice"), report_mapping),
            comment_choice=LLMDecisionParser._normalize_comment(data.get("comment_choice"), report_mapping),
            like_choice=LLMDecisionParser._normalize_ternary(data.get("like_choice"), report_mapping),
            reason=LLMDecisionParser._normalize_reason(data.get("reason")),
            reason_signal=LLMDecisionParser._normalize_reason_signal(data.get("reason_signal")),
            planned_action=LLMDecisionParser._normalize_action(data.get("planned_action")),
            public_comment=LLMDecisionParser._normalize_text(data.get("public_comment")),
            reply_target=LLMDecisionParser._normalize_text(data.get("reply_target")),
        )

    @staticmethod
    def _parse_json(response_text: str) -> Dict[str, Any]:
        """解析模型输出 JSON，失败时回退到正则抓取大括号块。"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not match:
                raise ValueError("模型输出中未找到可解析 JSON")
            return json.loads(match.group(0))

    @staticmethod
    def _normalize_binary_choice(
        value: Any,
        report_mapping: Dict[str, str] | None = None,
    ) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        aliases = {
            "positive": "positive",
            "negative": "negative",
            "report_1": (report_mapping or {}).get("report_1"),
            "report1": (report_mapping or {}).get("report_1"),
            "报道1": (report_mapping or {}).get("report_1"),
            "report_2": (report_mapping or {}).get("report_2"),
            "report2": (report_mapping or {}).get("report_2"),
            "报道2": (report_mapping or {}).get("report_2"),
        }
        return aliases.get(text)

    @staticmethod
    def _normalize_credible(value: Any, report_mapping: Dict[str, str] | None = None) -> str:
        if value in {"neutral", "中立"}:
            return "neutral"
        choice = LLMDecisionParser._normalize_binary_choice(value, report_mapping)
        return choice if choice in {"positive", "negative"} else "neutral"

    @staticmethod
    def _normalize_final_choice(data: Dict[str, Any], report_mapping: Dict[str, str] | None = None) -> str:
        explicit = LLMDecisionParser._normalize_binary_choice(data.get("final_choice"), report_mapping)
        if explicit in {"positive", "negative"}:
            return explicit
        credible = LLMDecisionParser._normalize_binary_choice(data.get("credible_choice"), report_mapping)
        if credible in {"positive", "negative"}:
            return credible
        comment = data.get("comment_choice")
        if comment in {"support_negative", "support_report_2", "support_report2", "支持报道2"}:
            return "negative"
        if comment in {"support_positive", "support_report_1", "support_report1", "支持报道1"}:
            return "positive"
        share = LLMDecisionParser._normalize_binary_choice(data.get("share_choice"), report_mapping)
        if share in {"positive", "negative"}:
            return share
        like = LLMDecisionParser._normalize_binary_choice(data.get("like_choice"), report_mapping)
        if like in {"positive", "negative"}:
            return like
        return "positive"

    @staticmethod
    def _normalize_ternary(value: Any, report_mapping: Dict[str, str] | None = None) -> str:
        if value in {"none", "无", "不选"}:
            return "none"
        choice = LLMDecisionParser._normalize_binary_choice(value, report_mapping)
        return choice if choice in {"positive", "negative"} else "none"

    @staticmethod
    def _normalize_comment(value: Any, report_mapping: Dict[str, str] | None = None) -> str:
        if value in {"neutral", "中立"}:
            return "neutral"
        text = str(value).strip().lower() if value is not None else ""
        positive = (report_mapping or {}).get("report_1") == "positive"
        aliases = {
            "support_positive": "support_positive",
            "support_negative": "support_negative",
            "support_report_1": "support_positive" if positive else "support_negative",
            "support_report1": "support_positive" if positive else "support_negative",
            "支持报道1": "support_positive" if positive else "support_negative",
            "support_report_2": "support_negative" if positive else "support_positive",
            "support_report2": "support_negative" if positive else "support_positive",
            "支持报道2": "support_negative" if positive else "support_positive",
        }
        return aliases.get(text, "neutral")

    @staticmethod
    def _normalize_action(value: Any) -> str:
        allowed = {"post_comment", "reply", "like", "repost", "observe"}
        return value if value in allowed else "observe"

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_reason(value: Any) -> str | None:
        text = LLMDecisionParser._normalize_text(value)
        if text is None:
            return None
        return text[:200]

    @staticmethod
    def _normalize_reason_signal(value: Any) -> str | None:
        allowed = {
            "official_reassurance",
            "personal_experience",
            "queue_pressure",
            "recovery_speed",
            "risk_underestimation",
            "balanced_evidence",
            "social_proof",
            "thread_majority",
            "unclear",
        }
        if value is None:
            return None
        text = str(value).strip()
        return text if text in allowed else "unclear"
