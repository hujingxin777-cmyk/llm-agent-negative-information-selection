from __future__ import annotations

"""P3 entry-state initialization from P2 comment text.

The initializer reads the P2 social comments as pre-discussion background, strips
like counts, asks the model once for three persona prototypes, and then maps
those prototypes back to individual P3 agents with small deterministic noise.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Mapping

from .environments import get_social_block
from .llm_client import is_non_retryable_model_error
from .prompts import build_system_message
from .schemas import AgentDynamicState, DiscussionAgent, ExperimentCell


_CACHE_LOCK = Lock()
_CACHE_CONDITION = Condition(_CACHE_LOCK)
_INFLIGHT_CACHE_KEYS: set[str] = set()
_LIKE_PATTERN = re.compile(r"👍\s*\d+\s*")
_LABEL_PATTERN = re.compile(r"^\s*【([^】]+)】\s*")


@dataclass(frozen=True)
class PrototypeState:
    persona_name: str
    latent_neg_tendency: float
    risk_arousal: float
    engagement_tendency: float
    last_final_choice: str
    reason: str
    initialization_method: str
    initialization_cached: bool = False


PERSONA_PROTOTYPE_DESCRIPTIONS = {
    "risk_sensitive": "更容易把恢复拖长、复诊、班级传播和资源压力视为早期风险信号；可控线索充分时也会降低担忧。",
    "authority_trusting": "更信任医生、医院、学校和正式应对线索；当个体经验与可核验事实反复印证时也会提高警觉。",
    "ordinary_user": "普通社交平台用户，不预设固定立场，会被更显眼、更具体的信息打动，但通常需要多个具体线索相互补充才会转向强烈判断。",
}

BASELINE_OPEN_PROTOTYPE_DESCRIPTION = (
    "主实验普通用户总体原型：不预设固定立场，会受讨论前材料影响；"
    "如果材料集中呈现风险经历，会提高风险警觉；如果材料集中呈现可控和恢复，会降低风险警觉。"
    "当材料同时包含恢复慢、复诊、反复观察和规范处理线索时，普通用户可以保持观望，"
    "但观望不等于低风险或完全可控。"
)


def strip_comment_likes(text: str) -> str:
    """Remove like counts and engineered labels from warm-start material."""
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        body = _LABEL_PATTERN.sub("", line)
        body = _LIKE_PATTERN.sub("", body).strip()
        if body:
            lines.append(body)
    return "\n".join(lines)


def _clip(value: object, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = (lower + upper) / 2
    return max(lower, min(upper, parsed))


def _stable_noise(seed: str, scale: float) -> float:
    if scale <= 0:
        return 0.0
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 2 * scale


def _choice_from_latent(latent: float) -> str:
    if latent > 0.54:
        return "negative"
    if latent < 0.46:
        return "positive"
    return "neutral"


class P2PrototypeInitializer:
    """Initialize persona prototypes from stripped P2 comment text."""

    def __init__(self, config: dict[str, Any], client: Any) -> None:
        self.config = config
        self.client = client
        self.p3_cfg = config.get("p3", {}) if isinstance(config.get("p3", {}), dict) else {}
        init_cfg = self.p3_cfg.get("p2_prototype_initialization", {})
        self.init_cfg = init_cfg if isinstance(init_cfg, dict) else {}

    def enabled(self) -> bool:
        return bool(self.p3_cfg.get("inherit_p2_state", False)) and bool(self.init_cfg.get("enabled", False))

    def initialize(
        self,
        cell: ExperimentCell,
        scenario: Mapping[str, Any],
        output_dir: Path,
    ) -> dict[str, PrototypeState]:
        if not self.enabled():
            return self._neutral_prototypes(cached=False, method="disabled")

        social_block = get_social_block(
            self.config,
            "high_interaction",
            cell.chat_atmosphere,
            cell.scenario_name,
        )
        p2_comment_text = (
            strip_comment_likes(social_block)
            if bool(self.init_cfg.get("strip_likes", True))
            else str(social_block or "")
        )
        prompt_version = str(self.init_cfg.get("prompt_version", "p2_comment_text_only_v1"))
        model_name = str(self.config.get("model", {}).get("model_name", "unknown_model"))
        cache_key = self._cache_key(cell, scenario, p2_comment_text, prompt_version, model_name)
        cache_path = output_dir / "cache" / "p3_prototype_initialization.json"
        cache_enabled = bool(self.init_cfg.get("cache_enabled", True))
        inflight_key = f"{cache_path.resolve()}::{cache_key}"

        if cache_enabled:
            with _CACHE_CONDITION:
                while True:
                    cached = self._read_cache_unlocked(cache_path).get(cache_key)
                    if isinstance(cached, dict):
                        return self._states_from_payload(cached, cached=True)
                    if inflight_key not in _INFLIGHT_CACHE_KEYS:
                        _INFLIGHT_CACHE_KEYS.add(inflight_key)
                        break
                    _CACHE_CONDITION.wait()

        payload: dict[str, Any]
        try:
            method = str(self.init_cfg.get("method", "llm_once")).lower()
            if method == "llm_once":
                try:
                    payload = self._run_llm(cell=cell, scenario=scenario, p2_comment_text=p2_comment_text)
                    payload["_prototype_initialization_method"] = "llm_once"
                except Exception as exc:
                    if is_non_retryable_model_error(exc):
                        raise
                    if not bool(self.init_cfg.get("fallback_to_rule", True)):
                        raise
                    payload = self._fallback_payload(cell, reason=f"rule_fallback_after_llm_error: {exc}")
            else:
                payload = self._fallback_payload(cell, reason=f"rule_fallback_method={method}")

            payload = self._apply_baseline_open_guardrail(payload, cell)
            payload = self._apply_persona_gradient_guardrail(payload, cell)
            states = self._states_from_payload(payload, cached=False)
            if cache_enabled:
                serializable = {
                    persona: {
                        "latent_neg_tendency": state.latent_neg_tendency,
                        "risk_arousal": state.risk_arousal,
                        "engagement_tendency": state.engagement_tendency,
                        "last_final_choice": state.last_final_choice,
                        "reason": state.reason,
                    }
                    for persona, state in states.items()
                }
                serializable["_prototype_initialization_method"] = next(iter(states.values())).initialization_method
                with _CACHE_CONDITION:
                    self._write_cache_unlocked(cache_path, cache_key, serializable)
            return states
        finally:
            if cache_enabled:
                with _CACHE_CONDITION:
                    _INFLIGHT_CACHE_KEYS.discard(inflight_key)
                    _CACHE_CONDITION.notify_all()

    def apply_to_agents(
        self,
        prototypes: Mapping[str, PrototypeState],
        cell: ExperimentCell,
        agents: list[DiscussionAgent],
        agent_states: dict[str, AgentDynamicState],
    ) -> dict[str, dict[str, object]]:
        bounds = self._value_bounds()
        noise_cfg = self.init_cfg.get("individual_noise", {})
        noise_cfg = noise_cfg if isinstance(noise_cfg, dict) else {}
        latent_scale = float(noise_cfg.get("latent_scale", 0.03))
        risk_scale = float(noise_cfg.get("risk_scale", 0.03))
        engagement_scale = float(noise_cfg.get("engagement_scale", 0.02))
        fallback = prototypes.get("ordinary_user") or next(iter(prototypes.values()))
        meta_by_persona: dict[str, dict[str, object]] = {}
        for agent in agents:
            prototype = prototypes.get(agent.persona_name, fallback)
            state = agent_states[agent.agent_id]
            latent = _clip(
                prototype.latent_neg_tendency
                + _stable_noise(f"{cell.scenario_name}|{cell.repeat_index}|{agent.agent_id}|latent", latent_scale),
                *bounds["latent_neg_tendency"],
            )
            risk = _clip(
                prototype.risk_arousal
                + _stable_noise(f"{cell.scenario_name}|{cell.repeat_index}|{agent.agent_id}|risk", risk_scale),
                *bounds["risk_arousal"],
            )
            engagement = _clip(
                prototype.engagement_tendency
                + _stable_noise(f"{cell.scenario_name}|{cell.repeat_index}|{agent.agent_id}|engagement", engagement_scale),
                *bounds["engagement_tendency"],
            )
            state.latent_neg_tendency = latent
            state.risk_arousal = risk
            state.engagement_tendency = engagement
            state.last_final_choice = _choice_from_latent(latent)
            state.last_choice = state.last_final_choice
            state.recent_choice_history = [state.last_final_choice]
            state.last_reason = prototype.reason
            meta_by_persona[agent.persona_name] = {
                "prototype_initialization_method": prototype.initialization_method,
                "prototype_initialization_cached": prototype.initialization_cached,
                "prototype_initialization_reason": prototype.reason,
            }
        return meta_by_persona

    def _run_llm(self, cell: ExperimentCell, scenario: Mapping[str, Any], p2_comment_text: str) -> dict[str, Any]:
        if cell.persona_name in {"baseline_open", "main_baseline_neutral"}:
            prototype_name = cell.persona_name
            prototype_description = (
                BASELINE_OPEN_PROTOTYPE_DESCRIPTION
                if prototype_name == "baseline_open"
                else "主实验中性普通用户总体原型：不预设风险或可控立场，比较风险线索与可控线索的具体性、代表性和一致性；少数个案不自动外推为整体。"
            )
            prototype_instruction = (
                f"请只估计主实验普通用户总体原型 {prototype_name}。不要拆分成风险敏感、权威信任或其他人格类型。\n"
                f"- {prototype_name}: {prototype_description}\n\n"
                "JSON 格式如下：\n"
                "{\n"
                f'  "{prototype_name}": {{"latent_neg_tendency": 0.0, "risk_arousal": 0.0, "engagement_tendency": 0.0, "last_final_choice": "positive|neutral|negative", "reason": ""}}\n'
                "}"
            )
        else:
            prototype_instruction = (
                "三类原型说明：\n"
                f"- risk_sensitive: {PERSONA_PROTOTYPE_DESCRIPTIONS['risk_sensitive']}\n"
                f"- authority_trusting: {PERSONA_PROTOTYPE_DESCRIPTIONS['authority_trusting']}\n"
                f"- ordinary_user: {PERSONA_PROTOTYPE_DESCRIPTIONS['ordinary_user']}\n\n"
                "请分别估计三类原型在看到同一批讨论前材料后的私下状态。"
                "不要套用固定排序，也不要把人格标签当作最终答案；"
                "请根据材料文本、报道叙事和每类原型的信息加工倾向给出数值。\n\n"
                "JSON 格式如下：\n"
                "{\n"
                '  "risk_sensitive": {"latent_neg_tendency": 0.0, "risk_arousal": 0.0, "engagement_tendency": 0.0, "last_final_choice": "positive|neutral|negative", "reason": ""},\n'
                '  "authority_trusting": {"latent_neg_tendency": 0.0, "risk_arousal": 0.0, "engagement_tendency": 0.0, "last_final_choice": "positive|neutral|negative", "reason": ""},\n'
                '  "ordinary_user": {"latent_neg_tendency": 0.0, "risk_arousal": 0.0, "engagement_tendency": 0.0, "last_final_choice": "positive|neutral|negative", "reason": ""}\n'
                "}"
            )
        prompt = (
            "你是社交仿真实验中的初始化器。请根据疾病议题背景、两种相互竞争的报道叙事、"
            "以及进入当前讨论前看到的评论材料，估计 P3 开始前的私下状态。\n"
            "注意：这些内容是进入论坛前已经读到的背景评论；同一批评论也可能作为 P3 初始只读评论可见，"
            "但它们不是可回复、点赞或转发的互动目标。文本中已去除点赞热度和工程标签。\n\n"
            f"背景信息：{scenario.get('background', '')}\n\n"
            f"偏可控/恢复较快叙事：{scenario.get('positive_narrative', '')}\n\n"
            f"偏风险/恢复拖长叙事：{scenario.get('negative_narrative', '')}\n\n"
            "讨论前评论材料：\n"
            f"{p2_comment_text or '无'}\n\n"
            f"{prototype_instruction}\n\n"
            "请输出 JSON 对象。数值含义：latent_neg_tendency 越高越偏向风险叙事；"
            "risk_arousal 越高表示风险唤起越强；engagement_tendency 越高表示越可能被激活参与。"
            "请严格使用三值映射：latent_neg_tendency < 0.46 表示 positive/偏可控，"
            "0.46 到 0.54 表示 neutral/观望，> 0.54 表示 negative/偏风险。"
            "不要因为两种叙事同时存在就机械输出 neutral。请先比较讨论前评论中哪一类线索更具体、"
            "更连续、更容易影响普通用户；只有两边确实接近时才使用 neutral。"
            "注意区分“最终态度”和“风险唤起”：last_final_choice 为 neutral 时，"
            "risk_arousal 仍可体现对恢复慢、复诊、夜间反复、班级请假或照护压力等线索的警觉；"
            "不要因为材料里出现规范治疗、按医嘱、逐步恢复等可控线索，就自动把 risk_arousal 压到很低。"
            "如果 reason 中写的是“谨慎观察”“继续观察”“平衡观点”“不能简单判断”，"
            "且最终态度确实无法判断，last_final_choice 应为 neutral，latent_neg_tendency 应落在 0.46 到 0.54。"
            "权威信任型也应根据正式线索与多条独立风险经验的相互印证程度形成相对判断，"
            "不要因为人格标签机械输出固定区间。"
            "不要输出任何实验条件标签、氛围标签或工程变量名。"
        )
        raw = self.client.generate_text(
            build_system_message(""),
            prompt,
            request_meta={"runner": "p3_prototype_initialization"},
        )
        return self._parse_json_object(raw)

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("prototype initialization response is not a JSON object")
        return payload

    def _apply_baseline_open_guardrail(
        self,
        payload: dict[str, Any],
        cell: ExperimentCell,
    ) -> dict[str, Any]:
        guard_cfg = self.init_cfg.get("baseline_open_guardrail", {})
        guard_cfg = guard_cfg if isinstance(guard_cfg, Mapping) else {}
        if not bool(guard_cfg.get("enabled", False)):
            return payload
        if cell.persona_name not in {"baseline_open", "main_baseline_neutral"}:
            return payload
        item = payload.get(cell.persona_name)
        if not isinstance(item, dict):
            return payload
        if cell.chat_atmosphere == "negative_leaning":
            self._raise_baseline_floor(
                item,
                latent_floor=float(guard_cfg.get("negative_latent_floor", 0.52)),
                risk_floor=float(guard_cfg.get("negative_risk_floor", 0.50)),
            )
        elif cell.chat_atmosphere == "positive_leaning":
            self._bound_baseline_range(
                item,
                latent_floor=float(guard_cfg.get("positive_latent_floor", 0.43)),
                latent_ceiling=float(guard_cfg.get("positive_latent_ceiling", 0.47)),
                risk_floor=float(guard_cfg.get("positive_risk_floor", 0.40)),
                risk_ceiling=float(guard_cfg.get("positive_risk_ceiling", 0.46)),
            )
        elif cell.chat_atmosphere == "none":
            self._bound_baseline_neutral_range(item)
        return payload

    def _apply_persona_gradient_guardrail(
        self,
        payload: dict[str, Any],
        cell: ExperimentCell,
    ) -> dict[str, Any]:
        guard_cfg = self.init_cfg.get("persona_gradient_guardrail", {})
        guard_cfg = guard_cfg if isinstance(guard_cfg, Mapping) else {}
        if not bool(guard_cfg.get("enabled", True)):
            return payload
        if cell.persona_name in {"baseline_open", "main_baseline_neutral"}:
            return payload
        required = ["risk_sensitive", "ordinary_user", "authority_trusting"]
        if not all(isinstance(payload.get(name), dict) for name in required):
            return payload

        min_gap = float(guard_cfg.get("min_latent_gap", 0.08))
        min_risk_gap = float(guard_cfg.get("min_risk_gap", 0.06))
        anchors = {
            "positive_leaning": {
                "risk_floor": 0.50,
                "ordinary_floor": 0.42,
                "ordinary_ceiling": 0.50,
                "authority_ceiling": 0.40,
            },
            "negative_leaning": {
                "risk_floor": 0.62,
                "ordinary_floor": 0.52,
                "ordinary_ceiling": 0.62,
                "authority_ceiling": 0.54,
            },
            "none": {
                "risk_floor": 0.56,
                "ordinary_floor": 0.48,
                "ordinary_ceiling": 0.54,
                "authority_ceiling": 0.44,
            },
        }
        anchor = anchors.get(cell.chat_atmosphere, anchors["none"])
        bounds = self._value_bounds()

        risk = payload["risk_sensitive"]
        ordinary = payload["ordinary_user"]
        authority = payload["authority_trusting"]

        ordinary_latent = _clip(
            ordinary.get("latent_neg_tendency", 0.5),
            float(anchor["ordinary_floor"]),
            float(anchor["ordinary_ceiling"]),
        )
        authority_latent = min(
            _clip(authority.get("latent_neg_tendency", 0.5), *bounds["latent_neg_tendency"]),
            ordinary_latent - min_gap,
            float(anchor["authority_ceiling"]),
        )
        risk_latent = max(
            _clip(risk.get("latent_neg_tendency", 0.5), *bounds["latent_neg_tendency"]),
            ordinary_latent + min_gap,
            float(anchor["risk_floor"]),
        )
        authority_latent = _clip(authority_latent, *bounds["latent_neg_tendency"])
        risk_latent = _clip(risk_latent, *bounds["latent_neg_tendency"])

        ordinary_risk = _clip(ordinary.get("risk_arousal", ordinary_latent), *bounds["risk_arousal"])
        authority_risk = min(
            _clip(authority.get("risk_arousal", authority_latent), *bounds["risk_arousal"]),
            ordinary_risk - min_risk_gap,
        )
        risk_risk = max(
            _clip(risk.get("risk_arousal", risk_latent), *bounds["risk_arousal"]),
            ordinary_risk + min_risk_gap,
        )

        self._set_guarded_state(risk, risk_latent, risk_risk, "按可选梯度配置调整风险敏感原型。")
        self._set_guarded_state(ordinary, ordinary_latent, ordinary_risk, "按可选梯度配置调整普通用户原型。")
        self._set_guarded_state(authority, authority_latent, authority_risk, "按可选梯度配置调整权威信任原型。")
        return payload

    @staticmethod
    def _set_guarded_state(item: dict[str, Any], latent: float, risk: float, note: str) -> None:
        item["latent_neg_tendency"] = latent
        item["risk_arousal"] = _clip(risk, 0.20, 0.80)
        item["last_final_choice"] = _choice_from_latent(latent)
        reason = str(item.get("reason") or "").strip()
        suffix = f"初始化人格梯度校准：{note}"
        item["reason"] = f"{reason}；{suffix}" if reason else suffix

    @staticmethod
    def _raise_baseline_floor(item: dict[str, Any], latent_floor: float, risk_floor: float) -> None:
        item["latent_neg_tendency"] = max(_clip(item.get("latent_neg_tendency", 0.5), 0.0, 1.0), latent_floor)
        item["risk_arousal"] = max(_clip(item.get("risk_arousal", 0.5), 0.0, 1.0), risk_floor)
        item["last_final_choice"] = _choice_from_latent(float(item["latent_neg_tendency"]))
        reason = str(item.get("reason") or "")
        item["reason"] = (
            f"{reason}；初始化规则校准：按可选 baseline 配置提高风险线索占优时的普通用户风险警觉。"
            if reason
            else "初始化规则校准：按可选 baseline 配置提高风险线索占优时的普通用户风险警觉。"
        )

    @staticmethod
    def _bound_baseline_range(
        item: dict[str, Any],
        latent_floor: float,
        latent_ceiling: float,
        risk_floor: float,
        risk_ceiling: float,
    ) -> None:
        item["latent_neg_tendency"] = min(
            max(_clip(item.get("latent_neg_tendency", 0.5), 0.0, 1.0), latent_floor),
            latent_ceiling,
        )
        item["risk_arousal"] = min(max(_clip(item.get("risk_arousal", 0.5), 0.0, 1.0), risk_floor), risk_ceiling)
        item["last_final_choice"] = _choice_from_latent(float(item["latent_neg_tendency"]))
        reason = str(item.get("reason") or "")
        item["reason"] = (
            f"{reason}；初始化规则校准：评论区可控/恢复线索占优，普通用户总体原型保持偏正但不归零。"
            if reason
            else "初始化规则校准：评论区可控/恢复线索占优，普通用户总体原型保持偏正但不归零。"
        )

    @staticmethod
    def _bound_baseline_neutral_range(item: dict[str, Any]) -> None:
        item["latent_neg_tendency"] = min(
            max(_clip(item.get("latent_neg_tendency", 0.5), 0.0, 1.0), 0.49),
            0.52,
        )
        item["risk_arousal"] = min(
            max(_clip(item.get("risk_arousal", 0.5), 0.0, 1.0), 0.47),
            0.53,
        )
        item["last_final_choice"] = _choice_from_latent(float(item["latent_neg_tendency"]))
        reason = str(item.get("reason") or "")
        item["reason"] = (
            f"{reason}；初始化规则校准：平衡氛围下普通用户总体原型应保持中性附近，避免被单次初始化拉到极端正向或极端负向。"
            if reason
            else "初始化规则校准：平衡氛围下普通用户总体原型应保持中性附近，避免被单次初始化拉到极端正向或极端负向。"
        )

    def _states_from_payload(self, payload: Mapping[str, Any], cached: bool) -> dict[str, PrototypeState]:
        bounds = self._value_bounds()
        method = str(payload.get("_prototype_initialization_method", "llm_once"))
        states: dict[str, PrototypeState] = {}
        persona_names = list(PERSONA_PROTOTYPE_DESCRIPTIONS)
        for single_persona_name in ("baseline_open", "main_baseline_neutral"):
            if single_persona_name in payload:
                persona_names.append(single_persona_name)
        for persona_name in persona_names:
            item = payload.get(persona_name, {})
            if not isinstance(item, Mapping):
                item = {}
            latent = _clip(item.get("latent_neg_tendency", 0.5), *bounds["latent_neg_tendency"])
            risk = _clip(item.get("risk_arousal", 0.5), *bounds["risk_arousal"])
            engagement = _clip(item.get("engagement_tendency", 0.5), *bounds["engagement_tendency"])
            choice = str(item.get("last_final_choice") or _choice_from_latent(latent)).strip().lower()
            if choice not in {"positive", "neutral", "negative"}:
                choice = _choice_from_latent(latent)
            states[persona_name] = PrototypeState(
                persona_name=persona_name,
                latent_neg_tendency=latent,
                risk_arousal=risk,
                engagement_tendency=engagement,
                last_final_choice=choice,
                reason=str(item.get("reason") or method),
                initialization_method=method,
                initialization_cached=cached,
            )
        return states

    def _fallback_payload(self, cell: ExperimentCell, reason: str) -> dict[str, Any]:
        anchor_name = self._fallback_anchor_name(cell.chat_atmosphere)
        anchors = self.init_cfg.get("fallback_anchor_without_likes", {})
        anchors = anchors if isinstance(anchors, Mapping) else {}
        base = anchors.get(anchor_name, {})
        if not isinstance(base, Mapping):
            base = {"latent_neg_tendency": 0.5, "risk_arousal": 0.5, "engagement_tendency": 0.5}
        payload: dict[str, Any] = {"_prototype_initialization_method": "rule_fallback"}
        if cell.persona_name in {"baseline_open", "main_baseline_neutral"}:
            latent = float(base.get("latent_neg_tendency", 0.5))
            risk = float(base.get("risk_arousal", 0.5))
            engagement = float(base.get("engagement_tendency", 0.5))
            payload[cell.persona_name] = {
                "latent_neg_tendency": latent,
                "risk_arousal": risk,
                "engagement_tendency": engagement,
                "last_final_choice": _choice_from_latent(latent),
                "reason": "模型初始化不可用时，根据讨论前评论文本与材料方向生成的规则兜底主实验普通原型。",
            }
            return payload
        adjustments = {
            "risk_sensitive": {"latent": 0.07, "risk": 0.07, "engagement": 0.02},
            "authority_trusting": {"latent": -0.06, "risk": -0.05, "engagement": -0.01},
            "ordinary_user": {"latent": 0.0, "risk": 0.0, "engagement": 0.0},
        }
        for persona_name, adjustment in adjustments.items():
            latent = float(base.get("latent_neg_tendency", 0.5)) + adjustment["latent"]
            risk = float(base.get("risk_arousal", 0.5)) + adjustment["risk"]
            engagement = float(base.get("engagement_tendency", 0.5)) + adjustment["engagement"]
            payload[persona_name] = {
                "latent_neg_tendency": latent,
                "risk_arousal": risk,
                "engagement_tendency": engagement,
                "last_final_choice": _choice_from_latent(latent),
                "reason": "模型初始化不可用时，根据讨论前评论文本与材料方向生成的规则兜底初始状态。",
            }
        return payload

    @staticmethod
    def _fallback_anchor_name(chat_atmosphere: str) -> str:
        if chat_atmosphere == "negative_leaning":
            return "negative_like"
        if chat_atmosphere == "positive_leaning":
            return "positive_like"
        return "neutral_like"

    def _neutral_prototypes(self, cached: bool, method: str) -> dict[str, PrototypeState]:
        return {
            persona_name: PrototypeState(
                persona_name=persona_name,
                latent_neg_tendency=0.5,
                risk_arousal=0.5,
                engagement_tendency=0.5,
                last_final_choice="neutral",
                reason=method,
                initialization_method=method,
                initialization_cached=cached,
            )
            for persona_name in PERSONA_PROTOTYPE_DESCRIPTIONS
        }

    def _value_bounds(self) -> dict[str, tuple[float, float]]:
        raw = self.init_cfg.get("value_bounds", {})
        raw = raw if isinstance(raw, Mapping) else {}
        defaults = {
            "latent_neg_tendency": (0.20, 0.80),
            "risk_arousal": (0.20, 0.80),
            "engagement_tendency": (0.30, 0.70),
        }
        bounds: dict[str, tuple[float, float]] = {}
        for field, default in defaults.items():
            value = raw.get(field, default)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                bounds[field] = (float(value[0]), float(value[1]))
            else:
                bounds[field] = default
        return bounds

    def _cache_key(
        self,
        cell: ExperimentCell,
        scenario: Mapping[str, Any],
        p2_comment_text: str,
        prompt_version: str,
        model_name: str,
    ) -> str:
        material = {
            "experiment_family": cell.experiment_family,
            "scenario_name": cell.scenario_name,
            "chat_atmosphere": cell.chat_atmosphere,
            "scenario_background_hash": self._hash_text(str(scenario.get("background", ""))),
            "p2_comment_text_hash": self._hash_text(p2_comment_text),
            "persona_set": sorted(PERSONA_PROTOTYPE_DESCRIPTIONS),
            "prompt_version": prompt_version,
            "model_name": model_name,
        }
        return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any]:
        with _CACHE_LOCK:
            return P2PrototypeInitializer._read_cache_unlocked(path)

    @staticmethod
    def _read_cache_unlocked(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_cache(path: Path, key: str, payload: Mapping[str, Any]) -> None:
        with _CACHE_LOCK:
            P2PrototypeInitializer._write_cache_unlocked(path, key, payload)

    @staticmethod
    def _write_cache_unlocked(path: Path, key: str, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = P2PrototypeInitializer._read_cache_unlocked(path)
        cache[key] = dict(payload)
        with path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _legacy_cache_write_removed(path: Path, key: str, payload: Mapping[str, Any]) -> None:
        # Kept only to make old stack traces easier to understand if external code
        # referenced the previous private implementation.
        P2PrototypeInitializer._write_cache(path, key, payload)
