from __future__ import annotations

"""Prototype-level panel and optional silent-agent updates for P3."""

import hashlib
from typing import Any, Mapping

from .schemas import AgentDynamicState, DiscussionAgent, ExperimentCell


def build_prototype_panel_rows(
    cell: ExperimentCell,
    agents: list[DiscussionAgent],
    agent_states: dict[str, AgentDynamicState],
    timestep: int,
    active_ids: set[str],
    init_meta_by_persona: Mapping[str, Mapping[str, object]],
    include_active_agents: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    persona_names = sorted({agent.persona_name for agent in agents})
    for persona_name in persona_names:
        group_agents = [agent for agent in agents if agent.persona_name == persona_name]
        active_count = sum(agent.agent_id in active_ids for agent in group_agents)
        silent_count = len(group_agents) - active_count
        included_agents = (
            group_agents if include_active_agents else [agent for agent in group_agents if agent.agent_id not in active_ids]
        )
        states = [agent_states[agent.agent_id] for agent in included_agents]
        meta = init_meta_by_persona.get(persona_name, {})
        rows.append(
            {
                "experiment_family": cell.experiment_family,
                "scenario_name": cell.scenario_name,
                "condition_name": cell.condition_name,
                "persona_name": cell.persona_name,
                "chat_atmosphere": cell.chat_atmosphere,
                "repeat": cell.repeat_index,
                "repeat_index": cell.repeat_index,
                "timestep": timestep,
                "prototype_id": f"{persona_name}_prototype",
                "agent_persona_name": persona_name,
                "persona_name_prototype": persona_name,
                "size": len(group_agents),
                "panel_size": len(states),
                "active_count": active_count,
                "silent_count": silent_count,
                "mean_latent_neg_tendency": _mean(states, "latent_neg_tendency"),
                "mean_risk_arousal": _mean(states, "risk_arousal"),
                "mean_engagement_tendency": _mean(states, "engagement_tendency"),
                "mean_fatigue": _mean(states, "fatigue"),
                "mean_visible_negative_pressure": _mean(states, "visible_negative_pressure"),
                "negative_ratio": _ratio(states, "negative"),
                "neutral_ratio": _ratio(states, "neutral"),
                "positive_ratio": _ratio(states, "positive"),
                "prototype_initialization_method": meta.get("prototype_initialization_method"),
                "prototype_initialization_cached": meta.get("prototype_initialization_cached"),
                "prototype_initialization_reason": meta.get("prototype_initialization_reason"),
            }
        )
    return rows


def apply_prototype_silent_update(
    config: dict[str, Any],
    cell: ExperimentCell,
    agents: list[DiscussionAgent],
    agent_states: dict[str, AgentDynamicState],
    active_ids: set[str],
    visible_negative_pressure: float | None,
    timestep: int,
) -> None:
    """Optional non-LLM silent update. Disabled by default."""
    p3_cfg = config.get("p3", {}) if isinstance(config.get("p3", {}), dict) else {}
    update_cfg = p3_cfg.get("prototype_silent_update", {})
    update_cfg = update_cfg if isinstance(update_cfg, dict) else {}
    if not bool(update_cfg.get("enabled", False)):
        return
    if visible_negative_pressure is None:
        return
    base_weight = float(update_cfg.get("base_weight", 0.02))
    noise_scale = float(update_cfg.get("noise_scale", 0.01))
    activation_threshold = float(update_cfg.get("activation_threshold", 0.65))
    pressure = max(0.0, min(1.0, float(visible_negative_pressure)))
    for agent in agents:
        if agent.agent_id in active_ids:
            continue
        state = agent_states[agent.agent_id]
        noise = _stable_noise(f"{cell.scenario_name}|{cell.repeat_index}|{timestep}|{agent.agent_id}|silent", noise_scale)
        state.latent_neg_tendency = _clip01(
            state.latent_neg_tendency + (pressure - 0.5) * base_weight + noise
        )
        state.risk_arousal = _clip01(state.risk_arousal + (pressure - 0.5) * base_weight * 0.5)
        if state.latent_neg_tendency >= activation_threshold:
            state.notification_score = _clip01(state.notification_score + 0.08)


def _mean(states: list[AgentDynamicState], field: str) -> float | None:
    if not states:
        return None
    return round(sum(float(getattr(state, field)) for state in states) / len(states), 4)


def _ratio(states: list[AgentDynamicState], category: str) -> float | None:
    if not states:
        return None
    count = sum(_latent_category(state.latent_neg_tendency) == category for state in states)
    return round(count / len(states), 4)


def _latent_category(value: float) -> str:
    if value >= 0.67:
        return "negative"
    if value <= 0.33:
        return "positive"
    return "neutral"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _stable_noise(seed: str, scale: float) -> float:
    if scale <= 0:
        return 0.0
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 2 * scale
