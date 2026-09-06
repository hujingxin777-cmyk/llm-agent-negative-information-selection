from __future__ import annotations

"""实验环境构造：社交材料、P3 线程热启动与可见上下文。"""

from collections import defaultdict
import re
from typing import Any, Dict, Iterable, Mapping

from .persona_variants import (
    get_persona_behavior_profile as _get_persona_behavior_profile,
    get_persona_behavior_text as _get_persona_behavior_text,
)
from .schemas import DiscussionAgent, HyperedgeSnapshot, ThreadEvent, ThreadSnapshot


DEFAULT_VISIBILITY_WEIGHTS = {
    "reply": 0.65,
    "repost": 0.45,
    "like": 0.20,
    "recency": 0.20,
}


def _base_activity_score(activity_level: str) -> float:
    mapping = {"high": 0.92, "medium": 0.66, "low": 0.38}
    return float(mapping.get(activity_level, 0.55))


def build_visibility_weights(config: Dict[str, Any]) -> dict[str, float]:
    """读取 P3 可见度权重，默认保持旧版口径。"""
    p3_cfg = config.get("p3", {})
    weights = dict(DEFAULT_VISIBILITY_WEIGHTS)
    configured = p3_cfg.get("visibility_weights", {})
    if isinstance(configured, dict):
        for key in weights:
            if key in configured:
                weights[key] = float(configured[key])
    if "like_visibility_weight" in p3_cfg:
        weights["like"] = float(p3_cfg["like_visibility_weight"])
    return weights


def get_social_block(
    config: Dict[str, Any],
    condition_name: str,
    chat_atmosphere: str,
    scenario_name: str | None = None,
) -> str:
    """按条件与氛围获取可见社交材料块。"""
    def _resolve(blocks: Mapping[str, Any], atmosphere: str) -> str:
        value = blocks.get(atmosphere, "")
        if isinstance(value, Mapping):
            if scenario_name and scenario_name in value:
                return str(value.get(scenario_name, ""))
            return str(value.get("default", ""))
        return str(value or "")

    social_blocks = config.get("social_blocks", {})
    if condition_name == "low_feedback":
        return _resolve(social_blocks.get("low_feedback", {}), chat_atmosphere)
    if condition_name == "high_interaction":
        bootstrap_source = config.get("p3", {}).get("bootstrap_source", "low_feedback")
        return _resolve(social_blocks.get(bootstrap_source, {}), chat_atmosphere)
    return ""


def describe_platform(condition_name: str) -> str:
    """返回平台条件的人类可读描述。"""
    if condition_name == "independent":
        return "P1 独立决策：无社交线索，无回复链，无动态过程。"
    if condition_name == "low_feedback":
        return "P2 静态社会证明：可见少量静态评论和热度线索，无回复链，无动态过程。"
    if condition_name == "high_interaction":
        return "P3 高阶互动：P2 评论作为论坛初始只读背景可见，agent 需要自主发新评论楼层，再围绕 agent 生成内容互动。"
    raise ValueError(f"未知条件: {condition_name}")


def get_persona_behavior_text(persona_name: str) -> str:
    """读取当前 persona variant 的行为文本。"""
    return _get_persona_behavior_text(persona_name)


def get_persona_behavior_profile(persona_name: str) -> dict[str, float]:
    """读取当前 persona variant 的参数画像。"""
    return _get_persona_behavior_profile(persona_name)

# 构建讨论代理人列表。
# 主实验默认不注入立场差异，所有 agent 从 neutral 起步，只保留活跃度分层。
# 在人格补充实验的 P3 中，使用 1:1:1 的人格混合讨论场，避免把整场讨论都绑定成单一 persona。
def build_discussion_agents(
    experiment_family: str,
    persona_name: str,
    config: Dict[str, Any] | None = None,
) -> list[DiscussionAgent]:
    """构造 P3 讨论参与者。

    - 默认：同一 persona + 活跃度分层
    - 混合场：三类 persona 均衡注入
    """
    p3_cfg = (config or {}).get("p3", {})
    configured_num_agents = p3_cfg.get("num_agents")
    if configured_num_agents is not None:
        num_agents = max(1, int(configured_num_agents))
        activity_mix = p3_cfg.get("activity_mix", {"high": 0.2, "medium": 0.5, "low": 0.3})
        high_n = int(round(num_agents * float(activity_mix.get("high", 0.2))))
        medium_n = int(round(num_agents * float(activity_mix.get("medium", 0.5))))
        high_n = min(high_n, num_agents)
        medium_n = min(medium_n, num_agents - high_n)
        low_n = num_agents - high_n - medium_n
        activity_levels = ["high"] * high_n + ["medium"] * medium_n + ["low"] * low_n
        if not activity_levels:
            activity_levels = ["medium"]
        if persona_name == "balanced_persona_mix":
            persona_cycle = ["risk_sensitive", "authority_trusting", "ordinary_user"]
        else:
            persona_cycle = [persona_name]
        return [
            DiscussionAgent(
                agent_id=f"agent_{index:03d}",
                persona_name=persona_cycle[(index - 1) % len(persona_cycle)],
                activity_level=activity_levels[(index - 1) % len(activity_levels)],
                base_activity_score=_base_activity_score(activity_levels[(index - 1) % len(activity_levels)]),
            )
            for index in range(1, num_agents + 1)
        ]

    if experiment_family in {"persona_supplement", "combined_supplement"} and persona_name == "balanced_persona_mix":
        mixed_seeds = [
            ("agent_01", "risk_sensitive", "high"),
            ("agent_02", "authority_trusting", "high"),
            ("agent_03", "ordinary_user", "high"),
            ("agent_04", "risk_sensitive", "medium"),
            ("agent_05", "authority_trusting", "medium"),
            ("agent_06", "ordinary_user", "medium"),
            ("agent_07", "risk_sensitive", "low"),
            ("agent_08", "authority_trusting", "low"),
            ("agent_09", "ordinary_user", "low"),
        ]
        return [
            DiscussionAgent(
                agent_id=agent_id,
                persona_name=agent_persona_name,
                activity_level=activity_level,
                base_activity_score=_base_activity_score(activity_level),
            )
            for agent_id, agent_persona_name, activity_level in mixed_seeds
        ]

    seeds = [
        ("agent_01", "high"),
        ("agent_02", "high"),
        ("agent_03", "high"),
        ("agent_04", "medium"),
        ("agent_05", "medium"),
        ("agent_06", "medium"),
        ("agent_07", "medium"),
        ("agent_08", "low"),
        ("agent_09", "low"),
    ]
    return [
        DiscussionAgent(
            agent_id=agent_id,
            persona_name=persona_name,
            activity_level=activity_level,
            base_activity_score=_base_activity_score(activity_level),
        )
        for agent_id, activity_level in seeds
    ]


def bootstrap_p3_events(
    config: Dict[str, Any],
    chat_atmosphere: str,
    scenario_name: str | None = None,
) -> list[ThreadEvent]:
    """构造 P3 初始公开线程。

    默认可通过 bootstrap_mode 控制。若不是 context_only/self_start 等空线程模式，
    则把 P2 可见评论作为 t=0 只读 seed；点赞热度可按配置剥离。
    """
    mode = str(config.get("p3", {}).get("bootstrap_mode", "context_only")).lower()
    if mode in {"context_only", "material_only", "no_seed", "none", "empty", "self_start"}:
        return []

    block = get_social_block(config, "high_interaction", chat_atmosphere, scenario_name)
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    events: list[ThreadEvent] = []
    current_root_event_id: str | None = None

    seed_like_policy = str(
        config.get("p3", {}).get("bootstrap_seed_like_policy", "inherit")
    ).lower()
    strip_seed_likes = seed_like_policy in {"strip", "zero", "ignore", "none", "no_like", "no_likes"}

    for index, line in enumerate(lines, start=1):
        actor_id, content, salience_score = _parse_bootstrap_line(
            line,
            index,
            keep_like_text=not strip_seed_likes,
        )
        if strip_seed_likes:
            salience_score = 0.0
        is_reply = "回复" in line and current_root_event_id is not None
        event_id = f"seed_{index}"
        if not is_reply:
            current_root_event_id = event_id
        events.append(
            ThreadEvent(
                timestep=0,
                event_id=event_id,
                actor_id=actor_id,
                action_type="reply" if is_reply else "post",
                content=content,
                reply_to=current_root_event_id if is_reply else None,
                # seed 评论只提供初始文本上下文，不由代码预先贴正负标签，
                # 让后续立场信号更多来自 agent 自身的判断与互动。
                signal_polarity=None,
                salience_score=salience_score,
                hyperedge_id=current_root_event_id if is_reply else event_id,
            )
        )
    return events


_BOOTSTRAP_LIKE_PATTERN = re.compile(r"👍\s*\d+\s*")


def _parse_bootstrap_line(line: str, index: int, keep_like_text: bool = True) -> tuple[str, str, float]:
    """解析 seed 行文本，提取作者、正文和显著性分数。"""
    text = line
    actor_id = f"seed_user_{index}"
    salience_score = 0.0
    if line.startswith("【") and "】" in line:
        actor_id = line[1 : line.index("】")].replace(" ", "_")
        text = line[line.index("】") + 1 :].strip()
    if "👍" in text:
        try:
            after_thumb = text.split("👍", 1)[1].strip()
            first_token = after_thumb.split()[0]
            salience_score = float(first_token)
        except (ValueError, IndexError):
            salience_score = 0.0
        if not keep_like_text:
            text = _BOOTSTRAP_LIKE_PATTERN.sub("", text, count=1).strip()
    return actor_id, text, salience_score


def _build_visibility_scores(
    events: list[ThreadEvent],
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """计算事件可见性评分（显著性 + 互动量 + 轻微时新性）。"""
    weights = weights or DEFAULT_VISIBILITY_WEIGHTS
    interaction_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"reply": 0, "like": 0, "repost": 0})
    total = max(len(events), 1)
    for event in events:
        if event.reply_to and event.action_type in {"reply", "like", "repost"}:
            interaction_counts[event.reply_to][event.action_type] += 1

    scores: dict[str, float] = {}
    for idx, event in enumerate(events):
        counts = interaction_counts.get(event.event_id, {})
        recency_bonus = (idx + 1) / total * float(weights.get("recency", 0.20))
        scores[event.event_id] = (
            float(event.salience_score)
            + counts.get("reply", 0) * float(weights.get("reply", 0.65))
            + counts.get("repost", 0) * float(weights.get("repost", 0.45))
            + counts.get("like", 0) * float(weights.get("like", 0.20))
            + recency_bonus
        )
    return scores


def _resolve_hyperedge_id(event: ThreadEvent, by_id: Mapping[str, ThreadEvent]) -> str:
    if event.hyperedge_id:
        return event.hyperedge_id
    if event.action_type == "post" and not event.reply_to:
        return event.event_id
    current = event
    seen: set[str] = set()
    while current.reply_to and current.reply_to not in seen:
        parent = by_id.get(current.reply_to)
        if parent is None:
            break
        if parent.hyperedge_id:
            return parent.hyperedge_id
        if parent.action_type == "post" and not parent.reply_to:
            return parent.event_id
        seen.add(current.reply_to)
        current = parent
    return event.reply_to or event.event_id


def _build_hyperedge_snapshots(events: list[ThreadEvent]) -> dict[str, HyperedgeSnapshot]:
    if not events:
        return {}

    by_id = {event.event_id: event for event in events}
    raw: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=lambda item: (item.timestep, item.event_id)):
        hyperedge_id = _resolve_hyperedge_id(event, by_id)
        bucket = raw.setdefault(
            hyperedge_id,
            {
                "hyperedge_id": hyperedge_id,
                "root_event_id": hyperedge_id,
                "root_actor_id": event.actor_id,
                "root_content": event.content,
                "latest_event_id": event.event_id,
                "latest_timestep": event.timestep,
                "members": set(),
                "event_count": 0,
                "dynamic_positive_events": 0,
                "dynamic_negative_events": 0,
                "dynamic_neutral_events": 0,
                "like_count": 0,
                "reply_count": 0,
                "post_count": 0,
                "repost_count": 0,
            },
        )
        if event.event_id == hyperedge_id:
            bucket["root_actor_id"] = event.actor_id
            bucket["root_content"] = event.content
        bucket["latest_event_id"] = event.event_id
        bucket["latest_timestep"] = event.timestep
        bucket["members"].add(event.actor_id)
        bucket["event_count"] += 1
        if event.timestep > 0:
            if event.signal_polarity == "positive":
                bucket["dynamic_positive_events"] += 1
            elif event.signal_polarity == "negative":
                bucket["dynamic_negative_events"] += 1
            else:
                bucket["dynamic_neutral_events"] += 1
            if event.action_type == "like":
                bucket["like_count"] += 1
            elif event.action_type == "reply":
                bucket["reply_count"] += 1
            elif event.action_type == "repost":
                bucket["repost_count"] += 1
            elif event.action_type == "post":
                bucket["post_count"] += 1

    snapshots: dict[str, HyperedgeSnapshot] = {}
    for hyperedge_id, bucket in raw.items():
        dynamic_positive = int(bucket["dynamic_positive_events"])
        dynamic_negative = int(bucket["dynamic_negative_events"])
        dynamic_neutral = int(bucket["dynamic_neutral_events"])
        dynamic_stance_total = dynamic_positive + dynamic_negative
        dynamic_negative_share = (
            dynamic_negative / dynamic_stance_total if dynamic_stance_total else None
        )
        dynamic_hps = (
            (dynamic_negative - dynamic_positive) / dynamic_stance_total
            if dynamic_stance_total
            else 0.0
        )
        dynamic_hcs = (
            max(dynamic_positive, dynamic_negative) / dynamic_stance_total
            if dynamic_stance_total
            else 0.0
        )
        dynamic_hvs = (
            float(bucket["like_count"]) * 1.0
            + float(bucket["reply_count"]) * 2.0
            + float(bucket["post_count"]) * 1.5
            + float(bucket["repost_count"]) * 2.5
        )
        snapshots[hyperedge_id] = HyperedgeSnapshot(
            hyperedge_id=hyperedge_id,
            root_event_id=str(bucket["root_event_id"]),
            root_actor_id=str(bucket["root_actor_id"]),
            root_content=str(bucket["root_content"]),
            latest_event_id=str(bucket["latest_event_id"]),
            latest_timestep=int(bucket["latest_timestep"]),
            member_count=len(bucket["members"]),
            event_count=int(bucket["event_count"]),
            dynamic_positive_events=dynamic_positive,
            dynamic_negative_events=dynamic_negative,
            dynamic_neutral_events=dynamic_neutral,
            dynamic_negative_share=dynamic_negative_share,
            dynamic_hps=dynamic_hps,
            dynamic_hcs=dynamic_hcs,
            dynamic_hvs=dynamic_hvs,
        )
    return snapshots


def _render_visible_hyperedge_summary(hyperedges: Iterable[HyperedgeSnapshot]) -> str:
    lines: list[str] = []
    for hyperedge in hyperedges:
        if hyperedge.dynamic_negative_share is None:
            stance = "立场尚弱"
        elif hyperedge.dynamic_negative_share >= 0.6:
            stance = "楼层偏负"
        elif hyperedge.dynamic_negative_share <= 0.4:
            stance = "楼层偏正"
        else:
            stance = "楼层分裂"
        lines.append(
            f"[楼={hyperedge.hyperedge_id}] {stance} | 可见度={hyperedge.dynamic_hvs:.2f} | "
            f"成员={hyperedge.member_count} | 最近更新=t{hyperedge.latest_timestep}"
        )
    return "\n".join(lines)


def _select_visible_events(
    events: list[ThreadEvent],
    max_events: int = 8,
    visibility_weights: Mapping[str, float] | None = None,
) -> list[ThreadEvent]:
    """选择可见事件窗口，并尽量保留回复链祖先。"""
    if len(events) <= max_events:
        return list(events)

    by_id = {event.event_id: event for event in events}
    scores = _build_visibility_scores(events, weights=visibility_weights)

    selected: list[ThreadEvent] = []
    selected_ids: set[str] = set()

    def add_with_chain(event: ThreadEvent) -> None:
        if event.event_id in selected_ids or len(selected) >= max_events:
            return
        if event.reply_to and event.reply_to in by_id and event.reply_to not in selected_ids:
            add_with_chain(by_id[event.reply_to])
        if event.event_id in selected_ids or len(selected) >= max_events:
            return
        selected.append(event)
        selected_ids.add(event.event_id)

    ranked = sorted(
        events,
        key=lambda event: (scores.get(event.event_id, 0.0), event.timestep),
        reverse=True,
    )
    for event in ranked:
        add_with_chain(event)
        if len(selected) >= max_events:
            break

    selected.sort(key=lambda event: (event.timestep, event.event_id))
    return selected


def render_visible_context(
    events: Iterable[ThreadEvent],
    max_events: int = 8,
    visibility_weights: Mapping[str, float] | None = None,
) -> str:
    """把可见事件渲染成 prompt 可读文本块。"""
    recent_events = _select_visible_events(
        list(events),
        max_events=max_events,
        visibility_weights=visibility_weights,
    )
    if not recent_events:
        return (
            "当前 P3 公开线程还没有 agent 生成的楼层。"
            "如果你要公开参与，请用 post_comment 自己建立一个新楼层；"
            "不要回复、点赞或转发参考材料。"
        )
    lines: list[str] = []
    for event in recent_events:
        readonly_seed = event.event_id.startswith("seed_")
        prefix = f"[t={event.timestep} | id={event.event_id}] {event.actor_id}"
        if readonly_seed:
            prefix += "（初始只读评论）"
        if event.reply_to:
            prefix += f" -> {event.reply_to}"
        if event.action_type == "like":
            body = event.content or "点赞了这条内容"
        elif event.action_type == "repost":
            body = event.content or "转发了这条内容"
        elif event.action_type == "observe":
            # Compatibility for old traces; current P3 runner keeps observe out
            # of public events, so this branch should not affect new runs.
            body = event.content or "本轮没有公开发言"
        else:
            body = event.content
        if readonly_seed:
            lines.append(f"{prefix}: {body}（只作为初始阅读材料，不要回复、点赞或转发这条）")
        else:
            lines.append(f"{prefix}: {body}")
    return "\n".join(lines)


def build_thread_snapshot(
    timestep: int,
    events: list[ThreadEvent],
    max_events: int = 8,
    visibility_weights: Mapping[str, float] | None = None,
) -> ThreadSnapshot:
    """构造某时间步的线程快照。"""
    visible_events = _select_visible_events(
        events,
        max_events=max_events,
        visibility_weights=visibility_weights,
    )
    hyperedge_map = _build_hyperedge_snapshots(events)
    visible_hyperedges: list[HyperedgeSnapshot] = []
    visible_hyperedge_ids: list[str] = []
    for event in visible_events:
        hyperedge_id = event.hyperedge_id or _resolve_hyperedge_id(event, {item.event_id: item for item in events})
        if hyperedge_id in hyperedge_map and hyperedge_id not in visible_hyperedge_ids:
            visible_hyperedge_ids.append(hyperedge_id)
            visible_hyperedges.append(hyperedge_map[hyperedge_id])

    weighted_pairs = [
        (
            hyperedge.dynamic_negative_share,
            hyperedge.dynamic_hvs if hyperedge.dynamic_hvs > 0 else 0.5,
        )
        for hyperedge in visible_hyperedges
        if hyperedge.dynamic_negative_share is not None
    ]
    negative_visibility_share = None
    if weighted_pairs:
        total_weight = sum(weight for _, weight in weighted_pairs) or 1.0
        negative_visibility_share = sum(share * weight for share, weight in weighted_pairs) / total_weight
    dominant_hyperedge_id = None
    if visible_hyperedges:
        dominant_hyperedge_id = max(
            visible_hyperedges,
            key=lambda item: (item.dynamic_hvs, item.latest_timestep, item.hyperedge_id),
        ).hyperedge_id
    return ThreadSnapshot(
        timestep=timestep,
        visible_events=visible_events,
        visible_context=render_visible_context(
            events,
            max_events=max_events,
            visibility_weights=visibility_weights,
        ),
        visible_hyperedges=visible_hyperedges,
        visible_hyperedge_ids=visible_hyperedge_ids,
        visible_hyperedge_summary=_render_visible_hyperedge_summary(visible_hyperedges),
        dominant_hyperedge_id=dominant_hyperedge_id,
        negative_visibility_share=negative_visibility_share,
    )
