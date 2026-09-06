from __future__ import annotations

"""P3 讨论线程时间步调度器。"""

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schemas import AgentDynamicState, DiscussionAgent, ThreadSnapshot


@dataclass
class StepSchedule:
    """单个时间步的活跃用户集合。"""

    timestep: int
    active_agents: list[DiscussionAgent]
    activation_scores: dict[str, float] | None = None


class ForumScheduler:
    """Mesa 风格的轻量时间步调度器。

    这里仍保持可复现，但加入轻量的活跃度差异：
    高活跃用户几乎每步都可能出现，中活跃用户隔步出现，
    低活跃用户更稀疏地进入讨论。
    """

    def __init__(self, agents: Sequence[DiscussionAgent]) -> None:
        self._agents = list(agents)

    def build_schedule(self, max_timestep: int) -> list[StepSchedule]:
        """按时间步生成完整调度计划。"""
        schedules: list[StepSchedule] = []
        for timestep in range(1, max_timestep + 1):
            schedules.append(
                StepSchedule(
                    timestep=timestep,
                    active_agents=self._pick_active_agents(timestep),
                )
            )
        return schedules

    @staticmethod
    def _stable_unit(seed: str) -> float:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def score_agents(
        self,
        timestep: int,
        agent_states: dict[str, AgentDynamicState],
        snapshot: ThreadSnapshot,
    ) -> dict[str, float]:
        thread_negative_share = snapshot.negative_visibility_share
        scores: dict[str, float] = {}
        for agent in self._agents:
            state = agent_states[agent.agent_id]
            disagreement = abs(
                state.latent_neg_tendency - (thread_negative_share if thread_negative_share is not None else 0.5)
            )
            cooldown_penalty = 0.28 if state.cooldown_steps_remaining > 0 else 0.0
            reentry_bonus = min(state.silence_steps, 4) * 0.02
            jitter = 0.02 * self._stable_unit(f"{agent.agent_id}|{timestep}|activation")
            score = (
                agent.base_activity_score * 0.34
                + state.engagement_tendency * 0.20
                + state.notification_score * 0.14
                + state.visible_negative_pressure * 0.14
                + disagreement * 0.08
                + reentry_bonus
                + jitter
                - state.fatigue * 0.18
                - cooldown_penalty
            )
            scores[agent.agent_id] = max(0.0, min(1.0, score))
        return scores

    def select_active_agents(
        self,
        timestep: int,
        agent_states: dict[str, AgentDynamicState],
        snapshot: ThreadSnapshot,
        max_active_agents: int = 3,
        exploration_slots: int = 0,
    ) -> StepSchedule:
        if not self._agents:
            return StepSchedule(timestep=timestep, active_agents=[], activation_scores={})
        scores = self.score_agents(timestep=timestep, agent_states=agent_states, snapshot=snapshot)
        ranked = sorted(
            self._agents,
            key=lambda agent: (
                scores.get(agent.agent_id, 0.0),
                self._stable_unit(f"{agent.agent_id}|{timestep}|rank"),
            ),
            reverse=True,
        )
        active_limit = max(1, min(max_active_agents, len(ranked)))
        explore_n = max(0, min(int(exploration_slots), active_limit - 1))
        core_n = active_limit - explore_n
        active_agents = ranked[:core_n]
        selected_ids = {agent.agent_id for agent in active_agents}
        if explore_n:
            candidates = [agent for agent in self._agents if agent.agent_id not in selected_ids]
            exploration_ranked = sorted(
                candidates,
                key=lambda agent: self._exploration_score(
                    agent=agent,
                    state=agent_states[agent.agent_id],
                    timestep=timestep,
                    base_score=scores.get(agent.agent_id, 0.0),
                    snapshot=snapshot,
                ),
                reverse=True,
            )
            active_agents.extend(exploration_ranked[:explore_n])
        return StepSchedule(
            timestep=timestep,
            active_agents=active_agents,
            activation_scores={agent.agent_id: scores.get(agent.agent_id, 0.0) for agent in ranked},
        )

    def _exploration_score(
        self,
        agent: DiscussionAgent,
        state: AgentDynamicState,
        timestep: int,
        base_score: float,
        snapshot: ThreadSnapshot,
    ) -> float:
        """Score long-silent users for exploration slots without making everyone active."""
        never_active_bonus = 0.32 if not state.was_ever_active else 0.0
        silence_bonus = min(state.silence_steps, 8) * 0.045
        pressure = snapshot.negative_visibility_share
        exposure_gap = (
            abs(state.latent_neg_tendency - pressure)
            if pressure is not None
            else abs(state.latent_neg_tendency - 0.5)
        )
        low_activity_bonus = {"low": 0.08, "medium": 0.04, "high": 0.0}.get(agent.activity_level, 0.02)
        jitter = 0.04 * self._stable_unit(f"{agent.agent_id}|{timestep}|explore")
        return (
            base_score * 0.35
            + never_active_bonus
            + silence_bonus
            + exposure_gap * 0.18
            + low_activity_bonus
            + jitter
            - state.fatigue * 0.08
        )

    def _pick_active_agents(self, timestep: int) -> list[DiscussionAgent]:
        """按活跃度节奏挑选当前步参与者，并做滑窗轮转。"""
        if not self._agents:
            return []
        cadence = {"high": 1, "medium": 2, "low": 3}
        eligible = [
            agent
            for agent in self._agents
            if (timestep - 1) % cadence.get(agent.activity_level, 2) == 0
        ]
        if not eligible:
            eligible = list(self._agents)

        window_size = min(3, len(eligible))
        start = (timestep - 1) % len(eligible)
        doubled = eligible + eligible
        return doubled[start : start + window_size]

    @staticmethod
    def summarize(schedule: Iterable[StepSchedule]) -> list[dict[str, object]]:
        """把调度对象转换为可导出行。"""
        rows: list[dict[str, object]] = []
        for item in schedule:
            rows.append(
                {
                    "timestep": item.timestep,
                    "active_agents": ",".join(agent.agent_id for agent in item.active_agents),
                    "n_active_agents": len(item.active_agents),
                }
            )
        return rows
