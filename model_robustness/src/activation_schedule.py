from __future__ import annotations

"""Export and replay P3 activation schedules for direct-effect audits."""

import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from .io_utils import append_jsonl
from .schemas import DiscussionAgent, ExperimentCell


class ActivationScheduleStore:
    _write_lock = Lock()

    def __init__(self, config: Mapping[str, Any]) -> None:
        activation = config.get("p3", {}).get("activation", {})
        replay_path = activation.get("replay_schedule_path")
        self.replay_path = Path(str(replay_path)).expanduser() if replay_path else None
        self._replay = self._load_replay(self.replay_path) if self.replay_path else {}

    @staticmethod
    def _key(cell: ExperimentCell, timestep: int) -> str:
        return "|".join(
            [
                cell.experiment_family,
                cell.scenario_name,
                cell.condition_name,
                cell.persona_name,
                cell.chat_atmosphere,
                str(cell.repeat_index),
                str(timestep),
            ]
        )

    def replay_agents(
        self,
        cell: ExperimentCell,
        timestep: int,
        agents: Sequence[DiscussionAgent],
        max_active_agents: int,
    ) -> list[DiscussionAgent] | None:
        if not self.replay_path:
            return None
        key = self._key(cell, timestep)
        if key not in self._replay:
            raise ValueError(f"激活序列缺少记录: {key}")
        agent_by_id = {agent.agent_id: agent for agent in agents}
        requested = self._replay[key]
        unknown = [agent_id for agent_id in requested if agent_id not in agent_by_id]
        if unknown:
            raise ValueError(f"激活序列包含未知 agent: {key}: {unknown}")
        if len(requested) > max_active_agents:
            raise ValueError(f"激活序列超过 max_active_agents: {key}")
        return [agent_by_id[agent_id] for agent_id in requested]

    def record(
        self,
        output_dir: Path,
        cell: ExperimentCell,
        timestep: int,
        active_agents: Sequence[DiscussionAgent],
        replayed: bool,
    ) -> None:
        row = {
            "schedule_key": self._key(cell, timestep),
            "experiment_family": cell.experiment_family,
            "scenario_name": cell.scenario_name,
            "condition_name": cell.condition_name,
            "persona_name": cell.persona_name,
            "chat_atmosphere": cell.chat_atmosphere,
            "repeat_index": cell.repeat_index,
            "timestep": timestep,
            "active_agent_ids": [agent.agent_id for agent in active_agents],
            "replayed": replayed,
        }
        with self._write_lock:
            append_jsonl(output_dir / "progress" / "activation_schedule.jsonl", row)

    @classmethod
    def _load_replay(cls, path: Path) -> dict[str, list[str]]:
        if not path.exists():
            raise FileNotFoundError(f"激活序列文件不存在: {path}")
        rows: dict[str, list[str]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                key = str(payload.get("schedule_key") or "")
                ids = payload.get("active_agent_ids")
                if not key or not isinstance(ids, list):
                    raise ValueError(f"激活序列格式错误: {path}:{line_number}")
                rows[key] = [str(agent_id) for agent_id in ids]
        return rows
