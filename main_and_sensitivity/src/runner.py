from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import re
import time
from typing import Any, Dict, List

from .checkpoints import CellCheckpointStore
from .activation_schedule import ActivationScheduleStore
from .environments import (
    build_visibility_weights,
    bootstrap_p3_events,
    build_discussion_agents,
    build_thread_snapshot,
    get_persona_behavior_profile,
    get_persona_behavior_text,
)
from .io_utils import write_csv, write_jsonl
from .llm_client import LLMClient, LLMDecisionParser
from .metrics import encode_metrics
from .p2_prototype_initializer import P2PrototypeInitializer
from .progress import ProgressTracker
from .prototype_agents import apply_prototype_silent_update, build_prototype_panel_rows
from .prompts import (
    _reason_focus_hint,
    build_p3_timestep_prompt,
    build_system_message,
    build_user_prompt,
)
from .scheduler import ForumScheduler, StepSchedule
from .run_guard import RunAbortError, RunGuard
from .schemas import (
    AgentDynamicState,
    DiscussionAgent,
    ExperimentCell,
    P3AgentPanelRow,
    P3RunRow,
    SimulationTraceRow,
    StaticRunRow,
    ThreadSnapshot,
    ThreadEvent,
)


class ExperimentPlanner:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def build_cells(self) -> List[ExperimentCell]:
        experiments = self.config["experiments"]
        p3_max_timestep = int(self.config.get("p3", {}).get("max_timestep", 10))
        cells: list[ExperimentCell] = []

        main = experiments["main"]
        if main.get("enabled", False):
            p3_main_persona_override = self.config.get("p3", {}).get("main_persona_override")
            for scenario_name in self._scenario_names(main["scenario_names"]):
                for condition_name in main["conditions"]:
                    atmosphere = "none"
                    if condition_name in ("low_feedback", "high_interaction"):
                        atmosphere = main["default_chat_atmosphere"][condition_name]
                    persona_name = main["default_persona"]
                    if condition_name == "high_interaction" and p3_main_persona_override:
                        persona_name = str(p3_main_persona_override)
                    for repeat_index in self._repeat_indices(int(main["repeats_per_cell"])):
                        cells.append(
                            ExperimentCell(
                                experiment_family="main",
                                scenario_name=scenario_name,
                                condition_name=condition_name,
                                persona_name=persona_name,
                                chat_atmosphere=atmosphere,
                                repeat_index=repeat_index,
                                max_timestep=p3_max_timestep if condition_name == "high_interaction" else 1,
                            )
                        )

        persona = experiments["persona_supplement"]
        if persona.get("enabled", False):
            persona_p3_mode = str(persona.get("p3_persona_mode", "balanced_mix"))
            for scenario_name in self._scenario_names(persona["scenario_names"]):
                for condition_name in persona["conditions"]:
                    atmosphere = "none"
                    if condition_name in ("low_feedback", "high_interaction"):
                        atmosphere = persona["fixed_chat_atmosphere"][condition_name]
                    persona_names = (
                        ["balanced_persona_mix"]
                        if condition_name == "high_interaction" and persona_p3_mode == "balanced_mix"
                        else persona["persona_names"]
                    )
                    for persona_name in persona_names:
                        for repeat_index in self._repeat_indices(int(persona["repeats_per_cell"])):
                            cells.append(
                                ExperimentCell(
                                    experiment_family="persona_supplement",
                                    scenario_name=scenario_name,
                                    condition_name=condition_name,
                                    persona_name=persona_name,
                                    chat_atmosphere=atmosphere,
                                    repeat_index=repeat_index,
                                    max_timestep=p3_max_timestep if condition_name == "high_interaction" else 1,
                                )
                            )

        atmosphere_exp = experiments["atmosphere_supplement"]
        if atmosphere_exp.get("enabled", False):
            reuse_main_none = bool(atmosphere_exp.get("use_main_baseline_for_none", False)) and main.get("enabled", False)
            configured_atmospheres = list(atmosphere_exp.get("chat_atmospheres", []))
            if reuse_main_none:
                configured_atmospheres = [item for item in configured_atmospheres if item != "none"]
            for scenario_name in self._scenario_names(atmosphere_exp["scenario_names"]):
                for condition_name in atmosphere_exp["conditions"]:
                    if reuse_main_none and condition_name == "independent":
                        continue
                    candidate_atmospheres = ["none"]
                    if condition_name in ("low_feedback", "high_interaction"):
                        candidate_atmospheres = configured_atmospheres
                    if not candidate_atmospheres:
                        continue
                    for chat_atmosphere in candidate_atmospheres:
                        for repeat_index in self._repeat_indices(int(atmosphere_exp["repeats_per_cell"])):
                            cells.append(
                                ExperimentCell(
                                    experiment_family="atmosphere_supplement",
                                    scenario_name=scenario_name,
                                    condition_name=condition_name,
                                    persona_name=atmosphere_exp["persona_name"],
                                    chat_atmosphere=chat_atmosphere,
                                    repeat_index=repeat_index,
                                    max_timestep=p3_max_timestep if condition_name == "high_interaction" else 1,
                                )
                            )

        combined = experiments.get("combined_supplement", {})
        if combined.get("enabled", False):
            reuse_persona_none = (
                bool(combined.get("use_persona_baseline_for_none", False))
                and persona.get("enabled", False)
            )
            combined_p3_mode = str(combined.get("p3_persona_mode", "balanced_mix"))
            for scenario_name in self._scenario_names(combined["scenario_names"]):
                for condition_name in combined["conditions"]:
                    if reuse_persona_none and condition_name == "independent":
                        continue
                    persona_names = (
                        ["balanced_persona_mix"]
                        if condition_name == "high_interaction" and combined_p3_mode == "balanced_mix"
                        else combined["persona_names"]
                    )
                    candidate_atmospheres = (
                        ["none"]
                        if condition_name == "independent"
                        else combined["chat_atmospheres"]
                    )
                    if reuse_persona_none:
                        candidate_atmospheres = [item for item in candidate_atmospheres if item != "none"]
                    if not candidate_atmospheres:
                        continue
                    for persona_name in persona_names:
                        for chat_atmosphere in candidate_atmospheres:
                            for repeat_index in self._repeat_indices(int(combined["repeats_per_cell"])):
                                cells.append(
                                    ExperimentCell(
                                        experiment_family="combined_supplement",
                                        scenario_name=scenario_name,
                                        condition_name=condition_name,
                                        persona_name=persona_name,
                                        chat_atmosphere=chat_atmosphere,
                                        repeat_index=repeat_index,
                                        max_timestep=p3_max_timestep if condition_name == "high_interaction" else 1,
                                    )
                                )

        return cells

    def _scenario_names(self, configured: list[str]) -> list[str]:
        requested = self.config.get("runtime", {}).get("scenario_filter")
        if not requested:
            return list(configured)
        unknown = sorted(set(requested) - set(configured))
        if unknown:
            raise ValueError(f"场景过滤包含当前实验未配置的场景: {unknown}")
        return [name for name in configured if name in requested]

    def _repeat_indices(self, repeats_per_cell: int) -> range:
        runtime = self.config.get("runtime", {})
        start = int(runtime.get("repeat_start", 1))
        end = int(runtime.get("repeat_end", repeats_per_cell))
        if start < 1 or end < start or end > repeats_per_cell:
            raise ValueError(
                "repeat 区间无效: "
                f"start={start}, end={end}, repeats_per_cell={repeats_per_cell}"
            )
        return range(start, end + 1)

    def export_plan(self) -> None:
        output_dir = Path(self.config["project"]["output_dir"])
        cells = self.build_cells()
        rows = [
            {
                "experiment_family": cell.experiment_family,
                "scenario_name": cell.scenario_name,
                "condition_name": cell.condition_name,
                "persona_name": cell.persona_name,
                "chat_atmosphere": cell.chat_atmosphere,
                "repeat_index": cell.repeat_index,
                "max_timestep": cell.max_timestep,
            }
            for cell in cells
        ]
        write_jsonl(output_dir / "plan" / "experiment_cells.jsonl", rows)
        write_csv(output_dir / "plan" / "experiment_cells.csv", rows)

        summary_rows = self._build_summary(rows)
        write_csv(output_dir / "plan" / "design_summary.csv", summary_rows)
        reuse_rows = self.build_reuse_aliases()
        if reuse_rows:
            write_jsonl(output_dir / "plan" / "reuse_aliases.jsonl", reuse_rows)
            write_csv(output_dir / "plan" / "reuse_aliases.csv", reuse_rows)

    def build_reuse_aliases(self) -> list[dict[str, object]]:
        """记录被省略的重复实验应从哪个已运行实验读取。"""
        experiments = self.config["experiments"]
        aliases: list[dict[str, object]] = []

        main = experiments.get("main", {})
        atmosphere = experiments.get("atmosphere_supplement", {})
        if (
            main.get("enabled", False)
            and atmosphere.get("enabled", False)
            and atmosphere.get("use_main_baseline_for_none", False)
        ):
            main_repeats = max(1, int(main.get("repeats_per_cell", 1)))
            target_repeats = max(1, int(atmosphere.get("repeats_per_cell", 1)))
            for scenario_name in atmosphere.get("scenario_names", []):
                for condition_name in atmosphere.get("conditions", []):
                    for repeat_index in range(1, target_repeats + 1):
                        aliases.append(
                            {
                                "reuse_scope": "B_none_uses_main_baseline",
                                "target_experiment_family": "atmosphere_supplement",
                                "target_scenario_name": scenario_name,
                                "target_condition_name": condition_name,
                                "target_persona_name": atmosphere.get("persona_name", "baseline_open"),
                                "target_chat_atmosphere": "none",
                                "target_repeat_index": repeat_index,
                                "source_experiment_family": "main",
                                "source_scenario_name": scenario_name,
                                "source_condition_name": condition_name,
                                "source_persona_name": main.get("default_persona", "baseline_open"),
                                "source_chat_atmosphere": "none",
                                "source_repeat_index": ((repeat_index - 1) % main_repeats) + 1,
                            }
                        )

        persona = experiments.get("persona_supplement", {})
        combined = experiments.get("combined_supplement", {})
        if (
            persona.get("enabled", False)
            and combined.get("enabled", False)
            and combined.get("use_persona_baseline_for_none", False)
        ):
            persona_repeats = max(1, int(persona.get("repeats_per_cell", 1)))
            target_repeats = max(1, int(combined.get("repeats_per_cell", 1)))
            for scenario_name in combined.get("scenario_names", []):
                for condition_name in combined.get("conditions", []):
                    persona_names = (
                        ["balanced_persona_mix"]
                        if condition_name == "high_interaction"
                        else combined.get("persona_names", [])
                    )
                    for persona_name in persona_names:
                        for repeat_index in range(1, target_repeats + 1):
                            aliases.append(
                                {
                                    "reuse_scope": "AB_none_uses_persona_baseline",
                                    "target_experiment_family": "combined_supplement",
                                    "target_scenario_name": scenario_name,
                                    "target_condition_name": condition_name,
                                    "target_persona_name": persona_name,
                                    "target_chat_atmosphere": "none",
                                    "target_repeat_index": repeat_index,
                                    "source_experiment_family": "persona_supplement",
                                    "source_scenario_name": scenario_name,
                                    "source_condition_name": condition_name,
                                    "source_persona_name": persona_name,
                                    "source_chat_atmosphere": "none",
                                    "source_repeat_index": ((repeat_index - 1) % persona_repeats) + 1,
                                }
                            )
        return aliases

    def _build_summary(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bucket: Dict[tuple[str, str], int] = {}
        unique_bucket: Dict[tuple[str, str], set[tuple[str, str, str]]] = {}
        for row in rows:
            key = (row["experiment_family"], row["condition_name"])
            bucket[key] = bucket.get(key, 0) + 1
            unique_bucket.setdefault(key, set()).add(
                (
                    row["scenario_name"],
                    row["persona_name"],
                    row["chat_atmosphere"],
                )
            )
        return [
            {
                "experiment_family": family,
                "condition_name": condition,
                "n_runs": count,
                "n_unique_cells": len(unique_bucket[(family, condition)]),
            }
            for (family, condition), count in sorted(bucket.items())
        ]


class DiscussionSimulator:
    """P3 论坛/推特话题讨论的轻量事件流模拟器。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def simulate_cell(self, cell: ExperimentCell) -> list[SimulationTraceRow]:
        if cell.condition_name != "high_interaction":
            return []

        agents = build_discussion_agents(cell.experiment_family, cell.persona_name, self.config)
        scheduler = ForumScheduler(agents)
        schedules = scheduler.build_schedule(cell.max_timestep)
        events: list[ThreadEvent] = bootstrap_p3_events(
            self.config,
            cell.chat_atmosphere,
            cell.scenario_name,
        )
        rows: list[SimulationTraceRow] = [
            SimulationTraceRow(
                experiment_family=cell.experiment_family,
                scenario_name=cell.scenario_name,
                condition_name=cell.condition_name,
                persona_name=cell.persona_name,
                chat_atmosphere=cell.chat_atmosphere,
                repeat_index=cell.repeat_index,
                timestep=event.timestep,
                event_id=event.event_id,
                actor_id=event.actor_id,
                action_type=event.action_type,
                reply_to=event.reply_to,
                hyperedge_id=event.hyperedge_id,
                signal_polarity=event.signal_polarity,
                content=event.content,
            )
            for event in events
        ]

        for step in schedules:
            snapshot = build_thread_snapshot(step.timestep, events)
            for position, agent in enumerate(step.active_agents, start=1):
                reply_target = events[-1].event_id if position > 1 else None
                polarity = self._infer_signal(cell.chat_atmosphere)
                event = ThreadEvent(
                    timestep=step.timestep,
                    event_id=f"t{step.timestep}_e{position}",
                    actor_id=agent.agent_id,
                    action_type="reply" if reply_target else "post",
                    content=self._compose_content(agent.agent_id, polarity, step.timestep),
                    reply_to=reply_target,
                    signal_polarity=polarity,
                    hyperedge_id=reply_target or f"t{step.timestep}_e{position}",
                )
                events.append(event)
                rows.append(
                    SimulationTraceRow(
                        experiment_family=cell.experiment_family,
                        scenario_name=cell.scenario_name,
                        condition_name=cell.condition_name,
                        persona_name=cell.persona_name,
                        chat_atmosphere=cell.chat_atmosphere,
                        repeat_index=cell.repeat_index,
                        timestep=step.timestep,
                        event_id=event.event_id,
                        actor_id=event.actor_id,
                        action_type=event.action_type,
                        reply_to=event.reply_to,
                        hyperedge_id=event.hyperedge_id,
                        signal_polarity=event.signal_polarity,
                        content=event.content,
                    )
                )
            _ = snapshot

        return rows

    def export_preview(self, output_dir: Path, cells: List[ExperimentCell]) -> None:
        p3_cells = [cell for cell in cells if cell.condition_name == "high_interaction"]
        if not p3_cells:
            return
        traces = self.simulate_cell(p3_cells[0])
        rows = [trace.__dict__ for trace in traces]
        write_jsonl(output_dir / "simulation" / "p3_thread_preview.jsonl", rows)
        write_csv(output_dir / "simulation" / "p3_thread_preview.csv", rows)

    @staticmethod
    def _infer_signal(chat_atmosphere: str) -> str:
        if chat_atmosphere == "negative_leaning":
            return "negative"
        if chat_atmosphere == "positive_leaning":
            return "positive"
        return "neutral"

    @staticmethod
    def _compose_content(agent_id: str, polarity: str, timestep: int) -> str:
        if polarity == "negative":
            return f"{agent_id} 在 t={timestep} 强调风险可能被低估。"
        if polarity == "positive":
            return f"{agent_id} 在 t={timestep} 强调整体情况仍可控。"
        return f"{agent_id} 在 t={timestep} 表示还需要继续观察。"


class StaticExperimentRunner:
    """真实模型调用的最小运行器，先覆盖 P1/P2。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.client = LLMClient(config)
        self.scenarios = {item["name"]: item for item in config["scenarios"]}
        self.personas = {item["name"]: item for item in config["personas"]}
        self.max_concurrency = max(1, int(config.get("runtime", {}).get("max_concurrency", 1)))
        self.batch_size = max(1, int(config.get("runtime", {}).get("batch_size", self.max_concurrency)))
        self.batch_pause_seconds = float(config.get("runtime", {}).get("batch_pause_seconds", 0))

    def run_cells(
        self,
        cells: List[ExperimentCell],
        output_dir: Path,
        family_filter: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        static_cells = [
            cell
            for cell in cells
            if cell.condition_name in ("independent", "low_feedback")
            and (family_filter is None or cell.experiment_family == family_filter)
        ]
        if limit is not None:
            static_cells = static_cells[:limit]

        rows: list[dict[str, object]] = []
        panel_rows: list[dict[str, object]] = []
        tracker = ProgressTracker(output_dir=output_dir, run_name="static", total_units=len(static_cells), unit_label="cell")
        checkpoints = CellCheckpointStore(output_dir=output_dir, run_name="static", config=self.config)
        run_guard = RunGuard(self.config)
        pending_cells: list[ExperimentCell] = []
        for cell in static_cells:
            cached = checkpoints.load(cell)
            if cached is None:
                pending_cells.append(cell)
                continue
            cached_row = cached.get("artifacts", {}).get("row")
            if isinstance(cached_row, dict):
                rows.append(cached_row)
                tracker.advance()
            else:
                pending_cells.append(cell)
        try:
            for batch_index, batch_cells in enumerate(self._iter_batches(pending_cells), start=1):
                run_guard.check()
                with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(len(batch_cells), 1))) as executor:
                    future_map = {
                        executor.submit(self._run_single_cell, cell, run_guard): cell
                        for cell in batch_cells
                    }
                    for future in as_completed(future_map):
                        tracker.set_current_cell(future_map[future])
                        row = future.result().__dict__
                        run_guard.record_result(bool(row.get("success")), str(row.get("error_message") or ""))
                        rows.append(row)
                        tracker.record_observation(row)
                        tracker.advance()
                        checkpoints.save(future_map[future], {"row": row})
                if self.batch_pause_seconds > 0 and batch_index * self.batch_size < len(pending_cells):
                    time.sleep(self.batch_pause_seconds)
        except BaseException:
            tracker.close(status="aborted")
            raise
        else:
            tracker.close(status="completed")

        if family_filter is None and limit is None:
            rows.extend(self._materialize_reused_static_rows(rows))

        rows.sort(
            key=lambda row: (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                str(row["condition_name"]),
                int(row["repeat_index"]),
            )
        )
        write_jsonl(output_dir / "results" / "static_results.jsonl", rows)
        write_csv(output_dir / "results" / "static_results.csv", rows)
        write_csv(output_dir / "results" / "static_summary.csv", self._build_summary(rows))
        self._assert_required_results(
            output_dir,
            ["static_results.jsonl", "static_results.csv", "static_summary.csv"],
        )
        return rows

    def _materialize_reused_static_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        aliases = [
            row
            for row in ExperimentPlanner(self.config).build_reuse_aliases()
            if str(row["target_condition_name"]) in {"independent", "low_feedback"}
        ]
        return self._materialize_alias_rows(rows, aliases)

    @staticmethod
    def _row_cell_key(row: dict[str, object], prefix: str = "") -> tuple[str, str, str, str, str, int]:
        return (
            str(row[f"{prefix}experiment_family"]),
            str(row[f"{prefix}scenario_name"]),
            str(row[f"{prefix}condition_name"]),
            str(row[f"{prefix}persona_name"]),
            str(row[f"{prefix}chat_atmosphere"]),
            int(row[f"{prefix}repeat_index"]),
        )

    @classmethod
    def _materialize_alias_rows(
        cls,
        rows: list[dict[str, object]],
        aliases: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not rows or not aliases:
            return []
        source_index: dict[tuple[str, str, str, str, str, int], list[dict[str, object]]] = {}
        existing_keys: set[tuple[str, str, str, str, str, int]] = set()
        for row in rows:
            key = cls._row_cell_key(row)
            source_index.setdefault(key, []).append(row)
            existing_keys.add(key)

        materialized: list[dict[str, object]] = []
        for alias in aliases:
            source_key = cls._row_cell_key(alias, prefix="source_")
            target_key = cls._row_cell_key(alias, prefix="target_")
            if target_key in existing_keys:
                continue
            source_rows = source_index.get(source_key, [])
            if not source_rows:
                continue
            for source_row in source_rows:
                target_row = dict(source_row)
                target_row["experiment_family"] = alias["target_experiment_family"]
                target_row["scenario_name"] = alias["target_scenario_name"]
                target_row["condition_name"] = alias["target_condition_name"]
                target_row["persona_name"] = alias["target_persona_name"]
                target_row["chat_atmosphere"] = alias["target_chat_atmosphere"]
                target_row["repeat_index"] = alias["target_repeat_index"]
                if "repeat" in target_row:
                    target_row["repeat"] = alias["target_repeat_index"]
                materialized.append(target_row)
            existing_keys.add(target_key)
        return materialized

    def _run_single_cell(self, cell: ExperimentCell, run_guard: RunGuard | None = None) -> StaticRunRow:
        scenario = self.scenarios[cell.scenario_name]
        persona = self.personas.get(cell.persona_name, self.personas["baseline_open"])
        report_mapping = self._build_report_mapping(cell)
        system_prompt = build_system_message(persona["profile"])
        user_prompt = build_user_prompt(
            config=self.config,
            scenario=scenario,
            condition_name=cell.condition_name,
            chat_atmosphere=cell.chat_atmosphere,
            report_mapping=report_mapping,
            persona_behavior_text=get_persona_behavior_text(cell.persona_name),
            persona_name=cell.persona_name,
            reason_focus_hint=_reason_focus_hint(cell.repeat_index - 1),
            social_shuffle_seed=(
                f"static|{cell.experiment_family}|{cell.scenario_name}|{cell.condition_name}|"
                f"{cell.persona_name}|{cell.chat_atmosphere}|{cell.repeat_index}"
            ),
        )
        try:
            if run_guard is not None:
                run_guard.check()
            raw_response = self.client.generate_text(
                system_prompt,
                user_prompt,
                request_meta={
                    "runner": "static",
                    "experiment_family": cell.experiment_family,
                    "scenario_name": cell.scenario_name,
                    "condition_name": cell.condition_name,
                    "persona_name": cell.persona_name,
                    "chat_atmosphere": cell.chat_atmosphere,
                    "repeat_index": cell.repeat_index,
                },
            )
            static_cfg = self.config.get("static_measurement", {})
            allow_neutral_final = str(static_cfg.get("final_choice_mode", "binary")).lower() == "ternary"
            decision = LLMDecisionParser.parse_decision(
                raw_response,
                report_mapping=report_mapping,
                allow_neutral_final=allow_neutral_final,
            )
            metrics = encode_metrics(decision)
            return StaticRunRow(
                experiment_family=cell.experiment_family,
                scenario_name=cell.scenario_name,
                condition_name=cell.condition_name,
                persona_name=cell.persona_name,
                chat_atmosphere=cell.chat_atmosphere,
                report_1_polarity=report_mapping["report_1"],
                report_2_polarity=report_mapping["report_2"],
                repeat_index=cell.repeat_index,
                timestep=1,
                success=True,
                error_message=None,
                final_choice=decision.final_choice,
                credible_choice=decision.credible_choice,
                share_choice=decision.share_choice,
                comment_choice=decision.comment_choice,
                like_choice=decision.like_choice,
                reason=decision.reason,
                reason_signal=decision.reason_signal,
                nsb=metrics.nsb,
                cb=metrics.cb,
                sb=metrics.sb,
                eb=metrics.eb,
                nci_comp=metrics.nci_comp,
                raw_response=raw_response,
            )
        except Exception as exc:
            return StaticRunRow(
                experiment_family=cell.experiment_family,
                scenario_name=cell.scenario_name,
                condition_name=cell.condition_name,
                persona_name=cell.persona_name,
                chat_atmosphere=cell.chat_atmosphere,
                report_1_polarity=report_mapping["report_1"],
                report_2_polarity=report_mapping["report_2"],
                repeat_index=cell.repeat_index,
                timestep=1,
                success=False,
                error_message=str(exc),
                final_choice="neutral",
                credible_choice="neutral",
                share_choice="none",
                comment_choice="neutral",
                like_choice="none",
                reason=None,
                reason_signal=None,
                nsb=None,
                cb=None,
                sb=None,
                eb=None,
                nci_comp=None,
                raw_response="",
            )

    @staticmethod
    def _assert_required_results(output_dir: Path, names: list[str]) -> None:
        missing = [name for name in names if not (output_dir / "results" / name).exists()]
        if missing:
            raise RuntimeError(f"结果目录缺少必要文件: {', '.join(missing)}")

    @staticmethod
    def _build_summary(rows: List[dict[str, object]]) -> list[dict[str, object]]:
        bucket: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
        for row in rows:
            if not row.get("success", True):
                continue
            key = (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                str(row["condition_name"]),
                str(row["persona_name"]),
                str(row["chat_atmosphere"]),
            )
            if key not in bucket:
                bucket[key] = {"n": 0.0, "nsb_sum": 0.0, "nci_comp_sum": 0.0}
            bucket[key]["n"] += 1.0
            bucket[key]["nsb_sum"] += float(row["nsb"])
            bucket[key]["nci_comp_sum"] += float(row["nci_comp"])
        return [
            {
                "experiment_family": family,
                "scenario_name": scenario,
                "condition_name": condition,
                "persona_name": persona_name,
                "chat_atmosphere": chat_atmosphere,
                "n_obs": int(stats["n"]),
                "mean_nsb": round(stats["nsb_sum"] / stats["n"], 4),
                "mean_nci_comp": round(stats["nci_comp_sum"] / stats["n"], 4),
            }
            for (family, scenario, condition, persona_name, chat_atmosphere), stats in sorted(bucket.items())
        ]

    def _iter_batches(self, cells: List[ExperimentCell]) -> List[List[ExperimentCell]]:
        return [cells[index:index + self.batch_size] for index in range(0, len(cells), self.batch_size)]

    @staticmethod
    def _build_report_mapping(cell: ExperimentCell) -> dict[str, str]:
        # Keep report order counterbalanced inside every cell, otherwise a 10-repeat
        # cell can accidentally compare different report-order distributions.
        if cell.repeat_index % 2 == 1:
            return {"report_1": "positive", "report_2": "negative"}
        return {"report_1": "negative", "report_2": "positive"}


class P3ExperimentRunner:
    """P3 的最小真实模型运行器。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.client = LLMClient(config)
        self.scenarios = {item["name"]: item for item in config["scenarios"]}
        self.personas = {item["name"]: item for item in config["personas"]}
        self.max_concurrency = max(1, int(config.get("runtime", {}).get("max_concurrency", 1)))
        self.batch_size = max(1, int(config.get("runtime", {}).get("batch_size", self.max_concurrency)))
        self.batch_pause_seconds = float(config.get("runtime", {}).get("batch_pause_seconds", 0))
        self.activation_schedules = ActivationScheduleStore(config)

    def run_cells(
        self,
        cells: List[ExperimentCell],
        output_dir: Path,
        family_filter: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        p3_cells = [
            cell
            for cell in cells
            if cell.condition_name == "high_interaction"
            and (family_filter is None or cell.experiment_family == family_filter)
        ]
        if limit is not None:
            p3_cells = p3_cells[:limit]

        rows: list[dict[str, object]] = []
        panel_rows: list[dict[str, object]] = []
        probe_rows: list[dict[str, object]] = []
        timestep_rows: list[dict[str, object]] = []
        prototype_panel_rows: list[dict[str, object]] = []
        total_actions = self._estimate_total_actions(p3_cells)
        tracker = ProgressTracker(output_dir=output_dir, run_name="p3", total_units=total_actions, unit_label="step")
        checkpoints = CellCheckpointStore(output_dir=output_dir, run_name="p3", config=self.config)
        run_guard = RunGuard(self.config)
        pending_cells: list[ExperimentCell] = []
        for cell in p3_cells:
            cached = checkpoints.load(cell)
            if cached is None:
                pending_cells.append(cell)
                continue
            artifacts = cached.get("artifacts", {})
            cached_rows = artifacts.get("rows", [])
            cached_panel_rows = artifacts.get("panel_rows", [])
            cached_probe_rows = artifacts.get("probe_rows", [])
            cached_timestep_rows = artifacts.get("timestep_rows", [])
            cached_prototype_rows = artifacts.get("prototype_panel_rows", [])
            if not isinstance(cached_rows, list):
                pending_cells.append(cell)
                continue
            rows.extend(item for item in cached_rows if isinstance(item, dict))
            panel_rows.extend(item for item in cached_panel_rows if isinstance(item, dict))
            probe_rows.extend(item for item in cached_probe_rows if isinstance(item, dict))
            timestep_rows.extend(item for item in cached_timestep_rows if isinstance(item, dict))
            prototype_panel_rows.extend(item for item in cached_prototype_rows if isinstance(item, dict))
            tracker.advance(len(cached_rows))

        def on_row(row: P3RunRow) -> None:
            tracker.record_observation(row.__dict__)
            tracker.advance()
            run_guard.record_result(row.success, row.error_message)

        try:
            for batch_index, batch_cells in enumerate(self._iter_batches(pending_cells), start=1):
                run_guard.check()
                with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(len(batch_cells), 1))) as executor:
                    future_map = {
                        executor.submit(
                            self._run_single_cell,
                            cell,
                            output_dir,
                            on_row,
                            run_guard,
                        ): cell
                        for cell in batch_cells
                    }
                    for future in as_completed(future_map):
                        tracker.set_current_cell(future_map[future])
                        (
                            cell_rows,
                            cell_panel_rows,
                            cell_probe_rows,
                            cell_timestep_rows,
                            cell_prototype_panel_rows,
                        ) = future.result()
                        serialized_rows = [item.__dict__ for item in cell_rows]
                        serialized_panel_rows = [item.__dict__ for item in cell_panel_rows]
                        rows.extend(serialized_rows)
                        panel_rows.extend(serialized_panel_rows)
                        probe_rows.extend(cell_probe_rows)
                        timestep_rows.extend(cell_timestep_rows)
                        prototype_panel_rows.extend(cell_prototype_panel_rows)
                        checkpoints.save(
                            future_map[future],
                            {
                                "rows": serialized_rows,
                                "panel_rows": serialized_panel_rows,
                                "probe_rows": cell_probe_rows,
                                "timestep_rows": cell_timestep_rows,
                                "prototype_panel_rows": cell_prototype_panel_rows,
                            },
                        )
                if self.batch_pause_seconds > 0 and batch_index * self.batch_size < len(pending_cells):
                    time.sleep(self.batch_pause_seconds)
        except BaseException:
            tracker.close(status="aborted")
            raise
        else:
            tracker.close(status="completed")

        if family_filter is None and limit is None:
            aliases = [
                row
                for row in ExperimentPlanner(self.config).build_reuse_aliases()
                if str(row["target_condition_name"]) == "high_interaction"
            ]
            rows.extend(StaticExperimentRunner._materialize_alias_rows(rows, aliases))
            panel_rows.extend(StaticExperimentRunner._materialize_alias_rows(panel_rows, aliases))
            probe_rows.extend(StaticExperimentRunner._materialize_alias_rows(probe_rows, aliases))
            timestep_rows.extend(StaticExperimentRunner._materialize_alias_rows(timestep_rows, aliases))
            prototype_panel_rows.extend(StaticExperimentRunner._materialize_alias_rows(prototype_panel_rows, aliases))

        rows.sort(
            key=lambda row: (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                int(row["repeat_index"]),
                int(row["timestep"]),
                str(row["event_id"]),
            )
        )
        write_jsonl(output_dir / "results" / "p3_results.jsonl", rows)
        write_csv(output_dir / "results" / "p3_results.csv", rows)
        if panel_rows:
            panel_rows.sort(
                key=lambda row: (
                    str(row["experiment_family"]),
                    str(row["scenario_name"]),
                    int(row["repeat_index"]),
                    int(row["timestep"]),
                    str(row["agent_id"]),
                )
            )
            write_jsonl(output_dir / "results" / "p3_agent_panel.jsonl", panel_rows)
            write_csv(output_dir / "results" / "p3_agent_panel.csv", panel_rows)
        if probe_rows:
            probe_rows.sort(
                key=lambda row: (
                    str(row["experiment_family"]),
                    str(row["scenario_name"]),
                    int(row["repeat_index"]),
                    str(row["phase"]),
                    str(row["agent_id"]),
                )
            )
            write_jsonl(output_dir / "results" / "p3_probe_results.jsonl", probe_rows)
            write_csv(output_dir / "results" / "p3_probe_results.csv", probe_rows)
        if timestep_rows:
            timestep_rows.sort(
                key=lambda row: (
                    str(row["experiment_family"]),
                    str(row["scenario_name"]),
                    int(row["repeat_index"]),
                    int(row["timestep"]),
                )
            )
            write_jsonl(output_dir / "results" / "p3_timestep_summary.jsonl", timestep_rows)
            write_csv(output_dir / "results" / "p3_timestep_summary.csv", timestep_rows)
        if prototype_panel_rows:
            prototype_panel_rows.sort(
                key=lambda row: (
                    str(row["experiment_family"]),
                    str(row["scenario_name"]),
                    int(row["repeat_index"]),
                    int(row["timestep"]),
                    str(row["prototype_id"]),
                )
            )
            write_jsonl(output_dir / "results" / "p3_prototype_panel.jsonl", prototype_panel_rows)
            write_csv(output_dir / "results" / "p3_prototype_panel.csv", prototype_panel_rows)
        p3_cell_summary = self._build_cell_summary(rows, probe_rows=probe_rows, panel_rows=panel_rows)
        write_csv(output_dir / "results" / "p3_cell_summary.csv", p3_cell_summary)
        write_csv(output_dir / "results" / "p3_agent_persona_summary.csv", self._build_agent_persona_summary(rows))
        write_csv(output_dir / "results" / "p3_summary.csv", self._build_summary(p3_cell_summary))
        StaticExperimentRunner._assert_required_results(
            output_dir,
            [
                "p3_results.jsonl",
                "p3_results.csv",
                "p3_agent_panel.csv",
                "p3_prototype_panel.csv",
                "p3_cell_summary.csv",
                "p3_summary.csv",
            ],
        )
        return rows

    def _run_single_cell(
        self,
        cell: ExperimentCell,
        output_dir: Path,
        on_row=None,
        run_guard: RunGuard | None = None,
    ) -> tuple[list[P3RunRow], list[P3AgentPanelRow], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        if run_guard is not None:
            run_guard.check()
        scenario = self.scenarios[cell.scenario_name]
        report_mapping = StaticExperimentRunner._build_report_mapping(cell)
        agents = build_discussion_agents(cell.experiment_family, cell.persona_name, self.config)
        scheduler = ForumScheduler(agents)
        # P3 先从热启动事件开始，再在时间步中逐步追加新事件。
        events: list[ThreadEvent] = bootstrap_p3_events(
            self.config,
            cell.chat_atmosphere,
            cell.scenario_name,
        )
        rows: list[P3RunRow] = []
        panel_rows: list[P3AgentPanelRow] = []
        probe_rows: list[dict[str, object]] = []
        timestep_rows: list[dict[str, object]] = []
        prototype_panel_rows: list[dict[str, object]] = []
        previous_nci_by_agent: dict[str, int] = {}
        behavior_profiles = {
            agent.agent_id: get_persona_behavior_profile(agent.persona_name)
            for agent in agents
        }
        # 这里保存每个 agent 的局部状态，让下一轮不至于像“重新开局”。
        agent_states = {
            agent.agent_id: AgentDynamicState(
                agent_id=agent.agent_id,
                last_choice="neutral",
                last_final_choice="neutral",
                recent_choice_history=[],
                recent_focus_history=[],
                last_visible_hyperedge_ids=[],
                participated_hyperedge_ids=[],
            )
            for agent in agents
        }
        p3_cfg = self.config.get("p3", {})
        prototype_initializer = P2PrototypeInitializer(self.config, self.client)
        prototype_init_meta_by_persona: dict[str, dict[str, object]] = {}
        if prototype_initializer.enabled():
            prototypes = prototype_initializer.initialize(cell=cell, scenario=scenario, output_dir=output_dir)
            prototype_init_meta_by_persona = prototype_initializer.apply_to_agents(
                prototypes=prototypes,
                cell=cell,
                agents=agents,
                agent_states=agent_states,
            )
        else:
            prototype_init_meta_by_persona = {
                agent.persona_name: {
                    "prototype_initialization_method": "disabled",
                    "prototype_initialization_cached": False,
                    "prototype_initialization_reason": "inherit_p2_state disabled",
                }
                for agent in agents
            }
        consensus_hold = 0
        stalled_steps = 0
        visible_context_limit = int(p3_cfg.get("visible_context_limit", 8))
        visibility_weights = build_visibility_weights(self.config)
        consensus_threshold = float(p3_cfg.get("consensus_threshold", 0.9))
        consensus_hold_steps = int(p3_cfg.get("consensus_hold_steps", 2))
        stalled_limit = int(p3_cfg.get("stalled_steps", 4))
        max_active_agents = int(p3_cfg.get("max_active_agents", 3))
        activation_cfg = p3_cfg.get("activation", {}) if isinstance(p3_cfg.get("activation", {}), dict) else {}
        exploration_slots = int(activation_cfg.get("exploration_slots", 0))
        stratified_ab_activation = bool(activation_cfg.get("stratified_ab_activation", False))
        early_persona_floor_steps = int(activation_cfg.get("early_persona_floor_steps", 0))
        dynamics_cfg = p3_cfg.get("dynamics", {})
        silent_update_cfg = p3_cfg.get("prototype_silent_update", {})
        silent_update_cfg = silent_update_cfg if isinstance(silent_update_cfg, dict) else {}
        silent_update_enabled = bool(silent_update_cfg.get("enabled", False))
        stop_reason: str | None = None
        probe_cfg = p3_cfg.get("private_probe", {})
        probe_phases = set(probe_cfg.get("phases", ["entry", "terminal"])) if probe_cfg.get("enabled", True) else set()
        mid_probe_interval = int(probe_cfg.get("mid_interval", 0)) if probe_cfg.get("enabled", True) else 0
        min_timestep_before_consensus_stop = int(p3_cfg.get("min_timestep_before_consensus_stop", 0))
        prototype_panel_cfg = p3_cfg.get("prototype_panel", {})
        prototype_panel_cfg = prototype_panel_cfg if isinstance(prototype_panel_cfg, dict) else {}
        prototype_panel_enabled = bool(prototype_panel_cfg.get("enabled", False))
        prototype_panel_include_active = bool(prototype_panel_cfg.get("include_active_agents", False))

        entry_snapshot = build_thread_snapshot(
            0,
            events,
            max_events=visible_context_limit,
            visibility_weights=visibility_weights,
        )
        for agent in agents:
            self._refresh_agent_from_snapshot(
                state=agent_states[agent.agent_id],
                snapshot=entry_snapshot,
                behavior_profile=behavior_profiles[agent.agent_id],
                dynamics_cfg=dynamics_cfg,
                persona_name=agent.persona_name,
                apply_tendency_update=False,
            )
        if prototype_panel_enabled:
            prototype_panel_rows.extend(
                build_prototype_panel_rows(
                    cell=cell,
                    agents=agents,
                    agent_states=agent_states,
                    timestep=0,
                    active_ids=set(),
                    init_meta_by_persona=prototype_init_meta_by_persona,
                    include_active_agents=prototype_panel_include_active,
                )
            )
        if "entry" in probe_phases:
            probe_rows.extend(
                self._run_population_probe(
                    cell=cell,
                    agents=agents,
                    agent_states=agent_states,
                    snapshot=entry_snapshot,
                    report_mapping=report_mapping,
                    scenario=scenario,
                    phase="entry",
                    mutate_state=False,
                )
            )

        for timestep in range(1, cell.max_timestep + 1):
            step_rows: list[P3RunRow] = []
            snapshot = build_thread_snapshot(
                timestep,
                events,
                max_events=visible_context_limit,
                visibility_weights=visibility_weights,
            )
            for agent in agents:
                self._refresh_agent_from_snapshot(
                    state=agent_states[agent.agent_id],
                    snapshot=snapshot,
                    behavior_profile=behavior_profiles[agent.agent_id],
                    dynamics_cfg=dynamics_cfg,
                    persona_name=agent.persona_name,
                    apply_tendency_update=False,
                )
            replayed_agents = self.activation_schedules.replay_agents(
                cell=cell,
                timestep=timestep,
                agents=agents,
                max_active_agents=max_active_agents,
            )
            if replayed_agents is not None:
                replay_scores = scheduler.score_agents(
                    timestep=timestep,
                    agent_states=agent_states,
                    snapshot=snapshot,
                )
                step = StepSchedule(
                    timestep=timestep,
                    active_agents=replayed_agents,
                    activation_scores=replay_scores,
                )
            elif self._uses_stratified_ab_activation(cell, stratified_ab_activation):
                step = self._select_stratified_persona_step(
                    scheduler=scheduler,
                    agents=agents,
                    timestep=timestep,
                    agent_states=agent_states,
                    snapshot=snapshot,
                    max_active_agents=max_active_agents,
                )
            else:
                step = scheduler.select_active_agents(
                    timestep=timestep,
                    agent_states=agent_states,
                    snapshot=snapshot,
                    max_active_agents=max_active_agents,
                    exploration_slots=exploration_slots,
                )
            self.activation_schedules.record(
                output_dir=output_dir,
                cell=cell,
                timestep=timestep,
                active_agents=step.active_agents,
                replayed=replayed_agents is not None,
            )
            active_ids = {agent.agent_id for agent in step.active_agents}
            activation_scores = step.activation_scores or {}
            active_rank = {agent.agent_id: rank for rank, agent in enumerate(step.active_agents, start=1)}
            for agent in step.active_agents:
                self._apply_snapshot_tendency_update(
                    state=agent_states[agent.agent_id],
                    snapshot=snapshot,
                    behavior_profile=behavior_profiles[agent.agent_id],
                    dynamics_cfg=dynamics_cfg,
                    persona_name=agent.persona_name,
                )
            thread_summary = self._build_thread_summary(events, report_mapping)
            step_signal_values: list[int] = []
            negative_share: float | None = None
            had_new_content = False
            for position, agent in enumerate(step.active_agents, start=1):
                try:
                    if run_guard is not None:
                        run_guard.check()
                    state = agent_states[agent.agent_id]
                    agent_persona = self.personas.get(agent.persona_name, self.personas["baseline_open"])
                    persona_behavior_text = get_persona_behavior_text(agent.persona_name)
                    persona_behavior_profile = get_persona_behavior_profile(agent.persona_name)
                    system_prompt = build_system_message(agent_persona["profile"])
                    user_prompt = build_p3_timestep_prompt(
                        config=self.config,
                        scenario=scenario,
                        persona_profile=agent_persona["profile"],
                        persona_behavior_text=persona_behavior_text,
                        chat_atmosphere=cell.chat_atmosphere,
                        report_mapping=report_mapping,
                        timestep=step.timestep,
                        visible_context=snapshot.visible_context,
                        thread_summary=thread_summary,
                        agent_state_text=self._render_agent_state(state),
                        reason_focus_hint=_reason_focus_hint(step.timestep + position - 2),
                        social_shuffle_seed=(
                            f"p3_static_ref|{cell.experiment_family}|{cell.scenario_name}|"
                            f"{cell.chat_atmosphere}|{cell.repeat_index}"
                        ),
                        agent_persona_name=agent.persona_name,
                    )
                    raw_response = self.client.generate_text(
                        system_prompt,
                        user_prompt,
                        request_meta={
                            "runner": "p3",
                            "experiment_family": cell.experiment_family,
                            "scenario_name": cell.scenario_name,
                            "condition_name": cell.condition_name,
                            "persona_name": cell.persona_name,
                            "agent_persona_name": agent.persona_name,
                            "chat_atmosphere": cell.chat_atmosphere,
                            "repeat_index": cell.repeat_index,
                            "timestep": timestep,
                            "actor_id": agent.agent_id,
                        },
                    )
                    decision = LLMDecisionParser.parse_p3_decision(raw_response, report_mapping=report_mapping)
                    previous_nci = previous_nci_by_agent.get(agent.agent_id)
                    _ = encode_metrics(
                        decision,
                        previous_nci_comp=previous_nci,
                    )

                    decision = self._calibrate_initial_seed_floor_decision(
                        decision=decision,
                        state=state,
                        visible_events=snapshot.visible_events,
                        actor_id=agent.agent_id,
                        timestep=timestep,
                        active_position=position,
                    )
                    decision = self._calibrate_runaway_consensus_decision(
                        decision=decision,
                        state=state,
                        visible_events=snapshot.visible_events,
                        actor_id=agent.agent_id,
                        timestep=timestep,
                        dynamics_cfg=dynamics_cfg,
                    )
                    decision = self._ensure_initial_floor_action(
                        decision=decision,
                        visible_events=snapshot.visible_events,
                        actor_id=agent.agent_id,
                        timestep=timestep,
                        enabled=bool(p3_cfg.get("force_initial_floor_creation", True)),
                    )
                    decision = self._ensure_ab_persona_floor_action(
                        decision=decision,
                        events=events,
                        actor_id=agent.agent_id,
                        agent_persona_name=agent.persona_name,
                        timestep=timestep,
                        enabled=(
                            self._uses_stratified_ab_activation(cell, stratified_ab_activation)
                            and early_persona_floor_steps > 0
                            and timestep <= early_persona_floor_steps
                        ),
                    )
                    decision, reply_target = self._promote_interaction_if_needed(
                        decision=decision,
                        visible_events=snapshot.visible_events,
                        actor_id=agent.agent_id,
                        timestep=timestep,
                        action_balance=p3_cfg.get("action_balance", {}),
                    )
                    decision, reply_target = self._balance_forum_action(
                        decision=decision,
                        visible_events=snapshot.visible_events,
                        state=state,
                        actor_id=agent.agent_id,
                        timestep=timestep,
                        action_balance=p3_cfg.get("action_balance", {}),
                        fallback_target=reply_target,
                    )
                    decision = self._apply_action_policy(
                        decision=decision,
                        visible_events=snapshot.visible_events,
                        timestep=timestep,
                        mode=str(p3_cfg.get("action_policy", {}).get("mode", "full")),
                    )
                    reply_target = self._resolve_reply_target(
                        decision=decision,
                        visible_events=snapshot.visible_events,
                    )
                    action_type = "reply" if decision.planned_action == "reply" else "post"
                    if decision.planned_action in {"like", "repost", "observe"}:
                        action_type = decision.planned_action
                    content = self._public_content_for_action(decision, action_type, agent.agent_id, timestep)
                    content, action_type, reply_target = self._deduplicate_comment(
                        content=content,
                        action_type=action_type,
                        reply_target=reply_target,
                        visible_events=snapshot.visible_events,
                        decision=decision,
                    )
                    public_signal = self._public_signal_from_decision(decision, action_type)
                    if action_type in {"post", "reply"} and content:
                        had_new_content = True
                    event = ThreadEvent(
                        timestep=timestep,
                        event_id=f"t{timestep}_e{position}",
                        actor_id=agent.agent_id,
                        action_type=action_type,
                        content=content,
                        reply_to=reply_target if action_type in {"reply", "like", "repost"} else None,
                        signal_polarity=public_signal,
                        salience_score=self._compute_event_salience(decision, timestep, action_type),
                    )
                    event = ThreadEvent(
                        timestep=event.timestep,
                        event_id=event.event_id,
                        actor_id=event.actor_id,
                        action_type=event.action_type,
                        content=event.content,
                        reply_to=event.reply_to,
                        signal_polarity=event.signal_polarity,
                        salience_score=event.salience_score,
                        hyperedge_id=self._resolve_event_hyperedge_id(event, events),
                    )
                    if self._enters_public_thread(action_type):
                        events.append(event)

                    if event.signal_polarity in {"positive", "negative"}:
                        step_signal_values.append(int(event.signal_polarity == "negative"))
                    negative_share = (
                        sum(step_signal_values) / len(step_signal_values)
                        if step_signal_values else None
                    )
                    metrics = encode_metrics(
                        decision,
                        previous_nci_comp=previous_nci,
                        group_negative_ratio=negative_share,
                        negative_signal_share=negative_share,
                    )
                    previous_nci_by_agent[agent.agent_id] = metrics.nci_comp
                    self._update_agent_state(
                        state=state,
                        decision=decision,
                        negative_share=negative_share,
                        behavior_profile=persona_behavior_profile,
                        dynamics_cfg=dynamics_cfg,
                        persona_name=agent.persona_name,
                        activation_score=activation_scores.get(agent.agent_id),
                        hyperedge_id=event.hyperedge_id,
                    )
                    step_rows.append(
                        P3RunRow(
                            experiment_family=cell.experiment_family,
                            scenario_name=cell.scenario_name,
                            condition_name=cell.condition_name,
                            persona_name=cell.persona_name,
                            agent_persona_name=agent.persona_name,
                            chat_atmosphere=cell.chat_atmosphere,
                            report_1_polarity=report_mapping["report_1"],
                            report_2_polarity=report_mapping["report_2"],
                            repeat_index=cell.repeat_index,
                            timestep=timestep,
                            event_id=event.event_id,
                            actor_id=event.actor_id,
                            action_type=event.action_type,
                            success=True,
                            error_message=None,
                            reply_to=event.reply_to,
                            hyperedge_id=event.hyperedge_id,
                            visible_hyperedge_ids="|".join(snapshot.visible_hyperedge_ids),
                            activation_score=activation_scores.get(agent.agent_id),
                            signal_polarity=event.signal_polarity,
                            content=event.content,
                            final_choice=decision.final_choice,
                            credible_choice=decision.credible_choice,
                            share_choice=decision.share_choice,
                            comment_choice=decision.comment_choice,
                            like_choice=decision.like_choice,
                            reason=decision.reason,
                            reason_signal=decision.reason_signal,
                            nsb=metrics.nsb,
                            cb=metrics.cb,
                            sb=metrics.sb,
                            eb=metrics.eb,
                            nci_comp=metrics.nci_comp,
                            scm=metrics.scm,
                            gcr=metrics.gcr,
                            ssa=metrics.ssa,
                            is_stop_step=False,
                            stop_reason=None,
                            raw_response=raw_response,
                        )
                    )
                    if on_row is not None:
                        on_row(step_rows[-1])
                except RunAbortError:
                    raise
                except Exception as exc:
                    failed_event = ThreadEvent(
                        timestep=timestep,
                        event_id=f"t{timestep}_e{position}",
                        actor_id=agent.agent_id,
                        action_type="observe",
                        content="",
                        reply_to=None,
                        signal_polarity=None,
                        salience_score=0.0,
                        hyperedge_id=None,
                    )
                    step_rows.append(
                        P3RunRow(
                            experiment_family=cell.experiment_family,
                            scenario_name=cell.scenario_name,
                            condition_name=cell.condition_name,
                            persona_name=cell.persona_name,
                            agent_persona_name=agent.persona_name,
                            chat_atmosphere=cell.chat_atmosphere,
                            report_1_polarity=report_mapping["report_1"],
                            report_2_polarity=report_mapping["report_2"],
                            repeat_index=cell.repeat_index,
                            timestep=timestep,
                            event_id=failed_event.event_id,
                            actor_id=failed_event.actor_id,
                            action_type=failed_event.action_type,
                            success=False,
                            error_message=str(exc),
                            reply_to=None,
                            hyperedge_id=None,
                            visible_hyperedge_ids="|".join(snapshot.visible_hyperedge_ids),
                            activation_score=activation_scores.get(agent.agent_id),
                            signal_polarity=failed_event.signal_polarity,
                            content=failed_event.content,
                            final_choice="neutral",
                            credible_choice="neutral",
                            share_choice="none",
                            comment_choice="neutral",
                            like_choice="none",
                            reason=None,
                            reason_signal=None,
                            nsb=None,
                            cb=None,
                            sb=None,
                            eb=None,
                            nci_comp=None,
                            scm=None,
                            gcr=None,
                            ssa=None,
                            is_stop_step=False,
                            stop_reason=None,
                            raw_response="",
                        )
                    )
                    if on_row is not None:
                        on_row(step_rows[-1])

            step_gcr = max(negative_share, 1 - negative_share) if negative_share is not None else None
            if step_gcr is not None and step_gcr >= consensus_threshold:
                consensus_hold += 1
            else:
                consensus_hold = 0

            if had_new_content:
                stalled_steps = 0
            else:
                stalled_steps += 1

            apply_prototype_silent_update(
                config=self.config,
                cell=cell,
                agents=agents,
                agent_states=agent_states,
                active_ids=active_ids,
                visible_negative_pressure=snapshot.negative_visibility_share,
                timestep=timestep,
            )
            self._settle_inactive_agents(
                agents=agents,
                active_ids=active_ids,
                agent_states=agent_states,
                behavior_profiles=behavior_profiles,
                refresh_private_choice=silent_update_enabled,
            )
            if prototype_panel_enabled:
                prototype_panel_rows.extend(
                    build_prototype_panel_rows(
                        cell=cell,
                        agents=agents,
                        agent_states=agent_states,
                        timestep=timestep,
                        active_ids=active_ids,
                        init_meta_by_persona=prototype_init_meta_by_persona,
                        include_active_agents=prototype_panel_include_active,
                    )
                )
            panel_rows.extend(
                self._build_agent_panel_rows(
                    cell=cell,
                    agents=agents,
                    snapshot=snapshot,
                    agent_states=agent_states,
                    activation_scores=activation_scores,
                    active_ids=active_ids,
                    active_rank=active_rank,
                    timestep=timestep,
                )
            )
            timestep_rows.append(
                self._build_timestep_summary_row(
                    cell=cell,
                    timestep=timestep,
                    step_rows=step_rows,
                    snapshot=snapshot,
                    agents=agents,
                    agent_states=agent_states,
                    active_ids=active_ids,
                )
            )
            if mid_probe_interval > 0 and timestep % mid_probe_interval == 0:
                probe_rows.extend(
                    self._run_population_probe(
                        cell=cell,
                        agents=agents,
                        agent_states=agent_states,
                        snapshot=snapshot,
                        report_mapping=report_mapping,
                        scenario=scenario,
                        phase=f"mid_t{timestep}",
                        mutate_state=True,
                    )
                )

            if consensus_hold >= consensus_hold_steps and timestep >= min_timestep_before_consensus_stop:
                stop_reason = "high_consensus"
            elif stalled_steps >= stalled_limit:
                stop_reason = "discussion_stalled"
            elif timestep >= cell.max_timestep:
                stop_reason = "max_timestep"

            if stop_reason and step_rows:
                last_row = step_rows[-1]
                step_rows[-1] = P3RunRow(
                    experiment_family=last_row.experiment_family,
                    scenario_name=last_row.scenario_name,
                    condition_name=last_row.condition_name,
                    persona_name=last_row.persona_name,
                    agent_persona_name=last_row.agent_persona_name,
                    chat_atmosphere=last_row.chat_atmosphere,
                    report_1_polarity=last_row.report_1_polarity,
                    report_2_polarity=last_row.report_2_polarity,
                    repeat_index=last_row.repeat_index,
                    timestep=last_row.timestep,
                    event_id=last_row.event_id,
                    actor_id=last_row.actor_id,
                    action_type=last_row.action_type,
                    success=last_row.success,
                    error_message=last_row.error_message,
                    reply_to=last_row.reply_to,
                    hyperedge_id=last_row.hyperedge_id,
                    visible_hyperedge_ids=last_row.visible_hyperedge_ids,
                    activation_score=last_row.activation_score,
                    signal_polarity=last_row.signal_polarity,
                    content=last_row.content,
                    final_choice=last_row.final_choice,
                    credible_choice=last_row.credible_choice,
                    share_choice=last_row.share_choice,
                    comment_choice=last_row.comment_choice,
                    like_choice=last_row.like_choice,
                    reason=last_row.reason,
                    reason_signal=last_row.reason_signal,
                    nsb=last_row.nsb,
                    cb=last_row.cb,
                    sb=last_row.sb,
                    eb=last_row.eb,
                    nci_comp=last_row.nci_comp,
                    scm=last_row.scm,
                    gcr=last_row.gcr,
                    ssa=last_row.ssa,
                    is_stop_step=True,
                    stop_reason=stop_reason,
                    raw_response=last_row.raw_response,
                )

            rows.extend(step_rows)
            if stop_reason:
                break

        terminal_snapshot = build_thread_snapshot(
            rows[-1].timestep if rows else 0,
            events,
            max_events=visible_context_limit,
            visibility_weights=visibility_weights,
        )
        if "terminal" in probe_phases:
            probe_rows.extend(
                self._run_population_probe(
                    cell=cell,
                    agents=agents,
                    agent_states=agent_states,
                    snapshot=terminal_snapshot,
                    report_mapping=report_mapping,
                    scenario=scenario,
                    phase="terminal",
                    mutate_state=False,
                )
            )

        return rows, panel_rows, probe_rows, timestep_rows, prototype_panel_rows

    @staticmethod
    def _apply_action_policy(decision, visible_events: list[ThreadEvent], timestep: int, mode: str):
        """Apply a legal action-channel ablation without changing private choices."""
        normalized = mode.strip().lower()
        if normalized == "full":
            return decision
        if normalized not in {"reply_chain", "post_only", "like_channel"}:
            raise ValueError(f"未知 P3 action_policy.mode: {mode}")
        if timestep == 1:
            return replace(decision, planned_action="post", reply_target=None)

        dynamic = [
            event
            for event in visible_events
            if not event.event_id.startswith("seed_") and event.action_type in {"post", "reply"}
        ]
        target = dynamic[-1].event_id if dynamic else None
        if normalized == "reply_chain":
            action = "reply" if target else "observe"
            return replace(decision, planned_action=action, reply_target=target)
        if normalized == "like_channel":
            action = "like" if target else "observe"
            return replace(decision, planned_action=action, reply_target=target)
        if decision.planned_action == "observe":
            return replace(decision, reply_target=None)
        return replace(decision, planned_action="post", reply_target=None)

    def _estimate_total_actions(self, cells: List[ExperimentCell]) -> int:
        total = 0
        max_active_agents = int(self.config.get("p3", {}).get("max_active_agents", 3))
        for cell in cells:
            agents = build_discussion_agents(cell.experiment_family, cell.persona_name, self.config)
            total += min(max_active_agents, len(agents)) * cell.max_timestep
        return max(total, 1)

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _resolve_event_hyperedge_id(event: ThreadEvent, existing_events: list[ThreadEvent]) -> str | None:
        if event.action_type == "observe":
            return None
        if event.action_type == "post" and not event.reply_to:
            return event.event_id
        by_id = {item.event_id: item for item in existing_events}
        current_id = event.reply_to
        seen: set[str] = set()
        while current_id and current_id not in seen:
            parent = by_id.get(current_id)
            if parent is None:
                return current_id
            if parent.hyperedge_id:
                return parent.hyperedge_id
            if parent.action_type == "post" and not parent.reply_to:
                return parent.event_id
            seen.add(current_id)
            current_id = parent.reply_to
        return event.reply_to or event.event_id

    @classmethod
    def _refresh_agent_from_snapshot(
        cls,
        state: AgentDynamicState,
        snapshot,
        behavior_profile: dict[str, float],
        dynamics_cfg: dict[str, object] | None = None,
        persona_name: str | None = None,
        apply_tendency_update: bool = True,
    ) -> None:
        dynamics_cfg = dynamics_cfg or {}
        state.last_visible_hyperedge_ids = list(snapshot.visible_hyperedge_ids)
        state.dominant_hyperedge_id = snapshot.dominant_hyperedge_id
        visible_pressure = (
            snapshot.negative_visibility_share
            if snapshot.negative_visibility_share is not None
            else state.latent_neg_tendency
        )
        state.visible_negative_pressure = cls._clip01(float(visible_pressure))
        if snapshot.negative_visibility_share is None:
            state.perceived_majority = "unclear"
        elif snapshot.negative_visibility_share >= 0.6:
            state.perceived_majority = "negative"
        elif snapshot.negative_visibility_share <= 0.4:
            state.perceived_majority = "positive"
        else:
            state.perceived_majority = "split"

        if apply_tendency_update:
            cls._apply_snapshot_tendency_update(
                state=state,
                snapshot=snapshot,
                behavior_profile=behavior_profile,
                dynamics_cfg=dynamics_cfg,
                persona_name=persona_name,
            )

        participated = set(state.participated_hyperedge_ids or [])
        visible = set(snapshot.visible_hyperedge_ids)
        if participated & visible:
            state.notification_score = cls._clip01(state.notification_score + 0.12)
        else:
            state.notification_score = cls._clip01(state.notification_score * 0.75)

    @classmethod
    def _apply_snapshot_tendency_update(
        cls,
        state: AgentDynamicState,
        snapshot,
        behavior_profile: dict[str, float],
        dynamics_cfg: dict[str, object] | None = None,
        persona_name: str | None = None,
    ) -> None:
        """Apply full public-pressure tendency updates to active agents only."""
        if snapshot.negative_visibility_share is None:
            return
        dynamics_cfg = dynamics_cfg or {}
        exposure_shift_weight = float(dynamics_cfg.get("exposure_shift_weight", 0.08))
        arousal_shift_weight = float(dynamics_cfg.get("arousal_shift_weight", 0.03))
        risk_shift_weight = float(dynamics_cfg.get("risk_shift_weight", 0.05))
        exposure_shift = (state.visible_negative_pressure - 0.5) * exposure_shift_weight
        exposure_shift *= behavior_profile.get("majority_sensitivity", 1.0)
        arousal_shift = (state.risk_arousal - 0.5) * arousal_shift_weight
        state.latent_neg_tendency = cls._clip01(
            state.latent_neg_tendency + exposure_shift + arousal_shift
        )
        risk_shift = (state.visible_negative_pressure - 0.5) * risk_shift_weight
        state.risk_arousal = cls._clip01(
            state.risk_arousal
            + risk_shift * behavior_profile.get("majority_sensitivity", 1.0)
        )
        authority_floor = float(dynamics_cfg.get("authority_trusting_exposure_pressure_floor", 0.5))
        if persona_name == "authority_trusting" and state.visible_negative_pressure > authority_floor:
            authority_weight = float(dynamics_cfg.get("authority_trusting_exposure_shift_weight", 0.035))
            authority_cap = float(dynamics_cfg.get("authority_trusting_exposure_shift_cap", 0.045))
            authority_shift = max(0.0, state.visible_negative_pressure - authority_floor) * authority_weight
            authority_shift = min(authority_cap, authority_shift)
            state.latent_neg_tendency = cls._clip01(state.latent_neg_tendency + authority_shift)
            state.risk_arousal = cls._clip01(state.risk_arousal + authority_shift * 0.6)

    @classmethod
    def _settle_inactive_agents(
        cls,
        agents: list,
        active_ids: set[str],
        agent_states: dict[str, AgentDynamicState],
        behavior_profiles: dict[str, dict[str, float]],
        refresh_private_choice: bool,
    ) -> None:
        for agent in agents:
            if agent.agent_id in active_ids:
                continue
            state = agent_states[agent.agent_id]
            profile = behavior_profiles[agent.agent_id]
            state.silence_steps += 1
            if state.cooldown_steps_remaining > 0:
                state.cooldown_steps_remaining -= 1
                state.participation_state = "cooldown"
            else:
                state.participation_state = "dormant"
            state.last_action = "observe"
            state.fatigue = cls._clip01(state.fatigue - 0.05)
            state.engagement_tendency = cls._clip01(
                state.engagement_tendency
                + (state.notification_score - 0.5) * 0.03 * profile.get("engagement_gain", 1.0)
            )
            if refresh_private_choice:
                if state.latent_neg_tendency > 0.54:
                    state.last_final_choice = "negative"
                elif state.latent_neg_tendency < 0.46:
                    state.last_final_choice = "positive"
                else:
                    state.last_final_choice = "neutral"

    @staticmethod
    def _build_agent_panel_rows(
        cell: ExperimentCell,
        agents: list,
        snapshot,
        agent_states: dict[str, AgentDynamicState],
        activation_scores: dict[str, float],
        active_ids: set[str],
        active_rank: dict[str, int],
        timestep: int,
    ) -> list[P3AgentPanelRow]:
        rows: list[P3AgentPanelRow] = []
        visible_ids = "|".join(snapshot.visible_hyperedge_ids)
        for agent in agents:
            state = agent_states[agent.agent_id]
            rows.append(
                P3AgentPanelRow(
                    experiment_family=cell.experiment_family,
                    scenario_name=cell.scenario_name,
                    condition_name=cell.condition_name,
                    persona_name=cell.persona_name,
                    agent_persona_name=agent.persona_name,
                    chat_atmosphere=cell.chat_atmosphere,
                    repeat_index=cell.repeat_index,
                    timestep=timestep,
                    agent_id=agent.agent_id,
                    is_active=agent.agent_id in active_ids,
                    participation_state=state.participation_state,
                    activation_score=activation_scores.get(agent.agent_id),
                    active_rank=active_rank.get(agent.agent_id),
                    visible_hyperedge_ids=visible_ids,
                    visible_hyperedge_count=len(snapshot.visible_hyperedge_ids),
                    dominant_hyperedge_id=snapshot.dominant_hyperedge_id,
                    visible_negative_pressure=snapshot.negative_visibility_share,
                    last_choice=state.last_choice,
                    last_final_choice=state.last_final_choice,
                    latent_neg_tendency=round(state.latent_neg_tendency, 4),
                    perceived_majority=state.perceived_majority,
                    risk_arousal=round(state.risk_arousal, 4),
                    engagement_tendency=round(state.engagement_tendency, 4),
                    fatigue=round(state.fatigue, 4),
                    notification_score=round(state.notification_score, 4),
                    silence_steps=state.silence_steps,
                    active_count=state.active_count,
                    was_ever_active=state.was_ever_active,
                )
            )
        return rows

    def _run_population_probe(
        self,
        cell: ExperimentCell,
        agents: list,
        agent_states: dict[str, AgentDynamicState],
        snapshot,
        report_mapping: dict[str, str],
        scenario: dict[str, Any],
        phase: str,
        mutate_state: bool = False,
    ) -> list[dict[str, object]]:
        probe_cfg = self.config.get("p3", {}).get("private_probe", {})
        mode = str(probe_cfg.get("mode", "rule")).lower()
        batch_size = max(1, int(probe_cfg.get("batch_size", 12)))
        rows: list[dict[str, object]] = []
        if mode == "batch_llm":
            for start in range(0, len(agents), batch_size):
                batch = agents[start : start + batch_size]
                try:
                    rows.extend(
                        self._run_batch_llm_probe(
                            cell=cell,
                            agents=batch,
                            agent_states=agent_states,
                            snapshot=snapshot,
                            report_mapping=report_mapping,
                            scenario=scenario,
                            phase=phase,
                            mutate_state=mutate_state,
                        )
                    )
                except Exception:
                    rows.extend(
                        self._rule_probe_rows(
                            cell=cell,
                            agents=batch,
                            agent_states=agent_states,
                            snapshot=snapshot,
                            phase=phase,
                            probe_mode="rule_fallback",
                            mutate_state=mutate_state,
                        )
                    )
            return rows
        return self._rule_probe_rows(
            cell=cell,
            agents=agents,
            agent_states=agent_states,
            snapshot=snapshot,
            phase=phase,
            probe_mode="rule",
            mutate_state=mutate_state,
        )

    def _run_batch_llm_probe(
        self,
        cell: ExperimentCell,
        agents: list,
        agent_states: dict[str, AgentDynamicState],
        snapshot,
        report_mapping: dict[str, str],
        scenario: dict[str, Any],
        phase: str,
        mutate_state: bool = False,
    ) -> list[dict[str, object]]:
        payload = []
        for agent in agents:
            state = agent_states[agent.agent_id]
            payload.append(
                {
                    "agent_id": agent.agent_id,
                    "persona": agent.persona_name,
                    "last_final_choice": state.last_final_choice,
                    "latent_neg_tendency": round(state.latent_neg_tendency, 3),
                    "risk_arousal": round(state.risk_arousal, 3),
                    "engagement_tendency": round(state.engagement_tendency, 3),
                    "visible_hyperedge_ids": state.last_visible_hyperedge_ids or [],
                    "was_ever_active": state.was_ever_active,
                    "active_count": state.active_count,
                }
            )
        positive_report = "report_1" if report_mapping.get("report_1") == "positive" else "report_2"
        negative_report = "report_1" if report_mapping.get("report_1") == "negative" else "report_2"
        prompt = (
            "你是社交仿真实验中的私下判断批量更新器。请根据线程楼层摘要和每个用户状态，"
            "只更新他们此刻的私下判断，不产生公开发言。\n\n"
            f"phase: {phase}\n"
            f"背景: {scenario['background']}\n"
            f"正向报道对应: {positive_report}; 负向报道对应: {negative_report}\n"
            f"可见评论楼摘要:\n{snapshot.visible_hyperedge_summary or '暂无显著评论楼'}\n\n"
            f"agents:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "只输出 JSON 数组，每个元素包含 agent_id, final_choice, credible_choice, "
            "share_choice, comment_choice, latent_neg_tendency, reason。"
        )
        raw_response = self.client.generate_text(
            build_system_message(""),
            prompt,
            request_meta={
                "runner": "p3_probe",
                "scenario_name": cell.scenario_name,
                "phase": phase,
                "repeat_index": cell.repeat_index,
            },
        )
        parsed = self._parse_probe_array(raw_response)
        by_agent = {str(item.get("agent_id")): item for item in parsed if isinstance(item, dict)}
        rows: list[dict[str, object]] = []
        for agent in agents:
            state = agent_states[agent.agent_id]
            item = by_agent.get(agent.agent_id, {})
            final_choice = self._normalize_probe_choice(item.get("final_choice"), state, report_mapping)
            credible_choice = self._normalize_probe_credible(item.get("credible_choice"), final_choice)
            share_choice = self._normalize_probe_share(item.get("share_choice"), final_choice, report_mapping)
            comment_choice = self._normalize_probe_comment(item.get("comment_choice"), final_choice, report_mapping)
            latent = self._clip01(float(item.get("latent_neg_tendency", state.latent_neg_tendency)))
            if mutate_state:
                state.latent_neg_tendency = latent
                state.last_final_choice = final_choice
            rows.append(
                self._probe_row(
                    cell=cell,
                    agent=agent,
                    state=state,
                    snapshot=snapshot,
                    phase=phase,
                    final_choice=final_choice,
                    credible_choice=credible_choice,
                    share_choice=share_choice,
                    comment_choice=comment_choice,
                    latent_neg_tendency=latent,
                    reason=str(item.get("reason") or ""),
                    probe_mode="batch_llm",
                )
            )
        return rows

    @staticmethod
    def _parse_probe_array(raw_response: str) -> list[dict[str, object]]:
        text = raw_response.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return []
            payload = json.loads(match.group(0))
        return payload if isinstance(payload, list) else []

    @staticmethod
    def _normalize_probe_choice(
        value: object,
        state: AgentDynamicState,
        report_mapping: dict[str, str] | None = None,
    ) -> str:
        text = str(value or "").strip().lower()
        if text in {"negative", "负向"}:
            return "negative"
        if text in {"positive", "正向"}:
            return "positive"
        if text in {"report_1", "report1", "报道1"}:
            mapped = (report_mapping or {}).get("report_1")
            if mapped in {"positive", "negative"}:
                return mapped
        if text in {"report_2", "report2", "报道2"}:
            mapped = (report_mapping or {}).get("report_2")
            if mapped in {"positive", "negative"}:
                return mapped
        if state.latent_neg_tendency > 0.54:
            return "negative"
        if state.latent_neg_tendency < 0.46:
            return "positive"
        return "neutral"

    @staticmethod
    def _normalize_probe_credible(value: object, final_choice: str) -> str:
        text = str(value or "").strip().lower()
        if text in {"positive", "negative", "neutral"}:
            return text
        return final_choice

    @staticmethod
    def _normalize_probe_share(
        value: object,
        final_choice: str,
        report_mapping: dict[str, str] | None = None,
    ) -> str:
        text = str(value or "").strip().lower()
        if text in {"positive", "negative", "none"}:
            return text
        if text in {"report_1", "report1", "报道1"}:
            return (report_mapping or {}).get("report_1") or final_choice
        if text in {"report_2", "report2", "报道2"}:
            return (report_mapping or {}).get("report_2") or final_choice
        return final_choice

    @staticmethod
    def _normalize_probe_comment(
        value: object,
        final_choice: str,
        report_mapping: dict[str, str] | None = None,
    ) -> str:
        text = str(value or "").strip().lower()
        if text in {"support_positive", "support_negative", "neutral"}:
            return text
        if text in {"support_report_1", "support_report1", "支持报道1"}:
            mapped = (report_mapping or {}).get("report_1")
            if mapped == "negative":
                return "support_negative"
            if mapped == "positive":
                return "support_positive"
        if text in {"support_report_2", "support_report2", "支持报道2"}:
            mapped = (report_mapping or {}).get("report_2")
            if mapped == "negative":
                return "support_negative"
            if mapped == "positive":
                return "support_positive"
        return "support_negative" if final_choice == "negative" else "support_positive"

    def _rule_probe_rows(
        self,
        cell: ExperimentCell,
        agents: list,
        agent_states: dict[str, AgentDynamicState],
        snapshot,
        phase: str,
        probe_mode: str,
        mutate_state: bool = False,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for agent in agents:
            state = agent_states[agent.agent_id]
            jitter = (self._stable_unit(f"{cell.scenario_name}|{cell.repeat_index}|{phase}|{agent.agent_id}") - 0.5) * 0.06
            latent = self._clip01(state.latent_neg_tendency + jitter)
            negative_threshold = 0.54
            positive_threshold = 0.46
            if latent > negative_threshold:
                final_choice = "negative"
            elif latent < positive_threshold:
                final_choice = "positive"
            else:
                final_choice = "neutral"
            if abs(latent - 0.5) < 0.06:
                credible_choice = "neutral"
            else:
                credible_choice = final_choice
            share_choice = final_choice if abs(latent - 0.5) >= 0.12 else "none"
            comment_choice = (
                "support_negative"
                if latent >= 0.6
                else "support_positive"
                if latent <= 0.4
                else "neutral"
            )
            if mutate_state:
                state.latent_neg_tendency = latent
                state.last_final_choice = final_choice
            rows.append(
                self._probe_row(
                    cell=cell,
                    agent=agent,
                    state=state,
                    snapshot=snapshot,
                    phase=phase,
                    final_choice=final_choice,
                    credible_choice=credible_choice,
                    share_choice=share_choice,
                    comment_choice=comment_choice,
                    latent_neg_tendency=latent,
                    reason="rule_probe_from_visible_hyperedges",
                    probe_mode=probe_mode,
                )
            )
        return rows

    @staticmethod
    def _probe_row(
        cell: ExperimentCell,
        agent,
        state: AgentDynamicState,
        snapshot,
        phase: str,
        final_choice: str,
        credible_choice: str,
        share_choice: str,
        comment_choice: str,
        latent_neg_tendency: float,
        reason: str,
        probe_mode: str,
    ) -> dict[str, object]:
        def ternary_metric(choice: str, positive_label: str) -> float | None:
            if choice == "negative":
                return 1.0
            if choice == positive_label:
                return 0.0
            if choice == "neutral":
                return 0.5
            return None

        nsb = ternary_metric(final_choice, "positive")
        cb = ternary_metric(credible_choice, "positive")
        # share=none and comment=neutral mean no explicit public amplification.
        sb = 1.0 if share_choice == "negative" else 0.0 if share_choice == "positive" else None
        eb_probe = ternary_metric(
            "negative" if comment_choice == "support_negative" else "positive" if comment_choice == "support_positive" else "neutral",
            "positive",
        )
        return {
            "experiment_family": cell.experiment_family,
            "scenario_name": cell.scenario_name,
            "condition_name": cell.condition_name,
            "persona_name": cell.persona_name,
            "agent_persona_name": agent.persona_name,
            "chat_atmosphere": cell.chat_atmosphere,
            "repeat_index": cell.repeat_index,
            "phase": phase,
            "timestep": snapshot.timestep,
            "agent_id": agent.agent_id,
            "was_ever_active": state.was_ever_active,
            "active_count": state.active_count,
            "visible_hyperedge_ids": "|".join(snapshot.visible_hyperedge_ids),
            "dominant_hyperedge_id": snapshot.dominant_hyperedge_id,
            "visible_negative_pressure": snapshot.negative_visibility_share,
            "latent_neg_tendency": round(latent_neg_tendency, 4),
            "final_choice": final_choice,
            "credible_choice": credible_choice,
            "share_choice": share_choice,
            "comment_choice": comment_choice,
            "nsb": nsb,
            "cb": cb,
            "sb": sb,
            "eb_probe": eb_probe,
            "probe_mode": probe_mode,
            "reason": reason,
        }

    @staticmethod
    def _build_timestep_summary_row(
        cell: ExperimentCell,
        timestep: int,
        step_rows: list[P3RunRow],
        snapshot,
        agents: list,
        agent_states: dict[str, AgentDynamicState],
        active_ids: set[str],
    ) -> dict[str, object]:
        valid_rows = [row for row in step_rows if row.success and row.nsb is not None]
        public_rows = [row for row in step_rows if row.action_type in {"post", "reply", "like", "repost"}]
        signed_public = [row for row in public_rows if row.signal_polarity in {"positive", "negative"}]
        negative_public = sum(row.signal_polarity == "negative" for row in signed_public)
        positive_public = sum(row.signal_polarity == "positive" for row in signed_public)
        nsb_active = (
            sum(float(row.nsb) for row in valid_rows) / len(valid_rows)
            if valid_rows
            else None
        )
        scm_values = [float(row.scm) for row in valid_rows if row.scm is not None]
        gcr_step = None
        ssa_step = None
        if signed_public:
            negative_share = negative_public / len(signed_public)
            gcr_step = max(negative_share, 1 - negative_share)
            ssa_step = (negative_share - 0.5) * 2
        active_states = [agent_states[agent.agent_id] for agent in agents if agent.agent_id in active_ids]
        inactive_states = [agent_states[agent.agent_id] for agent in agents if agent.agent_id not in active_ids]

        def mean_latent(states: list[AgentDynamicState]) -> float | None:
            if not states:
                return None
            return round(sum(state.latent_neg_tendency for state in states) / len(states), 4)

        return {
            "experiment_family": cell.experiment_family,
            "scenario_name": cell.scenario_name,
            "condition_name": cell.condition_name,
            "persona_name": cell.persona_name,
            "chat_atmosphere": cell.chat_atmosphere,
            "repeat_index": cell.repeat_index,
            "timestep": timestep,
            "n_agents": len(agents),
            "n_active_agents": len(active_ids),
            "n_inactive_agents": len(agents) - len(active_ids),
            "n_public_events": len(public_rows),
            "n_post": sum(row.action_type == "post" for row in step_rows),
            "n_reply": sum(row.action_type == "reply" for row in step_rows),
            "n_like": sum(row.action_type == "like" for row in step_rows),
            "n_repost": sum(row.action_type == "repost" for row in step_rows),
            "n_observe": sum(row.action_type == "observe" for row in step_rows),
            "nsb_active": round(nsb_active, 4) if nsb_active is not None else None,
            "gcr_step_public": round(gcr_step, 4) if gcr_step is not None else None,
            "ssa_step_public": round(ssa_step, 4) if ssa_step is not None else None,
            "scm_step_active": round(sum(scm_values) / len(scm_values), 4) if scm_values else None,
            "positive_public_signals": positive_public,
            "negative_public_signals": negative_public,
            "visible_hyperedge_count": len(snapshot.visible_hyperedge_ids),
            "dominant_hyperedge_id": snapshot.dominant_hyperedge_id,
            "visible_negative_pressure": snapshot.negative_visibility_share,
            "mean_latent_neg_all": mean_latent(list(agent_states.values())),
            "mean_latent_neg_active": mean_latent(active_states),
            "mean_latent_neg_inactive": mean_latent(inactive_states),
        }

    @staticmethod
    def _render_agent_state(state: AgentDynamicState) -> str:
        recent_choices = " -> ".join(state.recent_choice_history[-3:]) if state.recent_choice_history else "暂无"
        recent_focus = "；".join(state.recent_focus_history[-2:]) if state.recent_focus_history else "暂无"
        last_public = state.last_public_comment or "上一轮未公开发言"
        last_reason = state.last_reason or "暂无"
        last_target = state.last_reply_target or "无"
        return (
            f"- 上一轮选择倾向：{state.last_choice}\n"
            f"- 上一轮动作：{state.last_action}\n"
            f"- 你最近几轮的选择轨迹：{recent_choices}\n"
            f"- 你上一轮公开内容：{last_public}\n"
            f"- 你上一轮回复对象：{last_target}\n"
            f"- 你上一轮最看重的线索：{last_reason}\n"
            f"- 你最近关注的线索摘要：{recent_focus}\n"
            f"- 你感知到的多数意见：{state.perceived_majority}\n"
            f"- 当前风险唤起水平（0-1）：{state.risk_arousal:.2f}\n"
            f"- 当前参与倾向（0-1）：{state.engagement_tendency:.2f}\n"
            f"- 当前潜在负向倾向（0-1）：{state.latent_neg_tendency:.2f}\n"
            f"- 最近可见评论楼：{'、'.join(state.last_visible_hyperedge_ids or []) or '暂无'}"
        )

    @staticmethod
    def _uses_stratified_ab_activation(cell: ExperimentCell, enabled: bool) -> bool:
        return (
            enabled
            and cell.experiment_family in {"persona_supplement", "combined_supplement"}
            and cell.condition_name == "high_interaction"
            and cell.persona_name == "balanced_persona_mix"
        )

    @staticmethod
    def _balanced_persona_order(agents: list[DiscussionAgent]) -> list[str]:
        preferred_order = ["risk_sensitive", "authority_trusting", "ordinary_user"]
        observed = {agent.persona_name for agent in agents}
        return [name for name in preferred_order if name in observed] + sorted(observed - set(preferred_order))

    @staticmethod
    def _balanced_persona_from_agent_id(agent_id: str) -> str | None:
        match = re.search(r"(\d+)$", agent_id or "")
        if not match:
            return None
        persona_cycle = ["risk_sensitive", "authority_trusting", "ordinary_user"]
        return persona_cycle[(int(match.group(1)) - 1) % len(persona_cycle)]

    def _select_stratified_persona_step(
        self,
        scheduler: ForumScheduler,
        agents: list[DiscussionAgent],
        timestep: int,
        agent_states: dict[str, AgentDynamicState],
        snapshot: ThreadSnapshot,
        max_active_agents: int,
    ) -> StepSchedule:
        """AB 混合群体专用：每轮按人格分层激活，避免单一人格抢占早期楼层。"""
        if not agents:
            return StepSchedule(timestep=timestep, active_agents=[], activation_scores={})

        scores = scheduler.score_agents(timestep=timestep, agent_states=agent_states, snapshot=snapshot)
        active_limit = max(1, min(max_active_agents, len(agents)))
        persona_order = self._balanced_persona_order(agents)
        if not persona_order:
            return scheduler.select_active_agents(
                timestep=timestep,
                agent_states=agent_states,
                snapshot=snapshot,
                max_active_agents=max_active_agents,
                exploration_slots=0,
            )

        base_quota = active_limit // len(persona_order)
        remainder = active_limit % len(persona_order)
        quotas = {
            persona_name: base_quota + (1 if index < remainder else 0)
            for index, persona_name in enumerate(persona_order)
        }

        grouped: dict[str, list[DiscussionAgent]] = {persona_name: [] for persona_name in persona_order}
        for agent in agents:
            grouped.setdefault(agent.persona_name, []).append(agent)
        for persona_name, persona_agents in grouped.items():
            persona_agents.sort(
                key=lambda agent: (
                    scores.get(agent.agent_id, 0.0),
                    ForumScheduler._stable_unit(f"{agent.agent_id}|{timestep}|stratified_rank"),
                ),
                reverse=True,
            )

        selected_by_persona = {
            persona_name: grouped.get(persona_name, [])[: max(0, quotas.get(persona_name, 0))]
            for persona_name in persona_order
        }
        active_agents: list[DiscussionAgent] = []
        max_quota = max((len(items) for items in selected_by_persona.values()), default=0)
        for slot_index in range(max_quota):
            for persona_name in persona_order:
                persona_agents = selected_by_persona.get(persona_name, [])
                if slot_index < len(persona_agents):
                    active_agents.append(persona_agents[slot_index])

        selected_ids = {agent.agent_id for agent in active_agents}
        if len(active_agents) < active_limit:
            remaining = [agent for agent in agents if agent.agent_id not in selected_ids]
            remaining.sort(
                key=lambda agent: (
                    scores.get(agent.agent_id, 0.0),
                    ForumScheduler._stable_unit(f"{agent.agent_id}|{timestep}|stratified_fill"),
                ),
                reverse=True,
            )
            active_agents.extend(remaining[: active_limit - len(active_agents)])

        return StepSchedule(
            timestep=timestep,
            active_agents=active_agents[:active_limit],
            activation_scores={agent.agent_id: scores.get(agent.agent_id, 0.0) for agent in agents},
        )

    @staticmethod
    def _ensure_ab_persona_floor_action(
        decision,
        events: list[ThreadEvent],
        actor_id: str,
        agent_persona_name: str,
        timestep: int,
        enabled: bool,
    ):
        """前几轮确保混合群体中每种人格至少建立一个公开楼层。"""
        if not enabled:
            return decision

        existing_floor_personas = {
            persona_name
            for event in events
            if event.action_type == "post" and not event.reply_to
            for persona_name in [P3ExperimentRunner._balanced_persona_from_agent_id(event.actor_id)]
            if persona_name
        }
        if agent_persona_name in existing_floor_personas:
            return decision

        data = decision.__dict__.copy()
        signal = P3ExperimentRunner._signal_from_decision(decision)
        if signal == "neutral" and decision.final_choice in {"positive", "negative"}:
            signal = decision.final_choice
        if signal == "negative":
            data["comment_choice"] = "support_negative"
        elif signal == "positive":
            data["comment_choice"] = "support_positive"

        data["planned_action"] = "post_comment"
        data["reply_target"] = ""
        comment = (decision.public_comment or "").strip()
        if not comment:
            comment = P3ExperimentRunner._build_reason_based_comment(
                reason=decision.reason,
                signal=signal,
                seed=f"{actor_id}|{timestep}|ab_persona_floor",
            )
        if not comment:
            if signal == "negative":
                comment = "我先开个楼，想说这件事里的风险线索还是值得认真看。"
            elif signal == "positive":
                comment = "我也开个楼，从目前材料看我更倾向于情况仍可控。"
            else:
                comment = "我也开个楼，这件事可能要看不同线索怎么互相印证。"
        data["public_comment"] = comment
        return type(decision)(**data)

    @staticmethod
    def _update_agent_state(
        state: AgentDynamicState,
        decision,
        negative_share: float | None,
        behavior_profile: dict[str, float],
        dynamics_cfg: dict[str, object] | None = None,
        persona_name: str | None = None,
        activation_score: float | None = None,
        hyperedge_id: str | None = None,
    ) -> None:
        dynamics_cfg = dynamics_cfg or {}
        if decision.credible_choice in {"positive", "negative"}:
            state.last_choice = decision.credible_choice
        else:
            state.last_choice = "neutral"
        state.last_action = decision.planned_action
        state.last_final_choice = decision.final_choice
        state.last_reason = (decision.reason or "").strip() or None
        state.last_public_comment = (decision.public_comment or "").strip() or None
        state.last_reply_target = (decision.reply_target or "").strip() or None
        state.active_count += 1
        state.was_ever_active = True
        state.silence_steps = 0
        state.participation_state = "active"
        state.fatigue = min(1.0, state.fatigue + 0.08)
        if activation_score is not None and activation_score < 0.35:
            state.cooldown_steps_remaining = max(state.cooldown_steps_remaining, 1)
        if hyperedge_id:
            if state.participated_hyperedge_ids is None:
                state.participated_hyperedge_ids = []
            if hyperedge_id not in state.participated_hyperedge_ids:
                state.participated_hyperedge_ids.append(hyperedge_id)
                state.participated_hyperedge_ids = state.participated_hyperedge_ids[-6:]
        if state.recent_choice_history is None:
            state.recent_choice_history = []
        state.recent_choice_history.append(state.last_choice)
        state.recent_choice_history = state.recent_choice_history[-4:]
        if state.recent_focus_history is None:
            state.recent_focus_history = []
        if state.last_reason:
            state.recent_focus_history.append(state.last_reason)
            state.recent_focus_history = state.recent_focus_history[-3:]

        if negative_share is None:
            state.perceived_majority = "unclear"
        elif negative_share >= 0.6:
            state.perceived_majority = "negative"
        elif negative_share <= 0.4:
            state.perceived_majority = "positive"
        else:
            state.perceived_majority = "split"

        if decision.credible_choice == "negative":
            state.risk_arousal = min(
                1.0,
                state.risk_arousal + 0.15 * behavior_profile.get("negative_risk_gain", 1.0),
            )
        elif decision.credible_choice == "positive":
            state.risk_arousal = max(
                0.0,
                state.risk_arousal - 0.10 * behavior_profile.get("positive_relief_gain", 1.0),
            )

        if negative_share is not None:
            majority_risk_shift_weight = float(dynamics_cfg.get("majority_risk_shift_weight", 0.12))
            majority_latent_shift_weight = float(dynamics_cfg.get("majority_latent_shift_weight", 0.10))
            majority_shift = (
                (negative_share - 0.5)
                * majority_risk_shift_weight
                * behavior_profile.get("majority_sensitivity", 1.0)
            )
            state.risk_arousal = min(1.0, max(0.0, state.risk_arousal + majority_shift))
            latent_shift = (
                (negative_share - 0.5)
                * majority_latent_shift_weight
                * behavior_profile.get("majority_sensitivity", 1.0)
            )
            state.latent_neg_tendency = max(0.0, min(1.0, state.latent_neg_tendency + latent_shift))

        decision_anchor_weight = float(dynamics_cfg.get("decision_anchor_weight", 0.32))
        if decision.final_choice == "negative":
            negative_anchor_weight = decision_anchor_weight
            if persona_name == "authority_trusting":
                negative_anchor_weight *= float(dynamics_cfg.get("authority_trusting_negative_anchor_factor", 0.45))
            elif persona_name == "risk_sensitive":
                negative_anchor_weight *= float(dynamics_cfg.get("risk_sensitive_negative_anchor_factor", 1.10))
            elif persona_name == "ordinary_user":
                negative_anchor_weight *= float(dynamics_cfg.get("ordinary_user_negative_anchor_factor", 0.72))
            negative_anchor_weight = max(0.0, min(1.0, negative_anchor_weight))
            state.latent_neg_tendency = min(
                1.0,
                state.latent_neg_tendency * (1 - negative_anchor_weight) + negative_anchor_weight,
            )
        elif decision.final_choice == "positive":
            positive_anchor_weight = decision_anchor_weight * float(
                dynamics_cfg.get("positive_anchor_factor", 0.35)
            )
            if (
                persona_name == "authority_trusting"
                and negative_share is not None
                and negative_share >= float(dynamics_cfg.get("authority_trusting_exposure_pressure_floor", 0.40))
            ):
                relief_factor = float(dynamics_cfg.get("authority_trusting_positive_anchor_relief", 0.45))
                positive_anchor_weight *= max(0.0, min(1.0, relief_factor))
            positive_anchor_weight = max(0.0, min(1.0, positive_anchor_weight))
            state.latent_neg_tendency = max(
                0.0,
                state.latent_neg_tendency * (1 - positive_anchor_weight),
            )

        if decision.planned_action in {"post_comment", "reply", "repost"}:
            state.engagement_tendency = min(
                1.0,
                state.engagement_tendency + 0.10 * behavior_profile.get("engagement_gain", 1.0),
            )
        elif decision.planned_action == "like":
            state.engagement_tendency = min(
                1.0,
                state.engagement_tendency + 0.03 * behavior_profile.get("engagement_gain", 1.0),
            )
        elif decision.planned_action == "observe":
            state.engagement_tendency = max(
                0.0,
                state.engagement_tendency - 0.10 * behavior_profile.get("observe_decay", 1.0),
            )
        state.notification_score = max(0.0, state.notification_score * 0.5)

    @staticmethod
    def _build_thread_summary(events: list[ThreadEvent], report_mapping: dict[str, str]) -> str:
        if not events:
            return "线程刚开始，还没有形成明确讨论。"

        visible = events[-8:]
        dynamic_visible = [event for event in visible if not event.event_id.startswith("seed_")]
        readonly_seed_count = len(visible) - len(dynamic_visible)
        positive_count = sum(event.signal_polarity == "positive" for event in dynamic_visible)
        negative_count = sum(event.signal_polarity == "negative" for event in dynamic_visible)
        neutral_count = sum(event.signal_polarity == "neutral" for event in dynamic_visible)
        reply_count = sum(event.action_type == "reply" for event in dynamic_visible)
        like_count = sum(event.action_type == "like" for event in dynamic_visible)
        repost_count = sum(event.action_type == "repost" for event in dynamic_visible)
        post_count = sum(event.action_type == "post" for event in dynamic_visible)
        latest_topics = [event.content for event in visible if event.content][:2]
        interaction = (
            f"最近8条里有{readonly_seed_count}条初始只读评论、{post_count}条agent新评论、"
            f"{reply_count}条回复、{like_count}次点赞、{repost_count}次转发。"
        )
        positive_report = "报道1" if report_mapping.get("report_1") == "positive" else "报道2"
        negative_report = "报道1" if report_mapping.get("report_1") == "negative" else "报道2"
        total_signals = positive_count + negative_count + neutral_count
        if total_signals == 0:
            stance_counts = "这些可见内容里还没有形成由系统明确标记的立场信号，主要需要你自己理解文本内容。"
        else:
            stance_counts = (
                f"其中支持{positive_report}（偏可控/恢复较快）的信号约{positive_count}条，"
                f"支持{negative_report}（偏风险/恢复拖长）的信号约{negative_count}条，"
                f"中性或难判断的约{neutral_count}条。"
            )
        focus = "；".join(topic for topic in latest_topics if topic) or "暂无明确讨论焦点。"
        return f"{interaction}\n{stance_counts}\n最近显眼的话题点：{focus}"

    @staticmethod
    def _resolve_reply_target(decision, visible_events: list[ThreadEvent]) -> str | None:
        if decision.planned_action not in {"reply", "like", "repost"} or not visible_events:
            return None

        interactable_events = [
            event
            for event in visible_events
            if not event.event_id.startswith("seed_")
            and event.action_type in {"post", "reply"}
        ]
        if not interactable_events:
            return None

        valid_ids = {event.event_id for event in interactable_events}
        if decision.reply_target in valid_ids:
            return decision.reply_target

        desired_polarity = P3ExperimentRunner._signal_from_decision(decision)
        preferred_opposite = None
        if desired_polarity == "negative":
            preferred_opposite = "positive"
        elif desired_polarity == "positive":
            preferred_opposite = "negative"

        if preferred_opposite:
            # 优先回复最近窗口中“显眼且与自己相反”的评论，模拟争论与反驳。
            opposite_candidates = [
                event for event in interactable_events if event.signal_polarity == preferred_opposite
            ]
            if opposite_candidates:
                opposite_candidates.sort(
                    key=lambda event: (event.salience_score, event.timestep),
                    reverse=True,
                )
                return opposite_candidates[0].event_id

        signed_events = [
            event for event in interactable_events if event.signal_polarity in {"positive", "negative"}
        ]
        if signed_events:
            signed_events.sort(key=lambda event: (event.salience_score, event.timestep), reverse=True)
            return signed_events[0].event_id
        return interactable_events[-1].event_id

    @staticmethod
    def _promote_interaction_if_needed(
        decision,
        visible_events: list[ThreadEvent],
        actor_id: str,
        timestep: int,
        action_balance: dict[str, object] | None = None,
    ):
        if decision.planned_action != "observe" or not visible_events:
            return decision, None

        signal = P3ExperimentRunner._signal_from_decision(decision)
        if signal == "neutral":
            return decision, None

        data = decision.__dict__.copy()
        has_public_comment = bool((decision.public_comment or "").strip())
        action_balance = action_balance or {}
        observe_to_like_rate = float(action_balance.get("observe_to_like_rate", 1.0))
        observe_to_repost_rate = float(action_balance.get("observe_to_repost_rate", observe_to_like_rate))
        seed = f"{actor_id}|{timestep}|observe_promote|{len(visible_events)}|{signal}"
        roll = P3ExperimentRunner._stable_unit(seed)

        if decision.like_choice in {"positive", "negative"} and not has_public_comment and roll < observe_to_like_rate:
            data["planned_action"] = "like"
            data["public_comment"] = ""
        elif (
            decision.share_choice in {"positive", "negative"}
            and decision.like_choice == "none"
            and not has_public_comment
            and roll < observe_to_repost_rate
        ):
            data["planned_action"] = "repost"
            data["public_comment"] = ""
        elif decision.comment_choice in {"support_positive", "support_negative"} and has_public_comment:
            data["planned_action"] = "post_comment"
        else:
            return decision, None

        promoted = type(decision)(**data)
        return promoted, None

    @staticmethod
    def _ensure_initial_floor_action(
        decision,
        visible_events: list[ThreadEvent],
        actor_id: str,
        timestep: int,
        enabled: bool = True,
    ):
        """When P3 has only read-only seed comments, agents create their own floors first."""
        has_agent_floor = any(
            not event.event_id.startswith("seed_")
            and event.action_type in {"post", "reply"}
            for event in visible_events
        )
        if not enabled or has_agent_floor:
            return decision
        if decision.planned_action == "post_comment" and (decision.public_comment or "").strip():
            return decision

        data = decision.__dict__.copy()
        signal = P3ExperimentRunner._signal_from_decision(decision)
        if signal == "neutral" and decision.final_choice in {"positive", "negative"}:
            signal = decision.final_choice
        if signal == "negative":
            data["comment_choice"] = "support_negative"
        elif signal == "positive":
            data["comment_choice"] = "support_positive"

        data["planned_action"] = "post_comment"
        data["reply_target"] = ""
        comment = (decision.public_comment or "").strip()
        if not comment:
            comment = P3ExperimentRunner._build_reason_based_comment(
                reason=decision.reason,
                signal=signal,
                seed=f"{actor_id}|{timestep}|initial_floor",
            )
        if not comment:
            if signal == "negative":
                comment = "我先说下，我更担心这件事里的风险信号可能被低估了。"
            elif signal == "positive":
                comment = "我先说下，我更倾向于认为目前情况还在可控范围内。"
            else:
                comment = "我先占个楼，感觉这件事还需要结合更多具体情况判断。"
        data["public_comment"] = comment
        return type(decision)(**data)

    @staticmethod
    def _calibrate_initial_seed_floor_decision(
        decision,
        state: AgentDynamicState,
        visible_events: list[ThreadEvent],
        actor_id: str,
        timestep: int,
        active_position: int | None = None,
    ):
        """Prevent read-only seed comments from synchronizing all first floors.

        Before any agent-generated floor exists, P3 should reflect heterogeneous
        private entry states rather than letting one salient seed line make every
        active user post the same negative stance. This only applies to the first
        seed-only context and does not rewrite later interaction dynamics.
        """
        if timestep != 1:
            return decision
        has_dynamic_floor = any(
            not event.event_id.startswith("seed_") and event.action_type in {"post", "reply"}
            for event in visible_events
        )
        has_readonly_seed = any(event.event_id.startswith("seed_") for event in visible_events)
        if has_dynamic_floor or not has_readonly_seed:
            return decision
        if state.latent_neg_tendency > 0.54:
            return decision

        slot = ((active_position or 1) - 1) % 6
        if state.latent_neg_tendency < 0.46:
            target = "positive" if slot in {0, 1, 3, 5} else "negative"
        else:
            if state.risk_arousal >= 0.56:
                target = "positive" if slot in {1, 4} else "negative"
            else:
                target = "positive" if slot in {1, 3, 5} else "negative"

        if target == decision.final_choice and decision.comment_choice in {"support_positive", "support_negative"}:
            return decision

        data = decision.__dict__.copy()
        data["final_choice"] = target
        data["credible_choice"] = target if state.latent_neg_tendency < 0.46 or target == "negative" else "neutral"
        data["share_choice"] = target if target == "negative" else "none"
        data["like_choice"] = target
        data["comment_choice"] = "support_negative" if target == "negative" else "support_positive"
        data["planned_action"] = "post_comment"
        data["reply_target"] = ""
        if target == "positive":
            data["public_comment"] = "我先占个楼，按目前看到的材料，我还会把它理解成需要观察但总体可处理。"
            data["reason"] = (
                "我的进入讨论前状态还没有明显偏向风险，正式应对、规范治疗和多数恢复线索仍然能支撑轻度可控判断。"
            )
        else:
            data["public_comment"] = "我先占个楼，恢复慢和复诊这些具体提醒还是值得继续看。"
            data["reason"] = (
                "我的进入讨论前状态接近观望，seed 中恢复慢、复诊或照护压力这些线索让我保留轻度风险意识。"
            )
        return type(decision)(**data)

    @staticmethod
    def _calibrate_runaway_consensus_decision(
        decision,
        state: AgentDynamicState,
        visible_events: list[ThreadEvent],
        actor_id: str,
        timestep: int,
        dynamics_cfg: dict[str, object] | None = None,
    ):
        """Keep high-consensus threads from becoming a mechanical command.

        P3 should allow majority pressure, but not turn every mildly neutral
        agent into the same stance once the visible thread becomes one-sided.
        This only affects agents whose private latent state has not crossed a
        clear negative threshold.
        """
        if timestep <= 1 or decision.final_choice != "negative":
            return decision
        dynamics_cfg = dynamics_cfg or {}
        dynamic_events = [
            event
            for event in visible_events
            if not event.event_id.startswith("seed_") and event.signal_polarity in {"positive", "negative"}
        ]
        min_events = int(dynamics_cfg.get("runaway_consensus_min_events", 6))
        if len(dynamic_events) < min_events:
            return decision
        negative_share = sum(event.signal_polarity == "negative" for event in dynamic_events) / len(dynamic_events)
        threshold = float(dynamics_cfg.get("runaway_consensus_negative_share", 0.72))
        high_arousal_floor = float(dynamics_cfg.get("runaway_consensus_high_arousal_floor", 0.62))
        if (
            negative_share < threshold
            or state.latent_neg_tendency > 0.54
            or state.risk_arousal >= high_arousal_floor
        ):
            return decision

        if state.latent_neg_tendency <= 0.46:
            flip_rate = float(dynamics_cfg.get("runaway_consensus_positive_latent_flip_rate", 0.45))
        elif state.latent_neg_tendency <= 0.54:
            flip_rate = float(dynamics_cfg.get("runaway_consensus_neutral_latent_flip_rate", 0.30))
        else:
            flip_rate = 0.0
        roll = P3ExperimentRunner._stable_unit(
            f"{actor_id}|{timestep}|runaway_consensus|{len(dynamic_events)}|{negative_share:.2f}"
        )
        if roll >= flip_rate:
            return decision

        data = decision.__dict__.copy()
        data["final_choice"] = "positive"
        data["credible_choice"] = "positive" if state.latent_neg_tendency < 0.50 else "neutral"
        data["share_choice"] = "none"
        data["like_choice"] = "positive"
        data["comment_choice"] = "support_positive"
        if decision.planned_action in {"reply", "like", "repost"}:
            data["planned_action"] = "reply"
        else:
            data["planned_action"] = "post_comment"
        data["public_comment"] = "我理解大家在担心，但规范治疗、随访和多数逐步恢复这些线索也不能完全丢掉。"
        data["reason"] = (
            "可见讨论已经明显偏向风险，但我的私有状态还没有强负向；材料中的正式应对、规范治疗和多数恢复线索仍然值得保留。"
        )
        return type(decision)(**data)

    @staticmethod
    def _balance_forum_action(
        decision,
        visible_events: list[ThreadEvent],
        state: AgentDynamicState,
        actor_id: str,
        timestep: int,
        action_balance: dict[str, object] | None = None,
        fallback_target: str | None = None,
    ):
        """Keep forum actions realistic: reading/liking are cheaper than commenting."""
        action_balance = action_balance or {}
        if not visible_events:
            return decision, fallback_target
        dynamic_events = [
            event for event in visible_events
            if not event.event_id.startswith("seed_")
        ]
        has_agent_floor = any(
            event.action_type in {"post", "reply"}
            for event in dynamic_events
        )
        if not has_agent_floor:
            return decision, fallback_target

        if decision.planned_action == "like":
            return P3ExperimentRunner._upshift_like_when_thread_needs_comment(
                decision=decision,
                visible_events=dynamic_events,
                actor_id=actor_id,
                timestep=timestep,
                action_balance=action_balance,
                fallback_target=fallback_target,
            )

        if decision.planned_action not in {"post_comment", "reply"}:
            return decision, fallback_target

        recent = dynamic_events[-8:]
        recent_comment_count = sum(event.action_type in {"post", "reply"} for event in recent)
        recent_like_count = sum(event.action_type == "like" for event in recent)
        text = (decision.public_comment or "").strip()
        has_new_comment = bool(text)
        signal = P3ExperimentRunner._signal_from_decision(decision)
        seed = f"{actor_id}|{timestep}|{len(visible_events)}|{decision.planned_action}|{signal}"
        roll = P3ExperimentRunner._stable_unit(seed)

        # GA-S3 treats viewing as the common action and comments as costlier than likes.
        # When the visible thread is already comment-heavy, some low-novelty comments
        # become lightweight support or silent observation.
        comment_heavy = recent_comment_count >= 6 and recent_like_count <= 2
        low_engagement = state.engagement_tendency < 0.32
        repeated = has_new_comment and any(
            P3ExperimentRunner._is_near_duplicate(
                P3ExperimentRunner._normalize_text_for_similarity(text),
                P3ExperimentRunner._normalize_text_for_similarity(event.content),
            )
            for event in recent[-3:]
            if event.content
        )
        weak_public_expression = decision.comment_choice == "neutral" or not has_new_comment

        # After agents have created their own floors, route a bounded share of
        # additional comments into existing dynamic threads instead of opening
        # near-parallel floors. Timestep 1 remains floor-creation oriented.
        if decision.planned_action == "post_comment" and timestep >= int(
            action_balance.get("reply_after_timestep", 2)
        ):
            root_floor_count = sum(
                event.action_type == "post" and not event.reply_to
                for event in dynamic_events
            )
            post_to_reply_rate = float(action_balance.get("post_to_reply_rate", 0.0))
            min_dynamic_floors = int(action_balance.get("reply_min_dynamic_floors", 3))
            reply_roll = P3ExperimentRunner._stable_unit(
                f"{actor_id}|{timestep}|post_to_reply|{len(visible_events)}|{signal}"
            )
            if root_floor_count >= min_dynamic_floors and reply_roll < post_to_reply_rate:
                data = decision.__dict__.copy()
                data["planned_action"] = "reply"
                data["reply_target"] = ""
                replied = type(decision)(**data)
                target = P3ExperimentRunner._resolve_reply_target(replied, visible_events)
                if target:
                    data["reply_target"] = target
                    return type(decision)(**data), target

        if repeated and roll < 0.35:
            can_like = decision.like_choice in {"positive", "negative"}
            return P3ExperimentRunner._downshift_action(
                decision=decision,
                visible_events=visible_events,
                prefer_like=can_like and roll < 0.28,
            )

        if comment_heavy and low_engagement and roll < 0.15:
            can_like = decision.like_choice in {"positive", "negative"}
            return P3ExperimentRunner._downshift_action(
                decision=decision,
                visible_events=visible_events,
                prefer_like=can_like and roll < 0.10,
            )

        if comment_heavy and weak_public_expression and roll < 0.10:
            return P3ExperimentRunner._downshift_action(
                decision=decision,
                visible_events=visible_events,
                prefer_like=decision.like_choice in {"positive", "negative"} and roll < 0.08,
            )

        return decision, fallback_target

    @staticmethod
    def _upshift_like_when_thread_needs_comment(
        decision,
        visible_events: list[ThreadEvent],
        actor_id: str,
        timestep: int,
        action_balance: dict[str, object],
        fallback_target: str | None,
    ):
        signal = P3ExperimentRunner._signal_from_decision(decision)
        if signal not in {"positive", "negative"}:
            return decision, fallback_target

        recent = visible_events[-8:]
        recent_comment_count = sum(event.action_type in {"post", "reply"} for event in recent)
        recent_like_count = sum(event.action_type == "like" for event in recent)
        quiet_comment_threshold = int(action_balance.get("quiet_comment_threshold", 4))
        like_heavy_margin = int(action_balance.get("like_heavy_margin", 2))
        like_to_comment_rate = float(action_balance.get("like_to_comment_rate", 0.0))

        thread_needs_comment = (
            recent_comment_count < quiet_comment_threshold
            or recent_like_count >= recent_comment_count + like_heavy_margin
        )
        if not thread_needs_comment or like_to_comment_rate <= 0:
            return decision, fallback_target

        seed = f"{actor_id}|{timestep}|like_to_comment|{len(visible_events)}|{signal}"
        if P3ExperimentRunner._stable_unit(seed) >= like_to_comment_rate:
            return decision, fallback_target

        comment = P3ExperimentRunner._build_reason_based_comment(
            reason=decision.reason,
            signal=signal,
            seed=f"{actor_id}|{timestep}|like_to_comment_text",
        )
        if not comment:
            return decision, fallback_target

        target = P3ExperimentRunner._find_like_target_for_signal(signal, visible_events)
        data = decision.__dict__.copy()
        data["planned_action"] = "reply" if target else "post_comment"
        data["public_comment"] = comment
        data["reply_target"] = target or ""
        data["comment_choice"] = "support_negative" if signal == "negative" else "support_positive"
        upshifted = type(decision)(**data)
        return upshifted, target

    @staticmethod
    def _downshift_action(decision, visible_events: list[ThreadEvent], prefer_like: bool):
        data = decision.__dict__.copy()
        if prefer_like:
            data["planned_action"] = "like"
            data["public_comment"] = ""
            signal = P3ExperimentRunner._signal_from_decision(decision)
            if data.get("like_choice") not in {"positive", "negative"} and signal in {"positive", "negative"}:
                data["like_choice"] = signal
            downshifted = type(decision)(**data)
            return downshifted, P3ExperimentRunner._resolve_reply_target(downshifted, visible_events)

        data["planned_action"] = "observe"
        data["public_comment"] = ""
        data["reply_target"] = ""
        return type(decision)(**data), None

    @staticmethod
    def _compute_event_salience(decision, timestep: int, action_type: str | None = None) -> float:
        # salience_score 不是平台真实热度，而是用于近似“这条内容有多显眼”的内部排序量。
        score = 1.0
        public_action = action_type or decision.planned_action
        if public_action == "reply":
            score += 0.4
        if public_action in {"post", "post_comment", "reply"} and decision.comment_choice in {
            "support_positive",
            "support_negative",
        }:
            score += 0.4
        if decision.like_choice in {"positive", "negative"}:
            score += 0.2
        if decision.share_choice in {"positive", "negative"}:
            score += 0.3
        if decision.credible_choice in {"positive", "negative"}:
            score += 0.3
        score += timestep * 0.05
        return round(score, 3)

    @staticmethod
    def _stable_unit(seed: str) -> float:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    @staticmethod
    def _signal_from_decision(decision) -> str:
        if decision.comment_choice == "support_negative":
            return "negative"
        if decision.comment_choice == "support_positive":
            return "positive"
        if decision.credible_choice == "negative":
            return "negative"
        if decision.credible_choice == "positive":
            return "positive"
        return "neutral"

    @staticmethod
    def _enters_public_thread(action_type: str) -> bool:
        # observe 是内部判断记录，不是公开讨论事件，不能进入后续可见上下文。
        return action_type in {"post", "reply", "like", "repost"}

    @staticmethod
    def _public_content_for_action(decision, action_type: str, agent_id: str, timestep: int) -> str:
        if action_type not in {"post", "reply"}:
            return ""
        return decision.public_comment or P3ExperimentRunner._fallback_comment(decision, agent_id, timestep)

    @staticmethod
    def _public_signal_from_decision(decision, action_type: str) -> str | None:
        if action_type == "observe":
            return None
        if action_type == "like":
            if decision.like_choice in {"positive", "negative"}:
                return decision.like_choice
            signal = P3ExperimentRunner._signal_from_decision(decision)
            return signal if signal in {"positive", "negative"} else None
        if action_type == "repost":
            if decision.share_choice in {"positive", "negative"}:
                return decision.share_choice
            return P3ExperimentRunner._signal_from_decision(decision)
        return P3ExperimentRunner._signal_from_decision(decision)

    @staticmethod
    def _fallback_comment(decision, agent_id: str, timestep: int) -> str:
        signal = P3ExperimentRunner._signal_from_decision(decision)
        reason_based = P3ExperimentRunner._build_reason_based_comment(
            reason=decision.reason,
            signal=signal,
            seed=f"{agent_id}|{timestep}|fallback",
        )
        if reason_based:
            return reason_based

        if decision.comment_choice == "support_negative":
            variants = [
                "我更担心的是现实压力可能还会继续往上走。",
                "我还是觉得眼下这些风险信号不能太快忽略。",
                "我更在意的是那些看起来被低估的压力和拖长迹象。",
            ]
            return variants[P3ExperimentRunner._stable_variant_index(f"{agent_id}|{timestep}|neg", len(variants))]
        if decision.comment_choice == "support_positive":
            variants = [
                "我现在更相信整体还在可控范围内。",
                "我会更偏向那些说情况还稳得住的判断。",
                "至少从眼前这些线索看，我还不觉得已经失控了。",
            ]
            return variants[P3ExperimentRunner._stable_variant_index(f"{agent_id}|{timestep}|pos", len(variants))]
        variants = [
            "我还想再看看后面的讨论怎么走。",
            "这会儿我先不急着表态。",
            "我暂时还想多看几条再判断。",
        ]
        return variants[P3ExperimentRunner._stable_variant_index(f"{agent_id}|{timestep}|neu", len(variants))]

    @staticmethod
    def _deduplicate_comment(
        content: str,
        action_type: str,
        reply_target: str | None,
        visible_events: list[ThreadEvent],
        decision,
    ) -> tuple[str, str, str | None]:
        if action_type not in {"post", "reply"} or not content.strip():
            return content, action_type, reply_target

        normalized = P3ExperimentRunner._normalize_text_for_similarity(content)
        recent_texts = [
            P3ExperimentRunner._normalize_text_for_similarity(event.content)
            for event in visible_events[-3:]
            if event.content
        ]
        if any(P3ExperimentRunner._is_near_duplicate(normalized, other) for other in recent_texts):
            signal = P3ExperimentRunner._signal_from_decision(decision)
            alt_comment = P3ExperimentRunner._build_reason_based_comment(
                reason=decision.reason,
                signal=signal,
                seed=f"{reply_target or ''}|{len(visible_events)}|dedup",
            )
            if alt_comment and alt_comment.strip():
                alt_normalized = P3ExperimentRunner._normalize_text_for_similarity(alt_comment)
                if not any(P3ExperimentRunner._is_near_duplicate(alt_normalized, other) for other in recent_texts):
                    return alt_comment, action_type, reply_target
            if signal == "negative":
                return "我更担心的是现实里的压力累积，不只是表面上的平稳。", action_type, reply_target
            if signal == "positive":
                return "我还是会更相信眼下这些说明情况还稳得住的线索。", action_type, reply_target
            return "我想再等等看后面有没有新的信息。", "observe", None
        return content, action_type, reply_target

    @staticmethod
    def _find_like_target_for_signal(signal: str, visible_events: list[ThreadEvent]) -> str | None:
        if signal not in {"positive", "negative"}:
            return None
        candidates = [
            event
            for event in visible_events
            if not event.event_id.startswith("seed_")
            and event.signal_polarity == signal
            and event.action_type in {"post", "reply"}
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda event: (event.salience_score, event.timestep), reverse=True)
        return candidates[0].event_id

    @staticmethod
    def _build_reason_based_comment(reason: str | None, signal: str, seed: str = "") -> str:
        text = (reason or "").strip()
        if not text:
            return ""
        text = re.sub(r"^[我][觉得到认为感觉比较更会偏向倾向现在]*", "", text)
        text = text.strip("，。；;：:、 ")
        if not text:
            return ""

        clauses = [part.strip("，。；;：:、 ") for part in re.split(r"[。！？!?；;]", text) if part.strip()]
        base = ""
        for clause in clauses:
            if len(clause) >= 8:
                base = clause
                break
        if not base and clauses:
            base = clauses[0]
        if not base:
            return ""

        prefixes = {
            "negative": ["我更担心的是", "更让我在意的是", "我现在最怕的是"],
            "positive": ["我更相信的是", "更让我安心的是", "我现在更看重的是"],
            "neutral": ["我还在想的是", "我现在拿不准的是", "我还想再看看的是"],
        }
        prefix_list = prefixes.get(signal, prefixes["neutral"])
        prefix = prefix_list[P3ExperimentRunner._stable_variant_index(f"{seed}|{signal}", len(prefix_list))]
        return f"{prefix}{base}。"

    @staticmethod
    def _stable_variant_index(seed: str, n: int) -> int:
        if n <= 0:
            return 0
        return sum(ord(ch) for ch in seed) % n

    @staticmethod
    def _normalize_text_for_similarity(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[，。！？、,.!?：:；;“”\"'‘’（）()\\-\\s]+", "", text)
        return text

    @staticmethod
    def _is_near_duplicate(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left == right:
            return True
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if shorter and shorter in longer and len(shorter) / max(len(longer), 1) >= 0.7:
            return True
        overlap = len(set(shorter[i : i + 2] for i in range(max(len(shorter) - 1, 1))) & set(longer[i : i + 2] for i in range(max(len(longer) - 1, 1))))
        return overlap >= 4

    @staticmethod
    def _build_cell_summary(
        rows: List[dict[str, object]],
        probe_rows: List[dict[str, object]] | None = None,
        panel_rows: List[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        bucket: dict[tuple[str, str, str, str, str, int], list[dict[str, object]]] = {}
        for row in rows:
            key = (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                str(row["condition_name"]),
                str(row["persona_name"]),
                str(row["chat_atmosphere"]),
                int(row["repeat_index"]),
            )
            bucket.setdefault(key, []).append(row)

        probe_bucket: dict[tuple[str, str, str, str, str, int], list[dict[str, object]]] = {}
        for row in probe_rows or []:
            key = (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                str(row["condition_name"]),
                str(row["persona_name"]),
                str(row["chat_atmosphere"]),
                int(row["repeat_index"]),
            )
            probe_bucket.setdefault(key, []).append(row)

        panel_bucket: dict[tuple[str, str, str, str, str, int], list[dict[str, object]]] = {}
        for row in panel_rows or []:
            key = (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                str(row["condition_name"]),
                str(row["persona_name"]),
                str(row["chat_atmosphere"]),
                int(row["repeat_index"]),
            )
            panel_bucket.setdefault(key, []).append(row)

        def as_bool(value: object) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes"}
            return bool(value)

        def mean_numeric(rows_: list[dict[str, object]], field: str) -> float | None:
            values: list[float] = []
            for row in rows_:
                value = row.get(field)
                if value is None:
                    continue
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed):
                    values.append(parsed)
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        def choice_nsb(rows_: list[dict[str, object]], field: str = "last_final_choice") -> float | None:
            values: list[float] = []
            for row in rows_:
                choice = str(row.get(field) or "").lower()
                if choice == "negative":
                    values.append(1.0)
                elif choice == "positive":
                    values.append(0.0)
                elif choice == "neutral":
                    values.append(0.5)
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        def public_consensus(rows_: list[dict[str, object]]) -> tuple[float | None, float | None]:
            signed_public = [
                row
                for row in rows_
                if row.get("action_type") in {"post", "reply", "like", "repost"}
                and row.get("signal_polarity") in {"positive", "negative"}
            ]
            if not signed_public:
                return None, None
            negative_share = sum(row.get("signal_polarity") == "negative" for row in signed_public) / len(signed_public)
            return round(max(negative_share, 1 - negative_share), 4), round((negative_share - 0.5) * 2, 4)

        def terminal_persona_stats(rows_: list[dict[str, object]]) -> dict[str, str]:
            by_persona: dict[str, list[dict[str, object]]] = {}
            for row in rows_:
                persona = str(row.get("agent_persona_name") or row.get("persona_name") or "")
                if persona:
                    by_persona.setdefault(persona, []).append(row)
            nsb_by_persona: dict[str, object] = {}
            active_by_persona: dict[str, int] = {}
            latent_by_persona: dict[str, object] = {}
            negative_ratio_by_persona: dict[str, object] = {}
            for persona, persona_rows in sorted(by_persona.items()):
                nsb_by_persona[persona] = choice_nsb(persona_rows)
                active_by_persona[persona] = sum(as_bool(row.get("was_ever_active")) for row in persona_rows)
                latent_by_persona[persona] = mean_numeric(persona_rows, "latent_neg_tendency")
                latent_values = [
                    float(row.get("latent_neg_tendency"))
                    for row in persona_rows
                    if row.get("latent_neg_tendency") is not None
                ]
                negative_ratio_by_persona[persona] = (
                    round(sum(value >= 0.67 for value in latent_values) / len(latent_values), 4)
                    if latent_values
                    else None
                )
            return {
                "terminal_nsb_by_persona": json.dumps(nsb_by_persona, ensure_ascii=False, sort_keys=True),
                "active_count_by_persona": json.dumps(active_by_persona, ensure_ascii=False, sort_keys=True),
                "prototype_mean_latent_by_persona": json.dumps(latent_by_persona, ensure_ascii=False, sort_keys=True),
                "prototype_negative_ratio_by_persona": json.dumps(negative_ratio_by_persona, ensure_ascii=False, sort_keys=True),
            }

        def phase_probe_stats(
            key: tuple[str, str, str, str, str, int],
            phase: str,
        ) -> dict[str, object]:
            phase_rows = [row for row in probe_bucket.get(key, []) if str(row.get("phase")) == phase]
            if not phase_rows:
                return {
                    f"{phase}_all_agents_nsb": None,
                    f"{phase}_all_agents_cb": None,
                    f"{phase}_all_agents_sb": None,
                    f"{phase}_all_agents_eb_probe": None,
                    f"{phase}_active_agents_nsb": None,
                    f"{phase}_inactive_agents_nsb": None,
                    f"{phase}_all_agents_latent_neg_mean": None,
                    f"{phase}_active_agents_latent_neg_mean": None,
                    f"{phase}_inactive_agents_latent_neg_mean": None,
                    f"{phase}_n_agents": 0,
                    f"{phase}_all_agents_attitude_n": 0,
                    f"{phase}_active_agents_attitude_n": 0,
                    f"{phase}_inactive_agents_attitude_n": 0,
                }
            active_rows = [row for row in phase_rows if as_bool(row.get("was_ever_active"))]
            inactive_rows = [row for row in phase_rows if not as_bool(row.get("was_ever_active"))]

            def attitude_n(rows_: list[dict[str, object]]) -> int:
                return sum(str(row.get("final_choice") or "").lower() in {"negative", "positive"} for row in rows_)

            return {
                f"{phase}_all_agents_nsb": mean_numeric(phase_rows, "nsb"),
                f"{phase}_all_agents_cb": mean_numeric(phase_rows, "cb"),
                f"{phase}_all_agents_sb": mean_numeric(phase_rows, "sb"),
                f"{phase}_all_agents_eb_probe": mean_numeric(phase_rows, "eb_probe"),
                f"{phase}_active_agents_nsb": mean_numeric(active_rows, "nsb"),
                f"{phase}_inactive_agents_nsb": mean_numeric(inactive_rows, "nsb"),
                f"{phase}_all_agents_latent_neg_mean": mean_numeric(phase_rows, "latent_neg_tendency"),
                f"{phase}_active_agents_latent_neg_mean": mean_numeric(active_rows, "latent_neg_tendency"),
                f"{phase}_inactive_agents_latent_neg_mean": mean_numeric(inactive_rows, "latent_neg_tendency"),
                f"{phase}_n_agents": len(phase_rows),
                f"{phase}_all_agents_attitude_n": attitude_n(phase_rows),
                f"{phase}_active_agents_attitude_n": attitude_n(active_rows),
                f"{phase}_inactive_agents_attitude_n": attitude_n(inactive_rows),
            }

        def terminal_panel_stats(
            key: tuple[str, str, str, str, str, int],
        ) -> dict[str, object]:
            rows_ = panel_bucket.get(key, [])
            if not rows_:
                return {
                    "terminal_panel_all_last_choice_nsb": None,
                    "terminal_panel_active_last_choice_nsb": None,
                    "terminal_panel_inactive_last_choice_nsb": None,
                    "terminal_panel_never_active_last_choice_nsb": None,
                    "terminal_panel_all_latent_neg_mean": None,
                    "terminal_panel_active_latent_neg_mean": None,
                    "terminal_panel_inactive_latent_neg_mean": None,
                    "terminal_panel_never_active_latent_neg_mean": None,
                    "terminal_panel_active_n": 0,
                    "terminal_panel_inactive_n": 0,
                    "terminal_panel_never_active_n": 0,
                    "terminal_nsb_by_persona": "{}",
                    "active_count_by_persona": "{}",
                    "prototype_mean_latent_by_persona": "{}",
                    "prototype_negative_ratio_by_persona": "{}",
                }
            final_timestep = max(int(row["timestep"]) for row in rows_)
            terminal_rows = [row for row in rows_ if int(row["timestep"]) == final_timestep]
            active_rows = [row for row in terminal_rows if as_bool(row.get("is_active"))]
            inactive_rows = [row for row in terminal_rows if not as_bool(row.get("is_active"))]
            never_active_rows = [row for row in terminal_rows if not as_bool(row.get("was_ever_active"))]
            return {
                "terminal_panel_all_last_choice_nsb": choice_nsb(terminal_rows),
                "terminal_panel_active_last_choice_nsb": choice_nsb(active_rows),
                "terminal_panel_inactive_last_choice_nsb": choice_nsb(inactive_rows),
                "terminal_panel_never_active_last_choice_nsb": choice_nsb(never_active_rows),
                "terminal_panel_all_latent_neg_mean": mean_numeric(terminal_rows, "latent_neg_tendency"),
                "terminal_panel_active_latent_neg_mean": mean_numeric(active_rows, "latent_neg_tendency"),
                "terminal_panel_inactive_latent_neg_mean": mean_numeric(inactive_rows, "latent_neg_tendency"),
                "terminal_panel_never_active_latent_neg_mean": mean_numeric(never_active_rows, "latent_neg_tendency"),
                "terminal_panel_active_n": len(active_rows),
                "terminal_panel_inactive_n": len(inactive_rows),
                "terminal_panel_never_active_n": len(never_active_rows),
                **terminal_persona_stats(terminal_rows),
            }

        summaries: list[dict[str, object]] = []
        for key, cell_rows in sorted(bucket.items()):
            valid_rows = [row for row in cell_rows if row.get("success", True) and row.get("nsb") is not None]
            if not valid_rows:
                continue
            final_timestep = max(int(row["timestep"]) for row in valid_rows)
            final_timestep_rows = [
                row for row in valid_rows if int(row["timestep"]) == final_timestep
            ]
            stop_rows = [row for row in valid_rows if row.get("is_stop_step")]
            stop_row = stop_rows[-1] if stop_rows else final_timestep_rows[-1]
            gcr_rows = [float(row["gcr"]) for row in valid_rows if row.get("gcr") is not None]
            ssa_rows = [float(row["ssa"]) for row in valid_rows if row.get("ssa") is not None]
            scm_rows = [float(row["scm"]) for row in valid_rows if row.get("scm") is not None]
            terminal_positive = sum(row.get("signal_polarity") == "positive" for row in final_timestep_rows)
            terminal_negative = sum(row.get("signal_polarity") == "negative" for row in final_timestep_rows)
            terminal_neutral = sum(row.get("signal_polarity") == "neutral" for row in final_timestep_rows)
            terminal_reply = sum(row.get("action_type") == "reply" for row in final_timestep_rows)
            terminal_post = sum(row.get("action_type") == "post" for row in final_timestep_rows)
            terminal_like = sum(row.get("action_type") == "like" for row in final_timestep_rows)
            terminal_repost = sum(row.get("action_type") == "repost" for row in final_timestep_rows)
            final_gcr, final_ssa = public_consensus(final_timestep_rows)
            total_public_negative = sum(row.get("signal_polarity") == "negative" for row in valid_rows)
            total_public_positive = sum(row.get("signal_polarity") == "positive" for row in valid_rows)
            public_thread_negative_share = (
                round(total_public_negative / (total_public_negative + total_public_positive), 4)
                if total_public_negative + total_public_positive
                else None
            )
            terminal_active_nsb = round(
                sum(float(row["nsb"]) for row in final_timestep_rows) / len(final_timestep_rows),
                4,
            )
            entry_probe = phase_probe_stats(key, "entry")
            terminal_probe = phase_probe_stats(key, "terminal")
            terminal_panel = terminal_panel_stats(key)
            terminal_all_agents_nsb = terminal_probe.get("terminal_all_agents_nsb")
            entry_all_agents_nsb = entry_probe.get("entry_all_agents_nsb")
            delta_all_agents_nsb = None
            if terminal_all_agents_nsb is not None and entry_all_agents_nsb is not None:
                delta_all_agents_nsb = round(
                    float(terminal_all_agents_nsb) - float(entry_all_agents_nsb),
                    4,
                )
            terminal_all_latent = terminal_probe.get("terminal_all_agents_latent_neg_mean")
            entry_all_latent = entry_probe.get("entry_all_agents_latent_neg_mean")
            terminal_inactive_latent = terminal_probe.get("terminal_inactive_agents_latent_neg_mean")
            entry_inactive_latent = entry_probe.get("entry_inactive_agents_latent_neg_mean")
            delta_all_latent = (
                round(float(terminal_all_latent) - float(entry_all_latent), 4)
                if terminal_all_latent is not None and entry_all_latent is not None
                else None
            )
            delta_inactive_latent = (
                round(float(terminal_inactive_latent) - float(entry_inactive_latent), 4)
                if terminal_inactive_latent is not None and entry_inactive_latent is not None
                else None
            )
            final_nsb = terminal_all_agents_nsb if terminal_all_agents_nsb is not None else terminal_active_nsb
            if terminal_negative > terminal_positive:
                terminal_group_tendency = "support_report_2"
            elif terminal_positive > terminal_negative:
                terminal_group_tendency = "support_report_1"
            else:
                terminal_group_tendency = "split"
            if terminal_reply + terminal_post + terminal_like + terminal_repost == 0:
                terminal_community_atmosphere = "quiet"
            elif terminal_positive == terminal_negative and terminal_positive > 0:
                terminal_community_atmosphere = "mixed"
            elif terminal_negative > terminal_positive:
                terminal_community_atmosphere = "negative_leaning"
            elif terminal_positive > terminal_negative:
                terminal_community_atmosphere = "positive_leaning"
            else:
                terminal_community_atmosphere = "unclear"
            summaries.append(
                {
                    "experiment_family": key[0],
                    "scenario_name": key[1],
                    "condition_name": key[2],
                    "persona_name": key[3],
                    "cell_persona_design": (
                        "balanced_persona_mix"
                        if key[3] == "balanced_persona_mix"
                        else "single_persona"
                    ),
                    "chat_atmosphere": key[4],
                    "repeat_index": key[5],
                    "n_events": len(valid_rows),
                    "final_timestep": final_timestep,
                    "mean_nsb": round(sum(float(row["nsb"]) for row in valid_rows) / len(valid_rows), 4),
                    "mean_gcr": round(sum(gcr_rows) / len(gcr_rows), 4) if gcr_rows else None,
                    "mean_ssa": round(sum(ssa_rows) / len(ssa_rows), 4) if ssa_rows else None,
                    "mean_scm": round(sum(scm_rows) / len(scm_rows), 4) if scm_rows else None,
                    "mean_public_nsb_active": round(sum(float(row["nsb"]) for row in valid_rows) / len(valid_rows), 4),
                    "final_nsb": final_nsb,
                    "terminal_active_nsb": terminal_active_nsb,
                    "active_agents_terminal_nsb": terminal_active_nsb,
                    "terminal_all_agents_nsb": terminal_all_agents_nsb,
                    "entry_all_agents_nsb": entry_all_agents_nsb,
                    "delta_all_agents_nsb": delta_all_agents_nsb,
                    "delta_terminal_minus_entry": delta_all_agents_nsb,
                    "public_thread_negative_share": public_thread_negative_share,
                    "terminal_active_agents_nsb": terminal_probe.get("terminal_active_agents_nsb"),
                    "terminal_inactive_agents_nsb": terminal_probe.get("terminal_inactive_agents_nsb"),
                    "entry_all_agents_latent_neg_mean": entry_all_latent,
                    "terminal_all_agents_latent_neg_mean": terminal_all_latent,
                    "delta_all_agents_latent_neg_mean": delta_all_latent,
                    "entry_active_agents_latent_neg_mean": entry_probe.get("entry_active_agents_latent_neg_mean"),
                    "terminal_active_agents_latent_neg_mean": terminal_probe.get("terminal_active_agents_latent_neg_mean"),
                    "entry_inactive_agents_latent_neg_mean": entry_inactive_latent,
                    "terminal_inactive_agents_latent_neg_mean": terminal_inactive_latent,
                    "delta_inactive_agents_latent_neg_mean": delta_inactive_latent,
                    "entry_all_agents_attitude_n": entry_probe.get("entry_all_agents_attitude_n"),
                    "terminal_all_agents_attitude_n": terminal_probe.get("terminal_all_agents_attitude_n"),
                    "entry_active_agents_attitude_n": entry_probe.get("entry_active_agents_attitude_n"),
                    "terminal_active_agents_attitude_n": terminal_probe.get("terminal_active_agents_attitude_n"),
                    "entry_inactive_agents_attitude_n": entry_probe.get("entry_inactive_agents_attitude_n"),
                    "terminal_inactive_agents_attitude_n": terminal_probe.get("terminal_inactive_agents_attitude_n"),
                    **terminal_panel,
                    "inactive_shift": (
                        round(
                            float(terminal_probe["terminal_inactive_agents_nsb"])
                            - float(entry_probe["entry_inactive_agents_nsb"]),
                            4,
                        )
                        if terminal_probe.get("terminal_inactive_agents_nsb") is not None
                        and entry_probe.get("entry_inactive_agents_nsb") is not None
                        else None
                    ),
                    "active_shift": (
                        round(
                            float(terminal_probe["terminal_active_agents_nsb"])
                            - float(entry_probe["entry_active_agents_nsb"]),
                            4,
                        )
                        if terminal_probe.get("terminal_active_agents_nsb") is not None
                        and entry_probe.get("entry_active_agents_nsb") is not None
                        else None
                    ),
                    "terminal_all_agents_cb": terminal_probe.get("terminal_all_agents_cb"),
                    "terminal_all_agents_sb": terminal_probe.get("terminal_all_agents_sb"),
                    "terminal_all_agents_eb_probe": terminal_probe.get("terminal_all_agents_eb_probe"),
                    "probe_n_agents": terminal_probe.get("terminal_n_agents"),
                    "final_gcr": final_gcr,
                    "final_ssa": final_ssa,
                    "terminal_support_report_1_count": terminal_positive,
                    "terminal_support_report_2_count": terminal_negative,
                    "terminal_neutral_count": terminal_neutral,
                    "terminal_reply_count": terminal_reply,
                    "terminal_post_count": terminal_post,
                    "terminal_like_count": terminal_like,
                    "terminal_repost_count": terminal_repost,
                    "terminal_group_tendency": terminal_group_tendency,
                    "terminal_community_atmosphere": terminal_community_atmosphere,
                    "stop_reason": stop_row.get("stop_reason"),
                    "stop_timestep": stop_row.get("timestep"),
                }
            )
        return summaries

    @staticmethod
    def _build_summary(rows: List[dict[str, object]]) -> list[dict[str, object]]:
        def finite_float(value: object) -> float | None:
            try:
                parsed = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return parsed if math.isfinite(parsed) else None

        extra_summary_fields = [
            ("entry_all_agents_nsb", "mean_entry_all_agents_nsb"),
            ("delta_terminal_minus_entry", "mean_delta_terminal_minus_entry"),
            ("active_agents_terminal_nsb", "mean_active_agents_terminal_nsb"),
            ("public_thread_negative_share", "mean_public_thread_negative_share"),
            ("terminal_all_agents_latent_neg_mean", "mean_terminal_all_agents_latent_neg_mean"),
            ("terminal_active_agents_latent_neg_mean", "mean_terminal_active_agents_latent_neg_mean"),
            ("terminal_inactive_agents_latent_neg_mean", "mean_terminal_inactive_agents_latent_neg_mean"),
            ("delta_all_agents_latent_neg_mean", "mean_delta_all_agents_latent_neg_mean"),
            ("delta_inactive_agents_latent_neg_mean", "mean_delta_inactive_agents_latent_neg_mean"),
            ("terminal_panel_all_last_choice_nsb", "mean_terminal_panel_all_last_choice_nsb"),
            ("terminal_panel_active_last_choice_nsb", "mean_terminal_panel_active_last_choice_nsb"),
            ("terminal_panel_inactive_last_choice_nsb", "mean_terminal_panel_inactive_last_choice_nsb"),
            ("terminal_panel_never_active_last_choice_nsb", "mean_terminal_panel_never_active_last_choice_nsb"),
            ("terminal_panel_all_latent_neg_mean", "mean_terminal_panel_all_latent_neg_mean"),
            ("terminal_panel_active_latent_neg_mean", "mean_terminal_panel_active_latent_neg_mean"),
            ("terminal_panel_inactive_latent_neg_mean", "mean_terminal_panel_inactive_latent_neg_mean"),
            ("terminal_panel_never_active_latent_neg_mean", "mean_terminal_panel_never_active_latent_neg_mean"),
        ]
        persona_json_fields = [
            ("terminal_nsb_by_persona", "mean_terminal_nsb_by_persona"),
            ("active_count_by_persona", "mean_active_count_by_persona"),
            ("prototype_mean_latent_by_persona", "mean_prototype_latent_by_persona"),
            ("prototype_negative_ratio_by_persona", "mean_prototype_negative_ratio_by_persona"),
        ]

        bucket: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
        for row in rows:
            key = (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                str(row["condition_name"]),
                str(row["persona_name"]),
                str(row["chat_atmosphere"]),
                str(row.get("cell_persona_design", "single_persona")),
            )
            if key not in bucket:
                bucket[key] = {
                    "n_cells": 0,
                    "mean_nsb_sum": 0.0,
                    "mean_gcr_sum": 0.0,
                    "mean_gcr_n": 0,
                    "mean_ssa_sum": 0.0,
                    "mean_ssa_n": 0,
                    "final_nsb_sum": 0.0,
                    "final_nsb_n": 0,
                    "terminal_all_agents_nsb_sum": 0.0,
                    "terminal_all_agents_nsb_n": 0,
                    "delta_all_agents_nsb_sum": 0.0,
                    "delta_all_agents_nsb_n": 0,
                    "terminal_active_nsb_sum": 0.0,
                    "terminal_active_nsb_n": 0,
                    "terminal_inactive_agents_nsb_sum": 0.0,
                    "terminal_inactive_agents_nsb_n": 0,
                    "inactive_shift_sum": 0.0,
                    "inactive_shift_n": 0,
                    "final_gcr_sum": 0.0,
                    "final_gcr_n": 0,
                    "final_ssa_sum": 0.0,
                    "final_ssa_n": 0,
                    "final_timestep_sum": 0.0,
                    "stop_reason_counts": {},
                }
                for field, _ in extra_summary_fields:
                    bucket[key][f"{field}_sum"] = 0.0
                    bucket[key][f"{field}_n"] = 0
                for field, _ in persona_json_fields:
                    bucket[key][f"{field}_by_persona"] = {}
            stats = bucket[key]
            stats["n_cells"] = int(stats["n_cells"]) + 1
            mean_nsb = finite_float(row.get("mean_nsb"))
            if mean_nsb is not None:
                stats["mean_nsb_sum"] = float(stats["mean_nsb_sum"]) + mean_nsb
            mean_gcr = finite_float(row.get("mean_gcr"))
            if mean_gcr is not None:
                stats["mean_gcr_sum"] = float(stats["mean_gcr_sum"]) + mean_gcr
                stats["mean_gcr_n"] = int(stats["mean_gcr_n"]) + 1
            mean_ssa = finite_float(row.get("mean_ssa"))
            if mean_ssa is not None:
                stats["mean_ssa_sum"] = float(stats["mean_ssa_sum"]) + mean_ssa
                stats["mean_ssa_n"] = int(stats["mean_ssa_n"]) + 1
            final_nsb = finite_float(row.get("final_nsb"))
            if final_nsb is not None:
                stats["final_nsb_sum"] = float(stats["final_nsb_sum"]) + final_nsb
                stats["final_nsb_n"] = int(stats["final_nsb_n"]) + 1
            terminal_all_agents_nsb = finite_float(row.get("terminal_all_agents_nsb"))
            if terminal_all_agents_nsb is not None:
                stats["terminal_all_agents_nsb_sum"] = float(stats["terminal_all_agents_nsb_sum"]) + terminal_all_agents_nsb
                stats["terminal_all_agents_nsb_n"] = int(stats["terminal_all_agents_nsb_n"]) + 1
            delta_all_agents_nsb = finite_float(row.get("delta_all_agents_nsb"))
            if delta_all_agents_nsb is not None:
                stats["delta_all_agents_nsb_sum"] = float(stats["delta_all_agents_nsb_sum"]) + delta_all_agents_nsb
                stats["delta_all_agents_nsb_n"] = int(stats["delta_all_agents_nsb_n"]) + 1
            terminal_active_nsb = finite_float(row.get("terminal_active_nsb"))
            if terminal_active_nsb is not None:
                stats["terminal_active_nsb_sum"] = float(stats["terminal_active_nsb_sum"]) + terminal_active_nsb
                stats["terminal_active_nsb_n"] = int(stats["terminal_active_nsb_n"]) + 1
            terminal_inactive_agents_nsb = finite_float(row.get("terminal_inactive_agents_nsb"))
            if terminal_inactive_agents_nsb is not None:
                stats["terminal_inactive_agents_nsb_sum"] = float(stats["terminal_inactive_agents_nsb_sum"]) + terminal_inactive_agents_nsb
                stats["terminal_inactive_agents_nsb_n"] = int(stats["terminal_inactive_agents_nsb_n"]) + 1
            inactive_shift = finite_float(row.get("inactive_shift"))
            if inactive_shift is not None:
                stats["inactive_shift_sum"] = float(stats["inactive_shift_sum"]) + inactive_shift
                stats["inactive_shift_n"] = int(stats["inactive_shift_n"]) + 1
            for field, _ in extra_summary_fields:
                value = finite_float(row.get(field))
                if value is not None:
                    stats[f"{field}_sum"] = float(stats[f"{field}_sum"]) + value
                    stats[f"{field}_n"] = int(stats[f"{field}_n"]) + 1
            for field, _ in persona_json_fields:
                try:
                    parsed = json.loads(str(row.get(field) or "{}"))
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    field_bucket = dict(stats[f"{field}_by_persona"])
                    for persona_key, raw_value in parsed.items():
                        value = finite_float(raw_value)
                        if value is None:
                            continue
                        persona_stats = dict(field_bucket.get(str(persona_key), {"sum": 0.0, "n": 0}))
                        persona_stats["sum"] = float(persona_stats["sum"]) + value
                        persona_stats["n"] = int(persona_stats["n"]) + 1
                        field_bucket[str(persona_key)] = persona_stats
                    stats[f"{field}_by_persona"] = field_bucket
            final_gcr = finite_float(row.get("final_gcr"))
            if final_gcr is not None:
                stats["final_gcr_sum"] = float(stats["final_gcr_sum"]) + final_gcr
                stats["final_gcr_n"] = int(stats["final_gcr_n"]) + 1
            final_ssa = finite_float(row.get("final_ssa"))
            if final_ssa is not None:
                stats["final_ssa_sum"] = float(stats["final_ssa_sum"]) + final_ssa
                stats["final_ssa_n"] = int(stats["final_ssa_n"]) + 1
            final_timestep = finite_float(row.get("final_timestep")) or 0.0
            stats["final_timestep_sum"] = float(stats["final_timestep_sum"]) + final_timestep
            raw_stop_reason = row.get("stop_reason")
            stop_reason = str(raw_stop_reason) if raw_stop_reason is not None and str(raw_stop_reason) != "nan" else "unknown"
            stop_reason_counts = dict(stats["stop_reason_counts"])
            stop_reason_counts[stop_reason] = int(stop_reason_counts.get(stop_reason, 0)) + 1
            stats["stop_reason_counts"] = stop_reason_counts
        return [
            {
                "experiment_family": family,
                "scenario_name": scenario,
                "condition_name": condition,
                "persona_name": persona_name,
                "chat_atmosphere": chat_atmosphere,
                "cell_persona_design": cell_persona_design,
                "n_cells": int(stats["n_cells"]),
                "mean_nsb": round(float(stats["mean_nsb_sum"]) / int(stats["n_cells"]), 4),
                "mean_gcr": round(float(stats["mean_gcr_sum"]) / int(stats["mean_gcr_n"]), 4) if int(stats["mean_gcr_n"]) else None,
                "mean_ssa": round(float(stats["mean_ssa_sum"]) / int(stats["mean_ssa_n"]), 4) if int(stats["mean_ssa_n"]) else None,
                "mean_final_nsb": round(float(stats["final_nsb_sum"]) / int(stats["final_nsb_n"]), 4) if int(stats["final_nsb_n"]) else None,
                "mean_terminal_all_agents_nsb": round(float(stats["terminal_all_agents_nsb_sum"]) / int(stats["terminal_all_agents_nsb_n"]), 4) if int(stats["terminal_all_agents_nsb_n"]) else None,
                "mean_delta_all_agents_nsb": round(float(stats["delta_all_agents_nsb_sum"]) / int(stats["delta_all_agents_nsb_n"]), 4) if int(stats["delta_all_agents_nsb_n"]) else None,
                "mean_terminal_active_nsb": round(float(stats["terminal_active_nsb_sum"]) / int(stats["terminal_active_nsb_n"]), 4) if int(stats["terminal_active_nsb_n"]) else None,
                "mean_terminal_inactive_agents_nsb": round(float(stats["terminal_inactive_agents_nsb_sum"]) / int(stats["terminal_inactive_agents_nsb_n"]), 4) if int(stats["terminal_inactive_agents_nsb_n"]) else None,
                "mean_inactive_shift": round(float(stats["inactive_shift_sum"]) / int(stats["inactive_shift_n"]), 4) if int(stats["inactive_shift_n"]) else None,
                **{
                    output_name: (
                        round(float(stats[f"{field}_sum"]) / int(stats[f"{field}_n"]), 4)
                        if int(stats[f"{field}_n"])
                        else None
                    )
                    for field, output_name in extra_summary_fields
                },
                **{
                    output_name: json.dumps(
                        {
                            persona_key: round(float(value["sum"]) / int(value["n"]), 4)
                            for persona_key, value in sorted(dict(stats[f"{field}_by_persona"]).items())
                            if int(value["n"])
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    for field, output_name in persona_json_fields
                },
                "mean_final_gcr": round(float(stats["final_gcr_sum"]) / int(stats["final_gcr_n"]), 4) if int(stats["final_gcr_n"]) else None,
                "mean_final_ssa": round(float(stats["final_ssa_sum"]) / int(stats["final_ssa_n"]), 4) if int(stats["final_ssa_n"]) else None,
                "mean_final_timestep": round(float(stats["final_timestep_sum"]) / int(stats["n_cells"]), 2),
                "stop_reason_counts": "; ".join(
                    f"{reason}:{count}" for reason, count in sorted(dict(stats["stop_reason_counts"]).items())
                ),
            }
            for (family, scenario, condition, persona_name, chat_atmosphere, cell_persona_design), stats in sorted(bucket.items())
        ]

    @staticmethod
    def _build_agent_persona_summary(rows: List[dict[str, object]]) -> list[dict[str, object]]:
        bucket: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
        for row in rows:
            if not row.get("success", True) or row.get("nsb") is None:
                continue
            agent_persona_name = str(row.get("agent_persona_name") or "")
            if not agent_persona_name:
                continue
            key = (
                str(row["experiment_family"]),
                str(row["scenario_name"]),
                str(row["condition_name"]),
                agent_persona_name,
                str(row["chat_atmosphere"]),
            )
            if key not in bucket:
                bucket[key] = {
                    "n_rows": 0.0,
                    "nsb_sum": 0.0,
                    "cb_sum": 0.0,
                    "sb_sum": 0.0,
                    "eb_sum": 0.0,
                }
            stats = bucket[key]
            stats["n_rows"] += 1.0
            stats["nsb_sum"] += float(row["nsb"])
            stats["cb_sum"] += float(row["cb"])
            stats["sb_sum"] += float(row["sb"])
            stats["eb_sum"] += float(row["eb"])
        return [
            {
                "experiment_family": family,
                "scenario_name": scenario,
                "condition_name": condition,
                "agent_persona_name": agent_persona_name,
                "chat_atmosphere": chat_atmosphere,
                "n_rows": int(stats["n_rows"]),
                "mean_nsb": round(float(stats["nsb_sum"]) / float(stats["n_rows"]), 4),
                "mean_cb": round(float(stats["cb_sum"]) / float(stats["n_rows"]), 4),
                "mean_sb": round(float(stats["sb_sum"]) / float(stats["n_rows"]), 4),
                "mean_eb": round(float(stats["eb_sum"]) / float(stats["n_rows"]), 4),
            }
            for (family, scenario, condition, agent_persona_name, chat_atmosphere), stats in sorted(bucket.items())
        ]

    def _iter_batches(self, cells: List[ExperimentCell]) -> List[List[ExperimentCell]]:
        return [cells[index:index + self.batch_size] for index in range(0, len(cells), self.batch_size)]
