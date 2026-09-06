from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.activation_schedule import ActivationScheduleStore
from src.config import load_config
from src.llm_client import LLMClient, LLMDecisionParser
from src.runner import ExperimentPlanner, P3ExperimentRunner
from src.schemas import DiscussionAgent, ExperimentCell, P3DecisionOutput, ThreadEvent


ROOT = Path(__file__).resolve().parents[1]


class ExperimentControlTests(unittest.TestCase):
    def test_inherited_config_and_repeat_slice(self) -> None:
        config = load_config(ROOT / "configs" / "main_300_add_repeats_11_30.yaml")
        cells = ExperimentPlanner(config).build_cells()
        p3_cells = [cell for cell in cells if cell.condition_name == "high_interaction"]
        self.assertEqual(40, len(p3_cells))
        self.assertEqual(set(range(11, 31)), {cell.repeat_index for cell in p3_cells})

    def test_scale_config_filters_to_influenza(self) -> None:
        config = load_config(ROOT / "configs" / "main_scale_influenza_1000_r10.yaml")
        cells = ExperimentPlanner(config).build_cells()
        self.assertEqual({"influenza_a"}, {cell.scenario_name for cell in cells})
        self.assertEqual(1000, config["p3"]["num_agents"])
        self.assertEqual(50, config["p3"]["max_active_agents"])

    def test_action_ablation_never_targets_seed(self) -> None:
        decision = P3DecisionOutput(
            final_choice="negative",
            credible_choice="negative",
            share_choice="none",
            comment_choice="support_negative",
            planned_action="post",
            public_comment="test",
        )
        events = [
            ThreadEvent(0, "seed_1", "seed_user", "post", "seed"),
            ThreadEvent(1, "t1_e1", "agent_1", "post", "dynamic"),
        ]
        reply = P3ExperimentRunner._apply_action_policy(decision, events, 2, "reply_chain")
        like = P3ExperimentRunner._apply_action_policy(decision, events, 2, "like_channel")
        self.assertEqual(("reply", "t1_e1"), (reply.planned_action, reply.reply_target))
        self.assertEqual(("like", "t1_e1"), (like.planned_action, like.reply_target))

    def test_activation_schedule_round_trip(self) -> None:
        cell = ExperimentCell("main", "influenza_a", "high_interaction", "baseline", "none", 1, 8)
        agents = [DiscussionAgent(f"agent_{index}", "baseline", "medium") for index in range(1, 4)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            store = ActivationScheduleStore({"p3": {"activation": {}}})
            store.record(output, cell, 1, agents[:2], replayed=False)
            replay = ActivationScheduleStore(
                {"p3": {"activation": {"replay_schedule_path": str(output / "progress" / "activation_schedule.jsonl")}}}
            )
            selected = replay.replay_agents(cell, 1, agents, max_active_agents=2)
            self.assertEqual(["agent_1", "agent_2"], [agent.agent_id for agent in selected or []])

    def test_deepseek_robustness_config_uses_non_thinking_mode(self) -> None:
        config = load_config(ROOT / "configs" / "deepseek_v4_flash_main_r10.yaml")
        client = object.__new__(LLMClient)
        client.model_cfg = config["model"]
        kwargs = client._build_request_kwargs("system", "user")
        self.assertEqual("deepseek-v4-flash", kwargs["model"])
        self.assertEqual({"thinking": {"type": "disabled"}}, kwargs["extra_body"])
        self.assertEqual(2048, kwargs["max_tokens"])
        self.assertEqual(0.7, kwargs["temperature"])

    def test_gpt_robustness_config_uses_fixed_non_reasoning_snapshot(self) -> None:
        config = load_config(ROOT / "configs" / "gpt_5_1_main_r10.yaml")
        client = object.__new__(LLMClient)
        client.model_cfg = config["model"]
        kwargs = client._build_request_kwargs("system", "user")
        self.assertEqual("gpt-5.1-2025-11-13", kwargs["model"])
        self.assertEqual("none", kwargs["reasoning_effort"])
        self.assertEqual(2048, kwargs["max_completion_tokens"])
        self.assertNotIn("temperature", kwargs)

    def test_gpt_5_4_robustness_config_uses_fixed_non_reasoning_snapshot(self) -> None:
        config = load_config(ROOT / "configs" / "gpt_5_4_main_r10.yaml")
        client = object.__new__(LLMClient)
        client.model_cfg = config["model"]
        kwargs = client._build_request_kwargs("system", "user")
        self.assertEqual("gpt-5.4-2026-03-05", kwargs["model"])
        self.assertEqual("none", kwargs["reasoning_effort"])
        self.assertEqual(2048, kwargs["max_completion_tokens"])
        self.assertNotIn("temperature", kwargs)

    def test_model_robustness_configs_keep_main_design_constant(self) -> None:
        for config_name in (
            "deepseek_v4_flash_main_r10.yaml",
            "gpt_5_1_main_r10.yaml",
            "gpt_5_4_main_r10.yaml",
        ):
            config = load_config(ROOT / "configs" / config_name)
            cells = ExperimentPlanner(config).build_cells()
            self.assertEqual(60, len(cells))
            self.assertEqual(20, len([cell for cell in cells if cell.condition_name == "high_interaction"]))
            self.assertEqual(300, config["p3"]["num_agents"])
            self.assertEqual(15, config["p3"]["max_active_agents"])
            self.assertEqual(10, config["experiments"]["main"]["repeats_per_cell"])

    def test_smoke_report_mapping_matches_experiment_parser(self) -> None:
        decision = LLMDecisionParser.parse_decision(
            '{"final_choice":"report_2","credible_choice":"report_2"}',
            report_mapping={"report_1": "positive", "report_2": "negative"},
        )
        self.assertEqual("negative", decision.final_choice)
        self.assertEqual("negative", decision.credible_choice)


if __name__ == "__main__":
    unittest.main()
