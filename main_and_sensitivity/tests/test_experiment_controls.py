from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.activation_schedule import ActivationScheduleStore
from src.config import load_config
from src.llm_client import LLMDecisionParser
from src.metrics import encode_metrics
from src.runner import ExperimentPlanner, P3ExperimentRunner
from src.schemas import DiscussionAgent, ExperimentCell, P3DecisionOutput, ThreadEvent


ROOT = Path(__file__).resolve().parents[1]


class ExperimentControlTests(unittest.TestCase):
    def test_ternary_static_final_choice_encodes_neutral_as_half(self) -> None:
        decision = LLMDecisionParser.parse_decision(
            json.dumps(
                {
                    "final_choice": "neutral",
                    "credible_choice": "neutral",
                    "share_choice": "none",
                    "comment_choice": "neutral",
                    "like_choice": "none",
                }
            ),
            report_mapping={"report_1": "positive", "report_2": "negative"},
            allow_neutral_final=True,
        )
        self.assertEqual("neutral", decision.final_choice)
        self.assertEqual(0.5, encode_metrics(decision).nsb)

    def test_binary_static_mode_keeps_existing_fallback(self) -> None:
        decision = LLMDecisionParser.parse_decision(
            json.dumps(
                {
                    "final_choice": "neutral",
                    "credible_choice": "neutral",
                    "share_choice": "none",
                    "comment_choice": "neutral",
                    "like_choice": "none",
                }
            ),
            report_mapping={"report_1": "positive", "report_2": "negative"},
        )
        self.assertEqual("positive", decision.final_choice)

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


if __name__ == "__main__":
    unittest.main()
