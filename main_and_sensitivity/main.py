from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.env_utils import load_env_file
from src.llm_client import LLMClient, LLMDecisionParser
from src.metrics import encode_metrics
from src.persona_variants import apply_persona_variant_to_config, set_persona_variant
from src.prompts import build_prompt_preview
from src.prompts import build_system_message, build_user_prompt
from src.runner import DiscussionSimulator, ExperimentPlanner, P3ExperimentRunner, StaticExperimentRunner
from src.visualize import FigureBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="负面信息选择实验骨架")
    parser.add_argument(
        "--config",
        default="full_experiment_大规模实验.yaml",
        help="配置文件路径",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="生成实验设计清单")
    subparsers.add_parser("simulate-p3-preview", help="生成一个 P3 讨论线程预览")
    subparsers.add_parser("plot-results", help="根据结果文件生成论文图表")
    run_static = subparsers.add_parser("run-static", help="运行 P1/P2 的真实模型实验")
    run_static.add_argument("--env-file", default=".env")
    run_static.add_argument("--family", default=None)
    run_static.add_argument("--limit", type=int, default=None)
    run_p3 = subparsers.add_parser("run-p3", help="运行 P3 的最小真实模型实验")
    run_p3.add_argument("--env-file", default=".env")
    run_p3.add_argument("--family", default=None)
    run_p3.add_argument("--limit", type=int, default=None)
    smoke = subparsers.add_parser("model-smoke-test", help="调用一次真实模型做冒烟测试")
    smoke.add_argument("--condition", default="independent")
    smoke.add_argument("--scenario", default="mycoplasma_pneumonia")
    smoke.add_argument("--persona", default="baseline_open")
    smoke.add_argument("--chat-atmosphere", default="none")
    smoke.add_argument("--env-file", default=".env")

    preview = subparsers.add_parser("prompt-preview", help="预览提示词")
    preview.add_argument("--condition", required=True)
    preview.add_argument("--scenario", required=True)
    preview.add_argument("--persona", default="baseline_open")
    preview.add_argument("--chat-atmosphere", default="none")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))
    set_persona_variant(str(config.get("persona_runtime", {}).get("variant", "legacy")))
    config = apply_persona_variant_to_config(config)

    if args.command == "plan":
        planner = ExperimentPlanner(config)
        planner.export_plan()
        print("已生成实验设计清单。")
        return

    if args.command == "simulate-p3-preview":
        planner = ExperimentPlanner(config)
        cells = planner.build_cells()
        simulator = DiscussionSimulator(config)
        simulator.export_preview(Path(config["project"]["output_dir"]), cells)
        print("已生成 P3 讨论线程预览。")
        return

    if args.command == "plot-results":
        builder = FigureBuilder(Path(config["project"]["output_dir"]))
        generated = builder.build_all()
        if generated:
            print("已生成图表：")
            for name in generated:
                print(name)
        else:
            print("未找到可用于出图的结果文件。")
        return

    if args.command == "prompt-preview":
        content = build_prompt_preview(
            config=config,
            scenario_name=args.scenario,
            condition_name=args.condition,
            persona_name=args.persona,
            chat_atmosphere=args.chat_atmosphere,
            report_mapping={"report_1": "positive", "report_2": "negative"},
        )
        print(content)
        return

    if args.command == "run-static":
        load_env_file(args.env_file)
        planner = ExperimentPlanner(config)
        cells = planner.build_cells()
        runner = StaticExperimentRunner(config)
        rows = runner.run_cells(
            cells=cells,
            output_dir=Path(config["project"]["output_dir"]),
            family_filter=args.family,
            limit=args.limit,
        )
        print(f"已完成静态实验，生成 {len(rows)} 条结果。")
        return

    if args.command == "run-p3":
        load_env_file(args.env_file)
        planner = ExperimentPlanner(config)
        cells = planner.build_cells()
        runner = P3ExperimentRunner(config)
        rows = runner.run_cells(
            cells=cells,
            output_dir=Path(config["project"]["output_dir"]),
            family_filter=args.family,
            limit=args.limit,
        )
        print(f"已完成 P3 实验，生成 {len(rows)} 条结果。")
        return

    if args.command == "model-smoke-test":
        load_env_file(args.env_file)
        scenarios = {item["name"]: item for item in config["scenarios"]}
        personas = {item["name"]: item for item in config["personas"]}
        scenario = scenarios[args.scenario]
        persona = personas[args.persona]
        system_prompt = build_system_message(persona["profile"])
        user_prompt = build_user_prompt(
            config=config,
            scenario=scenario,
            condition_name=args.condition,
            chat_atmosphere=args.chat_atmosphere,
            report_mapping={"report_1": "positive", "report_2": "negative"},
        )
        client = LLMClient(config)
        raw_text = client.generate_text(system_prompt, user_prompt)
        decision = LLMDecisionParser.parse_decision(raw_text)
        metrics = encode_metrics(decision)
        print("=== RAW ===")
        print(raw_text)
        print("\n=== PARSED ===")
        print(decision)
        print("\n=== METRICS ===")
        print(metrics)
        return


if __name__ == "__main__":
    main()
