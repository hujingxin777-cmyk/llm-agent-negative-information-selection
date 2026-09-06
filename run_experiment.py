from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

from src.config import load_config
from src.env_utils import load_env_file
from src.exporter import ResultExporter
from src.io_utils import write_csv
from src.llm_client import LLMClient, LLMDecisionParser
from src.metrics import encode_metrics
from src.persona_variants import apply_persona_variant_to_config, set_persona_variant
from src.prompts import build_system_message, build_user_prompt
from src.runner import DiscussionSimulator, ExperimentPlanner, P3ExperimentRunner, StaticExperimentRunner
from src.visualize import FigureBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="负面信息选择实验统一启动脚本")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="单文件总配置路径",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="环境变量文件路径",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="只运行某个实验家族，如 main / persona_supplement / atmosphere_supplement / combined_supplement",
    )
    parser.add_argument(
        "--limit-static",
        type=int,
        default=None,
        help="静态实验最多运行多少个 cell",
    )
    parser.add_argument(
        "--limit-p3",
        type=int,
        default=None,
        help="P3 实验最多运行多少个 cell",
    )
    parser.add_argument("--enable-persona", action="store_true", help="启用 persona 补充实验")
    parser.add_argument("--enable-atmosphere", action="store_true", help="启用聊天氛围补充实验")
    parser.add_argument("--enable-combined", action="store_true", help="启用联合补充实验")
    parser.add_argument("--main-only", action="store_true", help="只保留主实验，关闭全部补充实验")
    parser.add_argument("--repeats", type=int, default=None, help="临时覆盖所有已启用实验的 repeats_per_cell")
    parser.add_argument("--repeat-start", type=int, default=None, help="只生成从该编号开始的 repeat")
    parser.add_argument("--repeat-end", type=int, default=None, help="只生成到该编号结束的 repeat")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_names",
        default=None,
        help="只运行指定场景；可重复传入，例如 --scenario influenza_a",
    )
    parser.add_argument("--output-dir", default=None, help="临时覆盖 project.output_dir")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="临时覆盖模型读取超时秒数")
    parser.add_argument("--connect-timeout-seconds", type=float, default=None, help="临时覆盖模型连接超时秒数")
    parser.add_argument(
        "--p3-main-persona",
        default=None,
        help="仅覆盖主实验 P3 的 persona，例如 balanced_persona_mix；不影响主实验 P1/P2。",
    )
    parser.add_argument(
        "--p3-static-reference",
        choices=["enabled", "disabled"],
        default=None,
        help="P3 是否在 prompt 中保留 P2 静态参考材料。",
    )
    parser.add_argument(
        "--p3-dynamics-profile",
        choices=["default", "damped"],
        default=None,
        help="P3 群体影响动力学档位；damped 用于诊断过快共识收敛。",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="生成实验设计清单")
    subparsers.add_parser("preview-p3", help="生成一个 P3 讨论线程预览")
    subparsers.add_parser("static", help="只运行静态实验 P1/P2")
    subparsers.add_parser("p3", help="只运行动态实验 P3")
    sensitivity = subparsers.add_parser("p3-sensitivity", help="运行 P3 active 人数/轮次敏感性实验")
    sensitivity.add_argument(
        "--active-levels",
        default=None,
        help="覆盖 p3.sensitivity.active_levels，例如 3,6,9",
    )
    sensitivity.add_argument(
        "--timestep-levels",
        default=None,
        help="覆盖 p3.sensitivity.timestep_levels，例如 10,15,20",
    )
    subparsers.add_parser("all", help="先运行静态实验，再运行 P3，并自动出图和导出")
    subparsers.add_parser("plot", help="根据结果文件生成图表")
    subparsers.add_parser("export", help="导出汇总表、图表与 manifest")
    smoke = subparsers.add_parser("smoke", help="单条模型调用诊断")
    smoke.add_argument("--condition", default="independent")
    smoke.add_argument("--scenario", default="mycoplasma_pneumonia")
    smoke.add_argument("--persona", default="baseline_open")
    smoke.add_argument("--chat-atmosphere", default="none")
    return parser


def run_plan(config: dict) -> None:
    planner = ExperimentPlanner(config)
    planner.export_plan()
    print("已生成实验设计清单。")


def run_preview(config: dict) -> None:
    planner = ExperimentPlanner(config)
    cells = planner.build_cells()
    simulator = DiscussionSimulator(config)
    simulator.export_preview(Path(config["project"]["output_dir"]), cells)
    print("已生成 P3 讨论线程预览。")


def run_static(config: dict, family: str | None, limit: int | None) -> None:
    planner = ExperimentPlanner(config)
    cells = planner.build_cells()
    runner = StaticExperimentRunner(config)
    rows = runner.run_cells(
        cells=cells,
        output_dir=Path(config["project"]["output_dir"]),
        family_filter=family,
        limit=limit,
    )
    print(f"已完成静态实验，生成 {len(rows)} 条结果。")


def run_p3(config: dict, family: str | None, limit: int | None) -> None:
    planner = ExperimentPlanner(config)
    cells = planner.build_cells()
    runner = P3ExperimentRunner(config)
    rows = runner.run_cells(
        cells=cells,
        output_dir=Path(config["project"]["output_dir"]),
        family_filter=family,
        limit=limit,
    )
    print(f"已完成 P3 实验，生成 {len(rows)} 条结果。")


def _parse_int_levels(raw_value: str | None, default_levels: list[int]) -> list[int]:
    values = default_levels if raw_value is None else [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    levels: list[int] = []
    for value in values:
        if value <= 0:
            raise ValueError(f"敏感性实验参数必须为正整数: {value}")
        if value not in levels:
            levels.append(value)
    if not levels:
        raise ValueError("敏感性实验至少需要一个参数档位")
    return levels


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def run_p3_sensitivity(
    config: dict,
    family: str | None,
    limit: int | None,
    active_levels_arg: str | None,
    timestep_levels_arg: str | None,
) -> None:
    p3_cfg = config.get("p3", {})
    sensitivity_cfg = p3_cfg.get("sensitivity", {})
    active_levels = _parse_int_levels(
        active_levels_arg,
        [int(item) for item in sensitivity_cfg.get("active_levels", [int(p3_cfg.get("max_active_agents", 3))])],
    )
    timestep_levels = _parse_int_levels(
        timestep_levels_arg,
        [int(item) for item in sensitivity_cfg.get("timestep_levels", [int(p3_cfg.get("max_timestep", 10))])],
    )
    base_output_dir = Path(config["project"]["output_dir"])
    sensitivity_root = base_output_dir / "sensitivity"
    summary_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []

    for max_active_agents in active_levels:
        for max_timestep in timestep_levels:
            label = f"active_{max_active_agents}_steps_{max_timestep}"
            run_config = deepcopy(config)
            run_config.setdefault("p3", {})["max_active_agents"] = max_active_agents
            run_config.setdefault("p3", {})["max_timestep"] = max_timestep
            run_config.setdefault("project", {})["output_dir"] = str(sensitivity_root / label)
            print(f"开始敏感性实验: {label}")
            run_p3(run_config, family=family, limit=limit)

            result_dir = Path(run_config["project"]["output_dir"]) / "results"
            for row in _read_csv_rows(result_dir / "p3_summary.csv"):
                summary_rows.append(
                    {
                        "sensitivity_label": label,
                        "max_active_agents": max_active_agents,
                        "max_timestep": max_timestep,
                        **row,
                    }
                )
            for row in _read_csv_rows(result_dir / "p3_cell_summary.csv"):
                cell_rows.append(
                    {
                        "sensitivity_label": label,
                        "max_active_agents": max_active_agents,
                        "max_timestep": max_timestep,
                        **row,
                    }
                )

    write_csv(sensitivity_root / "p3_sensitivity_summary.csv", summary_rows)
    write_csv(sensitivity_root / "p3_sensitivity_cell_summary.csv", cell_rows)
    print(f"已完成敏感性实验，共 {len(active_levels) * len(timestep_levels)} 组。")
    print(f"汇总文件: {sensitivity_root / 'p3_sensitivity_summary.csv'}")
    print(f"cell 汇总文件: {sensitivity_root / 'p3_sensitivity_cell_summary.csv'}")


def run_plot(config: dict) -> None:
    builder = FigureBuilder(Path(config["project"]["output_dir"]))
    generated = builder.build_all()
    if generated:
        print("已生成图表：")
        for name in generated:
            print(name)
    else:
        print("未找到可用于出图的结果文件。")


def run_export(config: dict) -> None:
    exported = ResultExporter(Path(config["project"]["output_dir"])).export_bundle()
    if exported:
        print("已导出文件：")
        for name in exported:
            print(name)
    else:
        print("当前没有可导出的结果文件。")


def run_smoke(config: dict, args) -> None:
    scenarios = {item["name"]: item for item in config["scenarios"]}
    personas = {item["name"]: item for item in config["personas"]}
    scenario = scenarios[args.scenario]
    persona = personas[args.persona]
    report_mapping = {"report_1": "positive", "report_2": "negative"}
    system_prompt = build_system_message(persona["profile"])
    user_prompt = build_user_prompt(
        config=config,
        scenario=scenario,
        condition_name=args.condition,
        chat_atmosphere=args.chat_atmosphere,
        report_mapping=report_mapping,
        persona_behavior_text="",
    )
    client = LLMClient(config)
    raw_text = client.generate_text(
        system_prompt,
        user_prompt,
        request_meta={
            "runner": "smoke",
            "scenario_name": args.scenario,
            "condition_name": args.condition,
            "persona_name": args.persona,
            "chat_atmosphere": args.chat_atmosphere,
        },
    )
    static_cfg = config.get("static_measurement", {})
    allow_neutral_final = str(static_cfg.get("final_choice_mode", "binary")).lower() == "ternary"
    decision = LLMDecisionParser.parse_decision(
        raw_text,
        report_mapping=report_mapping,
        allow_neutral_final=allow_neutral_final,
    )
    metrics = encode_metrics(decision)
    print("=== RAW ===")
    print(raw_text)
    print("\n=== PARSED ===")
    print(decision)
    print("\n=== METRICS ===")
    print(metrics)


def apply_cli_overrides(config: dict, args) -> dict:
    config = deepcopy(config)
    experiments = config.get("experiments", {})
    if args.main_only:
        for key in ("persona_supplement", "atmosphere_supplement", "combined_supplement"):
            if key in experiments:
                experiments[key]["enabled"] = False
    if args.enable_persona and "persona_supplement" in experiments:
        experiments["persona_supplement"]["enabled"] = True
    if args.enable_atmosphere and "atmosphere_supplement" in experiments:
        experiments["atmosphere_supplement"]["enabled"] = True
    if args.enable_combined and "combined_supplement" in experiments:
        experiments["combined_supplement"]["enabled"] = True
    if args.repeats is not None:
        for exp_cfg in experiments.values():
            if isinstance(exp_cfg, dict) and exp_cfg.get("enabled", False):
                exp_cfg["repeats_per_cell"] = args.repeats
    if args.repeat_start is not None:
        config.setdefault("runtime", {})["repeat_start"] = args.repeat_start
    if args.repeat_end is not None:
        config.setdefault("runtime", {})["repeat_end"] = args.repeat_end
    if args.scenario_names:
        config.setdefault("runtime", {})["scenario_filter"] = list(dict.fromkeys(args.scenario_names))
    if args.output_dir is not None:
        config.setdefault("project", {})["output_dir"] = args.output_dir
    if args.timeout_seconds is not None:
        config.setdefault("model", {})["timeout_seconds"] = args.timeout_seconds
    if args.connect_timeout_seconds is not None:
        config.setdefault("model", {})["connect_timeout_seconds"] = args.connect_timeout_seconds
    if args.p3_main_persona is not None:
        config.setdefault("p3", {})["main_persona_override"] = args.p3_main_persona
    if args.p3_static_reference is not None:
        config.setdefault("p3", {}).setdefault("static_reference", {})["enabled"] = (
            args.p3_static_reference == "enabled"
        )
    if args.p3_dynamics_profile == "damped":
        p3_cfg = config.setdefault("p3", {})
        p3_cfg["consensus_threshold"] = max(float(p3_cfg.get("consensus_threshold", 0.9)), 0.95)
        p3_cfg["consensus_hold_steps"] = max(int(p3_cfg.get("consensus_hold_steps", 2)), 4)
        p3_cfg.setdefault("dynamics", {}).update(
            {
                "exposure_shift_weight": 0.04,
                "arousal_shift_weight": 0.02,
                "risk_shift_weight": 0.025,
                "majority_risk_shift_weight": 0.06,
                "majority_latent_shift_weight": 0.05,
                "decision_anchor_weight": 0.20,
            }
        )
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(Path(args.config)), args)
    set_persona_variant(str(config.get("persona_runtime", {}).get("variant", "legacy")))
    config = apply_persona_variant_to_config(config)

    if args.command in {"static", "p3", "p3-sensitivity", "all", "smoke"}:
        load_env_file(args.env_file)

    if args.command == "plan":
        run_plan(config)
        return

    if args.command == "preview-p3":
        run_preview(config)
        return

    if args.command == "static":
        run_static(config, family=args.family, limit=args.limit_static)
        return

    if args.command == "p3":
        run_p3(config, family=args.family, limit=args.limit_p3)
        return

    if args.command == "p3-sensitivity":
        run_p3_sensitivity(
            config,
            family=args.family,
            limit=args.limit_p3,
            active_levels_arg=args.active_levels,
            timestep_levels_arg=args.timestep_levels,
        )
        return

    if args.command == "all":
        run_static(config, family=args.family, limit=args.limit_static)
        run_p3(config, family=args.family, limit=args.limit_p3)
        run_plot(config)
        run_export(config)
        return

    if args.command == "plot":
        run_plot(config)
        return

    if args.command == "export":
        run_export(config)
        return

    if args.command == "smoke":
        run_smoke(config, args)
        return


if __name__ == "__main__":
    main()
