from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT.parent / "outputs"
BASELINE = ROOT.parent.parent / "敏感性和稳定性检测" / "outputs" / "main_300_formal_r1_30"
SOURCES = {
    "GLM-4": BASELINE,
    "DeepSeek-V4-Flash": OUTPUT_ROOT / "deepseek_v4_flash_main_r10",
    "GPT-5.1": OUTPUT_ROOT / "gpt_5_1_main_r10",
    "GPT-5.4": OUTPUT_ROOT / "gpt_5_4_main_r10",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def main() -> None:
    rows: list[dict] = []
    for model, output in SOURCES.items():
        static = read_csv(output / "results" / "static_results.csv")
        p3 = read_csv(output / "results" / "p3_cell_summary.csv")
        if not static or not p3:
            continue
        scenarios = sorted({row["scenario_name"] for row in p3})
        for scenario in scenarios:
            p1 = describe([
                float(row["nsb"])
                for row in static
                if row["scenario_name"] == scenario and row["condition_name"] == "independent"
            ])
            p2 = describe([
                float(row["nsb"])
                for row in static
                if row["scenario_name"] == scenario and row["condition_name"] == "low_feedback"
            ])
            cell_rows = [row for row in p3 if row["scenario_name"] == scenario]
            entry = describe([float(row["entry_all_agents_nsb"]) for row in cell_rows])
            terminal = describe([float(row["terminal_all_agents_nsb"]) for row in cell_rows])
            delta = describe([float(row["delta_terminal_minus_entry"]) for row in cell_rows])
            public = describe([float(row["public_thread_negative_share"]) for row in cell_rows])
            p2_mean = float(p2["mean"]) if p2["mean"] is not None else None
            p3_terminal_mean = float(terminal["mean"]) if terminal["mean"] is not None else None
            p1_mean = float(p1["mean"]) if p1["mean"] is not None else None
            p2_minus_p1 = None if p1_mean is None or p2_mean is None else p2_mean - p1_mean
            p3_minus_p2 = None if p2_mean is None or p3_terminal_mean is None else p3_terminal_mean - p2_mean
            rows.append({
                "model": model,
                "scenario_name": scenario,
                "p1": p1,
                "p2": p2,
                "p3_entry": entry,
                "p3_terminal": terminal,
                "p3_delta": delta,
                "public_negative_share": public,
                "p2_minus_p1": p2_minus_p1,
                "p3_terminal_minus_p2": p3_minus_p2,
                # P1/P2/P3 are parallel experimental stages. The primary H3
                # robustness check is P3 terminal versus P2, not terminal versus
                # the private-state initialization diagnostic inside P3.
                "high_order_effect_preserved": bool(p3_minus_p2 is not None and p3_minus_p2 > 0),
                "within_p3_increase": bool(delta["mean"] is not None and float(delta["mean"]) > 0),
            })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "模型鲁棒性实验汇总_2026-07-13.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUTPUT_ROOT / "模型鲁棒性实验汇总_2026-07-13.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "model", "scenario_name", "p1_mean", "p1_sd", "p2_mean", "p2_sd",
            "p3_entry_mean", "p3_entry_sd", "p3_terminal_mean", "p3_terminal_sd",
            "p2_minus_p1", "p3_terminal_minus_p2", "p3_delta_mean", "p3_delta_sd",
            "public_negative_share_mean", "high_order_effect_preserved", "within_p3_increase",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row["model"],
                "scenario_name": row["scenario_name"],
                "p1_mean": row["p1"]["mean"],
                "p1_sd": row["p1"]["sd"],
                "p2_mean": row["p2"]["mean"],
                "p2_sd": row["p2"]["sd"],
                "p3_entry_mean": row["p3_entry"]["mean"],
                "p3_entry_sd": row["p3_entry"]["sd"],
                "p3_terminal_mean": row["p3_terminal"]["mean"],
                "p3_terminal_sd": row["p3_terminal"]["sd"],
                "p2_minus_p1": row["p2_minus_p1"],
                "p3_terminal_minus_p2": row["p3_terminal_minus_p2"],
                "p3_delta_mean": row["p3_delta"]["mean"],
                "p3_delta_sd": row["p3_delta"]["sd"],
                "public_negative_share_mean": row["public_negative_share"]["mean"],
                "high_order_effect_preserved": row["high_order_effect_preserved"],
                "within_p3_increase": row["within_p3_increase"],
            })

    def fmt(stats: dict) -> str:
        if stats["mean"] is None:
            return "-"
        return f'{float(stats["mean"]):.3f} ± {float(stats["sd"]):.3f}'

    lines = [
        "# Main 基座模型鲁棒性实验汇报",
        "",
        "本实验仅替换基座模型，保持实验材料、300 agents、active 15、P3 动力学和指标口径不变。",
        "",
        "## P1 / P2 / P3 主结果",
        "",
        "| 模型 | 场景 | P1 NSB | P2 NSB | P3 terminal | P2-P1 | P3-P2 | 高阶互动方向保持 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f'| {row["model"]} | {row["scenario_name"]} | {fmt(row["p1"])} | {fmt(row["p2"])} | '
            f'{fmt(row["p3_terminal"])} | {row["p2_minus_p1"]:.3f} | '
            f'{row["p3_terminal_minus_p2"]:.3f} | {"是" if row["high_order_effect_preserved"] else "否"} |'
        )
    all_high_order = all(row["high_order_effect_preserved"] for row in rows)
    correction_cases = [
        f'{row["model"]} / {row["scenario_name"]}'
        for row in rows
        if not row["within_p3_increase"]
    ]
    lines.extend([
        "",
        "## P3 内部过程诊断",
        "",
        "| 模型 | 场景 | P3 entry | P3 terminal | terminal-entry | 公开负向占比 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f'| {row["model"]} | {row["scenario_name"]} | {fmt(row["p3_entry"])} | '
            f'{fmt(row["p3_terminal"])} | {fmt(row["p3_delta"])} | {fmt(row["public_negative_share"])} |'
        )
    lines.extend([
        "",
        "## 判定口径",
        "",
        "P1、P2 与 P3 是平行实验阶段，因此核心 H3 鲁棒性检验使用 `P3 terminal - P2`。"
        "`P3 terminal - P3 entry` 只用于解释论坛内部从初始私有状态到终局的放大或纠偏，不单独决定 P3 实验是否支持 H3。",
        "",
        ("按这一口径，所有已完成的基座模型和场景均满足 `P3 terminal > P2`。"
         if all_high_order else
         "按这一口径，部分基座模型或场景未满足 `P3 terminal > P2`，需要按表中结果分开解释。"),
        (f'P3 内部出现 `terminal < entry` 的条件为：{"; ".join(correction_cases)}。'
         "这些条件应解释为论坛内部发生部分纠偏；是否支持 H3 仍以 `P3 terminal - P2` 判定。"
         if correction_cases else
         "所有已完成条件在 P3 内部均表现为 `terminal >= entry`。"),
        "",
    ])
    (OUTPUT_ROOT / "模型鲁棒性实验汇报_2026-07-13.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
