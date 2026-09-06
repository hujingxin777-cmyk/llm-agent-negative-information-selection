#!/usr/bin/env python3
"""Split each existing 300-judgment static cell into 10 fixed batches of 30."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy import stats


SCENARIO_LABELS = {
    "influenza_a": "甲流",
    "mycoplasma_pneumonia": "支原体肺炎",
}

PERSONA_LABELS = {
    "main_baseline_neutral": "主实验基线",
    "baseline_open": "普通基线",
    "risk_sensitive": "风险敏感型",
    "ordinary_user": "普通用户型",
    "authority_trusting": "权威信任型",
}

ATMOSPHERE_LABELS = {
    "none": "无初始氛围",
    "positive_leaning": "正向氛围",
    "negative_leaning": "负向氛围",
}

STAGE_LABELS = {
    "independent": "P1",
    "low_feedback": "P2",
}

EXPERIMENT_SOURCES = {
    "Main": "main_static_ternary_r300_2026-07-22",
    "A": "a_static_ternary_r300_2026-07-22",
    "B": "b_static_ternary_r300_2026-07-22",
    "AB": "ab_static_ternary_r300_2026-07-22",
}

CELL_KEYS = [
    "experiment",
    "scenario_name",
    "stage",
    "persona_name",
    "chat_atmosphere",
    "source_type",
    "source_experiment",
]


def t_interval(mean: float, sd: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return mean, mean
    margin = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return mean - margin, mean + margin


def load_direct(outputs_root: Path, experiment: str) -> pd.DataFrame:
    source_dir = EXPERIMENT_SOURCES[experiment]
    path = outputs_root / source_dir / "results" / "static_results.csv"
    data = pd.read_csv(path)
    if len(data) == 0 or not data["success"].astype(bool).all():
        raise RuntimeError(f"Static source is empty or contains failures: {path}")
    data = data.copy()
    data["experiment"] = experiment
    data["stage"] = data["condition_name"].map(STAGE_LABELS)
    data["source_type"] = "direct"
    data["source_experiment"] = experiment
    return data


def build_analysis_data(outputs_root: Path) -> pd.DataFrame:
    main = load_direct(outputs_root, "Main")
    a_data = load_direct(outputs_root, "A")
    b_direct = load_direct(outputs_root, "B")
    ab_direct = load_direct(outputs_root, "AB")

    # B uses the same baseline population and reuses Main P1/P2-none cells.
    b_reuse = main.copy()
    b_reuse["experiment"] = "B"
    b_reuse["persona_name"] = "baseline_open"
    b_reuse["source_type"] = "reused_baseline"
    b_reuse["source_experiment"] = "Main"

    # AB reuses the same-persona P1/P2-none cells from A.
    ab_reuse = a_data.copy()
    ab_reuse["experiment"] = "AB"
    ab_reuse["source_type"] = "reused_baseline"
    ab_reuse["source_experiment"] = "A"

    combined = pd.concat(
        [main, a_data, b_reuse, b_direct, ab_reuse, ab_direct],
        ignore_index=True,
    )
    combined["repeat_index"] = pd.to_numeric(combined["repeat_index"], errors="raise").astype(int)
    combined["nsb"] = pd.to_numeric(combined["nsb"], errors="raise").astype(float)
    combined["batch_id"] = ((combined["repeat_index"] - 1) // 30 + 1).astype(int)
    if not combined["batch_id"].between(1, 10).all():
        raise RuntimeError("Expected repeat_index in 1..300")
    return combined


def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batch_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for keys, subset in data.groupby(CELL_KEYS, sort=True, dropna=False):
        identity = dict(zip(CELL_KEYS, keys, strict=True))
        if len(subset) != 300:
            raise RuntimeError(f"Expected 300 rows, found {len(subset)} for {identity}")
        if subset["repeat_index"].nunique() != 300:
            raise RuntimeError(f"Repeat indices are not unique for {identity}")

        grouped = subset.groupby("batch_id", sort=True)["nsb"]
        batch_sizes = grouped.size()
        if len(batch_sizes) != 10 or not (batch_sizes == 30).all():
            raise RuntimeError(f"Expected 10 batches x 30 rows for {identity}")
        batch_means = grouped.mean()

        for batch_id, batch_mean in batch_means.items():
            batch = subset[subset["batch_id"] == batch_id]
            batch_rows.append(
                {
                    **identity,
                    "batch_id": int(batch_id),
                    "repeat_start": int(batch["repeat_index"].min()),
                    "repeat_end": int(batch["repeat_index"].max()),
                    "n": int(len(batch)),
                    "mean_nsb": float(batch_mean),
                    "sd_within_batch": float(batch["nsb"].std(ddof=1)),
                    "neutral_share": float((batch["final_choice"] == "neutral").mean()),
                }
            )

        values = subset["nsb"]
        individual_mean = float(values.mean())
        individual_sd = float(values.std(ddof=1))
        individual_ci_low, individual_ci_high = t_interval(individual_mean, individual_sd, len(values))
        batch_mean = float(batch_means.mean())
        batch_sd = float(batch_means.std(ddof=1))
        batch_ci_low, batch_ci_high = t_interval(batch_mean, batch_sd, len(batch_means))
        counts = subset["final_choice"].value_counts()

        cell_rows.append(
            {
                **identity,
                "scenario_label": SCENARIO_LABELS[identity["scenario_name"]],
                "persona_label": PERSONA_LABELS[identity["persona_name"]],
                "atmosphere_label": ATMOSPHERE_LABELS[identity["chat_atmosphere"]],
                "n_individual": int(len(values)),
                "mean_nsb": individual_mean,
                "sd_individual": individual_sd,
                "ci95_individual_low": individual_ci_low,
                "ci95_individual_high": individual_ci_high,
                "n_batches": int(len(batch_means)),
                "batch_size": 30,
                "mean_of_batch_means": batch_mean,
                "sd_between_batches": batch_sd,
                "se_between_batches": batch_sd / math.sqrt(len(batch_means)),
                "ci95_batch_low": batch_ci_low,
                "ci95_batch_high": batch_ci_high,
                "batch_mean_min": float(batch_means.min()),
                "batch_mean_max": float(batch_means.max()),
                "positive_share": float(counts.get("positive", 0) / len(values)),
                "neutral_share": float(counts.get("neutral", 0) / len(values)),
                "negative_share": float(counts.get("negative", 0) / len(values)),
            }
        )
        distribution_rows.append(
            {
                **identity,
                "positive_n": int(counts.get("positive", 0)),
                "neutral_n": int(counts.get("neutral", 0)),
                "negative_n": int(counts.get("negative", 0)),
            }
        )

    return pd.DataFrame(cell_rows), pd.DataFrame(batch_rows), pd.DataFrame(distribution_rows)


def add_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Mean"] = result["mean_nsb"].map(lambda value: f"{value:.3f}")
    result["个体SD"] = result["sd_individual"].map(lambda value: f"{value:.3f}")
    result["组间SD"] = result["sd_between_batches"].map(lambda value: f"{value:.3f}")
    result["组均值95%CI"] = result.apply(
        lambda row: f"[{row['ci95_batch_low']:.3f}, {row['ci95_batch_high']:.3f}]", axis=1
    )
    return result


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def make_report(summary: pd.DataFrame, output_dir: Path) -> None:
    report: list[str] = [
        "# 静态三元判断 300 次按 30 次分组的 SD 检查",
        "",
        "生成日期：2026-08-01",
        "",
        "## 1. 统计口径",
        "",
        "每个静态实验 cell 已有 300 次独立模型判断，按 `repeat_index` 固定分为 10 组：1-30、31-60，依此类推至 271-300。NSB 编码为正向 0、中性 0.5、负向 1。",
        "",
        "- `SD_individual`：300 个单次判断的样本标准差。",
        "- `SD_between_batches`：10 个 30 次组均值的样本标准差。",
        "- `95% CI_batch`：以 10 个组均值为分析单位，使用 t 分布（df=9）计算。",
        "- B 的 P1/none 复用 Main；AB 的 P1/none 复用 A，均明确标记，未当作新增数据。",
        "",
        "> 注意：10 组来自同一批 300 次结果的固定切分，不是 10 次独立启动的宏观重复。因此 `SD_between_batches` 可用于现有数据的分块敏感性检查，但不能完全替代真正独立的重复实验。",
        "",
        "表内统一格式：`Mean | SD_individual | SD_between_batches | 95% CI_batch`。",
    ]

    main = add_display_columns(summary[summary["experiment"] == "Main"])
    main["阶段"] = main["stage"]
    report.extend(
        [
            "",
            "## 2. Main 主实验",
            "",
            markdown_table(
                main,
                ["scenario_label", "阶段", "Mean", "个体SD", "组间SD", "组均值95%CI"],
                ["场景", "阶段", "Mean", "个体SD", "组间SD", "组均值95%CI"],
            ),
        ]
    )

    a_data = add_display_columns(summary[summary["experiment"] == "A"])
    a_data["阶段"] = a_data["stage"]
    report.extend(
        [
            "",
            "## 3. A 人格实验",
            "",
            markdown_table(
                a_data,
                ["scenario_label", "persona_label", "阶段", "Mean", "个体SD", "组间SD", "组均值95%CI"],
                ["场景", "人格", "阶段", "Mean", "个体SD", "组间SD", "组均值95%CI"],
            ),
        ]
    )

    b_data = add_display_columns(summary[summary["experiment"] == "B"])
    b_data["条件"] = b_data.apply(
        lambda row: "P1 独立" if row["stage"] == "P1" else f"P2 {row['atmosphere_label']}", axis=1
    )
    report.extend(
        [
            "",
            "## 4. B 氛围实验",
            "",
            markdown_table(
                b_data,
                ["scenario_label", "条件", "source_type", "Mean", "个体SD", "组间SD", "组均值95%CI"],
                ["场景", "条件", "数据来源", "Mean", "个体SD", "组间SD", "组均值95%CI"],
            ),
        ]
    )

    ab_data = add_display_columns(summary[summary["experiment"] == "AB"])
    ab_data["条件"] = ab_data.apply(
        lambda row: "P1 独立" if row["stage"] == "P1" else f"P2 {row['atmosphere_label']}", axis=1
    )
    report.extend(
        [
            "",
            "## 5. AB 人格与氛围交互实验",
            "",
            markdown_table(
                ab_data,
                ["scenario_label", "persona_label", "条件", "source_type", "Mean", "个体SD", "组间SD", "组均值95%CI"],
                ["场景", "人格", "条件", "数据来源", "Mean", "个体SD", "组间SD", "组均值95%CI"],
            ),
        ]
    )

    largest = (
        summary[summary["source_type"] == "direct"]
        .sort_values("sd_between_batches", ascending=False)
        .head(10)
        .copy()
    )
    largest["cell"] = largest.apply(
        lambda row: f"{row['experiment']} / {row['scenario_label']} / {row['persona_label']} / {row['stage']} {row['atmosphere_label']}",
        axis=1,
    )
    largest["组间SD"] = largest["sd_between_batches"].map(lambda value: f"{value:.3f}")
    largest["组均值范围"] = largest.apply(
        lambda row: f"{row['batch_mean_min']:.3f}-{row['batch_mean_max']:.3f}", axis=1
    )
    report.extend(
        [
            "",
            "## 6. 波动最大的 10 个 cell",
            "",
            markdown_table(largest, ["cell", "组间SD", "组均值范围"], ["Cell", "SD_between_batches", "10组均值范围"]),
            "",
            "## 7. 判断",
            "",
            "该分块结果可以补充展示现有 300 次判断内部的稳定程度，并解决只给均值、完全没有波动信息的问题。论文中建议同时给出个体层离散度与分块组均值的 95% CI，但应明确写成“将 300 次判断固定分为 10 个等规模批次的敏感性统计”，不要将其表述成 10 次完全独立实验。若审稿人要求运行层面的可重复性，仍需真正独立启动多个宏观重复。",
            "",
            "原始 300 次总体均值与 10 个组均值的平均严格一致；本分析没有修改任何实验结果。",
        ]
    )

    (output_dir / "静态三元300次_每30次分组_SD检查汇报_2026-08-01.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = build_analysis_data(args.outputs_root)
    summary, batch_means, distributions = summarize(data)
    summary.to_csv(args.output_dir / "static_cell_sd_summary.csv", index=False, encoding="utf-8-sig")
    batch_means.to_csv(args.output_dir / "static_batch_means_10x30.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(args.output_dir / "static_choice_counts.csv", index=False, encoding="utf-8-sig")
    make_report(summary, args.output_dir)
    print(f"Wrote {len(summary)} composite cell summaries and {len(batch_means)} batch rows to {args.output_dir}")


if __name__ == "__main__":
    main()
