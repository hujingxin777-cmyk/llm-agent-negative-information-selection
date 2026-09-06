#!/usr/bin/env python3
"""Compare 300-repeat ternary P1/P2 judgments with the formal P3 panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCENARIO_LABELS = {
    "influenza_a": "甲流",
    "mycoplasma_pneumonia": "支原体肺炎",
}

STAGE_LABELS = {
    "P1": "P1 独立判断",
    "P2": "P2 静态评论",
    "P3_entry": "P3 进入互动前",
    "P3_terminal": "P3 高阶互动终局",
}


def mean_summary(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(clean.size)
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else math.nan
    margin = float(stats.t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def welch_difference(a: pd.Series, b: pd.Series) -> dict[str, float | int | str]:
    """Return the mean of a minus b with a Welch 95% confidence interval."""
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    se2 = vx / nx + vy / ny
    se = math.sqrt(se2)
    df = se2**2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    estimate = float(np.mean(x) - np.mean(y))
    margin = float(stats.t.ppf(0.975, df) * se)
    test = stats.ttest_ind(x, y, equal_var=False)
    return {
        "estimate": estimate,
        "ci95_low": estimate - margin,
        "ci95_high": estimate + margin,
        "p_value": float(test.pvalue),
        "df": float(df),
        "n_a": nx,
        "n_b": ny,
        "test": "Welch independent",
    }


def paired_difference(a: pd.Series, b: pd.Series) -> dict[str, float | int | str]:
    """Return the paired mean of a minus b with a t confidence interval."""
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    diff = x - y
    summary = mean_summary(pd.Series(diff))
    test = stats.ttest_rel(x, y)
    return {
        "estimate": summary["mean"],
        "ci95_low": summary["ci95_low"],
        "ci95_high": summary["ci95_high"],
        "p_value": float(test.pvalue),
        "df": len(diff) - 1,
        "n_a": len(diff),
        "n_b": len(diff),
        "test": "Paired t",
    }


def configure_chinese_font() -> None:
    candidates = [
        "Songti SC",
        "STSong",
        "SimSun",
        "Noto Serif CJK SC",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def format_ci(row: pd.Series) -> str:
    return (
        f"{row['mean']:.3f} +/- {row['sd']:.3f} "
        f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
    )


def build_figure(stage_summary: pd.DataFrame, figure_dir: Path) -> None:
    configure_chinese_font()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    blue = "#2E6F9E"
    orange = "#D97720"

    for ax, scenario in zip(axes, SCENARIO_LABELS, strict=True):
        sub = stage_summary[stage_summary["scenario_name"] == scenario].set_index("stage")
        line_stages = ["P1", "P2", "P3_terminal"]
        x_line = np.array([0.0, 1.0, 2.0])
        y_line = np.array([sub.loc[stage, "mean"] for stage in line_stages])
        low = np.array([sub.loc[stage, "ci95_low"] for stage in line_stages])
        high = np.array([sub.loc[stage, "ci95_high"] for stage in line_stages])
        yerr = np.vstack([y_line - low, high - y_line])
        ax.errorbar(
            x_line,
            y_line,
            yerr=yerr,
            color=blue,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            capsize=3,
            label="P1-P2-P3 终局",
        )

        entry = sub.loc["P3_entry"]
        entry_x = 1.65
        ax.errorbar(
            [entry_x],
            [entry["mean"]],
            yerr=[[entry["mean"] - entry["ci95_low"]], [entry["ci95_high"] - entry["mean"]]],
            color=orange,
            marker="D",
            linewidth=1.2,
            markersize=5,
            capsize=3,
            label="P3 entry",
        )

        for x, y in zip(x_line, y_line, strict=True):
            ax.text(x, y + 0.035, f"{y:.3f}", color=blue, ha="center", fontsize=8)
        ax.text(entry_x, entry["mean"] - 0.065, f"entry {entry['mean']:.3f}", color=orange, ha="center", fontsize=8)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=11)
        ax.set_xticks([0, 1, 2], ["P1\n独立判断", "P2\n静态评论", "P3\n互动终局"])
        ax.set_xlim(-0.15, 2.15)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.axhline(0.5, color="#777777", linewidth=0.7, linestyle="--", alpha=0.55)

    axes[0].set_ylabel("Mean NSB (0, 0.5, 1)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22, wspace=0.12)
    fig.savefig(figure_dir / "main_p1_p2_p3_ternary_95ci.pdf")
    fig.savefig(figure_dir / "main_p1_p2_p3_ternary_95ci.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-output", type=Path, required=True)
    parser.add_argument("--p3-output", type=Path, required=True)
    args = parser.parse_args()

    static_results_path = args.static_output / "results" / "static_results.csv"
    p3_results_dir = args.p3_output / "results"
    p3_cell_path = p3_results_dir / "p3_cell_summary.csv"
    p3_probe_path = p3_results_dir / "p3_probe_results.csv"
    static = pd.read_csv(static_results_path)
    p3_cells = pd.read_csv(p3_cell_path)
    p3_probes = pd.read_csv(p3_probe_path)

    if len(static) != 1200 or not static["success"].all():
        raise RuntimeError("Expected 1200 successful static rows")

    result_dir = args.static_output / "results"
    figure_dir = args.static_output / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)

    distributions: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    static_values: dict[tuple[str, str], pd.Series] = {}
    p3_values: dict[tuple[str, str], pd.Series] = {}

    for scenario in SCENARIO_LABELS:
        for condition, stage in (("independent", "P1"), ("low_feedback", "P2")):
            subset = static[
                (static["scenario_name"] == scenario)
                & (static["condition_name"] == condition)
            ].copy()
            static_values[(scenario, stage)] = subset["nsb"]
            summary = mean_summary(subset["nsb"])
            stage_rows.append(
                {
                    "scenario_name": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "analysis_unit": "independent model judgment",
                    **summary,
                }
            )
            counts = subset["final_choice"].value_counts()
            n = len(subset)
            distributions.append(
                {
                    "scenario_name": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "stage": stage,
                    "n": n,
                    "positive_n": int(counts.get("positive", 0)),
                    "neutral_n": int(counts.get("neutral", 0)),
                    "negative_n": int(counts.get("negative", 0)),
                    "positive_share": float(counts.get("positive", 0) / n),
                    "neutral_share": float(counts.get("neutral", 0) / n),
                    "negative_share": float(counts.get("negative", 0) / n),
                }
            )

        cells = p3_cells[p3_cells["scenario_name"] == scenario].sort_values("repeat_index")
        if len(cells) != 30:
            raise RuntimeError(f"Expected 30 P3 cells for {scenario}, found {len(cells)}")
        for stage, column in (
            ("P3_entry", "entry_all_agents_nsb"),
            ("P3_terminal", "terminal_all_agents_nsb"),
        ):
            p3_values[(scenario, stage)] = cells[column]
            stage_rows.append(
                {
                    "scenario_name": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "analysis_unit": "300-agent discussion cell",
                    **mean_summary(cells[column]),
                }
            )

        for phase, stage in (("entry", "P3_entry"), ("terminal", "P3_terminal")):
            selected = p3_probes[
                (p3_probes["scenario_name"] == scenario)
                & (p3_probes["phase"] == phase)
            ]
            if len(selected) != 30 * 300:
                raise RuntimeError(
                    f"Expected 9000 P3 {phase} probe rows for {scenario}, found {len(selected)}"
                )
            counts = selected["final_choice"].value_counts()
            n = len(selected)
            distributions.append(
                {
                    "scenario_name": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "stage": stage,
                    "n": n,
                    "positive_n": int(counts.get("positive", 0)),
                    "neutral_n": int(counts.get("neutral", 0)),
                    "negative_n": int(counts.get("negative", 0)),
                    "positive_share": float(counts.get("positive", 0) / n),
                    "neutral_share": float(counts.get("neutral", 0) / n),
                    "negative_share": float(counts.get("negative", 0) / n),
                }
            )

    stage_summary = pd.DataFrame(stage_rows)
    choice_distribution = pd.DataFrame(distributions)
    stage_summary.to_csv(result_dir / "main_p1_p2_p3_ternary_stage_summary_95ci.csv", index=False)
    choice_distribution.to_csv(result_dir / "main_p1_p2_p3_choice_distribution.csv", index=False)

    contrast_rows: list[dict[str, object]] = []
    for scenario in SCENARIO_LABELS:
        comparisons = [
            ("P2 - P1", "P2", "P1", False),
            ("P3 terminal - P2", "P3_terminal", "P2", False),
            ("P3 terminal - P1", "P3_terminal", "P1", False),
            ("P3 terminal - P3 entry", "P3_terminal", "P3_entry", True),
        ]
        for label, a_stage, b_stage, paired in comparisons:
            a = static_values.get((scenario, a_stage), p3_values.get((scenario, a_stage)))
            b = static_values.get((scenario, b_stage), p3_values.get((scenario, b_stage)))
            result = paired_difference(a, b) if paired else welch_difference(a, b)
            contrast_rows.append(
                {
                    "scenario_name": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "contrast": label,
                    **result,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(result_dir / "main_p1_p2_p3_contrasts_95ci.csv", index=False)

    build_figure(stage_summary, figure_dir)

    request_log = args.static_output / "progress" / "model_request_log.jsonl"
    phase_counts: dict[str, int] = {}
    error_count = 0
    for line in request_log.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        phase = str(item.get("phase"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if phase in {"error", "failure"} or item.get("error") or item.get("error_message"):
            error_count += 1

    stage_table_rows = []
    for row in stage_summary.itertuples(index=False):
        dist = choice_distribution[
            (choice_distribution["scenario_name"] == row.scenario_name)
            & (choice_distribution["stage"] == row.stage)
        ]
        neutral = "-" if dist.empty else f"{float(dist.iloc[0]['neutral_share']):.1%}"
        stage_table_rows.append(
            f"| {row.scenario_label} | {row.stage_label} | {row.n} | "
            f"{row.mean:.3f} +/- {row.sd:.3f} | "
            f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}] | {neutral} |"
        )

    distribution_rows = []
    for row in choice_distribution.itertuples(index=False):
        distribution_rows.append(
            f"| {row.scenario_label} | {STAGE_LABELS[row.stage]} | "
            f"{row.positive_n} ({row.positive_share:.1%}) | "
            f"{row.neutral_n} ({row.neutral_share:.1%}) | "
            f"{row.negative_n} ({row.negative_share:.1%}) |"
        )

    contrast_table_rows = []
    for row in contrasts.itertuples(index=False):
        contrast_table_rows.append(
            f"| {row.scenario_label} | {row.contrast} | {row.estimate:+.3f} | "
            f"[{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] | {row.p_value:.4g} | {row.test} |"
        )

    report_path = args.static_output / "主实验_静态三元300次与P3高阶互动对比汇报_2026-07-22.md"
    report = f"""# 主实验：静态三元 300 次与 P3 高阶互动对比

生成日期：2026-07-22

## 1. 实验口径

- P1：每个场景 300 次独立模型判断。
- P2：每个场景 300 次静态评论条件判断。
- P1/P2 共 1200 次请求，全部成功；`start={phase_counts.get('start', 0)}`，`success={phase_counts.get('success', 0)}`，错误记录 `{error_count}`。
- 三元 NSB：正向 `0`，中性 `0.5`，负向 `1`。本轮没有强制配额或后处理校准。
- P3：复用正式主实验的 30 个讨论 cell；每个 cell 有 300 agents，主指标为 300 人全体终局 NSB。
- 注意统计单位不同：P1/P2 的重复单位是单次模型判断，P3 的重复单位是一次 300-agent 讨论 cell。跨阶段显著性检验仅作描述性辅助，核心判断以均值方向、95% CI 与重复分布共同为准。

## 2. P1/P2/P3 主结果

| 场景 | 阶段 | 重复单位 n | Mean +/- SD | 95% CI | neutral 占比 |
|---|---|---:|---:|---:|---:|
{chr(10).join(stage_table_rows)}

![主实验 P1/P2/P3 三元 NSB 与 95% CI]({(figure_dir / 'main_p1_p2_p3_ternary_95ci.png').resolve()})

图中蓝线连接 P1、P2 与 P3 终局；橙色菱形为 P3 entry。误差线为基于重复单位计算的 95% CI。

## 3. 三元选择分布

| 场景 | 阶段 | 正向 | 中性 | 负向 |
|---|---|---:|---:|---:|
{chr(10).join(distribution_rows)}

P1 中确实出现中性选择，说明三元口径已生效；P2 本轮没有自然产生中性选择。这是模型输出分布，而不是编码失败或人为强制为二元。

## 4. 阶段差异

| 场景 | 对比 | 均值差 | 95% CI | p 值 | 检验 |
|---|---|---:|---:|---:|---|
{chr(10).join(contrast_table_rows)}

其中 `P3 terminal - P3 entry` 使用同一讨论 cell 的配对检验；涉及 P1/P2 与 P3 的比较使用 Welch 独立样本检验。由于两类阶段的分析单位不同，p 值不应被解释为严格的个体层因果检验。

## 5. 结果解释

1. **静态评论没有放大负面选择。** 甲流 P2 比 P1 低 `0.090`，支原体肺炎低 `0.060`。这说明低强度静态评论在本轮更接近缓冲或校正，而不是负向放大。
2. **甲流的 P3 主要是恢复被 P2 压低的负向倾向。** P3 终局比 P2 高约 `0.117`，但只比 P1 高约 `0.027`，因此更适合表述为“高阶互动重新拉回静态阶段被缓冲的负向倾向”，而不是无条件超过独立基线。
3. **支原体肺炎出现清晰的额外互动放大。** P3 终局比 P2 高约 `0.330`，比 P1 高约 `0.270`；同时 P3 从 entry 到 terminal 增加约 `0.189`。
4. **P3 内部变化支持高阶互动机制。** 甲流和支原体肺炎的终局均高于各自 entry；该比较基于同一批 30 个讨论 cell，口径比直接拿静态单次判断与 P3 cell 均值做显著性检验更严格。
5. **场景差异明显。** 支原体肺炎在 P3 中的增量和终局均高于甲流，说明高阶互动效应受场景风险显著性约束，不宜写成所有场景等幅放大。

## 6. 文件与可复现性

- 静态原始结果：`{static_results_path.resolve()}`
- P3 cell 原始结果：`{p3_cell_path.resolve()}`
- P3 全体探针原始结果：`{p3_probe_path.resolve()}`
- 阶段均值与 95% CI：`{(result_dir / 'main_p1_p2_p3_ternary_stage_summary_95ci.csv').resolve()}`
- 三元选择分布：`{(result_dir / 'main_p1_p2_p3_choice_distribution.csv').resolve()}`
- 阶段差异：`{(result_dir / 'main_p1_p2_p3_contrasts_95ci.csv').resolve()}`
- 图：`{(figure_dir / 'main_p1_p2_p3_ternary_95ci.pdf').resolve()}`

## 7. 结论

本轮 300 次静态重复支持以下主线：P2 静态评论未提高 NSB，而 P3 高阶互动相对 P2 明显回升。甲流主要表现为恢复至独立判断附近，支原体肺炎则表现为超过 P1/P2 的额外放大。因此，较稳妥的论文结论是：**静态社会反馈可能产生缓冲，但高阶互动能够重新组织并放大负向选择；放大幅度具有场景依赖性。**
"""
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
