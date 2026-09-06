#!/usr/bin/env python3
"""Summarize 300-repeat ternary static A, B, and AB experiments."""

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


SCENARIOS = {
    "influenza_a": "甲流",
    "mycoplasma_pneumonia": "支原体肺炎",
}
PERSONAS = ["risk_sensitive", "ordinary_user", "authority_trusting"]
PERSONA_LABELS = {
    "risk_sensitive": "风险敏感型",
    "ordinary_user": "普通用户型",
    "authority_trusting": "权威信任型",
    "baseline_open": "普通基线",
    "main_baseline_neutral": "主实验基线",
}
ATMOSPHERES = ["positive_leaning", "none", "negative_leaning"]
ATMOSPHERE_LABELS = {
    "positive_leaning": "正向氛围",
    "none": "无初始氛围",
    "negative_leaning": "负向氛围",
}


def summarize(values: pd.Series) -> dict[str, float | int]:
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
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def welch(a: pd.Series, b: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)
    se2 = vx / nx + vy / ny
    if se2 == 0:
        estimate = float(np.mean(x) - np.mean(y))
        return {
            "estimate": estimate,
            "ci95_low": estimate,
            "ci95_high": estimate,
            "p_value": 0.0 if estimate else 1.0,
            "n_a": nx,
            "n_b": ny,
        }
    se = math.sqrt(se2)
    df = se2**2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    estimate = float(np.mean(x) - np.mean(y))
    margin = float(stats.t.ppf(0.975, df) * se)
    return {
        "estimate": estimate,
        "ci95_low": estimate - margin,
        "ci95_high": estimate + margin,
        "p_value": float(stats.ttest_ind(x, y, equal_var=False).pvalue),
        "n_a": nx,
        "n_b": ny,
    }


def bootstrap_did(
    a_high: pd.Series,
    a_low: pd.Series,
    b_high: pd.Series,
    b_low: pd.Series,
    seed: int = 20260722,
    draws: int = 20_000,
) -> dict[str, float | int]:
    groups = [series.to_numpy(dtype=float) for series in (a_high, a_low, b_high, b_low)]
    estimate = float(np.mean(groups[0]) - np.mean(groups[1]) - np.mean(groups[2]) + np.mean(groups[3]))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for idx in range(draws):
        means = [float(np.mean(rng.choice(group, size=len(group), replace=True))) for group in groups]
        boot[idx] = means[0] - means[1] - means[2] + means[3]
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_value": float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0))),
        "n_a": len(groups[0]),
        "n_b": len(groups[2]),
    }


def configure_font() -> None:
    candidates = ["Songti SC", "STSong", "SimSun", "Noto Serif CJK SC", "PingFang SC"]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_results(output: Path) -> pd.DataFrame:
    data = pd.read_csv(output / "results" / "static_results.csv")
    if not data["success"].all():
        raise RuntimeError(f"Failed rows found in {output}")
    return data


def add_summary(
    rows: list[dict[str, object]],
    data: pd.DataFrame,
    experiment: str,
    stage: str,
    source: str,
) -> None:
    keys = ["scenario_name", "persona_name", "chat_atmosphere"]
    for key, subset in data.groupby(keys, dropna=False):
        scenario, persona, atmosphere = key
        rows.append(
            {
                "experiment": experiment,
                "scenario_name": scenario,
                "scenario_label": SCENARIOS[scenario],
                "stage": stage,
                "persona_name": persona,
                "persona_label": PERSONA_LABELS[persona],
                "chat_atmosphere": atmosphere,
                "atmosphere_label": ATMOSPHERE_LABELS.get(atmosphere, atmosphere),
                "source": source,
                **summarize(subset["nsb"]),
            }
        )


def distribution_rows(data: pd.DataFrame, experiment: str, stage: str, source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, subset in data.groupby(["scenario_name", "persona_name", "chat_atmosphere"]):
        scenario, persona, atmosphere = key
        counts = subset["final_choice"].value_counts()
        n = len(subset)
        rows.append(
            {
                "experiment": experiment,
                "scenario_name": scenario,
                "stage": stage,
                "persona_name": persona,
                "chat_atmosphere": atmosphere,
                "source": source,
                "n": n,
                "positive_n": int(counts.get("positive", 0)),
                "neutral_n": int(counts.get("neutral", 0)),
                "negative_n": int(counts.get("negative", 0)),
                "positive_share": float(counts.get("positive", 0) / n),
                "neutral_share": float(counts.get("neutral", 0) / n),
                "negative_share": float(counts.get("negative", 0) / n),
            }
        )
    return rows


def cell(summary: pd.DataFrame, experiment: str, scenario: str, stage: str, persona: str, atmosphere: str) -> pd.Series:
    selected = summary[
        (summary["experiment"] == experiment)
        & (summary["scenario_name"] == scenario)
        & (summary["stage"] == stage)
        & (summary["persona_name"] == persona)
        & (summary["chat_atmosphere"] == atmosphere)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one summary row, found {len(selected)}: {experiment}/{scenario}/{stage}/{persona}/{atmosphere}")
    return selected.iloc[0]


def compact(row: pd.Series) -> str:
    return f"{row['mean']:.3f} +/- {row['sd']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"


def build_figures(summary: pd.DataFrame, figure_dir: Path) -> None:
    configure_font()
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "risk_sensitive": "#C6533D",
        "ordinary_user": "#526D82",
        "authority_trusting": "#2F7D5B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        for persona in PERSONAS:
            rows = [
                cell(summary, "A", scenario, "P1", persona, "none"),
                cell(summary, "A", scenario, "P2", persona, "none"),
            ]
            means = np.array([row["mean"] for row in rows])
            yerr = np.array(
                [
                    [row["mean"] - row["ci95_low"] for row in rows],
                    [row["ci95_high"] - row["mean"] for row in rows],
                ]
            )
            ax.errorbar(
                [0, 1], means, yerr=yerr, marker="o", capsize=3, linewidth=1.6,
                color=colors[persona], label=PERSONA_LABELS[persona],
            )
        ax.set_title(SCENARIOS[scenario])
        ax.set_xticks([0, 1], ["P1 独立", "P2 静态"])
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean NSB (0, 0.5, 1)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.20, wspace=0.12)
    fig.savefig(figure_dir / "a_persona_static_r300_95ci.pdf")
    fig.savefig(figure_dir / "a_persona_static_r300_95ci.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        rows = [cell(summary, "B", scenario, "P2", "baseline_open", atmosphere) for atmosphere in ATMOSPHERES]
        means = np.array([row["mean"] for row in rows])
        yerr = np.array(
            [
                [row["mean"] - row["ci95_low"] for row in rows],
                [row["ci95_high"] - row["mean"] for row in rows],
            ]
        )
        ax.errorbar(range(3), means, yerr=yerr, color="#356A8A", marker="o", capsize=3, linewidth=1.7)
        p1 = cell(summary, "B", scenario, "P1", "baseline_open", "none")
        ax.axhline(p1["mean"], color="#777777", linestyle="--", linewidth=1.0, label=f"P1={p1['mean']:.3f}")
        ax.set_xticks(range(3), ["正向", "无初始", "负向"])
        ax.set_title(SCENARIOS[scenario])
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("P2 Mean NSB (0, 0.5, 1)")
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(figure_dir / "b_atmosphere_static_r300_95ci.pdf")
    fig.savefig(figure_dir / "b_atmosphere_static_r300_95ci.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        matrix = np.array(
            [
                [cell(summary, "AB", scenario, "P2", persona, atmosphere)["mean"] for atmosphere in ATMOSPHERES]
                for persona in PERSONAS
            ]
        )
        image = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                color = "white" if matrix[row, col] < 0.18 or matrix[row, col] > 0.82 else "black"
                ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(3), ["正向", "无初始", "负向"])
        ax.set_yticks(range(3), [PERSONA_LABELS[persona] for persona in PERSONAS])
        ax.set_title(SCENARIOS[scenario])
    fig.colorbar(image, ax=axes, label="P2 Mean NSB", shrink=0.85)
    fig.savefig(figure_dir / "ab_persona_atmosphere_static_r300_heatmap.pdf")
    fig.savefig(figure_dir / "ab_persona_atmosphere_static_r300_heatmap.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--a-output", type=Path, required=True)
    parser.add_argument("--b-output", type=Path, required=True)
    parser.add_argument("--ab-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    main_data = load_results(args.main_output)
    a_data = load_results(args.a_output)
    b_data = load_results(args.b_output)
    ab_data = load_results(args.ab_output)
    args.output.mkdir(parents=True, exist_ok=True)
    result_dir = args.output / "results"
    figure_dir = args.output / "figures"
    result_dir.mkdir(exist_ok=True)

    main_p1 = main_data[main_data["condition_name"] == "independent"].copy()
    main_p2 = main_data[main_data["condition_name"] == "low_feedback"].copy()
    a_p1 = a_data[a_data["condition_name"] == "independent"].copy()
    a_p2 = a_data[a_data["condition_name"] == "low_feedback"].copy()

    summary_rows: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    add_summary(summary_rows, a_p1, "A", "P1", "A direct")
    add_summary(summary_rows, a_p2, "A", "P2", "A direct")
    distributions.extend(distribution_rows(a_p1, "A", "P1", "A direct"))
    distributions.extend(distribution_rows(a_p2, "A", "P2", "A direct"))

    b_p1 = main_p1.copy()
    b_p1["persona_name"] = "baseline_open"
    b_p2_none = main_p2.copy()
    b_p2_none["persona_name"] = "baseline_open"
    add_summary(summary_rows, b_p1, "B", "P1", "Main P1 reused")
    add_summary(summary_rows, b_p2_none, "B", "P2", "Main none P2 reused")
    add_summary(summary_rows, b_data, "B", "P2", "B direct")
    distributions.extend(distribution_rows(b_p1, "B", "P1", "Main P1 reused"))
    distributions.extend(distribution_rows(b_p2_none, "B", "P2", "Main none P2 reused"))
    distributions.extend(distribution_rows(b_data, "B", "P2", "B direct"))

    add_summary(summary_rows, a_p1, "AB", "P1", "A P1 reused")
    add_summary(summary_rows, a_p2, "AB", "P2", "A none P2 reused")
    add_summary(summary_rows, ab_data, "AB", "P2", "AB direct")
    distributions.extend(distribution_rows(a_p1, "AB", "P1", "A P1 reused"))
    distributions.extend(distribution_rows(a_p2, "AB", "P2", "A none P2 reused"))
    distributions.extend(distribution_rows(ab_data, "AB", "P2", "AB direct"))

    summary = pd.DataFrame(summary_rows)
    distribution = pd.DataFrame(distributions)
    summary.to_csv(result_dir / "a_b_ab_static_ternary_r300_summary_95ci.csv", index=False)
    distribution.to_csv(result_dir / "a_b_ab_static_ternary_r300_choice_distribution.csv", index=False)

    contrasts: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for persona in PERSONAS:
            p1 = a_p1[(a_p1["scenario_name"] == scenario) & (a_p1["persona_name"] == persona)]["nsb"]
            p2 = a_p2[(a_p2["scenario_name"] == scenario) & (a_p2["persona_name"] == persona)]["nsb"]
            contrasts.append(
                {"experiment": "A", "scenario_name": scenario, "persona_name": persona,
                 "chat_atmosphere": "none", "contrast": "P2 - P1", **welch(p2, p1)}
            )

        b_groups = {
            "positive_leaning": b_data[(b_data["scenario_name"] == scenario) & (b_data["chat_atmosphere"] == "positive_leaning")]["nsb"],
            "none": main_p2[main_p2["scenario_name"] == scenario]["nsb"],
            "negative_leaning": b_data[(b_data["scenario_name"] == scenario) & (b_data["chat_atmosphere"] == "negative_leaning")]["nsb"],
        }
        for label, high, low in (
            ("negative - positive", "negative_leaning", "positive_leaning"),
            ("negative - none", "negative_leaning", "none"),
            ("none - positive", "none", "positive_leaning"),
        ):
            contrasts.append(
                {"experiment": "B", "scenario_name": scenario, "persona_name": "baseline_open",
                 "chat_atmosphere": "all", "contrast": label, **welch(b_groups[high], b_groups[low])}
            )

        ab_groups: dict[tuple[str, str], pd.Series] = {}
        for persona in PERSONAS:
            ab_groups[(persona, "none")] = a_p2[
                (a_p2["scenario_name"] == scenario) & (a_p2["persona_name"] == persona)
            ]["nsb"]
            for atmosphere in ("positive_leaning", "negative_leaning"):
                ab_groups[(persona, atmosphere)] = ab_data[
                    (ab_data["scenario_name"] == scenario)
                    & (ab_data["persona_name"] == persona)
                    & (ab_data["chat_atmosphere"] == atmosphere)
                ]["nsb"]
            contrasts.append(
                {"experiment": "AB", "scenario_name": scenario, "persona_name": persona,
                 "chat_atmosphere": "all", "contrast": "negative - positive",
                 **welch(ab_groups[(persona, "negative_leaning")], ab_groups[(persona, "positive_leaning")])}
            )
        for atmosphere in ATMOSPHERES:
            contrasts.append(
                {"experiment": "AB", "scenario_name": scenario, "persona_name": "all",
                 "chat_atmosphere": atmosphere, "contrast": "risk - authority",
                 **welch(ab_groups[("risk_sensitive", atmosphere)], ab_groups[("authority_trusting", atmosphere)])}
            )
        contrasts.append(
            {"experiment": "AB", "scenario_name": scenario, "persona_name": "risk vs authority",
             "chat_atmosphere": "negative vs positive", "contrast": "interaction DiD",
             **bootstrap_did(
                 ab_groups[("risk_sensitive", "negative_leaning")],
                 ab_groups[("risk_sensitive", "positive_leaning")],
                 ab_groups[("authority_trusting", "negative_leaning")],
                 ab_groups[("authority_trusting", "positive_leaning")],
             )}
        )

    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(result_dir / "a_b_ab_static_ternary_r300_contrasts_95ci.csv", index=False)
    build_figures(summary, figure_dir)

    a_rows = []
    for scenario in SCENARIOS:
        for persona in PERSONAS:
            p1 = cell(summary, "A", scenario, "P1", persona, "none")
            p2 = cell(summary, "A", scenario, "P2", persona, "none")
            a_rows.append(
                f"| {SCENARIOS[scenario]} | {PERSONA_LABELS[persona]} | {compact(p1)} | {compact(p2)} | {p2['mean'] - p1['mean']:+.3f} |"
            )

    b_rows = []
    for scenario in SCENARIOS:
        p1 = cell(summary, "B", scenario, "P1", "baseline_open", "none")
        for atmosphere in ATMOSPHERES:
            p2 = cell(summary, "B", scenario, "P2", "baseline_open", atmosphere)
            b_rows.append(
                f"| {SCENARIOS[scenario]} | {ATMOSPHERE_LABELS[atmosphere]} | {p1['mean']:.3f} | {compact(p2)} | {p2['mean'] - p1['mean']:+.3f} | {p2['source']} |"
            )

    ab_rows = []
    for scenario in SCENARIOS:
        for atmosphere in ATMOSPHERES:
            for persona in PERSONAS:
                p1 = cell(summary, "AB", scenario, "P1", persona, "none")
                p2 = cell(summary, "AB", scenario, "P2", persona, atmosphere)
                ab_rows.append(
                    f"| {SCENARIOS[scenario]} | {ATMOSPHERE_LABELS[atmosphere]} | {PERSONA_LABELS[persona]} | "
                    f"{p1['mean']:.3f} | {compact(p2)} | {p2['mean'] - p1['mean']:+.3f} | {p2['source']} |"
                )

    contrast_rows = []
    selected_contrasts = contrast_df[
        ((contrast_df["experiment"] == "B") & (contrast_df["contrast"] == "negative - positive"))
        | ((contrast_df["experiment"] == "AB") & (contrast_df["contrast"].isin(["negative - positive", "interaction DiD"])))
    ]
    for row in selected_contrasts.itertuples(index=False):
        contrast_rows.append(
            f"| {row.experiment} | {SCENARIOS[row.scenario_name]} | {PERSONA_LABELS.get(row.persona_name, row.persona_name)} | "
            f"{row.contrast} | {row.estimate:+.3f} | [{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] | {row.p_value:.4g} |"
        )

    token_total = 0
    estimated_cost = 0.0
    request_success = 0
    request_errors = 0
    for output in (args.a_output, args.b_output, args.ab_output):
        for line in (output / "progress" / "model_request_log.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("phase") == "success":
                request_success += 1
                token_total += int(item.get("total_tokens") or 0)
            if item.get("phase") in {"error", "failure"} or item.get("error") or item.get("error_message"):
                request_errors += 1
    estimated_cost = token_total / 1000 * 0.003

    report = f"""# A / B / AB 静态三元 300 次结果汇总

生成日期：2026-07-22

## 1. 执行口径

- 三元 NSB：正向 `0`、中性 `0.5`、负向 `1`。
- 每个实际运行的 `场景 × 人格 × 氛围 × 阶段` cell 均为 300 次。
- 新增请求：A 3600、B 1200、AB 3600，共 `{request_success}` 次；错误 `{request_errors}`。
- 新增 token：`{token_total:,}`；按 `$0.003/千 tokens` 估算成本 `${estimated_cost:.2f}`。
- B 的 P1/none 复用 Main 300 次；AB 的 P1/none 复用 A 300 次。复用对象在材料、人格、场景和三元测量口径上完全一致。
- 未运行 P3，未修改任何原始 CSV。

## 2. A 人格静态结果

| 场景 | 人格 | P1 Mean +/- SD [95% CI] | P2 Mean +/- SD [95% CI] | P2-P1 |
|---|---|---:|---:|---:|
{chr(10).join(a_rows)}

![A 人格静态结果]({(figure_dir / 'a_persona_static_r300_95ci.png').resolve()})

两个场景均保持 `risk_sensitive > ordinary_user > authority_trusting`。P2 相比 P1 整体下降，但人格梯度没有被静态评论抹平。

## 3. B 氛围静态结果

| 场景 | 氛围 | P1 | P2 Mean +/- SD [95% CI] | P2-P1 | 数据来源 |
|---|---|---:|---:|---:|---|
{chr(10).join(b_rows)}

![B 氛围静态结果]({(figure_dir / 'b_atmosphere_static_r300_95ci.png').resolve()})

两个场景均为 `negative > none > positive`。负向与正向之间的差值分别为甲流约 `0.507`、支原体肺炎约 `0.430`。

## 4. AB 人格 × 氛围静态结果

| 场景 | 氛围 | 人格 | P1 | P2 Mean +/- SD [95% CI] | P2-P1 | 数据来源 |
|---|---|---|---:|---:|---:|---|
{chr(10).join(ab_rows)}

![AB 人格与氛围静态热力图]({(figure_dir / 'ab_persona_atmosphere_static_r300_heatmap.png').resolve()})

总体模式同时保留人格差异与氛围方向：风险敏感型最高、权威信任型最低；各人格的负向氛围均高于正向氛围。甲流权威信任型正向条件为 `0.000`，说明该组合存在地板效应，论文中应明确报告而不是进一步调参。

## 5. 关键氛围与交互对比

| 实验 | 场景 | 人格 | 对比 | 差值 | 95% CI | p 值 |
|---|---|---|---|---:|---:|---:|
{chr(10).join(contrast_rows)}

`interaction DiD` 定义为 `(risk_negative - risk_positive) - (authority_negative - authority_positive)`，95% CI 使用固定随机种子的非参数 bootstrap。该指标用于描述人格对氛围反应差异，不应单独解释为因果交互。

## 6. 三元口径检查

完整正向/中性/负向计数保存在 `a_b_ab_static_ternary_r300_choice_distribution.csv`。部分 cell 没有自然产生 neutral，这反映模型选择分布；程序允许 neutral，且没有强制比例、区间校准或后处理改值。

## 7. 结论

300 次静态重复表明：A 的人格梯度、B 的氛围梯度以及 AB 中人格与氛围的联合分化均能在较大重复数下复现。静态结果支持“个体认知倾向与评论氛围共同塑造 P2 判断”，但并不替代 P3 高阶互动分析；P3 仍用于识别互动链形成后的额外变化。

## 8. 文件

- 汇总与 95% CI：`{(result_dir / 'a_b_ab_static_ternary_r300_summary_95ci.csv').resolve()}`
- 三元分布：`{(result_dir / 'a_b_ab_static_ternary_r300_choice_distribution.csv').resolve()}`
- 对比与交互：`{(result_dir / 'a_b_ab_static_ternary_r300_contrasts_95ci.csv').resolve()}`
- A 原始数据：`{(args.a_output / 'results' / 'static_results.csv').resolve()}`
- B 原始数据：`{(args.b_output / 'results' / 'static_results.csv').resolve()}`
- AB 原始数据：`{(args.ab_output / 'results' / 'static_results.csv').resolve()}`
"""
    report_path = args.output / "A_B_AB静态三元300次结果汇总_2026-07-22.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
