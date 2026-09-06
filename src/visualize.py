from __future__ import annotations

"""图表构建器：根据结果表生成论文主图与补充图。"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import write_csv


class FigureBuilder:
    """集中生成项目所需图表与对应数据表。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.results_dir = output_dir / "results"
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._apply_publication_style()

    def _apply_publication_style(self) -> None:
        """统一 Matplotlib 出图风格（中文字体、线宽、网格等）。"""
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Hiragino Sans GB", "Arial Unicode MS", "STHeiti", "Songti SC"],
                "axes.unicode_minus": False,
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "grid.linewidth": 0.6,
                "grid.alpha": 0.25,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.bbox": "tight",
            }
        )

    @staticmethod
    def _platform_label(name: str) -> str:
        mapping = {
            "independent": "P1 独立决策",
            "low_feedback": "P2 静态社会证明",
            "high_interaction": "P3 高阶互动",
        }
        return mapping.get(name, name)

    @staticmethod
    def _atmosphere_label(name: str) -> str:
        mapping = {
            "none": "平衡氛围",
            "positive_leaning": "正向倾斜",
            "negative_leaning": "负向倾斜",
        }
        return mapping.get(name, name)

    @staticmethod
    def _observed_atmosphere_order(values: pd.Series) -> list[str]:
        base_order = ["none", "positive_leaning", "negative_leaning"]
        observed = {str(item) for item in values.dropna().unique()}
        ordered = [item for item in base_order if item in observed]
        ordered.extend(sorted(observed - set(base_order)))
        return ordered

    @staticmethod
    def _inject_reused_persona_none_for_combined(
        combined_df: pd.DataFrame,
        persona_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """AB 实验省略 none 时，用 A 实验 none 作为组合实验平衡基线。"""
        if combined_df.empty or persona_df.empty or "chat_atmosphere" not in combined_df.columns:
            return combined_df
        has_combined_none = (
            (combined_df.get("experiment_family", pd.Series(dtype=str)).astype(str) == "combined_supplement")
            & (combined_df["chat_atmosphere"].astype(str) == "none")
        ).any()
        if has_combined_none:
            return combined_df
        source = persona_df.copy()
        if "experiment_family" in source.columns:
            source = source[source["experiment_family"].astype(str) == "persona_supplement"].copy()
        if "chat_atmosphere" in source.columns:
            source = source[source["chat_atmosphere"].astype(str) == "none"].copy()
        if source.empty:
            return combined_df
        source["experiment_family"] = "combined_supplement"
        source["chat_atmosphere"] = "none"
        shared_cols = [col for col in combined_df.columns if col in source.columns]
        if not shared_cols:
            return combined_df
        return pd.concat([combined_df, source[shared_cols]], ignore_index=True, sort=False)

    def build_all(self) -> list[str]:
        """按顺序生成全部图，返回成功生成的文件名。"""
        generated: list[str] = []
        merged = self._load_merged_results()
        if merged is None or merged.empty:
            return generated

        if self._plot_platform_nsb(merged):
            generated.append("fig1_platform_nsb.png")
        if self._plot_p3_dynamics():
            generated.append("fig2_p3_dynamics.png")
        if self._plot_persona_heterogeneity(merged):
            generated.append("figS1_persona_heterogeneity.png")
        if self._plot_chat_atmosphere(merged):
            generated.append("figS2_chat_atmosphere.png")
        if self._plot_nci_robustness(merged):
            generated.append("figS3_nci_comp_robustness.png")
        if self._plot_combined_heatmap(merged):
            generated.append("figS4_combined_heatmap.png")
        if self._plot_p3_stop_heatmap():
            generated.append("figS5_p3_stop_heatmap.png")
        if self._plot_p3_terminal_metrics():
            generated.append("figS6_p3_terminal_metrics.png")
        if self._plot_p3_terminal_atmosphere():
            generated.append("figS7_p3_terminal_atmosphere.png")
        if self._plot_p3_individual_trajectories():
            generated.append("figS8_p3_individual_trajectories.png")
        if self._plot_p3_action_mix():
            generated.append("figS9_p3_action_mix.png")
        if self._plot_p3_interaction_effects():
            generated.append("figS10_p3_interaction_effects.png")
        return generated

    def _load_merged_results(self) -> Optional[pd.DataFrame]:
        """加载并合并 static + p3 结果，供多图复用。"""
        frames: list[pd.DataFrame] = []
        static_path = self.results_dir / "static_results.csv"
        p3_path = self.results_dir / "p3_results.csv"
        if static_path.exists():
            frames.append(pd.read_csv(static_path))
        if p3_path.exists():
            frames.append(pd.read_csv(p3_path))
        if not frames:
            return None
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged["condition_name"] = merged["condition_name"].astype(str)
        merged["experiment_family"] = merged["experiment_family"].astype(str)
        return merged

    def _plot_platform_nsb(self, df: pd.DataFrame) -> bool:
        # 主图比较的是“最终选择”口径：P1/P2 是静态最终判断，P3 是每个讨论单元的终局 final_nsb。
        # P3 的逐轮过程均值保留在 fig2 / figS8，不混入三平台主对比。
        summary_records: list[dict[str, object]] = []

        static_summary_path = self.results_dir / "static_summary.csv"
        if static_summary_path.exists():
            static_df = pd.read_csv(static_summary_path)
            static_df = static_df[static_df["experiment_family"] == "main"].copy()
            for condition in ["independent", "low_feedback"]:
                subset = static_df[static_df["condition_name"] == condition].copy()
                if subset.empty:
                    continue
                weight_col = "n_obs" if "n_obs" in subset.columns else None
                if weight_col:
                    total_n = float(subset[weight_col].sum())
                    mean_nsb = float((subset["mean_nsb"] * subset[weight_col]).sum() / total_n) if total_n else float("nan")
                    n = int(total_n)
                else:
                    mean_nsb = float(subset["mean_nsb"].mean())
                    n = int(len(subset))
                summary_records.append(
                    {
                        "condition_name": condition,
                        "mean_nsb": mean_nsb,
                        "n": n,
                        "metric_source": "static_final_choice",
                    }
                )
        else:
            static_df = df[(df["experiment_family"] == "main") & (df["condition_name"] != "high_interaction")].copy()
            if "nsb" in static_df.columns and not static_df.empty:
                for row in (
                    static_df.groupby("condition_name", as_index=False)
                    .agg(mean_nsb=("nsb", "mean"), n=("nsb", "size"))
                    .to_dict(orient="records")
                ):
                    row["metric_source"] = "static_final_choice"
                    summary_records.append(row)

        p3_cell_path = self.results_dir / "p3_cell_summary.csv"
        if p3_cell_path.exists():
            p3_df = pd.read_csv(p3_cell_path)
            p3_df = p3_df[p3_df["experiment_family"] == "main"].copy()
            if not p3_df.empty and "final_nsb" in p3_df.columns:
                summary_records.append(
                    {
                        "condition_name": "high_interaction",
                        "mean_nsb": float(p3_df["final_nsb"].mean()),
                        "n": int(p3_df["final_nsb"].count()),
                        "metric_source": "p3_terminal_final_nsb",
                    }
                )
        else:
            p3_df = df[(df["experiment_family"] == "main") & (df["condition_name"] == "high_interaction")].copy()
            if "nsb" in p3_df.columns and not p3_df.empty:
                summary_records.append(
                    {
                        "condition_name": "high_interaction",
                        "mean_nsb": float(p3_df["nsb"].mean()),
                        "n": int(p3_df["nsb"].count()),
                        "metric_source": "p3_process_nsb_fallback",
                    }
                )

        if not summary_records:
            return False
        summary = pd.DataFrame(summary_records)
        order = ["independent", "low_feedback", "high_interaction"]
        summary["condition_name"] = pd.Categorical(summary["condition_name"], categories=order, ordered=True)
        summary = summary.sort_values("condition_name")

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [self._platform_label(item) for item in summary["condition_name"].astype(str)]
        colors = ["#4C78A8", "#72B7B2", "#E45756"]
        ax.bar(labels, summary["mean_nsb"], color=colors, width=0.62)
        ax.set_ylabel("终局平均 NSB")
        ax.set_xlabel("平台条件")
        ax.set_title("图1 三平台负向选择偏向（P3终局口径）比较", loc="left", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y")
        for i, row in enumerate(summary.to_dict(orient="records")):
            ax.text(i, row["mean_nsb"] + 0.02, f"{row['mean_nsb']:.2f}", ha="center")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig1_platform_nsb.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "fig1_platform_nsb_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_p3_dynamics(self) -> bool:
        p3_path = self.results_dir / "p3_results.csv"
        if not p3_path.exists():
            return False
        df = pd.read_csv(p3_path)
        if df.empty:
            return False
        # P3 只报告时间轨迹，不把 P1/P2 伪装成动态过程。
        summary = (
            df.groupby("timestep", as_index=False)
            .agg(mean_gcr=("gcr", "mean"), mean_ssa=("ssa", "mean"), mean_scm=("scm", "mean"))
        )
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(summary["timestep"], summary["mean_gcr"], marker="o", color="#4C78A8", linewidth=1.8)
        axes[0].set_ylabel("GCR")
        axes[0].set_title("图2A 群体共识比率（GCR）时间轨迹", loc="left", fontweight="bold")
        axes[0].axhline(0.9, color="gray", linestyle="--", linewidth=1)
        axes[0].grid(axis="y")

        axes[1].plot(summary["timestep"], summary["mean_ssa"], marker="o", color="#E45756", linewidth=1.8)
        axes[1].set_ylabel("SSA")
        axes[1].set_title("图2B 社交信号非对称性（SSA）时间轨迹", loc="left", fontweight="bold")
        axes[1].grid(axis="y")

        axes[2].plot(summary["timestep"], summary["mean_scm"], marker="o", color="#54A24B", linewidth=1.8)
        axes[2].set_ylabel("SCM")
        axes[2].set_xlabel("时间步")
        axes[2].set_title("图2C 立场变化幅度（SCM）时间轨迹", loc="left", fontweight="bold")
        axes[2].grid(axis="y")

        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig2_p3_dynamics.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "fig2_p3_dynamics_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_persona_heterogeneity(self, df: pd.DataFrame) -> bool:
        static_summary_path = self.results_dir / "static_summary.csv"
        p3_persona_path = self.results_dir / "p3_agent_persona_summary.csv"
        frames: list[pd.DataFrame] = []
        if static_summary_path.exists():
            static_df = pd.read_csv(static_summary_path)
            static_df = static_df[static_df["experiment_family"] == "persona_supplement"].copy()
            if not static_df.empty:
                static_df = static_df.rename(columns={"persona_name": "persona_axis"})[
                    ["persona_axis", "condition_name", "mean_nsb"]
                ]
                frames.append(static_df)
        if p3_persona_path.exists():
            p3_df = pd.read_csv(p3_persona_path)
            p3_df = p3_df[p3_df["experiment_family"] == "persona_supplement"].copy()
            if not p3_df.empty:
                p3_df = p3_df.rename(columns={"agent_persona_name": "persona_axis"})[
                    ["persona_axis", "condition_name", "mean_nsb"]
                ]
                frames.append(p3_df)
        if not frames:
            return False
        summary = pd.concat(frames, ignore_index=True)
        summary = summary.groupby(["persona_axis", "condition_name"], as_index=False).agg(mean_nsb=("mean_nsb", "mean"))
        pivot = summary.pivot(index="persona_axis", columns="condition_name", values="mean_nsb").fillna(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([self._platform_label(item) for item in pivot.columns], rotation=15)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("图S1 个性异质性热力图", loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, label="平均 NSB")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS1_persona_heterogeneity.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS1_persona_heterogeneity_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_chat_atmosphere(self, df: pd.DataFrame) -> bool:
        static_summary_path = self.results_dir / "static_summary.csv"
        p3_summary_path = self.results_dir / "p3_summary.csv"
        frames: list[pd.DataFrame] = []
        if static_summary_path.exists():
            static_df = pd.read_csv(static_summary_path)
            static_df = static_df[static_df["experiment_family"] == "atmosphere_supplement"].copy()
            if not static_df.empty:
                frames.append(static_df[["chat_atmosphere", "condition_name", "mean_nsb"]])
        if p3_summary_path.exists():
            p3_df = pd.read_csv(p3_summary_path)
            p3_df = p3_df[p3_df["experiment_family"] == "atmosphere_supplement"].copy()
            if not p3_df.empty:
                frames.append(p3_df[["chat_atmosphere", "condition_name", "mean_nsb"]])
        if not frames:
            return False
        summary = pd.concat(frames, ignore_index=True)
        summary = summary.groupby(["chat_atmosphere", "condition_name"], as_index=False).agg(mean_nsb=("mean_nsb", "mean"))
        if "none" not in {str(item) for item in summary["chat_atmosphere"].dropna().unique()}:
            baseline_frames: list[pd.DataFrame] = []
            if static_summary_path.exists():
                static_df = pd.read_csv(static_summary_path)
                main_static = static_df[
                    (static_df["experiment_family"] == "main")
                    & (static_df["condition_name"].isin(["independent", "low_feedback"]))
                ].copy()
                if not main_static.empty:
                    main_static["chat_atmosphere"] = "none"
                    baseline_frames.append(main_static[["chat_atmosphere", "condition_name", "mean_nsb"]])
            if p3_summary_path.exists():
                p3_df = pd.read_csv(p3_summary_path)
                main_p3 = p3_df[
                    (p3_df["experiment_family"] == "main")
                    & (p3_df["condition_name"] == "high_interaction")
                ].copy()
                if not main_p3.empty:
                    main_p3["chat_atmosphere"] = "none"
                    baseline_frames.append(main_p3[["chat_atmosphere", "condition_name", "mean_nsb"]])
            if baseline_frames:
                baseline_summary = pd.concat(baseline_frames, ignore_index=True)
                baseline_summary = baseline_summary.groupby(
                    ["chat_atmosphere", "condition_name"],
                    as_index=False,
                ).agg(mean_nsb=("mean_nsb", "mean"))
                summary = pd.concat([summary, baseline_summary], ignore_index=True)
        order = self._observed_atmosphere_order(summary["chat_atmosphere"])
        if not order:
            return False
        summary["chat_atmosphere"] = pd.Categorical(summary["chat_atmosphere"], categories=order, ordered=True)
        summary = summary.sort_values(["chat_atmosphere", "condition_name"])

        fig, ax = plt.subplots(figsize=(8, 5))
        platforms = list(summary["condition_name"].drop_duplicates())
        x = range(len(order))
        width = 0.35
        palette = {"low_feedback": "#72B7B2", "high_interaction": "#E45756"}
        for idx, platform in enumerate(platforms):
            platform_data = summary[summary["condition_name"] == platform].set_index("chat_atmosphere").reindex(order)
            ax.bar(
                [item + (idx - 0.5) * width for item in x],
                platform_data["mean_nsb"],
                width=width,
                label=self._platform_label(platform),
                color=palette.get(platform, "#4C78A8"),
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels([self._atmosphere_label(item) for item in order])
        ax.set_ylabel("平均 NSB")
        ax.set_xlabel("聊天氛围")
        ax.set_title("图S2 聊天氛围效应", loc="left", fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS2_chat_atmosphere.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS2_chat_atmosphere_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_nci_robustness(self, df: pd.DataFrame) -> bool:
        if "nci_comp" not in df.columns:
            return False
        summary = (
            df.groupby("condition_name", as_index=False)
            .agg(mean_nci_comp=("nci_comp", "mean"))
        )
        order = ["independent", "low_feedback", "high_interaction"]
        summary["condition_name"] = pd.Categorical(summary["condition_name"], categories=order, ordered=True)
        summary = summary.sort_values("condition_name")

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [self._platform_label(item) for item in summary["condition_name"].astype(str)]
        ax.bar(labels, summary["mean_nci_comp"], color="#A0CBE8", width=0.62)
        ax.set_ylabel("平均 NCI_comp")
        ax.set_xlabel("平台条件")
        ax.set_title("图S3 复合负向选择指标（NCI_comp）稳健性检验", loc="left", fontweight="bold")
        ax.grid(axis="y")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS3_nci_comp_robustness.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS3_nci_comp_robustness_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_combined_heatmap(self, df: pd.DataFrame) -> bool:
        static_summary_path = self.results_dir / "static_summary.csv"
        p3_persona_path = self.results_dir / "p3_agent_persona_summary.csv"
        frames: list[pd.DataFrame] = []
        if static_summary_path.exists():
            static_df = pd.read_csv(static_summary_path)
            combined_static = static_df[static_df["experiment_family"] == "combined_supplement"].copy()
            persona_static = static_df[static_df["experiment_family"] == "persona_supplement"].copy()
            combined_static = self._inject_reused_persona_none_for_combined(combined_static, persona_static)
            if not combined_static.empty:
                combined_static["persona_atmosphere"] = combined_static["persona_name"].astype(str) + " | " + combined_static["chat_atmosphere"].astype(str)
                frames.append(combined_static[["persona_atmosphere", "condition_name", "mean_nsb"]])
        if p3_persona_path.exists():
            p3_df = pd.read_csv(p3_persona_path)
            combined_p3 = p3_df[p3_df["experiment_family"] == "combined_supplement"].copy()
            persona_p3 = p3_df[p3_df["experiment_family"] == "persona_supplement"].copy()
            combined_p3 = self._inject_reused_persona_none_for_combined(combined_p3, persona_p3)
            if not combined_p3.empty:
                combined_p3["persona_atmosphere"] = combined_p3["agent_persona_name"].astype(str) + " | " + combined_p3["chat_atmosphere"].astype(str)
                frames.append(combined_p3[["persona_atmosphere", "condition_name", "mean_nsb"]])
        if not frames:
            return False
        summary = pd.concat(frames, ignore_index=True)
        summary = summary.groupby(["persona_atmosphere", "condition_name"], as_index=False).agg(mean_nsb=("mean_nsb", "mean"))
        pivot = summary.pivot(index="persona_atmosphere", columns="condition_name", values="mean_nsb").fillna(0)

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([self._platform_label(item) for item in pivot.columns], rotation=15)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("图S4 联合补充实验热力图", loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, label="平均 NSB")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS4_combined_heatmap.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS4_combined_heatmap_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_p3_stop_heatmap(self) -> bool:
        p3_path = self.results_dir / "p3_cell_summary.csv"
        if not p3_path.exists():
            return False
        df = pd.read_csv(p3_path)
        if df.empty or "stop_reason" not in df.columns:
            return False
        summary = (
            df.groupby(["chat_atmosphere", "stop_reason"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        pivot = summary.pivot(index="chat_atmosphere", columns="stop_reason", values="count").fillna(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=15)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([self._atmosphere_label(item) for item in pivot.index])
        ax.set_title("图S5 P3 平台停止原因分布热力图", loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, label="频数")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS5_p3_stop_heatmap.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS5_p3_stop_heatmap_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_p3_terminal_metrics(self) -> bool:
        p3_path = self.results_dir / "p3_cell_summary.csv"
        if not p3_path.exists():
            return False
        df = pd.read_csv(p3_path)
        if df.empty:
            return False
        summary = (
            df.groupby(["chat_atmosphere", "cell_persona_design"], as_index=False)
            .agg(
                mean_final_nsb=("final_nsb", "mean"),
                mean_terminal_all_latent=(
                    "terminal_all_agents_latent_neg_mean"
                    if "terminal_all_agents_latent_neg_mean" in df.columns
                    else "final_nsb",
                    "mean",
                ),
                mean_panel_last_choice=(
                    "terminal_panel_all_last_choice_nsb"
                    if "terminal_panel_all_last_choice_nsb" in df.columns
                    else "final_nsb",
                    "mean",
                ),
                mean_final_gcr=("final_gcr", "mean"),
            )
        )
        order = self._observed_atmosphere_order(summary["chat_atmosphere"])
        if not order:
            return False
        summary["chat_atmosphere"] = pd.Categorical(summary["chat_atmosphere"], categories=order, ordered=True)
        summary = summary.sort_values(["cell_persona_design", "chat_atmosphere"])

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes_flat = axes.flatten()
        labels = [
            f"{self._atmosphere_label(atm)}\n({design})"
            for atm, design in zip(summary["chat_atmosphere"].astype(str), summary["cell_persona_design"].astype(str))
        ]
        plot_specs = [
            ("mean_final_nsb", "#E45756", "图S6A Probe二值终局 NSB", "Probe binary NSB"),
            ("mean_terminal_all_latent", "#B279A2", "图S6B 全体潜在负向倾向", "Latent negativity"),
            ("mean_panel_last_choice", "#F58518", "图S6C Panel last-choice NSB", "Panel last-choice NSB"),
            ("mean_final_gcr", "#4C78A8", "图S6D P3终局共识比率", "终局 GCR"),
        ]
        for ax, (column, color, title, ylabel) in zip(axes_flat, plot_specs):
            ax.bar(labels, summary[column], color=color, width=0.62)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
            ax.grid(axis="y")
            ax.tick_params(axis="x", rotation=20)

        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS6_p3_terminal_metrics.png", dpi=300)
        plt.close(fig)
        write_csv(
            self.figures_dir / "figS6_p3_terminal_metrics_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_p3_terminal_atmosphere(self) -> bool:
        p3_path = self.results_dir / "p3_cell_summary.csv"
        if not p3_path.exists():
            return False
        df = pd.read_csv(p3_path)
        if df.empty or "terminal_community_atmosphere" not in df.columns:
            return False

        summary = (
            df.groupby(["chat_atmosphere", "terminal_community_atmosphere"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        pivot = summary.pivot(
            index="chat_atmosphere",
            columns="terminal_community_atmosphere",
            values="count",
        ).fillna(0)

        order = self._observed_atmosphere_order(summary["chat_atmosphere"])
        if not order:
            return False
        atmosphere_order = ["positive_leaning", "negative_leaning", "mixed", "quiet", "unclear"]
        pivot = pivot.reindex(order).fillna(0)
        pivot = pivot.reindex(columns=[c for c in atmosphere_order if c in pivot.columns] + [c for c in pivot.columns if c not in atmosphere_order])

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=15)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([self._atmosphere_label(item) for item in pivot.index])
        ax.set_title("图S7 P3终局社群氛围分布热力图", loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, label="频数")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS7_p3_terminal_atmosphere.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS7_p3_terminal_atmosphere_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_p3_individual_trajectories(self) -> bool:
        p3_path = self.results_dir / "p3_results.csv"
        if not p3_path.exists():
            return False
        df = pd.read_csv(p3_path)
        if df.empty or "actor_id" not in df.columns:
            return False

        cell_summary = (
            df.groupby(["scenario_name", "repeat_index"], as_index=False)
            .agg(final_timestep=("timestep", "max"), mean_nsb=("nsb", "mean"))
        )
        if cell_summary.empty:
            return False

        representative_rows: list[dict] = []
        for scenario in sorted(cell_summary["scenario_name"].dropna().unique()):
            subset = cell_summary[cell_summary["scenario_name"] == scenario].copy()
            median_timestep = subset["final_timestep"].median()
            subset["dist_to_median"] = (subset["final_timestep"] - median_timestep).abs()
            representative_rows.append(
                subset.sort_values(["dist_to_median", "repeat_index"]).iloc[0].to_dict()
            )

        if not representative_rows:
            return False

        fig, axes = plt.subplots(
            len(representative_rows),
            1,
            figsize=(10, 3.8 * len(representative_rows)),
            sharex=False,
        )
        if len(representative_rows) == 1:
            axes = [axes]

        export_rows: list[dict] = []
        palette = [
            "#4C78A8",
            "#F58518",
            "#E45756",
            "#72B7B2",
            "#54A24B",
            "#EECA3B",
            "#B279A2",
            "#FF9DA6",
            "#9D755D",
        ]

        for ax, rep in zip(axes, representative_rows):
            scenario = rep["scenario_name"]
            repeat_index = int(rep["repeat_index"])
            subset = df[(df["scenario_name"] == scenario) & (df["repeat_index"] == repeat_index)].copy()
            subset = subset.sort_values(["actor_id", "timestep"])
            actor_ids = [item for item in sorted(subset["actor_id"].dropna().unique())]

            for idx, actor_id in enumerate(actor_ids):
                actor_df = subset[subset["actor_id"] == actor_id].copy()
                ax.plot(
                    actor_df["timestep"],
                    actor_df["nsb"],
                    marker="o",
                    linewidth=1.6,
                    markersize=3.8,
                    color=palette[idx % len(palette)],
                    label=actor_id,
                    alpha=0.95,
                )
                for row in actor_df.to_dict(orient="records"):
                    export_rows.append(
                        {
                            "scenario_name": scenario,
                            "repeat_index": repeat_index,
                            "actor_id": actor_id,
                            "timestep": row["timestep"],
                            "nsb": row["nsb"],
                            "action_type": row.get("action_type"),
                            "final_choice": row.get("final_choice"),
                        }
                    )

            title_scenario = "甲流" if scenario == "influenza_a" else "支原体肺炎"
            ax.set_title(
                f"图S8 {title_scenario}：代表性轮次中的个体 NSB 轨迹（repeat={repeat_index}）",
                loc="left",
                fontweight="bold",
            )
            ax.set_ylabel("个体 NSB")
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["偏报道1", "偏报道2"])
            ax.grid(axis="y")
            ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
            ax.set_xlabel("时间步")

        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS8_p3_individual_trajectories.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS8_p3_individual_trajectories_data.csv",
            export_rows,
        )
        return True

    def _plot_p3_action_mix(self) -> bool:
        p3_path = self.results_dir / "p3_results.csv"
        if not p3_path.exists():
            return False
        df = pd.read_csv(p3_path)
        if df.empty or "action_type" not in df.columns:
            return False

        order = ["post", "reply", "like", "repost", "observe"]
        summary = (
            df.groupby(["scenario_name", "action_type"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        totals = summary.groupby("scenario_name", as_index=False)["count"].sum().rename(columns={"count": "total"})
        summary = summary.merge(totals, on="scenario_name", how="left")
        summary["share"] = summary["count"] / summary["total"].replace(0, 1)

        pivot = summary.pivot(index="scenario_name", columns="action_type", values="share").fillna(0.0)
        pivot = pivot.reindex(columns=order, fill_value=0.0)
        if pivot.empty:
            return False

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {
            "post": "#4C78A8",
            "reply": "#F58518",
            "like": "#54A24B",
            "repost": "#B279A2",
            "observe": "#9D9D9D",
        }
        bottom = pd.Series(0.0, index=pivot.index)
        for action in order:
            values = pivot[action]
            ax.bar(pivot.index, values, bottom=bottom, label=action, color=colors[action], width=0.6)
            bottom = bottom + values

        ax.set_ylim(0, 1)
        ax.set_ylabel("动作占比")
        ax.set_title("图S9 P3 动作构成：like / reply / repost 的使用比例", loc="left", fontweight="bold")
        ax.grid(axis="y")
        ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS9_p3_action_mix.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS9_p3_action_mix_data.csv",
            summary.to_dict(orient="records"),
        )
        return True

    def _plot_p3_interaction_effects(self) -> bool:
        p3_path = self.results_dir / "p3_results.csv"
        cell_path = self.results_dir / "p3_cell_summary.csv"
        if not p3_path.exists() or not cell_path.exists():
            return False
        df = pd.read_csv(p3_path)
        cell_df = pd.read_csv(cell_path)
        if df.empty or cell_df.empty:
            return False

        group_cols = [
            "experiment_family",
            "scenario_name",
            "condition_name",
            "persona_name",
            "chat_atmosphere",
            "repeat_index",
        ]
        action_stats = (
            df.groupby(group_cols + ["action_type"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        pivot = action_stats.pivot(index=group_cols, columns="action_type", values="count").fillna(0).reset_index()
        for col in ["reply", "like", "post", "repost", "observe"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot["n_actions"] = pivot[["reply", "like", "post", "repost", "observe"]].sum(axis=1)
        pivot["reply_like_share"] = (
            pivot[["reply", "like"]].sum(axis=1) / pivot["n_actions"].replace(0, 1)
        )
        pivot["interactive_share"] = (
            pivot[["reply", "like", "post", "repost"]].sum(axis=1) / pivot["n_actions"].replace(0, 1)
        )

        merged = cell_df.merge(pivot, on=group_cols, how="left")
        if merged.empty:
            return False

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        colors = {"influenza_a": "#E45756", "mycoplasma_pneumonia": "#4C78A8"}
        for scenario, sub in merged.groupby("scenario_name"):
            axes[0].scatter(
                sub["reply_like_share"],
                sub["final_nsb"],
                label=scenario,
                color=colors.get(scenario, "#4C78A8"),
                alpha=0.8,
                s=42,
            )
            axes[1].scatter(
                sub["reply_like_share"],
                sub["final_ssa"],
                label=scenario,
                color=colors.get(scenario, "#4C78A8"),
                alpha=0.8,
                s=42,
            )
        axes[0].set_xlabel("reply + like 占比")
        axes[0].set_ylabel("final NSB")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("图S10A reply/like 强度与终局 NSB", loc="left", fontweight="bold")
        axes[0].grid(True, alpha=0.25)

        axes[1].set_xlabel("reply + like 占比")
        axes[1].set_ylabel("final SSA")
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[1].set_title("图S10B reply/like 强度与终局 SSA", loc="left", fontweight="bold")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figS10_p3_interaction_effects.png", dpi=300)
        plt.close(fig)

        write_csv(
            self.figures_dir / "figS10_p3_interaction_effects_data.csv",
            merged[
                group_cols
                + [
                    "reply",
                    "like",
                    "post",
                    "repost",
                    "observe",
                    "reply_like_share",
                    "interactive_share",
                    "final_nsb",
                    "final_ssa",
                    "mean_nsb",
                    "mean_ssa",
                ]
            ].to_dict(orient="records"),
        )
        return True
