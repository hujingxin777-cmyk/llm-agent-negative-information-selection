from __future__ import annotations

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REASON_KEYWORDS = [
    "门诊",
    "排队",
    "拥挤",
    "恢复",
    "恢复慢",
    "恢复较快",
    "风险",
    "被低估",
    "并发症",
    "医院",
    "疾控",
    "专家",
    "正式通报",
    "身边",
    "朋友",
    "家长",
    "老师",
    "高热",
    "咳嗽",
    "反复",
    "就医压力",
    "评论",
    "点赞",
    "讨论",
    "可控",
    "真实情况",
]


class FinalReportBuilder:
    def __init__(self, demo_dir: Path) -> None:
        self.demo_dir = demo_dir
        self.data_dir = demo_dir / "data"
        self.final_figures_dir = self.data_dir / "final_figures"
        self.final_figures_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()

    def _apply_style(self) -> None:
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
                "grid.alpha": 0.22,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.bbox": "tight",
            }
        )

    def _bar_labels(self, ax, bars, fmt: str = "{:.2f}", dy: float = 0.02) -> None:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + dy,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _encode_image(self, path: Path) -> str:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def _load_summary_tables(self) -> dict[str, pd.DataFrame]:
        tables = {
            "main_static": pd.read_csv(self.data_dir / "static_summary.csv"),
            "main_p3": pd.read_csv(self.data_dir / "p3_summary.csv"),
            "main_term": pd.read_csv(self.data_dir / "p3_terminal_summary.csv"),
            "persona_static": pd.read_csv(self.data_dir / "supplements/persona_supplement/results/static_summary.csv"),
            "persona_p3": pd.read_csv(self.data_dir / "supplements/persona_supplement/results/p3_summary.csv"),
            "persona_agent": pd.read_csv(self.data_dir / "supplements/persona_supplement/results/p3_agent_persona_summary.csv"),
            "persona_cell": pd.read_csv(self.data_dir / "supplements/persona_supplement/results/p3_cell_summary.csv"),
            "atm_static": pd.read_csv(self.data_dir / "supplements/atmosphere_supplement/results/static_summary.csv"),
            "atm_p3": pd.read_csv(self.data_dir / "supplements/atmosphere_supplement/results/p3_summary.csv"),
            "atm_cell": pd.read_csv(self.data_dir / "supplements/atmosphere_supplement/results/p3_cell_summary.csv"),
            "comb_static": pd.read_csv(self.data_dir / "supplements/combined_supplement/results/static_summary.csv"),
            "comb_p3": pd.read_csv(self.data_dir / "supplements/combined_supplement/results/p3_summary.csv"),
            "comb_agent": pd.read_csv(self.data_dir / "supplements/combined_supplement/results/p3_agent_persona_summary.csv"),
            "comb_cell": pd.read_csv(self.data_dir / "supplements/combined_supplement/results/p3_cell_summary.csv"),
        }
        tables = {name: self._canonicalize_persona_names(df) for name, df in tables.items()}
        tables["atm_static"] = self._inject_main_baseline_atmosphere(
            tables["atm_static"],
            tables["main_static"],
            condition_names={"independent", "low_feedback"},
        )
        tables["atm_p3"] = self._inject_main_baseline_atmosphere(
            tables["atm_p3"],
            tables["main_p3"],
            condition_names={"high_interaction"},
        )
        tables["comb_static"] = self._inject_persona_baseline_combined(
            tables["comb_static"],
            tables["persona_static"],
        )
        tables["comb_p3"] = self._inject_persona_baseline_combined(
            tables["comb_p3"],
            tables["persona_p3"],
        )
        tables["comb_agent"] = self._inject_persona_baseline_combined(
            tables["comb_agent"],
            tables["persona_agent"],
        )
        tables["comb_cell"] = self._inject_persona_baseline_combined(
            tables["comb_cell"],
            tables["persona_cell"],
        )
        return tables

    @staticmethod
    def _canonicalize_persona_names(df: pd.DataFrame) -> pd.DataFrame:
        """兼容旧结果中的 neutral_prudent，并统一显示为 ordinary_user。"""
        out = df.copy()
        for col in ("persona_name", "agent_persona_name", "target_persona_name"):
            if col in out.columns:
                out[col] = out[col].replace({"neutral_prudent": "ordinary_user"})
        return out

    @staticmethod
    def _inject_main_baseline_atmosphere(
        atmosphere_df: pd.DataFrame,
        main_df: pd.DataFrame,
        condition_names: set[str],
    ) -> pd.DataFrame:
        """B 实验省略 none 时，用主实验 none 作为 B 的平衡 baseline。"""
        if "chat_atmosphere" in atmosphere_df.columns and (atmosphere_df["chat_atmosphere"] == "none").any():
            return atmosphere_df
        baseline = main_df[main_df["condition_name"].isin(condition_names)].copy()
        if baseline.empty:
            return atmosphere_df
        baseline["experiment_family"] = "atmosphere_supplement"
        baseline["chat_atmosphere"] = "none"
        baseline["persona_name"] = "baseline_open"
        shared_cols = [col for col in atmosphere_df.columns if col in baseline.columns]
        if not shared_cols:
            return atmosphere_df
        baseline = baseline[shared_cols]
        return pd.concat([atmosphere_df, baseline], ignore_index=True, sort=False)

    @staticmethod
    def _inject_persona_baseline_combined(
        combined_df: pd.DataFrame,
        persona_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """AB 实验省略 none 时，用 A 实验 none 作为 AB 的平衡 baseline。"""
        if "chat_atmosphere" in combined_df.columns and (combined_df["chat_atmosphere"] == "none").any():
            return combined_df
        baseline = persona_df.copy()
        if "chat_atmosphere" in baseline.columns:
            baseline = baseline[baseline["chat_atmosphere"] == "none"].copy()
        if baseline.empty:
            return combined_df
        baseline["experiment_family"] = "combined_supplement"
        shared_cols = [col for col in combined_df.columns if col in baseline.columns]
        if not shared_cols:
            return combined_df
        baseline = baseline[shared_cols]
        return pd.concat([combined_df, baseline], ignore_index=True, sort=False)

    def _build_cross_experiment_p3_figure(self, tables: dict[str, pd.DataFrame]) -> Path:
        main = tables["main_p3"].copy()
        persona = tables["persona_p3"].copy()
        atm = tables["atm_p3"].copy()
        comb = tables["comb_p3"].copy()

        rows: list[dict[str, object]] = []
        for _, row in main.iterrows():
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "label": "主实验",
                    "mean_nsb": row["mean_nsb"],
                    "mean_final_timestep": row["mean_final_timestep"],
                }
            )
        for _, row in persona.iterrows():
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "label": "A实验",
                    "mean_nsb": row["mean_nsb"],
                    "mean_final_timestep": row["mean_final_timestep"],
                }
            )
        atmosphere_labels = {
            "negative_leaning": "B负向",
            "none": "B平衡",
            "positive_leaning": "B正向",
        }
        for _, row in atm.iterrows():
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "label": atmosphere_labels[row["chat_atmosphere"]],
                    "mean_nsb": row["mean_nsb"],
                    "mean_final_timestep": row["mean_final_timestep"],
                }
            )
        combined_labels = {
            "negative_leaning": "AB负向",
            "none": "AB平衡",
            "positive_leaning": "AB正向",
        }
        for _, row in comb.iterrows():
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "label": combined_labels[row["chat_atmosphere"]],
                    "mean_nsb": row["mean_nsb"],
                    "mean_final_timestep": row["mean_final_timestep"],
                }
            )

        df = pd.DataFrame(rows)
        order = ["主实验", "A实验", "B负向", "B平衡", "B正向", "AB负向", "AB平衡", "AB正向"]
        palette = ["#4C78A8", "#7A5195", "#E45756", "#72B7B2", "#54A24B", "#B279A2", "#8E6C8A", "#6F9EAF"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        panels = [
            ("influenza_a", "mean_nsb", "甲流：P3平均 NSB", (0, 0), (0, 1)),
            ("influenza_a", "mean_final_timestep", "甲流：P3平均结束时间步", (0, 1), None),
            ("mycoplasma_pneumonia", "mean_nsb", "支原体肺炎：P3平均 NSB", (1, 0), (0, 1)),
            ("mycoplasma_pneumonia", "mean_final_timestep", "支原体肺炎：P3平均结束时间步", (1, 1), None),
        ]
        for scenario, metric, title, ax_idx, ylim in panels:
            ax = axes[ax_idx]
            sub = df[df["scenario_name"] == scenario].copy()
            sub["label"] = pd.Categorical(sub["label"], categories=order, ordered=True)
            sub = sub.sort_values("label")
            bars = ax.bar(sub["label"].astype(str), sub[metric], color=palette, width=0.68)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(axis="y")
            ax.tick_params(axis="x", rotation=20)
            if metric == "mean_nsb":
                ax.set_ylabel("平均 NSB")
                ax.set_ylim(0, 1.05)
                self._bar_labels(ax, bars, "{:.2f}", 0.02)
            else:
                ax.set_ylabel("平均结束时间步")
                ax.set_ylim(0, max(sub[metric].max() + 1.2, 10.5))
                self._bar_labels(ax, bars, "{:.2f}", 0.12)

        fig.suptitle("跨实验 P3 负向收敛与讨论时长对比", fontsize=14, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout()
        out = self.final_figures_dir / "全实验_P3收敛与时长对比.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _build_combined_agent_heatmap(self, tables: dict[str, pd.DataFrame]) -> Path:
        agent = tables["comb_agent"].copy()
        summary = (
            agent.groupby(["agent_persona_name", "chat_atmosphere"], as_index=False)
            .agg(mean_nsb=("mean_nsb", "mean"))
        )
        order_cols = ["negative_leaning", "none", "positive_leaning"]
        order_rows = ["risk_sensitive", "ordinary_user", "authority_trusting"]
        pivot = summary.pivot(index="agent_persona_name", columns="chat_atmosphere", values="mean_nsb").reindex(index=order_rows, columns=order_cols)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(order_cols)))
        ax.set_xticklabels(["负向倾斜", "平衡氛围", "正向倾斜"])
        ax.set_yticks(range(len(order_rows)))
        ax.set_yticklabels(["风险敏感型", "普通用户型", "权威信任型"])
        ax.set_title("混合实验 P3：人格 × 初始氛围 交叉热力图", loc="left", fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, label="平均 NSB")
        fig.tight_layout()
        out = self.final_figures_dir / "混合实验_P3人格氛围交叉热力图.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _build_terminal_comparison_figure(self, tables: dict[str, pd.DataFrame]) -> Path:
        atm_cell = tables["atm_cell"].copy()
        comb_cell = tables["comb_cell"].copy()

        def build_pivot(df: pd.DataFrame, title: str):
            summary = (
                df.groupby(["chat_atmosphere", "terminal_community_atmosphere"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            order_rows = ["negative_leaning", "none", "positive_leaning"]
            order_cols = ["negative_leaning", "positive_leaning"]
            pivot = summary.pivot(index="chat_atmosphere", columns="terminal_community_atmosphere", values="count").reindex(index=order_rows, columns=order_cols).fillna(0)
            return pivot

        pivot_b = build_pivot(atm_cell, "B实验")
        pivot_c = build_pivot(comb_cell, "混合实验")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, pivot, title in zip(axes, [pivot_b, pivot_c], ["B实验终局氛围", "混合实验终局氛围"]):
            im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(["终局负向", "终局正向"])
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(["起点负向", "起点平衡", "起点正向"])
            ax.set_title(title, loc="left", fontweight="bold")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    ax.text(j, i, f"{int(pivot.iloc[i, j])}", ha="center", va="center", color="black", fontweight="bold")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="轮次数")
        fig.suptitle("B实验与混合实验：终局氛围对比", fontsize=14, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout()
        out = self.final_figures_dir / "B与混合实验_终局氛围对比.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _build_main_static_heatmap(self, tables: dict[str, pd.DataFrame]) -> Path:
        static = tables["main_static"].copy()
        p3 = tables["main_p3"][["scenario_name", "mean_nsb"]].copy()
        p3["condition_name"] = "high_interaction"
        merged = pd.concat(
            [
                static[["scenario_name", "condition_name", "mean_nsb"]],
                p3[["scenario_name", "condition_name", "mean_nsb"]],
            ],
            ignore_index=True,
        )
        order_rows = ["influenza_a", "mycoplasma_pneumonia"]
        order_cols = ["independent", "low_feedback", "high_interaction"]
        pivot = (
            merged.pivot(index="scenario_name", columns="condition_name", values="mean_nsb")
            .reindex(index=order_rows, columns=order_cols)
        )

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(order_cols)))
        ax.set_xticklabels(["P1", "P2", "P3"])
        ax.set_yticks(range(len(order_rows)))
        ax.set_yticklabels(["甲流", "支原体肺炎"])
        ax.set_title("主实验：场景 × 平台 平均 NSB 热力图", loc="left", fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="平均 NSB")
        fig.tight_layout()
        out = self.final_figures_dir / "主实验_场景平台热力图_标数.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _build_persona_static_heatmap(self, tables: dict[str, pd.DataFrame]) -> Path:
        static = tables["persona_static"].copy()
        static["scene_platform"] = static["scenario_name"].map(
            {
                "influenza_a": "甲流",
                "mycoplasma_pneumonia": "支原体肺炎",
            }
        ) + "-" + static["condition_name"].map({"independent": "P1", "low_feedback": "P2"})
        order_rows = ["risk_sensitive", "ordinary_user", "authority_trusting"]
        order_cols = ["甲流-P1", "甲流-P2", "支原体肺炎-P1", "支原体肺炎-P2"]
        pivot = (
            static.pivot(index="persona_name", columns="scene_platform", values="mean_nsb")
            .reindex(index=order_rows, columns=order_cols)
        )

        fig, ax = plt.subplots(figsize=(9, 4.8))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(order_cols)))
        ax.set_xticklabels(order_cols)
        ax.set_yticks(range(len(order_rows)))
        ax.set_yticklabels(["风险敏感型", "普通用户型", "权威信任型"])
        ax.set_title("A实验：人格 × 静态平台 平均 NSB 热力图", loc="left", fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="平均 NSB")
        fig.tight_layout()
        out = self.final_figures_dir / "A实验_人格静态热力图_标数.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _build_atmosphere_static_heatmap(self, tables: dict[str, pd.DataFrame]) -> Path:
        static = tables["atm_static"].copy()
        order_rows = ["influenza_a", "mycoplasma_pneumonia"]
        order_cols = ["negative_leaning", "none", "positive_leaning"]
        pivot = (
            static.pivot(index="scenario_name", columns="chat_atmosphere", values="mean_nsb")
            .reindex(index=order_rows, columns=order_cols)
        )

        fig, ax = plt.subplots(figsize=(8, 4.8))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(order_cols)))
        ax.set_xticklabels(["负向氛围", "平衡氛围", "正向氛围"])
        ax.set_yticks(range(len(order_rows)))
        ax.set_yticklabels(["甲流", "支原体肺炎"])
        ax.set_title("B实验：场景 × 聊天气氛 平均 NSB 热力图", loc="left", fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="平均 NSB")
        fig.tight_layout()
        out = self.final_figures_dir / "B实验_氛围静态热力图_标数.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _build_combined_static_heatmap(self, tables: dict[str, pd.DataFrame]) -> Path:
        static = tables["comb_static"].copy()
        order_rows = ["risk_sensitive", "ordinary_user", "authority_trusting"]
        order_cols = ["negative_leaning", "none", "positive_leaning"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, scenario, title in zip(
            axes,
            ["influenza_a", "mycoplasma_pneumonia"],
            ["甲流", "支原体肺炎"],
        ):
            sub = static[static["scenario_name"] == scenario]
            pivot = (
                sub.pivot(index="persona_name", columns="chat_atmosphere", values="mean_nsb")
                .reindex(index=order_rows, columns=order_cols)
            )
            im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(order_cols)))
            ax.set_xticklabels(["负向氛围", "平衡氛围", "正向氛围"])
            ax.set_yticks(range(len(order_rows)))
            ax.set_yticklabels(["风险敏感型", "普通用户型", "权威信任型"])
            ax.set_title(title, loc="left", fontweight="bold")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iloc[i, j]
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="平均 NSB")
        fig.suptitle("混合实验：静态阶段 人格 × 氛围 平均 NSB 热力图", fontsize=14, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout()
        out = self.final_figures_dir / "混合实验_静态交互热力图_标数.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _build_b_vs_combined_dynamic_comparison(self, tables: dict[str, pd.DataFrame]) -> Path:
        atm = tables["atm_p3"].copy()
        comb = tables["comb_p3"].copy()
        rows: list[dict[str, object]] = []
        for _, row in atm.iterrows():
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "chat_atmosphere": row["chat_atmosphere"],
                    "group": "B实验",
                    "mean_nsb": row["mean_nsb"],
                    "mean_final_timestep": row["mean_final_timestep"],
                }
            )
        for _, row in comb.iterrows():
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "chat_atmosphere": row["chat_atmosphere"],
                    "group": "混合实验",
                    "mean_nsb": row["mean_nsb"],
                    "mean_final_timestep": row["mean_final_timestep"],
                }
            )
        df = pd.DataFrame(rows)
        order = ["negative_leaning", "none", "positive_leaning"]
        label_map = {"negative_leaning": "负向起点", "none": "平衡起点", "positive_leaning": "正向起点"}

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        panels = [
            ("influenza_a", "mean_nsb", "甲流：B实验 vs 混合实验（P3 平均 NSB）", axes[0, 0]),
            ("influenza_a", "mean_final_timestep", "甲流：B实验 vs 混合实验（平均结束时间步）", axes[0, 1]),
            ("mycoplasma_pneumonia", "mean_nsb", "支原体肺炎：B实验 vs 混合实验（P3 平均 NSB）", axes[1, 0]),
            ("mycoplasma_pneumonia", "mean_final_timestep", "支原体肺炎：B实验 vs 混合实验（平均结束时间步）", axes[1, 1]),
        ]
        palette = {"B实验": "#E45756", "混合实验": "#4C78A8"}
        width = 0.36
        x = list(range(len(order)))
        for scenario, metric, title, ax in panels:
            sub = df[df["scenario_name"] == scenario].copy()
            for idx, group in enumerate(["B实验", "混合实验"]):
                part = sub[sub["group"] == group].set_index("chat_atmosphere").reindex(order)
                bars = ax.bar(
                    [i + (idx - 0.5) * width for i in x],
                    part[metric],
                    width=width,
                    label=group,
                    color=palette[group],
                )
                if metric == "mean_nsb":
                    self._bar_labels(ax, bars, "{:.2f}", 0.02)
                else:
                    self._bar_labels(ax, bars, "{:.2f}", 0.12)
            ax.set_xticks(x)
            ax.set_xticklabels([label_map[o] for o in order])
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(axis="y")
            if metric == "mean_nsb":
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("平均 NSB")
            else:
                ax.set_ylim(0, max(sub[metric].max() + 1.2, 10.5))
                ax.set_ylabel("平均结束时间步")
            ax.legend(frameon=False)

        fig.suptitle("B实验与混合实验：动态阶段的两类关键差异", fontsize=14, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout()
        out = self.final_figures_dir / "B与混合实验_动态阶段关键差异.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _load_all_reason_rows(self) -> pd.DataFrame:
        result_paths = [
            self.data_dir / "static_results.csv",
            self.data_dir / "p3_results.csv",
            self.data_dir / "supplements/persona_supplement/results/static_results.csv",
            self.data_dir / "supplements/persona_supplement/results/p3_results.csv",
            self.data_dir / "supplements/atmosphere_supplement/results/static_results.csv",
            self.data_dir / "supplements/atmosphere_supplement/results/p3_results.csv",
            self.data_dir / "supplements/combined_supplement/results/static_results.csv",
            self.data_dir / "supplements/combined_supplement/results/p3_results.csv",
        ]
        frames: list[pd.DataFrame] = []
        for path in result_paths:
            if path.exists():
                frame = pd.read_csv(path)
                frame["source_file"] = path.name
                frames.append(frame)
        df = pd.concat(frames, ignore_index=True, sort=False)
        return df.dropna(subset=["reason", "final_choice"]).copy()

    def _normalize_reason_choice(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def _row_choice(row: pd.Series) -> str | None:
            choice = row.get("final_choice")
            if choice in ("positive", "negative"):
                return choice
            if choice == "report_1":
                return row.get("report_1_polarity")
            if choice == "report_2":
                return row.get("report_2_polarity")
            return None

        df["choice_polarity"] = df.apply(_row_choice, axis=1)
        return df[df["choice_polarity"].isin(["positive", "negative"])].copy()

    def _reason_keyword_counts(self, subset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        reason_series = subset["reason"].astype(str)
        for keyword in REASON_KEYWORDS:
            count = int(reason_series.str.contains(keyword, regex=False).sum())
            if count > 0:
                rows.append({"keyword": keyword, "count": count})
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["keyword", "count"])
        return out.sort_values(["count", "keyword"], ascending=[False, True]).head(12)

    def _build_reason_choice_comparison(self) -> Path:
        df = self._normalize_reason_choice(self._load_all_reason_rows())
        positive = self._reason_keyword_counts(df[df["choice_polarity"] == "positive"])
        negative = self._reason_keyword_counts(df[df["choice_polarity"] == "negative"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        panels = [
            (axes[0], positive, "最终选择积极一侧：高频原因", "#4C78A8"),
            (axes[1], negative, "最终选择消极一侧：高频原因", "#E45756"),
        ]
        for ax, data, title, color in panels:
            if data.empty:
                ax.text(0.5, 0.5, "暂无足够文本", ha="center", va="center")
                ax.axis("off")
                continue
            data = data.sort_values("count", ascending=True)
            bars = ax.barh(data["keyword"], data["count"], color=color, alpha=0.85)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(axis="x")
            ax.set_xlabel("出现次数")
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=9)
        fig.suptitle("全实验：选积极 vs 选消极 的原因对照图", fontsize=14, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout()
        out = self.final_figures_dir / "全实验_积极消极原因对照图.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def build_enhanced_html(self) -> Path:
        tables = self._load_summary_tables()
        fig_main_static = self._build_main_static_heatmap(tables)
        fig_persona_static = self._build_persona_static_heatmap(tables)
        fig_atm_static = self._build_atmosphere_static_heatmap(tables)
        fig_comb_static = self._build_combined_static_heatmap(tables)
        fig_cross = self._build_cross_experiment_p3_figure(tables)
        fig_heatmap = self._build_combined_agent_heatmap(tables)
        fig_terminal = self._build_terminal_comparison_figure(tables)
        fig_b_vs_combined = self._build_b_vs_combined_dynamic_comparison(tables)
        fig_reason = self._build_reason_choice_comparison()

        def _pick(df: pd.DataFrame, **conds) -> pd.Series:
            mask = pd.Series([True] * len(df))
            for key, value in conds.items():
                mask &= df[key] == value
            return df[mask].iloc[0]

        def _fmt(value: float) -> str:
            return f"{value:.2f}"

        main_static = tables["main_static"]
        main_p3 = tables["main_p3"]
        persona_static = tables["persona_static"]
        persona_p3 = tables["persona_p3"]
        persona_agent = tables["persona_agent"]
        atm_static = tables["atm_static"]
        atm_p3 = tables["atm_p3"]
        comb_static = tables["comb_static"]
        comb_p3 = tables["comb_p3"]
        comb_agent = tables["comb_agent"]
        comb_cell = tables["comb_cell"]

        ms_inf_p1 = _pick(main_static, scenario_name="influenza_a", condition_name="independent")
        ms_inf_p2 = _pick(main_static, scenario_name="influenza_a", condition_name="low_feedback")
        ms_myc_p1 = _pick(main_static, scenario_name="mycoplasma_pneumonia", condition_name="independent")
        ms_myc_p2 = _pick(main_static, scenario_name="mycoplasma_pneumonia", condition_name="low_feedback")
        mp_inf = _pick(main_p3, scenario_name="influenza_a")
        mp_myc = _pick(main_p3, scenario_name="mycoplasma_pneumonia")

        ps_inf_np_p1 = _pick(persona_static, scenario_name="influenza_a", condition_name="independent", persona_name="ordinary_user")
        ps_inf_rs_p1 = _pick(persona_static, scenario_name="influenza_a", condition_name="independent", persona_name="risk_sensitive")
        ps_inf_at_p1 = _pick(persona_static, scenario_name="influenza_a", condition_name="independent", persona_name="authority_trusting")
        ps_myc_np_p1 = _pick(persona_static, scenario_name="mycoplasma_pneumonia", condition_name="independent", persona_name="ordinary_user")
        pp_inf = _pick(persona_p3, scenario_name="influenza_a")
        pp_myc = _pick(persona_p3, scenario_name="mycoplasma_pneumonia")
        pa_inf_rs = _pick(persona_agent, scenario_name="influenza_a", agent_persona_name="risk_sensitive")
        pa_inf_at = _pick(persona_agent, scenario_name="influenza_a", agent_persona_name="authority_trusting")
        pa_inf_np = _pick(persona_agent, scenario_name="influenza_a", agent_persona_name="ordinary_user")

        as_inf_neg = _pick(atm_static, scenario_name="influenza_a", chat_atmosphere="negative_leaning")
        as_inf_none = _pick(atm_static, scenario_name="influenza_a", chat_atmosphere="none")
        as_inf_pos = _pick(atm_static, scenario_name="influenza_a", chat_atmosphere="positive_leaning")
        as_myc_neg = _pick(atm_static, scenario_name="mycoplasma_pneumonia", chat_atmosphere="negative_leaning")
        as_myc_none = _pick(atm_static, scenario_name="mycoplasma_pneumonia", chat_atmosphere="none")
        as_myc_pos = _pick(atm_static, scenario_name="mycoplasma_pneumonia", chat_atmosphere="positive_leaning")
        ap_inf_neg = _pick(atm_p3, scenario_name="influenza_a", chat_atmosphere="negative_leaning")
        ap_inf_pos = _pick(atm_p3, scenario_name="influenza_a", chat_atmosphere="positive_leaning")
        ap_myc_neg = _pick(atm_p3, scenario_name="mycoplasma_pneumonia", chat_atmosphere="negative_leaning")
        ap_myc_pos = _pick(atm_p3, scenario_name="mycoplasma_pneumonia", chat_atmosphere="positive_leaning")

        cs_inf_at_none = _pick(comb_static, scenario_name="influenza_a", persona_name="authority_trusting", chat_atmosphere="none")
        cs_inf_at_neg = _pick(comb_static, scenario_name="influenza_a", persona_name="authority_trusting", chat_atmosphere="negative_leaning")
        cs_inf_np_none = _pick(comb_static, scenario_name="influenza_a", persona_name="ordinary_user", chat_atmosphere="none")
        cs_inf_np_pos = _pick(comb_static, scenario_name="influenza_a", persona_name="ordinary_user", chat_atmosphere="positive_leaning")
        cs_inf_np_neg = _pick(comb_static, scenario_name="influenza_a", persona_name="ordinary_user", chat_atmosphere="negative_leaning")
        cp_inf_none = _pick(comb_p3, scenario_name="influenza_a", chat_atmosphere="none")
        cp_inf_pos = _pick(comb_p3, scenario_name="influenza_a", chat_atmosphere="positive_leaning")
        cp_inf_neg = _pick(comb_p3, scenario_name="influenza_a", chat_atmosphere="negative_leaning")
        cp_myc_none = _pick(comb_p3, scenario_name="mycoplasma_pneumonia", chat_atmosphere="none")
        cp_myc_pos = _pick(comb_p3, scenario_name="mycoplasma_pneumonia", chat_atmosphere="positive_leaning")
        cp_myc_neg = _pick(comb_p3, scenario_name="mycoplasma_pneumonia", chat_atmosphere="negative_leaning")
        ca_inf_rs_pos = _pick(comb_agent, scenario_name="influenza_a", agent_persona_name="risk_sensitive", chat_atmosphere="positive_leaning")
        ca_inf_at_pos = _pick(comb_agent, scenario_name="influenza_a", agent_persona_name="authority_trusting", chat_atmosphere="positive_leaning")
        ca_inf_np_pos = _pick(comb_agent, scenario_name="influenza_a", agent_persona_name="ordinary_user", chat_atmosphere="positive_leaning")

        main_case = tables["main_term"][
            (tables["main_term"]["scenario_name"] == "influenza_a")
            & (tables["main_term"]["repeat_index"] == 3)
        ].iloc[0]
        persona_case = tables["persona_cell"][
            (tables["persona_cell"]["scenario_name"] == "influenza_a")
            & (tables["persona_cell"]["repeat_index"] == 7)
        ].iloc[0]
        comb_case_pos = comb_cell[
            (comb_cell["scenario_name"] == "influenza_a")
            & (comb_cell["chat_atmosphere"] == "positive_leaning")
            & (comb_cell["repeat_index"] == 2)
        ].iloc[0]
        comb_case_neg = comb_cell[
            (comb_cell["scenario_name"] == "influenza_a")
            & (comb_cell["chat_atmosphere"] == "positive_leaning")
            & (comb_cell["repeat_index"] == 8)
        ].iloc[0]

        fig_main_dynamic = self.demo_dir / "data/figures/主实验_P3动态趋势折线图.png"
        fig_persona_dynamic = self.demo_dir / "data/supplements/persona_supplement/figures/figS8_p3_individual_trajectories.png"
        fig_atm_dynamic = self.demo_dir / "data/supplements/atmosphere_supplement/figures/figS7_p3_terminal_atmosphere.png"

        html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>全实验阶段性汇报（按实验展开）</title>
  <style>
    :root {{
      --bg: #f4f1ea; --paper: #fffdf8; --ink: #1f2a37; --muted: #5a6875;
      --line: #d8d1c4; --accent: #8a3b2d; --accent-soft: #f3e7de;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; color:var(--ink); font-family:"PingFang SC","Noto Serif CJK SC","Source Han Serif SC",serif;
      background: radial-gradient(circle at top left, #efe6d8 0, transparent 28%), linear-gradient(180deg, #f6f3ed 0%, #efe8dd 100%); line-height:1.72; }}
    .page {{ max-width:1120px; margin:28px auto 60px; background:var(--paper); border:1px solid var(--line); border-radius:20px; overflow:hidden; box-shadow:0 18px 48px rgba(31,42,55,.08); }}
    .hero {{ padding:42px 48px 28px; background:linear-gradient(135deg,#f2e7da 0%,#fbf8f2 58%,#eef3f2 100%); border-bottom:1px solid var(--line); }}
    .content {{ padding:30px 48px 44px; }}
    h1 {{ margin:0 0 12px; font-size:34px; line-height:1.25; }}
    h2 {{ margin:30px 0 14px; font-size:24px; border-left:4px solid var(--accent); padding-left:10px; }}
    h3 {{ margin:22px 0 10px; font-size:18px; color:#243746; }}
    p {{ margin:10px 0; }}
    ul {{ margin:10px 0 18px 20px; padding:0; }}
    li {{ margin:6px 0; }}
    .lead {{ margin:0; color:var(--muted); font-size:17px; max-width:920px; }}
    .callout {{ margin:16px 0; padding:14px 16px; border:1px solid var(--line); border-left:5px solid var(--accent); border-radius:12px; background:var(--accent-soft); }}
    .figure {{ margin:22px 0 30px; border:1px solid var(--line); border-radius:16px; background:#fff; overflow:hidden; }}
    .figure img {{ display:block; width:100%; height:auto; background:#fff; }}
    .figcap {{ padding:14px 18px 16px; border-top:1px solid var(--line); background:#fcfaf5; }}
    .figcap strong {{ display:block; margin-bottom:6px; }}
    .datatable {{ width:100%; border-collapse:collapse; margin:14px 0 24px; font-size:14px; }}
    .datatable th, .datatable td {{ border:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
    .datatable th {{ background:#f5f1e8; font-weight:700; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>全实验阶段性汇报（按实验展开）</h1>
      <p class="lead">这版汇报不再把所有图混在一起，而是按“主实验 → A实验 → B实验 → 混合实验”的顺序逐段展开。每一段都明确回答三个问题：发现了什么、结论是什么、哪张图能证明。最后再做跨实验对比和 reason 原因展示。</p>
    </section>
    <section class="content">
      <h2>1. 全实验最核心的结论</h2>
      <div class="callout"><strong>一句话总括</strong> 主实验证明 <code>P3</code> 是最强的负向收敛放大器；A实验证明人格差异会显著改变谁更容易被带偏；B实验证明聊天氛围能强力改变静态起点，但动态互动会削弱这种保护；混合实验则说明人格异质性能减缓、但不能完全阻止负向收敛。</div>

      <h2>2. 主实验：平台结构效应</h2>
      <h3>2.1 主要发现</h3>
      <ul>
        <li><strong>P1 / P2</strong> 的结果受事件背景影响很大，说明材料基线并不一致。甲流场景从 <code>{_fmt(ms_inf_p1['mean_nsb'])}</code> 到 <code>{_fmt(ms_inf_p2['mean_nsb'])}</code>，变化不大；支原体肺炎场景则从 <code>{_fmt(ms_myc_p1['mean_nsb'])}</code> 到 <code>{_fmt(ms_myc_p2['mean_nsb'])}</code>，本身就偏负。</li>
        <li><strong>P3</strong> 一进入动态互动，两场景都会明显抬高负向偏向：甲流 <code>{_fmt(mp_inf['mean_nsb'])}</code>，支原体肺炎 <code>{_fmt(mp_myc['mean_nsb'])}</code>。</li>
        <li>主实验的讨论结束得很快，平均只需要 <code>{_fmt(mp_inf['mean_final_timestep'])}</code> 和 <code>{_fmt(mp_myc['mean_final_timestep'])}</code> 个时间步，说明同质讨论场很容易快速形成单边共识。</li>
      </ul>
      <div class="callout"><strong>总体结论</strong> 主实验回答的是“平台结构本身会不会改变选择”。当前结果说明：<code>P2</code> 的静态社会证明增量并不总是稳定单向，但 <code>P3</code> 的动态互动效应非常强，会把讨论快速推向偏负向的高共识状态。</div>
      <div class="figure"><img src="{self._encode_image(fig_main_static)}" alt="主实验场景平台热力图" /><div class="figcap"><strong>图1 主实验：场景 × 平台 平均 NSB 热力图</strong> 这张图把主实验最核心的四个静态结果和两个动态结果都放在一个热力图里，而且格子里直接标了数。它最适合用来开场说明：P1 / P2 的基线受场景影响较大，但真正把整体抬高的是 P3。</div></div>
      <div class="figure"><img src="{self._encode_image(fig_main_dynamic)}" alt="主实验 P3 动态趋势图" /><div class="figcap"><strong>图2 主实验：P3 动态趋势折线图</strong> 这张图是主实验的过程证据。它说明群体共识和负向偏向主要在前几个时间步快速抬升，后段波动更多是因为剩余轮次变少，因此汇报时应重点看前 3 到 4 个时间步。</div></div>

      <h2>3. A实验：人格异质性</h2>
      <h3>3.1 主要发现</h3>
      <ul>
        <li>静态阶段人格差异非常强：在甲流场景的 <code>P1</code> 中，<code>authority_trusting = {_fmt(ps_inf_at_p1['mean_nsb'])}</code>，<code>ordinary_user = {_fmt(ps_inf_np_p1['mean_nsb'])}</code>，<code>risk_sensitive = {_fmt(ps_inf_rs_p1['mean_nsb'])}</code>。</li>
        <li>进入 <code>P3</code> 后，整体均值被拉回中间：甲流 <code>{_fmt(pp_inf['mean_nsb'])}</code>、支原体肺炎 <code>{_fmt(pp_myc['mean_nsb'])}</code>，同时平均讨论时长上升到 <code>{_fmt(pp_inf['mean_final_timestep'])}</code> 和 <code>{_fmt(pp_myc['mean_final_timestep'])}</code>。</li>
        <li>按个体人格拆开后，差异并没有消失：甲流场景中，<code>risk_sensitive = {_fmt(pa_inf_rs['mean_nsb'])}</code>、<code>ordinary_user = {_fmt(pa_inf_np['mean_nsb'])}</code>、<code>authority_trusting = {_fmt(pa_inf_at['mean_nsb'])}</code>。</li>
      </ul>
      <div class="callout"><strong>总体结论</strong> A实验说明人格差异不是小修饰，而是决定“谁更容易被动态讨论拉动”的核心变量。混合人格会显著拉长讨论，但不会抹掉人格之间的方向差异。</div>
      <div class="figure"><img src="{self._encode_image(fig_persona_static)}" alt="A实验人格静态热力图" /><div class="figcap"><strong>图3 A实验：人格 × 静态平台 平均 NSB 热力图</strong> 这张图把 A实验静态阶段的关键数值直接标在格子里。最重要的模式是：风险敏感型更容易被风险证据打动，权威信任型更看重正式来源，普通用户型体现普通社交平台用户的基线可塑性。</div></div>
      <div class="figure"><img src="{self._encode_image(fig_persona_dynamic)}" alt="A实验 P3 个体轨迹图" /><div class="figcap"><strong>图4 A实验：P3 个体轨迹图</strong> 这张图说明人格差异并不只存在于均值表里，而是在动态讨论的个体轨迹上持续存在。也正因为场内始终保留不同人格的声音，A实验的轮次才会比主实验长得多。</div></div>

      <h2>4. B实验：聊天氛围效应</h2>
      <h3>4.1 主要发现</h3>
      <ul>
        <li>静态阶段三档氛围几乎把方向完全拉开。甲流场景中：负向氛围 <code>{_fmt(as_inf_neg['mean_nsb'])}</code>、平衡氛围 <code>{_fmt(as_inf_none['mean_nsb'])}</code>、正向氛围 <code>{_fmt(as_inf_pos['mean_nsb'])}</code>。</li>
        <li>支原体肺炎场景也一样：负向氛围 <code>{_fmt(as_myc_neg['mean_nsb'])}</code>、平衡氛围 <code>{_fmt(as_myc_none['mean_nsb'])}</code>、正向氛围 <code>{_fmt(as_myc_pos['mean_nsb'])}</code>。</li>
        <li>但进入 <code>P3</code> 后，正向起点的保护作用明显减弱。比如甲流中，负向起点 <code>{_fmt(ap_inf_neg['mean_nsb'])}</code>，正向起点也达到 <code>{_fmt(ap_inf_pos['mean_nsb'])}</code>；支原体肺炎中，负向起点 <code>{_fmt(ap_myc_neg['mean_nsb'])}</code>，正向起点仍有 <code>{_fmt(ap_myc_pos['mean_nsb'])}</code>。</li>
      </ul>
      <div class="callout"><strong>总体结论</strong> B实验说明：聊天氛围对静态选择非常有效，但一旦进入 P3，高阶互动的力量比静态氛围更强，初始正向氛围无法稳定守住终局方向。</div>
      <div class="figure"><img src="{self._encode_image(fig_atm_static)}" alt="B实验聊天氛围热力图" /><div class="figcap"><strong>图5 B实验：场景 × 聊天气氛 平均 NSB 热力图</strong> 这张图把三档氛围在两个场景中的数值直接写在格子里。它最适合证明氛围操纵本身是成功的：负向氛围推负，正向氛围推正，平衡氛围落在中间。</div></div>
      <div class="figure"><img src="{self._encode_image(fig_atm_dynamic)}" alt="B实验终局氛围图" /><div class="figcap"><strong>图6 B实验：P3 终局社群氛围图</strong> 这张图提供动态阶段的终局证据。它说明多数轮次最终还是回到偏负向的社群氛围，正向起点并不能保证终局仍然正向。</div></div>

      <h2>5. 混合实验：人格 × 氛围</h2>
      <h3>5.1 主要发现</h3>
      <ul>
        <li>静态阶段出现非常典型的交互：在甲流场景中，<code>authority_trusting</code> 在平衡氛围下是 <code>{_fmt(cs_inf_at_none['mean_nsb'])}</code>，但在负向氛围下会翻到 <code>{_fmt(cs_inf_at_neg['mean_nsb'])}</code>；<code>ordinary_user</code> 从正向氛围的 <code>{_fmt(cs_inf_np_pos['mean_nsb'])}</code>，到平衡氛围的 <code>{_fmt(cs_inf_np_none['mean_nsb'])}</code>，再到负向氛围的 <code>{_fmt(cs_inf_np_neg['mean_nsb'])}</code>。</li>
        <li>进入 <code>P3</code> 后，混合实验仍会整体偏负，但明显比 B实验更慢、更弱。甲流中：负向起点 <code>{_fmt(cp_inf_neg['mean_nsb'])}</code>，平衡起点 <code>{_fmt(cp_inf_none['mean_nsb'])}</code>，正向起点 <code>{_fmt(cp_inf_pos['mean_nsb'])}</code>；平均结束时间步分别是 <code>{_fmt(cp_inf_neg['mean_final_timestep'])}</code>、<code>{_fmt(cp_inf_none['mean_final_timestep'])}</code>、<code>{_fmt(cp_inf_pos['mean_final_timestep'])}</code>。</li>
        <li>不同人格仍然保留鲜明分化：在甲流 + 正向氛围的 P3 中，<code>risk_sensitive = {_fmt(ca_inf_rs_pos['mean_nsb'])}</code>，<code>ordinary_user = {_fmt(ca_inf_np_pos['mean_nsb'])}</code>，<code>authority_trusting = {_fmt(ca_inf_at_pos['mean_nsb'])}</code>。</li>
      </ul>
      <div class="callout"><strong>总体结论</strong> 混合实验说明：人格与氛围不是彼此独立叠加，而是共同决定讨论从哪里出发、会不会快速收敛、以及最后会偏到什么程度。人格异质性确实能减缓收敛，但不能完全阻止负向终局。</div>
      <div class="figure"><img src="{self._encode_image(fig_comb_static)}" alt="混合实验静态热力图" /><div class="figcap"><strong>图7 混合实验：静态阶段人格 × 氛围 平均 NSB 热力图</strong> 这张图最能证明混合实验的价值，而且现在每个格子里都直接有数值。它不是简单说哪一类人格更高，而是展示同一种氛围面对不同人格会出现完全不同的模式，因此这里真正成立的是强交互，而不是单一主效应。</div></div>
      <div class="figure"><img src="{self._encode_image(fig_heatmap)}" alt="混合实验动态交叉热力图" /><div class="figcap"><strong>图8 混合实验：P3 人格 × 初始氛围交叉热力图</strong> 这张图把动态阶段的人格差异写得更清楚：风险敏感型几乎不受正向氛围保护，权威信任型最能维持在报道1一侧，普通用户型则体现普通社交平台用户被讨论过程拉动的中间地带。</div></div>

      <h2>6. 所有实验放在一起，应该怎么比</h2>
      <div class="figure"><img src="{self._encode_image(fig_cross)}" alt="跨实验 P3 对比图" /><div class="figcap"><strong>图9 跨实验 P3 收敛与讨论时长对比</strong> 这张图适合做总汇报主图。它把主实验、A实验、B实验和混合实验放到同一坐标系里比较，一眼就能看出：主实验和 B实验更容易快速偏负收敛，而 A实验和混合实验会把讨论拉长。</div></div>
      <p>这张图回答的是“不同实验到底改变了什么”。主实验告诉我们平台结构本身会放大负向收敛；A实验告诉我们人格异质性会显著拖长讨论；B实验告诉我们静态正向氛围在 P3 中并不稳；混合实验则说明人格和氛围同时存在时，讨论仍然偏负，但收敛强度被明显拉低。</p>

      <div class="figure"><img src="{self._encode_image(fig_b_vs_combined)}" alt="B与混合实验动态阶段关键差异图" /><div class="figcap"><strong>图10 B实验与混合实验：动态阶段关键差异图</strong> 这张图才是 B实验和混合实验最应该放在一起比较的主图。上排看平均 NSB，下排看平均结束时间步。它直接告诉我们：混合实验有没有把 B实验那种“强负向收敛”拉回一点，以及人格异质性有没有把讨论拖长。</div></div>
      <p>从这张图可以看出，B实验的正向起点进入 P3 后仍然整体偏负；而混合实验在 <code>positive_leaning</code> 和 <code>none</code> 条件下的平均 NSB 更低、讨论时长更长。也就是说，<strong>人格异质性改变的首先是过程强度，其次才是终局方向</strong>。</p>

      <div class="figure"><img src="{self._encode_image(fig_terminal)}" alt="B与混合实验终局氛围对照图" /><div class="figcap"><strong>图11 B实验与混合实验：终局氛围对照图</strong> 这张图作为补充证据保留。它说明即使混合实验把讨论拖长、把均值拉回中间，很多轮次最终仍然会走向负向终局，所以异质性更像“缓冲”和“减速”，而不是彻底逆转。</div></div>

      <h2>7. 选积极的人和选消极的人，原因分别是什么</h2>
      <div class="figure"><img src="{self._encode_image(fig_reason)}" alt="积极消极原因对照图" /><div class="figcap"><strong>图12 全实验：选积极 vs 选消极 的原因对照图</strong> 左边是最终选择积极一侧时最常出现的原因词，右边是最终选择消极一侧时最常出现的原因词。这张图是从主实验、A实验、B实验、混合实验的全部结果里重新抓取并统一统计后的对照条形图，不再混用旧的 report_1 / report_2 口径。</div></div>
      <p>从全实验汇总统计看，选积极的人最常抓住的词是 <strong>恢复（1309）</strong>、<strong>医院（1175）</strong>、<strong>评论（1008）</strong>、<strong>可控（716）</strong>、<strong>疾控（482）</strong>、<strong>恢复较快（421）</strong>；这说明他们更容易被“情况可控、恢复较快、医疗系统正在应对”的线索打动。</p>
      <p>相对地，选消极的人最常抓住的词是 <strong>恢复（2887）</strong>、<strong>评论（1608）</strong>、<strong>风险（1584）</strong>、<strong>反复（1410）</strong>、<strong>门诊（1075）</strong>、<strong>身边（975）</strong>、<strong>咳嗽（596）</strong>、<strong>被低估（570）</strong>、<strong>恢复慢（526）</strong>、<strong>高热（501）</strong>；这说明他们更容易被“恢复拖长、风险偏高、身边经验和门诊压力”这类具体风险线索吸引。</p>
      <p>这里有一个值得强调的点：<strong>“恢复”</strong> 这个词两边都高频出现，但它在两类选择中的语义方向并不一样。选积极的人更多是把它理解成“恢复较快、整体可控”；选消极的人则更常把它理解成“恢复拖长、反复不好”。这正说明模型不是机械复制词，而是在不同选择方向下抓住了不同的解释框架。</p>

      <h2>8. 代表性案例分析</h2>
      <h3>案例1：主实验中的快速负向收敛</h3>
      <p>主实验 `influenza_a, repeat=3` 只用了 <strong>{int(main_case['final_timestep'])}</strong> 个时间步就结束，终局为 <strong>{main_case['terminal_group_tendency']}</strong> / <strong>{main_case['terminal_community_atmosphere']}</strong>。这类案例说明：在没有额外异质性干预时，P3 可以非常快地形成单边共识。</p>

      <h3>案例2：A实验中的长讨论但仍偏负</h3>
      <p>A实验 `influenza_a, repeat=7` 一直拖到 <strong>{int(persona_case['final_timestep'])}</strong> 个时间步，终局仍为 <strong>{persona_case['terminal_group_tendency']}</strong>。这说明人格异质性会拉长讨论，但不一定改变主方向，它更像是延缓单边收敛。</p>

      <h3>案例3：混合实验中同一正向起点却走向不同终局</h3>
      <p>在混合实验 `influenza_a + positive_leaning` 条件下，`repeat=2` 最终停在 <strong>{comb_case_pos['terminal_group_tendency']}</strong> / <strong>{comb_case_pos['terminal_community_atmosphere']}</strong>；而 `repeat=8` 则走向 <strong>{comb_case_neg['terminal_group_tendency']}</strong> / <strong>{comb_case_neg['terminal_community_atmosphere']}</strong>。这说明当人格与氛围同时进入讨论场时，互动路径依赖会显著增强，同样的起点并不保证同样的终局。</p>

      <h2>9. 论文修改建议</h2>
      <ul>
        <li><strong>结果章节顺序</strong>：建议严格按“主实验 → A实验 → B实验 → 混合实验”的顺序展开。这样逻辑最清楚：先讲平台主效应，再讲边界条件，最后讲交互。</li>
        <li><strong>主文图配置</strong>：主文优先保留 5 张图：主实验平台对比、主实验 P3 动态趋势、A实验人格异质性、B实验氛围效应、混合实验人格 × 氛围热力图。</li>
        <li><strong>跨实验比较写法</strong>：B实验和混合实验不要只比终局氛围，应该先比 <code>P3</code> 的平均 NSB 和平均结束时间步，再用终局图做补充。</li>
        <li><strong>案例分析写法</strong>：结果或讨论部分建议保留三类案例：主实验的快速高共识、A实验的长讨论、混合实验的同一起点分化终局。</li>
        <li><strong>reason 展示</strong>：当前主文或汇报里优先展示“选积极 vs 选消极”的原因图；人格、氛围维度的 reason 细分图可以放到补充材料。</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""

        out = self.demo_dir / "全实验阶段性汇报_按实验展开.html"
        out.write_text(html)
        legacy = self.demo_dir / "全实验结果与论文修改建议_增强版.html"
        legacy.write_text(html)
        return out


def main() -> None:
    demo_dir = Path(__file__).resolve().parents[2] / "demo"
    builder = FinalReportBuilder(demo_dir)
    out = builder.build_enhanced_html()
    print(out)


if __name__ == "__main__":
    main()
