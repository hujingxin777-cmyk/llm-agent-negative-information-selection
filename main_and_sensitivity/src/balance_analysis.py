from __future__ import annotations

"""P1 材料平衡性审计。"""

from pathlib import Path

import pandas as pd

from .io_utils import write_csv


class BalanceAnalyzer:
    """导出 P1 材料平衡检查表，快速判断是否存在系统性偏负。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.results_dir = output_dir / "results"

    def export_p1_balance_table(self) -> list[str]:
        static_path = self.results_dir / "static_results.csv"
        if not static_path.exists():
            return []
        df = pd.read_csv(static_path)
        if df.empty or "final_choice" not in df.columns:
            return []
        subset = df[
            (df["condition_name"] == "independent")
            & (df["success"].astype(str).str.lower() == "true")
        ].copy()
        if subset.empty:
            return []
        summary = (
            subset.groupby(["experiment_family", "scenario_name", "persona_name"], as_index=False)
            .agg(
                n_obs=("final_choice", "size"),
                negative_share=("nsb", "mean"),
                positive_share=("final_choice", lambda s: (s == "positive").mean()),
            )
        )
        path = self.results_dir / "p1_balance_audit.csv"
        write_csv(path, summary.to_dict(orient="records"))
        return [path.name]
