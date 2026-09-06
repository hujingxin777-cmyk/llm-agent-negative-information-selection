from __future__ import annotations

"""结果打包导出：摘要表、图表与线程报告。"""

import shutil
from pathlib import Path

import pandas as pd

from .balance_analysis import BalanceAnalyzer
from .io_utils import ensure_dir, write_csv, write_json
from .reason_analysis import ReasonAnalyzer
from .thread_report import ThreadReportBuilder


class ResultExporter:
    """把结果表、设计清单和图表打成一个轻量导出包。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.results_dir = output_dir / "results"
        self.figures_dir = output_dir / "figures"
        self.plan_dir = output_dir / "plan"
        self.export_dir = output_dir / "export_bundle"

    def export_bundle(self) -> list[str]:
        """构建导出包并返回导出的文件名列表。"""
        ensure_dir(self.export_dir)
        copied: list[str] = []
        ReasonAnalyzer(self.output_dir).export_reason_tables()
        BalanceAnalyzer(self.output_dir).export_p1_balance_table()
        thread_exports = ThreadReportBuilder(self.output_dir).export_reports()
        copied.extend(Path(item).name for item in thread_exports)
        for source in [
            self.plan_dir / "design_summary.csv",
            self.plan_dir / "reuse_aliases.csv",
            self.results_dir / "static_summary.csv",
            self.results_dir / "p3_summary.csv",
            self.results_dir / "p3_cell_summary.csv",
            self.results_dir / "p3_prototype_panel.csv",
            self.results_dir / "p3_prototype_panel.jsonl",
            self.results_dir / "p3_agent_persona_summary.csv",
            self.results_dir / "static_reason_keywords.csv",
            self.results_dir / "p3_reason_keywords.csv",
            self.results_dir / "p1_balance_audit.csv",
        ]:
            if source.exists():
                target = self.export_dir / source.name
                shutil.copy2(source, target)
                copied.append(target.name)

        for pattern in ("*.png", "*_data.csv"):
            for source in sorted(self.figures_dir.glob(pattern)):
                target = self.export_dir / source.name
                shutil.copy2(source, target)
                copied.append(target.name)

        thread_dir = self.output_dir / "thread_reports"
        if thread_dir.exists():
            target_dir = self.export_dir / "thread_reports"
            ensure_dir(target_dir)
            for source in sorted(thread_dir.glob("*.md")):
                target = target_dir / source.name
                shutil.copy2(source, target)
                copied.append(str(target.relative_to(self.export_dir)))

        summary_rows = self._build_export_summary()
        if summary_rows:
            write_csv(self.export_dir / "export_summary.csv", summary_rows)
            copied.append("export_summary.csv")

        write_json(
            self.export_dir / "manifest.json",
            {
                "output_dir": str(self.output_dir),
                "exported_files": copied,
            },
        )
        copied.append("manifest.json")
        return copied

    def _build_export_summary(self) -> list[dict[str, object]]:
        """生成导出文件摘要（行数 + 字段）。"""
        rows: list[dict[str, object]] = []
        for file_name in (
            "static_summary.csv",
            "p3_summary.csv",
            "p3_cell_summary.csv",
            "p3_prototype_panel.csv",
            "p3_agent_persona_summary.csv",
            "p1_balance_audit.csv",
        ):
            path = self.results_dir / file_name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            rows.append(
                {
                    "source_file": file_name,
                    "n_rows": len(df),
                    "columns": ", ".join(df.columns.astype(str).tolist()),
                }
            )
        return rows
