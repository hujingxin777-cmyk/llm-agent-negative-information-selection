from __future__ import annotations

"""理由文本关键词统计导出。"""

from collections import Counter
from pathlib import Path

import pandas as pd

from .io_utils import write_csv


REASON_KEYWORDS = [
    "门诊",
    "拥挤",
    "排队",
    "恢复",
    "恢复慢",
    "风险",
    "被低估",
    "并发症",
    "正式通报",
    "医院",
    "疾控",
    "专家",
    "可控",
    "全面失控",
    "真实情况",
]


class ReasonAnalyzer:
    """按条件与最终倾向统计 reason 中的高频关键词。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.results_dir = output_dir / "results"

    def export_reason_tables(self) -> list[str]:
        """导出 static 与 P3 的关键词计数表。"""
        exported: list[str] = []
        static_path = self.results_dir / "static_results.csv"
        if static_path.exists():
            df = pd.read_csv(static_path)
            rows = self._build_reason_keyword_rows(df, source_name="static")
            if rows:
                path = self.results_dir / "static_reason_keywords.csv"
                write_csv(path, rows)
                exported.append(path.name)

        p3_path = self.results_dir / "p3_results.csv"
        if p3_path.exists():
            df = pd.read_csv(p3_path)
            rows = self._build_reason_keyword_rows(df, source_name="p3")
            if rows:
                path = self.results_dir / "p3_reason_keywords.csv"
                write_csv(path, rows)
                exported.append(path.name)
        return exported

    def _build_reason_keyword_rows(self, df: pd.DataFrame, source_name: str) -> list[dict[str, object]]:
        """构建关键词计数明细行。"""
        if df.empty or "reason" not in df.columns:
            return []
        valid = df.dropna(subset=["reason"]).copy()
        if valid.empty:
            return []
        rows: list[dict[str, object]] = []
        for (condition_name, final_choice), group in valid.groupby(["condition_name", "final_choice"]):
            counter = Counter()
            for reason in group["reason"].astype(str):
                for keyword in REASON_KEYWORDS:
                    if keyword in reason:
                        counter[keyword] += 1
            for keyword, count in counter.most_common():
                rows.append(
                    {
                        "source_name": source_name,
                        "condition_name": condition_name,
                        "final_choice": final_choice,
                        "keyword": keyword,
                        "count": count,
                    }
                )
        return rows
