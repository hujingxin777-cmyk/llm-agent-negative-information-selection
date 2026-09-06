from __future__ import annotations

"""将 P3 事件明细导出为可阅读的线程式 Markdown 报告。"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir


class ThreadReportBuilder:
    """构建并导出每个实验单元的线程报告。"""

    def __init__(self, output_dir: Path) -> None:
        # 输出目录结构约定：
        # - results/: 读取 p3_results.csv、p3_cell_summary.csv
        # - thread_reports/: 写出每个单元的 Markdown 线程
        self.output_dir = output_dir
        self.results_dir = output_dir / "results"
        self.report_dir = output_dir / "thread_reports"

    def export_reports(self) -> list[str]:
        """导出所有线程报告，并返回相对输出目录的文件路径列表。"""
        p3_path = self.results_dir / "p3_results.csv"
        cell_path = self.results_dir / "p3_cell_summary.csv"
        if not p3_path.exists() or not cell_path.exists():
            return []

        p3_df = pd.read_csv(p3_path)
        cell_df = pd.read_csv(cell_path)
        if p3_df.empty or cell_df.empty:
            return []

        ensure_dir(self.report_dir)
        exported: list[str] = []
        index_rows: list[str] = [
            "# P3 论坛式线程导出",
            "",
            "下面每个链接对应一个 `P3` 单元，可直接像看论坛帖一样查看最终讨论结果。",
            "",
        ]

        group_cols = [
            "experiment_family",
            "scenario_name",
            "condition_name",
            "persona_name",
            "chat_atmosphere",
            "repeat_index",
        ]
        # 逐个单元过滤出对应事件，按时间步与事件编号排序后渲染。
        for _, cell in cell_df.sort_values(group_cols).iterrows():
            mask = pd.Series(True, index=p3_df.index)
            for col in group_cols:
                mask &= p3_df[col].astype(str) == str(cell[col])
            thread_df = p3_df[mask].copy().sort_values(["timestep", "event_id"])
            if thread_df.empty:
                continue

            filename = self._file_name(cell)
            path = self.report_dir / filename
            path.write_text(self._render_thread_markdown(cell.to_dict(), thread_df), encoding="utf-8")
            exported.append(str(path.relative_to(self.output_dir)))
            title = self._thread_title(cell.to_dict())
            index_rows.append(f"- [{title}](./{filename})")

        (self.report_dir / "README.md").write_text("\n".join(index_rows) + "\n", encoding="utf-8")
        exported.insert(0, str((self.report_dir / "README.md").relative_to(self.output_dir)))
        return exported

    def _file_name(self, cell: dict) -> str:
        """为每个实验单元生成可落盘、可跨平台使用的文件名。"""
        scenario = self._slug(str(cell["scenario_name"]))
        atmosphere = self._slug(str(cell["chat_atmosphere"]))
        repeat = self._slug(str(cell["repeat_index"]))
        family = self._slug(str(cell["experiment_family"]))
        return f"{family}_{scenario}_{atmosphere}_repeat{repeat}.md"

    def _thread_title(self, cell: dict) -> str:
        """报告标题（保留原始文本，便于阅读）。"""
        return (
            f"{cell['experiment_family']} | {cell['scenario_name']} | "
            f"{cell['chat_atmosphere']} | repeat={cell['repeat_index']}"
        )

    def _render_thread_markdown(self, cell: dict, thread_df: pd.DataFrame) -> str:
        """将单元数据渲染为 Markdown。

        - 公开动作：post/reply/like/repost
        - 静默动作：observe（仅内部判断）
        """
        public_rows = thread_df[thread_df["action_type"].isin(["post", "reply", "like", "repost"])].copy()
        silent_rows = thread_df[thread_df["action_type"] == "observe"].copy()

        lines: list[str] = [
            f"# {self._thread_title(cell)}",
            "",
            "## 单元摘要",
            "",
            f"- 终止原因：`{cell.get('stop_reason', '')}`",
            f"- 最终时间步：`{cell.get('final_timestep', '')}`",
            f"- `mean_nsb = {cell.get('mean_nsb', '')}`",
            f"- `final_nsb = {cell.get('final_nsb', '')}`",
            f"- `mean_gcr = {cell.get('mean_gcr', '')}`",
            f"- `mean_ssa = {cell.get('mean_ssa', '')}`",
            "",
            "## 公开讨论线程",
            "",
        ]

        if public_rows.empty:
            lines.append("这轮没有公开评论、回复、点赞或转发事件。")
        else:
            children: dict[str, list[dict]] = defaultdict(list)
            by_id: dict[str, dict] = {}
            roots: list[dict] = []
            for row in public_rows.to_dict(orient="records"):
                by_id[str(row["event_id"])] = row
            for row in public_rows.to_dict(orient="records"):
                reply_to = str(row.get("reply_to") or "").strip()
                if reply_to and reply_to in by_id:
                    children[reply_to].append(row)
                else:
                    roots.append(row)

            for node in roots:
                self._append_event(lines, node, children, depth=0, visited=set())

        lines.extend(
            [
                "",
                "## 未公开但有内部判断的观察记录",
                "",
            ]
        )

        if silent_rows.empty:
            lines.append("这轮没有 `observe` 记录。")
        else:
            for row in silent_rows.to_dict(orient="records"):
                lines.extend(
                    [
                        f"### {row['event_id']} | t={row['timestep']} | {row['actor_id']}",
                        f"- 动作：`observe`",
                        f"- 最终倾向：`{row.get('final_choice', '')}`",
                        f"- 可信判断：`{row.get('credible_choice', '')}`",
                        f"- 点赞倾向：`{row.get('like_choice', '')}`",
                        f"- 原因：{self._safe_text(row.get('reason')) or '（空）'}",
                        "",
                    ]
                )

        return "\n".join(lines).strip() + "\n"

    def _append_event(
        self,
        lines: list[str],
        node: dict,
        children: dict[str, list[dict]],
        depth: int,
        visited: set[str],
    ) -> None:
        """深度优先写入事件树；通过 `visited` 防止异常数据导致循环递归。"""
        node_id = str(node.get("event_id", ""))
        if node_id and node_id in visited:
            lines.append(f"{'  ' * depth}- **[WARN]** 检测到循环引用，event_id=`{node_id}`，已截断子树。")
            lines.append("")
            return

        local_visited = set(visited)
        if node_id:
            local_visited.add(node_id)

        indent = "  " * depth
        action = str(node.get("action_type", ""))
        badges = {
            "post": "POST",
            "reply": "REPLY",
            "like": "LIKE",
            "repost": "REPOST",
        }
        body = self._safe_text(node.get("content"))
        if action == "like":
            body = body or "点赞了上面的内容"
        elif action == "repost":
            body = body or "转发了上面的内容"
        elif not body:
            body = "（无公开文本）"

        lines.append(f"{indent}- **[{badges.get(action, action.upper())}]** `{node['event_id']}` by `{node['actor_id']}`")
        lines.append(f"{indent}  {body}")
        lines.append(
            f"{indent}  倾向：`final={node.get('final_choice','')}` / `credible={node.get('credible_choice','')}` / `like={node.get('like_choice','')}`"
        )
        reason = self._safe_text(node.get("reason"))
        if reason:
            lines.append(f"{indent}  原因：{reason}")
        lines.append("")

        for child in sorted(children.get(str(node["event_id"]), []), key=lambda item: (item["timestep"], item["event_id"])):
            self._append_event(lines, child, children, depth + 1, local_visited)

    @staticmethod
    def _slug(value: str) -> str:
        """将文本转为安全片段：仅保留字母/数字/下划线/短横线。"""
        safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", value.strip())
        return safe.strip("_") or "na"

    @staticmethod
    def _safe_text(value: object) -> str:
        """将空值/NaN 统一转换为空字符串，避免渲染异常。"""
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).strip()
