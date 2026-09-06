from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = [
    "p3_summary.csv",
    "p3_cell_summary.csv",
    "p3_agent_panel.csv",
    "p3_prototype_panel.csv",
    "p3_results.csv",
    "p3_timestep_summary.csv",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", default="full", choices=["full", "reply_chain", "post_only", "like_channel"])
    parser.add_argument("--min-reply-ratio", type=float, default=0.0)
    args = parser.parse_args()

    result_dir = args.output_dir / "results"
    issues: list[str] = []
    missing = [name for name in REQUIRED if not (result_dir / name).exists()]
    if missing:
        issues.append(f"missing result files: {missing}")

    action_counts: dict[str, int] = {}
    seed_dynamic = seed_targets = failed_rows = 0
    if not missing:
        event_rows = rows(result_dir / "p3_results.csv")
        for row in event_rows:
            action = row.get("action_type", "")
            action_counts[action] = action_counts.get(action, 0) + 1
            seed_dynamic += int(row.get("event_id", "").startswith("seed_"))
            seed_targets += int(action in {"reply", "like", "repost"} and row.get("reply_to", "").startswith("seed_"))
            failed_rows += int(row.get("success", "").lower() not in {"true", "1"})

        if seed_dynamic:
            issues.append(f"seed rows mixed into dynamic events: {seed_dynamic}")
        if seed_targets:
            issues.append(f"dynamic actions target seed: {seed_targets}")
        if failed_rows:
            issues.append(f"failed model rows: {failed_rows}")

        forbidden = {
            "reply_chain": {"post", "like", "repost"},
            "post_only": {"reply", "like", "repost"},
            "like_channel": {"post", "reply", "repost"},
        }.get(args.mode, set())
        # Every ablation builds floors at t=1; only enforce the channel after t=1.
        if forbidden:
            illegal = sum(
                row.get("action_type") in forbidden and int(row.get("timestep", "0")) >= 2
                for row in event_rows
            )
            if illegal:
                issues.append(f"forbidden post-t1 actions for {args.mode}: {illegal}")

        total = len(event_rows)
        reply_ratio = action_counts.get("reply", 0) / total if total else 0.0
        if reply_ratio < args.min_reply_ratio:
            issues.append(f"reply ratio {reply_ratio:.4f} below {args.min_reply_ratio:.4f}")

        early = sum(
            row.get("stop_reason") == "high_consensus" and int(float(row.get("stop_timestep") or 0)) < 5
            for row in rows(result_dir / "p3_cell_summary.csv")
        )
        if early:
            issues.append(f"high_consensus before timestep 5: {early}")
    else:
        reply_ratio = 0.0
        early = 0

    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "passed": not issues,
        "issues": issues,
        "action_counts": action_counts,
        "reply_ratio": reply_ratio,
        "seed_dynamic_rows": seed_dynamic,
        "seed_target_rows": seed_targets,
        "failed_model_rows": failed_rows,
        "early_high_consensus": early,
    }
    report_path = args.output_dir / "stage_check_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
