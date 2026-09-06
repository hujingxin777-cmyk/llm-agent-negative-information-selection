from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_RESULTS = (
    "static_results.csv",
    "static_summary.csv",
    "p3_results.csv",
    "p3_summary.csv",
    "p3_cell_summary.csv",
    "p3_agent_panel.csv",
    "p3_prototype_panel.csv",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_run_started_at(output_dir: Path) -> str | None:
    """Return the start of the latest checkpoint-resume attempt.

    Request logs are append-only across resumes. Using the latest progress
    trackers keeps historical failures auditable without making a successful
    recovery fail its final gate.
    """
    started_at: list[str] = []
    for name in ("static_progress.json", "p3_progress.json"):
        path = output_dir / "progress" / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("started_at")
        if value:
            started_at.append(str(value))
    return min(started_at) if started_at else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--min-reply-ratio", type=float, default=0.08)
    args = parser.parse_args()

    output_dir = args.output_dir
    result_dir = output_dir / "results"
    issues: list[str] = []
    missing = [name for name in REQUIRED_RESULTS if not (result_dir / name).exists()]
    if missing:
        issues.append(f"missing result files: {missing}")

    seed_dynamic = seed_targets = failed_rows = early_consensus = 0
    action_counts: dict[str, int] = {}
    if not missing:
        rows = read_csv(result_dir / "p3_results.csv")
        for row in rows:
            action = row.get("action_type", "")
            action_counts[action] = action_counts.get(action, 0) + 1
            seed_dynamic += int(row.get("event_id", "").startswith("seed_"))
            seed_targets += int(
                action in {"reply", "like", "repost"} and row.get("reply_to", "").startswith("seed_")
            )
            failed_rows += int(row.get("success", "").lower() not in {"true", "1"})
        early_consensus = sum(
            row.get("stop_reason") == "high_consensus" and int(float(row.get("stop_timestep") or 0)) < 5
            for row in read_csv(result_dir / "p3_cell_summary.csv")
        )
        reply_ratio = action_counts.get("reply", 0) / len(rows) if rows else 0.0
        if reply_ratio < args.min_reply_ratio:
            issues.append(f"reply ratio {reply_ratio:.4f} below {args.min_reply_ratio:.4f}")
    else:
        reply_ratio = 0.0

    if seed_dynamic:
        issues.append(f"seed rows mixed into dynamic events: {seed_dynamic}")
    if seed_targets:
        issues.append(f"dynamic actions target seed: {seed_targets}")
    if failed_rows:
        issues.append(f"failed model rows: {failed_rows}")
    if early_consensus:
        issues.append(f"high_consensus before timestep 5: {early_consensus}")

    log_rows = read_jsonl(output_dir / "progress" / "model_request_log.jsonl")
    request_window_started_at = latest_run_started_at(output_dir)
    current_log_rows = [
        row
        for row in log_rows
        if request_window_started_at is None or str(row.get("started_at") or "") >= request_window_started_at
    ]
    all_errors = [row for row in log_rows if row.get("phase") in {"error", "rate_limit"}]
    all_successes = [row for row in log_rows if row.get("phase") == "success"]
    errors = [row for row in current_log_rows if row.get("phase") in {"error", "rate_limit"}]
    successes = [row for row in current_log_rows if row.get("phase") == "success"]
    # A checkpoint-only validation run makes no fresh API calls. In that case,
    # use prior successful requests solely to verify the resolved model name;
    # current-window errors and final result rows still determine the gate.
    model_verification_successes = successes or all_successes
    resolved_models = sorted({
        str(row.get("resolved_model_name"))
        for row in model_verification_successes
        if row.get("resolved_model_name")
    })
    if errors:
        issues.append(f"model request errors: {len(errors)}")
    if not all_successes:
        issues.append("no successful model requests logged")
    if resolved_models and not any(args.expected_model in model for model in resolved_models):
        issues.append(f"resolved model mismatch: {resolved_models}")

    summary = {
        "output_dir": str(output_dir.resolve()),
        "expected_model": args.expected_model,
        "resolved_models": resolved_models,
        "passed": not issues,
        "issues": issues,
        "action_counts": action_counts,
        "reply_ratio": reply_ratio,
        "seed_dynamic_rows": seed_dynamic,
        "seed_target_rows": seed_targets,
        "failed_model_rows": failed_rows,
        "early_high_consensus": early_consensus,
        "request_successes": len(successes),
        "request_errors": len(errors),
        "historical_request_successes": len(all_successes) - len(successes),
        "historical_request_errors": len(all_errors) - len(errors),
        "request_window_started_at": request_window_started_at,
        "checkpoint_only_validation": not successes and bool(all_successes),
    }
    path = output_dir / "model_stage_check_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
