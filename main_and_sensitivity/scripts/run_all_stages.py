from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT.parent / "outputs"
ENV_FILE = ROOT / ".env"
STATE_PATH = OUTPUT_ROOT / "full_pipeline_state_2026-07-13.json"
LOG_PATH = OUTPUT_ROOT / "full_pipeline_2026-07-13.log"

STAGES = [
    ("main_probe", "configs/main_300_probe.yaml", "p3", "../outputs/main_300_probe", "full", 0.08),
    ("main_formal_30", "configs/main_300_formal_r1_30.yaml", "all", "../outputs/main_300_formal_r1_30", "full", 0.08),
    ("transfer_s0", "configs/main_transfer_s0.yaml", "p3", "../outputs/main_transfer_s0", "full", 0.08),
    ("transfer_s1", "configs/main_transfer_s1.yaml", "p3", "../outputs/main_transfer_s1", "full", 0.08),
    ("transfer_s3", "configs/main_transfer_s3.yaml", "p3", "../outputs/main_transfer_s3", "full", 0.08),
    ("fixed_s0_common5", "configs/main_fixed_schedule_s0_influenza_common5.yaml", "p3", "../outputs/main_fixed_schedule_s0_influenza_common5", "full", 0.08),
    ("fixed_s2_common5", "configs/main_fixed_schedule_s2_influenza_common5.yaml", "p3", "../outputs/main_fixed_schedule_s2_influenza_common5", "full", 0.08),
    ("ablation_reply", "configs/main_ablation_reply_chain.yaml", "p3", "../outputs/main_ablation_reply_chain", "reply_chain", 0.50),
    ("ablation_post", "configs/main_ablation_post_only.yaml", "p3", "../outputs/main_ablation_post_only", "post_only", 0.0),
    ("ablation_like", "configs/main_ablation_like_channel.yaml", "p3", "../outputs/main_ablation_like_channel", "like_channel", 0.0),
    ("scale_1000_probe", "configs/main_scale_influenza_1000_probe.yaml", "p3", "../outputs/main_scale_influenza_1000_probe", "full", 0.08),
    ("scale_1000_full", "configs/main_scale_influenza_1000_r10.yaml", "p3", "../outputs/main_scale_influenza_1000_r10", "full", 0.08),
]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def save(state: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str], log) -> None:
    log.write(f"[{now()}] RUN {' '.join(command)}\n")
    log.flush()
    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    if not ENV_FILE.exists():
        raise SystemExit(f"env file missing: {ENV_FILE}")
    state = {"started_at": now(), "status": "running", "current_stage": None, "stages": []}
    if STATE_PATH.exists():
        previous = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        completed = {item["name"] for item in previous.get("stages", []) if item.get("passed")}
    else:
        completed = set()
    save(state)

    with LOG_PATH.open("a", encoding="utf-8") as log:
        try:
            for name, config, command, output, mode, min_reply in STAGES:
                if name in completed:
                    state["stages"].append({"name": name, "passed": True, "reused_completed_stage": True})
                    continue
                state["current_stage"] = name
                stage = {"name": name, "config": config, "output": output, "started_at": now(), "passed": False}
                state["stages"].append(stage)
                save(state)
                run(
                    [sys.executable, "run_experiment.py", "--config", config, "--env-file", str(ENV_FILE), command],
                    log,
                )
                run(
                    [
                        sys.executable,
                        "scripts/check_stage.py",
                        output,
                        "--mode",
                        mode,
                        "--min-reply-ratio",
                        str(min_reply),
                    ],
                    log,
                )
                stage.update({"ended_at": now(), "passed": True})
                save(state)
        except BaseException as exc:
            state.update({"status": "blocked", "ended_at": now(), "blocking_error": repr(exc)})
            save(state)
            raise
        state.update({"status": "completed", "current_stage": None, "ended_at": now()})
        save(state)


if __name__ == "__main__":
    main()
