from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT.parent / "outputs"
ENV_FILE = ROOT / ".env"
STATE_PATH = OUTPUT_ROOT / "model_robustness_state_2026-07-13.json"
LOG_PATH = OUTPUT_ROOT / "model_robustness_pipeline_2026-07-13.log"

STAGES = (
    ("deepseek_smoke", "configs/deepseek_v4_flash_probe.yaml", "smoke", "../outputs/deepseek_v4_flash_smoke", None),
    ("deepseek_probe", "configs/deepseek_v4_flash_probe.yaml", "all", "../outputs/deepseek_v4_flash_probe", "deepseek-v4-flash"),
    ("deepseek_full", "configs/deepseek_v4_flash_main_r10.yaml", "all", "../outputs/deepseek_v4_flash_main_r10", "deepseek-v4-flash"),
    ("gpt_smoke", "configs/gpt_5_1_probe.yaml", "smoke", "../outputs/gpt_5_1_smoke", None),
    ("gpt_probe", "configs/gpt_5_1_probe.yaml", "all", "../outputs/gpt_5_1_probe", "gpt-5.1"),
    ("gpt_full", "configs/gpt_5_1_main_r10.yaml", "all", "../outputs/gpt_5_1_main_r10", "gpt-5.1"),
    ("gpt54_smoke", "configs/gpt_5_4_probe.yaml", "smoke", "../outputs/gpt_5_4_smoke", None),
    ("gpt54_probe", "configs/gpt_5_4_probe.yaml", "all", "../outputs/gpt_5_4_probe", "gpt-5.4"),
    ("gpt54_full", "configs/gpt_5_4_main_r10.yaml", "all", "../outputs/gpt_5_4_main_r10", "gpt-5.4"),
)


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
    previous = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    completed = {item["name"] for item in previous.get("stages", []) if item.get("passed")}
    state = {"started_at": now(), "status": "running", "current_stage": None, "stages": []}
    save(state)

    with LOG_PATH.open("a", encoding="utf-8") as log:
        try:
            for name, config, command, output, expected_model in STAGES:
                if name in completed:
                    state["stages"].append({"name": name, "passed": True, "reused_completed_stage": True})
                    continue
                state["current_stage"] = name
                stage = {"name": name, "config": config, "output": output, "started_at": now(), "passed": False}
                state["stages"].append(stage)
                save(state)

                command_args = [
                    sys.executable,
                    "run_experiment.py",
                    "--config",
                    config,
                    "--env-file",
                    str(ENV_FILE),
                    "--output-dir",
                    output,
                    "--main-only",
                    command,
                ]
                if command == "smoke":
                    command_args.extend(["--condition", "independent", "--scenario", "influenza_a", "--persona", "baseline_open"])
                run(command_args, log)
                if expected_model:
                    run(
                        [
                            sys.executable,
                            "scripts/check_model_stage.py",
                            output,
                            "--expected-model",
                            expected_model,
                            "--min-reply-ratio",
                            "0.08",
                        ],
                        log,
                    )
                stage.update({"ended_at": now(), "passed": True})
                save(state)
        except BaseException as exc:
            state.update({"status": "blocked", "ended_at": now(), "blocking_error": repr(exc)})
            save(state)
            raise

        run([sys.executable, "scripts/summarize_model_robustness.py"], log)
        state.update({"status": "completed", "current_stage": None, "ended_at": now()})
        save(state)


if __name__ == "__main__":
    main()
