from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.environments import build_discussion_agents
from src.runner import ExperimentPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate P3 model calls and token cost without running experiments")
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--tokens-per-call", type=int, default=3800)
    parser.add_argument("--price-per-1k", type=float, default=0.003)
    args = parser.parse_args()

    print("config\tp3_cells\tmax_model_calls\testimated_tokens\testimated_cost")
    for path in args.configs:
        config = load_config(path)
        cells = [cell for cell in ExperimentPlanner(config).build_cells() if cell.condition_name == "high_interaction"]
        max_active = int(config.get("p3", {}).get("max_active_agents", 3))
        calls = 0
        for cell in cells:
            population = build_discussion_agents(cell.experiment_family, cell.persona_name, config)
            calls += min(max_active, len(population)) * cell.max_timestep
        tokens = calls * args.tokens_per_call
        cost = tokens / 1000 * args.price_per_1k
        print(f"{path.name}\t{len(cells)}\t{calls}\t{tokens}\t{cost:.2f}")


if __name__ == "__main__":
    main()
