# Negative Information Selection in Evolving Online Discussions with LLM Agents

This repository contains the simulation code and experiment configurations associated with the manuscript *Negative Information Selection in Evolving Online Discussions with LLM Agents*.

The study compares information selection under three controlled conditions:

- **P1: individual judgement** without social cues;
- **P2: static social feedback** with fixed comments and like counts;
- **P3: evolving multi-agent discussion** in which comments, replies, likes, memory, and agent states are updated over multiple rounds.

The repository currently provides the executable code and the exact configuration families used for the reported main, sensitivity, ablation, scale, and cross-model analyses. Curated result data and a manuscript table/figure reproduction manifest will be added to `data/` after the output archive has been checked and de-identified.

## Repository structure

```text
.
├── main_and_sensitivity/   # Main experiment, static ternary scoring, sensitivity and ablation analyses
├── model_robustness/       # Cross-model robustness runs and model-specific request handling
└── data/                   # Curated result files (to be added)
```

Two code snapshots are retained because the cross-model runs required model-specific API parameter handling, whereas the later static-measurement analyses added an explicit neutral category. Keeping the snapshots separate preserves the code actually used for each analysis.

## Requirements

- Python 3.10 or later
- An API key for the selected OpenAI-compatible model endpoint
- Sufficient API quota for the requested number of agents and independent runs

Create and activate a virtual environment, then install the dependencies in the relevant snapshot:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r main_and_sensitivity/requirements.txt
```

## API configuration

Copy the example environment file in the snapshot you intend to run:

```bash
cd main_and_sensitivity
cp .env.example .env
```

Add only the key required by the selected configuration. `.env` files and generated model-request logs are excluded from version control.

Supported environment-variable names in the supplied configurations include:

- `ZHIPU_API_KEY`
- `DEEPSEEK_API_KEY`
- `ANYAPI_API_KEY`

## Verification without API calls

From either code-snapshot directory:

```bash
python -m compileall -q run_experiment.py main.py src scripts tests
python -m unittest discover -s tests -v
python run_experiment.py --config configs/main_300_formal_r1_30.yaml plan
```

These commands check syntax, local controls, configuration inheritance, and experiment-plan generation. They do not call a model endpoint.

## Main experiment

The formal main configuration uses 300 agents, 15 active agents per round, up to 15 rounds, two public-health scenarios, and 30 independent runs per condition:

```bash
cd main_and_sensitivity
python run_experiment.py \
  --config configs/main_300_formal_r1_30.yaml \
  --env-file .env \
  all
```

Generated files are written to the output directory declared in the selected YAML configuration.

## Sensitivity and ablation configurations

The `main_and_sensitivity/configs/` directory contains configurations for:

- activation rates of 2%, 5% (main setting), 8%, and 10%;
- populations of 100, 300, 600, and 1,000 agents;
- state-update and scheduling checks;
- reply-chain, post-only, and like-channel action ablations;
- binary and ternary static measurement runs.

Run any configuration using the same command pattern:

```bash
cd main_and_sensitivity
python run_experiment.py --config configs/CONFIG_FILE.yaml --env-file .env all
```

Some configurations are restricted to P3 and should use `p3` in place of `all`.

## Cross-model robustness

The `model_robustness/` snapshot contains configurations for GLM-4, DeepSeek-V4-Flash, GPT-5.1, and GPT-5.4. Model availability, snapshot aliases, and third-party gateway behaviour may change over time; verify the endpoint and model identifier before rerunning.

```bash
cd model_robustness
python run_experiment.py \
  --config configs/deepseek_v4_flash_main_r10.yaml \
  --env-file .env \
  --main-only \
  all
```

## Reproducibility notes

- Treat an independent run, rather than an individual agent, as the replication unit when aggregating condition-level outcomes.
- Record the requested and resolved model identifiers, sampling settings, code commit, configuration file, and run index for every rerun.
- LLM outputs may vary across providers and model revisions even when prompts and random seeds are held constant.
- Full-scale experiments incur API costs. Inspect the plan and start with a one-run probe before launching a formal configuration.

## Data status

The source code contains no real participant data. Curated simulation outputs and scripts mapping files to manuscript tables and figures are being prepared. See [`data/README.md`](data/README.md).
