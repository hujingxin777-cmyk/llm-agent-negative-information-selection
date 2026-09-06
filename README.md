# LLM-Agent Negative Information Selection

This repository contains the executable code for the manuscript **“Negative Information Selection in Evolving Online Discussions with LLM Agents.”**

The simulation compares three information-selection environments:

- **P1 — Individual judgement:** agents evaluate two reports without social information.
- **P2 — Static feedback:** agents additionally observe fixed comments and like counts.
- **P3 — Evolving discussion:** agents participate in a multi-round discussion in which posts, replies and likes become visible to later agents.

## Files

```text
.
├── src/                 # Simulation, prompts, metrics and result-export modules
├── run_experiment.py    # Command-line entry point
├── config.yaml          # Reproducible example configuration
├── requirements.txt     # Python dependencies
├── .env.example         # API-key variable template
└── README.md
```

Generated results, API logs, local environment files and intermediate analysis files are excluded from this repository.

## Requirements

- Python 3.10 or later
- An API key for the model endpoint specified in `config.yaml`

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Configure the API

Create a local environment file:

```bash
cp .env.example .env
```

The supplied configuration uses GLM-4 through the Zhipu OpenAI-compatible endpoint, so set:

```text
ZHIPU_API_KEY=your_api_key_here
```

Do not commit `.env`. To use another OpenAI-compatible service, edit the `model` section of `config.yaml`, including `model_name`, `api_base` and `api_key_env`. The client also supports `max_completion_tokens`, omitted temperature values, `reasoning_effort` and provider-specific `extra_body` fields.

## Verify the installation without API calls

Compile the source code:

```bash
python3 -m compileall -q run_experiment.py src
```

Generate the full experiment plan:

```bash
python3 run_experiment.py --config config.yaml plan
```

Successful plan generation creates design files under `outputs/plan/` and does not call the model API.

## Run a low-cost check

Inspect one model response before launching an experiment:

```bash
python3 run_experiment.py \
  --config config.yaml \
  --env-file .env \
  smoke \
  --condition independent \
  --scenario influenza_a \
  --persona baseline_open
```

This command makes one API request.

## Run the experiment

The following commands use the settings in `config.yaml`:

```bash
# P1 and P2 only
python3 run_experiment.py --config config.yaml --env-file .env static

# P3 only
python3 run_experiment.py --config config.yaml --env-file .env p3

# P1, P2 and P3, followed by export and plotting
python3 run_experiment.py --config config.yaml --env-file .env all
```

For an inexpensive trial, override the number of independent runs and limit the number of experimental cells:

```bash
python3 run_experiment.py \
  --config config.yaml \
  --env-file .env \
  --main-only \
  --repeats 1 \
  --limit-static 1 \
  --limit-p3 1 \
  all
```

Full-scale runs can make many API requests and incur substantial charges. Always generate the plan and run a limited trial first.

## Main configuration

The supplied configuration contains the two public-health scenarios used in the study, the four agent profiles, three initial comment climates and the three platform conditions. Its P3 setting uses 300 agents, 15 active agents per round and a maximum of 15 rounds. Model sampling and retry settings are defined in the same YAML file so that each run can be traced to one configuration.

Outputs are written to `outputs/` and include experiment plans, run-level results, interaction logs, derived summaries and figures. These generated files are ignored by Git.
