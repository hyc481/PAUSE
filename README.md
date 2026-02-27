## PAUSE Benchmark

PAUSE is a benchmark for **personal AI assistants** that act as tool-using agents inside a unified service environment.
It focuses on:

- Stateful reasoning over user profiles, wearable data, records, and shopping state.
- Respecting per-user configuration and permissions.
- Long-horizon, tool-heavy interactions with a simulated user.

This repository contains both the **sandbox environment** (tools, configuration, store builder) and the **pipeline** for generating tasks, trajectories, annotations, and evaluation results.

---

## Getting started

### Prerequisites

- Python 3.10+ (recommended).
- Access to at least one LLM backend (OpenAI / Azure / Gemini / DeepSeek, etc.).
- Install Python dependencies (example, adapt to your setup):

```bash
pip install -r requirements.txt  # or your own env management
```

Configure model endpoints and API keys in `bench/utils/clients.py` or via environment variables, depending on your local setup.

### Quick start

There are two high-level flows:

- **Prepare a run** (generate tasks, run trajectories, annotate, and collect passed tasks)
  - Script: `scripts/run_prepare.sh`
- **Test models on prepared tasks** (inference + evaluation only)
  - Script: `scripts/run_testrun.sh`

Both scripts respect `RUN_NAME` and can be pointed to a different `RUN_DIR` via the `RUN_DIR` environment variable if needed.

---

## Project layout (`bench/`)

- **`backend/`** – Sandbox tools and static configuration.
  - `tools/`:
    - `platform_tools.py`: health, lifestyle, records, and plotting tools exposed as platform APIs.
    - `shopping_tools.py`: shopping, cart, wallet, transaction tools.
    - `source_tools.py`: source-level wearable / raw-data tools.
    - `med_tools.py`: medical / provider-facing tools.
    - `user_tools.py`: user-side actions (permissions, wallet top-ups, checkout authorization).
  - `configuration/`:
    - `configuration.py`: base user profile, system settings, and environment configuration.
    - `shopping_state.py`: initial shopping wallet, cart, and transaction state.

- **`pipeline/`** – Task and trajectory generation.
  - `build_store.py`: builds a **store** object from configuration + wearable tables in `backend/data/`.
  - `build_user/`: utilities to synthesize user profile, meals, sports, system settings, and source assignment.
  - `generate_task/`:
    - `task_branch_base.py`: common branch interface (build prompt → call LLM → postprocess).
    - `branches/*.py`: task generators for shopping, wearable data, and lifestyle records.
    - `orchestrator.py`: runs multiple branches and aggregates tasks.
    - `generate_tasks.py`: CLI entry to build a store and generate tasks.
  - `run_user_tasks.py`: runs simulated **assistant–user** dialogues for each task and saves raw trajectories.
  - `annotate_trajs.py`: annotates trajectories with LLM-as-judge (summary, error analysis, selection).
  - `extract_annotated_tasks.py`: injects `annotated_log_path` into tasks and filters to passed tasks.

- **`evaluation/`** – Scoring and metrics.
  - `evaluate_trajs.py`: LLM-based evaluation for general tasks (semantic and trajectory-level metrics).
  - `evaluate_trajs_shopping.py`: rule-based evaluation for shopping tasks based on final state.
  - `summarize_shopping_scores.py`: aggregates shopping scores across tasks and models.

- **`analysis/`** – Optional analysis tools.
  - `analyze_traj_stats.py`: tool-call and dialogue-round statistics over trajectories.
  - `analyze_selected_traj_stats.py`: statistics for selected trajectories only.
  - `consistency_analysis.py`, `error_classification.py`: utilities for inspecting annotated / evaluated outputs.

- **`utils/`** – Shared infrastructure.
  - `clients.py`: LLM client and routing config.
  - `agent_runner.py`: the main assistant–user runner that wraps tools and simulates multi-turn dialogues.
  - `paths.py`: central path definitions (`BENCH_ROOT`, `RUNS_DIR`, `SNAPSHOTS_DIR`, `DATA_DIR`, `TOOL_SCHEMAS_DIR`, …).
  - `resolver.py`, `shopping_query_sampling.py`: shopping catalog utilities.
  - `generate_ids.py`: ID generators for stateful entities.
  - `misc.py`, `schema_adapter.py`, `parse_tool_call.py`: helper utilities for prompts, tool schemas, and tool-call parsing.

- **`prompts/`**
  - `agent_interplay_prompt.py`: system and role prompts for the assistant–user simulation.
  - `generation_prompt.py`: prompts for task and evaluation generation.

---

## Typical workflow

1. **Prepare a run (`scripts/run_prepare.sh`)**
   - Generates tasks using `pipeline/generate_task/generate_tasks.py`.
   - Runs trajectories for those tasks with `pipeline/run_user_tasks.py`.
   - Annotates trajectories with `pipeline/annotate_trajs.py`.
   - Produces:
     - `bench/runs/<run_name>/tasks/`
     - `bench/runs/<run_name>/logs/`
     - `bench/runs/<run_name>/annotated_logs/`
     - `bench/runs/<run_name>/saved_tasks/` (tasks + `annotated_log_path`).

2. **Run a test round (`scripts/run_testrun.sh`)**
   - Uses the prepared `saved_tasks/` as input.
   - Runs `pipeline/run_user_tasks.py` with `--inference` to produce `inference_logs/`.
   - Runs `evaluation/evaluate_trajs.py` to produce `evaluated_tasks/`.
   - All outputs live under:
     - `bench/runs/<run_name>/inference_logs/`
     - `bench/runs/<run_name>/evaluated_tasks/`.

You can then use the scripts under `analysis/` to inspect trajectories, selections, and evaluation results, or plug your own analysis on top of the generated JSON files.

