## PAUSE

PAUSE (Personal AI Assistants in Unified Service Environments) is a user-centric benchmark for evaluating tool-augmented personal AI assistants in realistic, stateful service settings. Unlike benchmarks that fragment service context or abstract away user state, PAUSE stresses what real deployments require: reasoning over persistent user state, respecting user-specific configurations and permissions, and sustaining long-horizon, constraint-aware interactions across multiple services.

### What PAUSE Evaluates
PAUSE captures core challenges of real-world assistants by requiring agents to coordinate actions across heterogeneous, user-owned resources while staying consistent with environment state and authorization constraints over multi-turn interactions. It also includes explicit user–agent interaction via realistic user simulation, enabling evaluation beyond static tool execution.

### Evaluation Framework
PAUSE adopts a multi-regime evaluation framework aligned with task characteristics: open-ended service management tasks are scored with semantic and trajectory-level behavioral metrics, while constraint-intensive tasks support deterministic, state-based verification.

### Tool Sandbox Specification
PAUSE simulates a personal healthcare management platform integrating wearable data, meal logs, activity records, appointments, and configurable user profiles. The sandbox enforces persistent user state, permission-gated tools, subscription constraints, and multi-turn task execution. Tasks require coordinated tool calls, temporal reasoning, and strict consistency with system configurations.

### Open Source
This repository provides the full PAUSE pipeline, including **generation**, **annotation**, and **evaluation**, as well as the **180 benchmark tasks** used in the paper.

---
## Leaderboard

We report task statistics and model performance on (1) data & log tracking (easy/hard) and (2) shopping tasks. Metrics include task completion (TC), targets achieved (TA), and trajectory overlap precision/recall/F1 (Pre/Rec/F1).

### Table 1. Data & Log Tracking Task Statistics

| Task Type | Avg. Rounds | Assistant TC | User TC |
|---|---:|---:|---:|
| Data & Log Tracking (Easy) | 2.11 | 12.85 | 0.00 |
| Data & Log Tracking (Hard) | 5.23 | 22.07 | 3.28 |

### Table 2. Performance on Data & Log Tracking (Easy)

| Model | TC | TA | Pre | Rec | F1 |
|---|---:|---:|---:|---:|---:|
| Gemini-3-Flash | 85.72% | 95.98% | 0.841 | 0.796 | 0.796 |
| Gemini-3-Pro | 92.06% | 98.51% | **0.849** | 0.869 | 0.844 |
| Gemini-2.5-Pro | 66.70% | 90.23% | 0.733 | 0.751 | 0.711 |
| GPT-5 | **95.26%** | **98.85%** | 0.847 | **0.880** | 0.856 |
| GPT-5-Mini | 92.07% | 98.37% | 0.895 | 0.876 | **0.880** |
| GPT-4.1 | 33.34% | 71.18% | 0.669 | 0.584 | 0.588 |
| GPT-4.1-Mini | 28.56% | 62.41% | 0.648 | 0.564 | 0.582 |
| DeepSeek-V3.2-Thinking | 69.86% | 85.00% | 0.718 | 0.744 | 0.720 |
| DeepSeek-V3.2 | 47.57% | 72.14% | 0.560 | 0.623 | 0.578 |

### Table 3. Performance on Data & Log Tracking (Hard)

| Model | TC | TA | Pre | Rec | F1 |
|---|---:|---:|---:|---:|---:|
| Gemini-3-Flash | **59.12%** | **77.48%** | **0.584** | 0.492 | **0.517** |
| Gemini-3-Pro | 48.39% | 77.25% | 0.511 | 0.427 | 0.439 |
| Gemini-2.5-Pro | 19.33% | 57.76% | 0.516 | 0.343 | 0.379 |
| GPT-5 | 47.34% | 72.96% | 0.494 | **0.555** | 0.479 |
| GPT-5-Mini | 43.84% | 66.96% | 0.439 | 0.495 | 0.406 |
| GPT-4.1 | 17.56% | 57.00% | 0.401 | 0.279 | 0.303 |
| GPT-4.1-Mini | 10.53% | 38.81% | 0.436 | 0.293 | 0.331 |
| DeepSeek-V3.2-Thinking | 35.11% | 55.71% | 0.297 | 0.326 | 0.296 |
| DeepSeek-V3.2 | 14.06% | 40.02% | 0.257 | 0.245 | 0.231 |

### Table 4. Performance on Shopping Tasks

| Model | Score | PID | Qty_Size | Voucher | Balance |
|---|---:|---:|---:|---:|---:|
| Gemini-3-Flash | 0.590 | 0.850 | 0.417 | 0.567 | 0.417 |
| Gemini-3-Pro | **0.721** | 0.901 | **0.600** | 0.567 | **0.583** |
| Gemini-2.5-Pro | 0.377 | 0.700 | 0.192 | 0.351 | 0.192 |
| GPT-5 | 0.691 | **0.901** | 0.582 | **0.620** | 0.565 |
| GPT-5-Mini | 0.473 | 0.750 | 0.267 | 0.517 | 0.233 |
| GPT-4.1 | 0.197 | 0.350 | 0.050 | 0.250 | 0.050 |
| GPT-4.1-Mini | 0.183 | 0.383 | 0.050 | 0.167 | 0.033 |
| DeepSeek-V3.2-Thinking | 0.550 | 0.808 | 0.350 | 0.550 | 0.350 |
| Deepseek-V3.2 | 0.417 | 0.792 | 0.150 | 0.417 | 0.150 |
---
## Getting started

### Prerequisites

- Python 3.10+ (required by `google-genai`).
- Access to at least one LLM backend (OpenAI / Azure / Gemini / DeepSeek, etc.).

### Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `openai`, `google-genai`, `pandas`. See `requirements.txt` for pinned versions.

### Configuration

Configure model endpoints and API keys in `bench/utils/clients.py` or via environment variables (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`), depending on your local setup.

### Quick start

There are two high-level flows:

- **Prepare a run** (generate tasks, run trajectories, annotate, and collect passed tasks)
  - Script: `scripts/run_prepare.sh`
- **Test models on prepared tasks** (inference + evaluation only)
  - Script: `scripts/run_testrun.sh`

Both scripts respect `RUN_NAME` and can be pointed to a different `RUN_DIR` via the `RUN_DIR` environment variable if needed.

---

## Key scripts and CLI (used by `run_prepare.sh` / `run_testrun.sh`)

Override run location with `RUN_NAME` and optionally `RUN_DIR`.

**`bench/pipeline/generate_task/generate_tasks.py`**  
Builds the user store (profile + wearable data), runs the task orchestrator over one or all branches (e.g. shopping, wearable_data_casual), and writes task JSON into the given directory.  
CLI: `--profile_id`, `--user_num`, `--runs_per_branch`, `--branch` (e.g. `all` or `shopping`), `--output_dir`.

**`bench/pipeline/run_user_tasks.py`**  
Runs the simulated assistant–user dialogue for each task: loads task files, invokes the agent runner with sandbox tools, and saves raw trajectories (logs). With `--inference`, runs in inference-only mode for model evaluation.  
CLI: `--tasks_files` (directory or glob of task JSONs), `--branch`, `--output_dir`; add `--inference` for testrun.

**`bench/pipeline/annotate_trajs.py`**  
Annotates trajectory logs with an LLM-as-judge: summary, error analysis, and pass/fail selection.  
CLI: `--logs_glob`, `--output_dir`, `--rerun`.

**`bench/pipeline/extract_annotated_tasks.py`**  
Merges annotation results back into the task list (adds `annotated_log_path`), filters to passed tasks, and writes the resulting task set to a directory for downstream inference/eval.  
CLI: `--tasks_dir`, `--annotated_logs_dir`, `--tasks_saved_dir`.

**`bench/evaluation/evaluate_trajs.py`**  
Scores trajectories (semantic and trajectory-level metrics via LLM) and writes evaluation outputs.  
CLI: `--tasks_dir`, `--evaluated_tasks_dir`.

**Flows.** Prepare (run_prepare.sh): generate → run_user_tasks → annotate → extract; outputs under `runs/<RUN_NAME>/{tasks,logs,annotated_logs,saved_tasks}`. Testrun (run_testrun.sh): run_user_tasks with `--inference` → evaluate; reads `saved_tasks/`, writes `inference_logs/` and `evaluated_tasks/`.

---

## Client Configs and Agent Interplay

These two modules under `bench/utils/` underpin the pipeline: they provide LLM access and the assistant–user dialogue runner. They have no CLI; they are used by the scripts above.

**`bench/utils/clients.py`**  
Central configuration and caller for all LLM backends. It holds:

- **Client registry** (`CLIENTS`): named clients (OpenAI, Azure, Gemini, DeepSeek, etc.) keyed by string (e.g. `"default"`, `"gemini1"`). Configure endpoints and API keys here or via environment variables.
- **Routes**: which client+model to use for each role.  
  - `GEN_ROUTE`: task generation (used by `generate_tasks.py`).  
  - `ANN_ROUTES`: annotation / LLM-as-judge (used by `annotate_trajs.py`, `evaluate_trajs.py`).  
  - `TRAJ_MODEL_ROUTES` / `INFERENCE_TRAJ_MODEL_ROUTES`: assistant model, user sim model, and optional user-valid model for trajectory runs (used by `run_user_tasks.py`).
- **Unified API**: `call_llm(client, model=..., messages=...)` and `call_llm_with_tools(...)` so pipeline code does not branch on provider. Helpers: `get_gen_route()`, `get_ann_route()` / `get_ann_routes()`, `get_traj_model_routes(inference=True|False)`.

**`bench/utils/agent_runner.py`**  
Implements the **two-agent dialogue** (assistant + simulated user) for a single task. Used by `run_user_tasks.py` for every task file.

- **AgentRunner** loads tool schemas (platform + user tools), wraps the sandbox tools (platform, med, source, user, shopping) with the current `store`, and drives multi-turn chat: assistant receives the task and can call tools; the user agent reacts with natural language or tool use (e.g. checkout authorization). Optionally runs a **user_valid** step (LLM check that the user agent’s behavior is consistent with the task).
- **Main entry** `run(store, task_instruction_text, targets=..., label=..., max_rounds=..., traj_route=..., assistant_tool_allowlist=..., assistant_policy=..., user_policy=..., user_valid=..., ...)`: runs one trajectory and returns a result dict with messages, tool calls, and metadata. `traj_route` is one of the route dicts from `get_traj_model_routes(inference)` (assistant/client/model and user/client/model).
- **Output**: trajectory (full message list, tool calls, state snapshots) is then saved by `run_user_tasks.py` and later annotated or evaluated.