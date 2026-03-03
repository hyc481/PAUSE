#!/usr/bin/env bash
set -e

###############################################################################
# PAUSE test run (inference + evaluation)
#
# This script runs 2 steps:
#   1) Run inference on saved tasks -> inference_logs
#   2) Evaluate inference logs -> evaluated_tasks
#
# Prerequisite: run_prepare.sh must have been run to produce saved_tasks.
#
# Output directory:
#   bench/runs/<RUN_NAME>/{inference_logs,evaluated_tasks}
#
# Tip:
#   Prefer `pip install -e .` or set PYTHONPATH outside this script:
#     export PYTHONPATH="/path/to/PAUSE:${PYTHONPATH}"
###############################################################################

############################
# Runtime configuration
############################
BRANCH="all"

############################
# Run identity
############################
RUN_NAME="testrun"

############################
# Paths
############################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$PROJECT_ROOT/bench"

# Set RUN_DIR to an absolute path if you want outputs elsewhere.
RUN_DIR="${RUN_DIR:-$BENCH_DIR/runs/$RUN_NAME}"

SAVED_TASKS_DIR="$RUN_DIR/saved_tasks"
INFERENCE_LOGS_DIR="$RUN_DIR/inference_logs"
EVALUATED_TASKS_DIR="$RUN_DIR/evaluated_tasks"

RUN_TASKS_SCRIPT="$BENCH_DIR/pipeline/run_user_tasks.py"
EVALUATE_TASKS_SCRIPT="$BENCH_DIR/evaluation/evaluate_trajs.py"

mkdir -p "$INFERENCE_LOGS_DIR" "$EVALUATED_TASKS_DIR"

# Optional (only if you do NOT use editable install):
# export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"

############################
# 1) Inference runs
############################
echo "[1/2] Running inference on saved tasks..."
python "$RUN_TASKS_SCRIPT" \
  --tasks_files "$SAVED_TASKS_DIR" \
  --branch "$BRANCH" \
  --output_dir "$INFERENCE_LOGS_DIR" \
  --inference

############################
# 2) Evaluate inference logs
############################
echo "[2/2] Evaluating inference logs..."
python "$EVALUATE_TASKS_SCRIPT" \
  --tasks_dir "$SAVED_TASKS_DIR" \
  --evaluated_tasks_dir "$EVALUATED_TASKS_DIR"

echo "Done."
echo "Run dir:              $RUN_DIR"
echo "Saved tasks dir:      $SAVED_TASKS_DIR"
echo "Inference logs dir:   $INFERENCE_LOGS_DIR"
echo "Evaluated tasks dir:  $EVALUATED_TASKS_DIR"
