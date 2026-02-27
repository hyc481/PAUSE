#!/usr/bin/env bash
set -e

# Test run: run inference on saved tasks and evaluate trajectories.

export PYTHONPATH=PYTHONPATH:"/home/chy/state_aware_bench"

############################
# Runtime Configuration
############################
RUN_NAME="testrun2"

############################
# Paths
############################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$PROJECT_ROOT/bench"

# Set RUN_DIR to an absolute path if you want outputs elsewhere.
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/bench/runs/$RUN_NAME}"

SAVED_TASKS_DIR="$RUN_DIR/saved_tasks"
INFERENCE_LOGS_DIR="$RUN_DIR/inference_logs"
EVALUATED_TASKS_DIR="$RUN_DIR/evaluated_tasks"

RUN_TASKS_SCRIPT="$BENCH_DIR/pipeline/run_user_tasks.py"
EVALUATE_TASKS_SCRIPT="$BENCH_DIR/evaluation/evaluate_trajs.py"

mkdir -p "$INFERENCE_LOGS_DIR" "$EVALUATED_TASKS_DIR"

############################
# 1. inference runs
############################

echo "[1/2] Running passed user tasks and generating inference logs..."
python "$RUN_TASKS_SCRIPT" \
  --tasks_files "$SAVED_TASKS_DIR" \
  --branch all \
  --output_dir "$INFERENCE_LOGS_DIR" \
  --inference

############################
# 2. evaluate all inference logs in this run
############################

echo "[2/2] Evaluating inference logs..."
python "$EVALUATE_TASKS_SCRIPT" \
  --tasks_dir "$SAVED_TASKS_DIR" \
  --evaluated_tasks_dir "$EVALUATED_TASKS_DIR"

echo "Testrun done."
echo "Run dir: $RUN_DIR"
echo "Saved tasks dir: $SAVED_TASKS_DIR"
echo "Inference logs dir: $INFERENCE_LOGS_DIR"
echo "Evaluated tasks dir: $EVALUATED_TASKS_DIR"

