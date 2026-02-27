#!/usr/bin/env bash
set -e

# Prepare pipeline: generate tasks, run trajectories, annotate, and
# collect passed tasks with annotations.

export PYTHONPATH=PYTHONPATH:"/home/chy/state_aware_bench"

############################
# Runtime Configuration
############################
PROFILE_ID="developing"
USER_NUM=5
RUNS_PER_BRANCH=4

############################
# Run identity
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

TASKS_DIR="$RUN_DIR/tasks"
LOGS_DIR="$RUN_DIR/logs"
ANNOTATED_DIR="$RUN_DIR/annotated_logs"
SAVED_TASKS_DIR="$RUN_DIR/saved_tasks"

GEN_TASKS_SCRIPT="$BENCH_DIR/pipeline/generate_task/generate_tasks.py"
RUN_TASKS_SCRIPT="$BENCH_DIR/pipeline/run_user_tasks.py"
ANNOTATE_SCRIPT="$BENCH_DIR/pipeline/annotate_trajs.py"
COLLECT_TASKS_SCRIPT="$BENCH_DIR/pipeline/extract_annotated_tasks.py"

mkdir -p "$TASKS_DIR" "$LOGS_DIR" "$ANNOTATED_DIR" "$SAVED_TASKS_DIR"

############################
# 1. generate tasks
############################

echo "[1/4] Generating tasks..."

python "$GEN_TASKS_SCRIPT" \
  --profile_id "$PROFILE_ID" \
  --user_num "$USER_NUM" \
  --runs_per_branch "$RUNS_PER_BRANCH" \
  --output_dir "$TASKS_DIR"

############################
# 2. run all tasks in this run
############################

echo "[2/4] Running user tasks and generating logs..."
python "$RUN_TASKS_SCRIPT" \
  --tasks_files "$TASKS_DIR" \
  --branch all \
  --output_dir "$LOGS_DIR"

############################
# 3. annotate all logs in this run
############################

echo "[3/4] Annotating logs..."
python "$ANNOTATE_SCRIPT" \
  --logs_glob "$LOGS_DIR" \
  --output_dir "$ANNOTATED_DIR" \
  --rerun

############################
# 4. collect passed tasks with annotations
############################

echo "[4/4] Collecting and saving passed tasks with annotations..."
python "$COLLECT_TASKS_SCRIPT" \
  --tasks_dir "$TASKS_DIR" \
  --annotated_logs_dir "$ANNOTATED_DIR" \
  --tasks_saved_dir "$SAVED_TASKS_DIR"

echo "Prepare pipeline done."
echo "Run dir: $RUN_DIR"
echo "Tasks dir: $TASKS_DIR"
echo "Logs dir: $LOGS_DIR"
echo "Annotated dir: $ANNOTATED_DIR"
echo "Saved tasks dir: $SAVED_TASKS_DIR"

