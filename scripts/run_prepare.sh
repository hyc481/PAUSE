#!/usr/bin/env bash
set -e

###############################################################################
# PAUSE pipeline runner (end-to-end)
#
# This script runs 4 steps:
#   1) generate tasks
#   2) run tasks to produce trajectories/logs
#   3) annotate trajectories
#   4) collect passed tasks with annotations
#
# Output directory:
#   bench/runs/<RUN_NAME>/{tasks,logs,annotated_logs,saved_tasks}
#
# Tip:
#   Prefer `pip install -e .` or set PYTHONPATH outside this script:
#     export PYTHONPATH="/path/to/PAUSE:${PYTHONPATH}"
###############################################################################

############################
# Runtime configuration
############################
PROFILE_ID="developing"
NUM_USERS=5
RUNS_PER_BRANCH=5
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

TASKS_DIR="$RUN_DIR/tasks"
LOGS_DIR="$RUN_DIR/logs"
ANNOTATED_DIR="$RUN_DIR/annotated_logs"
SAVED_TASKS_DIR="$RUN_DIR/saved_tasks"

GEN_TASKS_SCRIPT="$BENCH_DIR/pipeline/generate_task/generate_tasks.py"
RUN_TASKS_SCRIPT="$BENCH_DIR/pipeline/run_user_tasks.py"
ANNOTATE_SCRIPT="$BENCH_DIR/pipeline/annotate_trajs.py"
COLLECT_TASKS_SCRIPT="$BENCH_DIR/pipeline/extract_annotated_tasks.py"

mkdir -p "$TASKS_DIR" "$LOGS_DIR" "$ANNOTATED_DIR" "$SAVED_TASKS_DIR"

# Optional (only if you do NOT use editable install):
# export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"

############################
# 1) Generate tasks
############################
echo "[1/4] Generating tasks..."
python "$GEN_TASKS_SCRIPT" \
  --profile_id "$PROFILE_ID" \
  --user_num "$NUM_USERS" \
  --runs_per_branch "$RUNS_PER_BRANCH" \
  --branch "$BRANCH" \
  --output_dir "$TASKS_DIR"

############################
# 2) Run tasks -> logs
############################
echo "[2/4] Running tasks and generating logs..."
# Note: `--tasks_files` may accept either a directory or a file/glob list,
# depending on your implementation. Here we pass the tasks directory.
python "$RUN_TASKS_SCRIPT" \
  --tasks_files "$TASKS_DIR" \
  --branch "$BRANCH" \
  --output_dir "$LOGS_DIR"

############################
# 3) Annotate logs
############################
echo "[3/4] Annotating logs..."
python "$ANNOTATE_SCRIPT" \
  --logs_glob "$LOGS_DIR" \
  --output_dir "$ANNOTATED_DIR" \
  --rerun

############################
# 4) Collect passed tasks with annotations
############################
echo "[4/4] Collecting passed tasks with annotations..."
python "$COLLECT_TASKS_SCRIPT" \
  --tasks_dir "$TASKS_DIR" \
  --annotated_logs_dir "$ANNOTATED_DIR" \
  --tasks_saved_dir "$SAVED_TASKS_DIR"

echo "Done."
echo "Run dir:        $RUN_DIR"
echo "Tasks dir:      $TASKS_DIR"
echo "Logs dir:       $LOGS_DIR"
echo "Annotated dir:  $ANNOTATED_DIR"
echo "Saved tasks dir:$SAVED_TASKS_DIR"