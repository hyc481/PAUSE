#!/usr/bin/env bash
set -e

# There are two debug approaches. The first is to run the whole pipeline. Modify RUN_NAME, and output for tasks, logs_failed_models,
# annotated_logs will be stored under bench/runs/RUN_NAME. The second approach is to run each script individually, which
# is used for in-depth debugging. Outputs are stored under bench/generations.

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
# Path
############################

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Set RUN_DIR to an absolute path if you want outputs elsewhere.
RUN_DIR="${RUN_DIR:-$BASE_DIR/runs/$RUN_NAME}"

TASKS_DIR="$RUN_DIR/tasks"
LOGS_DIR="$RUN_DIR/logs"
INFERENCE_LOGS_DIR="$RUN_DIR/inference_logs"
ANNOTATED_DIR="$RUN_DIR/annotated_logs"
SAVED_TASKS_DIR="$RUN_DIR/saved_tasks"
EVALUATED_TASKS_DIR="$RUN_DIR/evaluated_tasks"

GEN_TASKS_SCRIPT="$BASE_DIR/backend/generate_task/generate_tasks.py"
RUN_TASKS_SCRIPT="$BASE_DIR/run_user_tasks.py"
ANNOTATE_SCRIPT="$BASE_DIR/annotate_trajs.py"
COLLECT_TASKS_SCRIPT="$BASE_DIR/extract_annotated_tasks.py"
EVALUATE_TASKS_SCRIPT="$BASE_DIR/evaluate_trajs.py"

mkdir -p "$TASKS_DIR" "$LOGS_DIR" "$ANNOTATED_DIR"

############################
# 1. generate tasks
############################

echo "[1/6] Generating tasks..."

python "$GEN_TASKS_SCRIPT" \
  --profile_id "$PROFILE_ID" \
  --user_num $USER_NUM \
  --runs_per_branch $RUNS_PER_BRANCH \
  --output_dir "$TASKS_DIR"

############################
# 2. run all tasks in this run
############################

echo "[2/6] Running user tasks and generating logs..."
echo $TASKS_DIR
python "$RUN_TASKS_SCRIPT" \
  --tasks_files "$TASKS_DIR" \
  --branch all \
  --output_dir "$LOGS_DIR" \

############################
# 3. annotate all logs in this run
############################

echo "[3/6] Annotating logs..."
python "$ANNOTATE_SCRIPT" \
  --logs_glob "$LOGS_DIR" \
  --output_dir "$ANNOTATED_DIR" \
  --rerun

############################
# 4. collect passed tasks with annotations
############################

echo "[4/6] Collecting and saving passed tasks with annotations..."
python "$COLLECT_TASKS_SCRIPT" \
  --tasks_dir "$TASKS_DIR" \
  --annotated_logs_dir "$ANNOTATED_DIR" \
  --tasks_saved_dir "$SAVED_TASKS_DIR"

############################
# 5. inference runs
############################

echo "[5/6] Running passed user tasks and generating inference logs..."
python "$RUN_TASKS_SCRIPT" \
  --tasks_files "$SAVED_TASKS_DIR" \
  --branch all \
  --output_dir "$INFERENCE_LOGS_DIR" \
  --inference

############################
# 6. evaluate all inference logs in this run
############################

echo "[6/6] Evaluating inference logs..."
python "$EVALUATE_TASKS_SCRIPT" \
  --tasks_dir "$SAVED_TASKS_DIR" \
  --evaluated_tasks_dir "$EVALUATED_TASKS_DIR"

############################
# done
############################

echo "Done."
echo "Run dir: $RUN_DIR"
echo "Tasks dir: $TASKS_DIR"
echo "Logs dir: $LOGS_DIR"
echo "Annotated dir: $ANNOTATED_DIR"
