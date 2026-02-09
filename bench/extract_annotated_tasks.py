from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _is_failure(payload: Dict[str, Any]) -> bool:
    final_selection = payload.get("final_selection") or {}
    selection = payload.get("selection") or {}
    status = str(final_selection.get("status") or selection.get("status") or "").lower()
    return status in {"failed", "failure"}


def _parse_task_ref(path: Path, payload: Dict[str, Any]) -> Tuple[str, str, int] | None:
    """
    Return (tasks_filename, branch, task_index) for an annotated log path.
    """
    branch = payload.get("branch")
    if not branch:
        return None

    match = re.search(r"_(\d+)\.json$", path.name)
    if not match:
        return None
    task_idx = int(match.group(1))

    suffix = f"_{branch}_{task_idx}.json"
    if path.name.endswith(suffix):
        tasks_filename = path.name[: -len(suffix)] + ".json"
        return tasks_filename, branch, task_idx

    # Fallback: strip trailing _{idx}.json and assume remaining is tasks file stem.
    base = path.name[: match.start()]
    tasks_filename = base + ".json"
    return tasks_filename, branch, task_idx


def _filter_tasks(
    payload: Dict[str, Any],
    keep_map: Dict[str, Dict[int, str]],
) -> Dict[str, Any]:
    tasks_by_branch = payload.get("tasks_by_branch") or {}
    new_tasks_by_branch: Dict[str, List[Dict[str, Any]]] = {}

    for branch, tasks in tasks_by_branch.items():
        branch_keep = keep_map.get(branch) or {}
        kept_tasks: List[Dict[str, Any]] = []
        for idx, t in enumerate(tasks):
            ann_path = branch_keep.get(idx)
            if not ann_path:
                continue
            new_t = dict(t)
            new_t["annotated_log_path"] = ann_path
            kept_tasks.append(new_t)
        if kept_tasks:
            new_tasks_by_branch[branch] = kept_tasks

    new_tasks: List[Dict[str, Any]] = []
    for branch in tasks_by_branch.keys():
        new_tasks.extend(new_tasks_by_branch.get(branch, []))

    payload["tasks_by_branch"] = new_tasks_by_branch
    payload["tasks"] = new_tasks
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_dir", default="/home/chy/state_aware_bench/bench/runs/testrun4/tasks")
    parser.add_argument("--annotated_logs_dir", default="/home/chy/state_aware_bench/bench/runs/testrun4/annotated_logs")
    parser.add_argument("--tasks_saved_dir", default="/home/chy/state_aware_bench/bench/runs/testrun4/saved_tasks")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    annotated_logs_dir = Path(args.annotated_logs_dir)
    tasks_saved_dir = Path(args.tasks_saved_dir)

    task_files = sorted(tasks_dir.rglob("*.json"))
    if not task_files:
        raise ValueError(f"No task files found under {tasks_dir}")

    tasks_by_name: Dict[str, Path] = {}
    for p in task_files:
        if p.name in tasks_by_name:
            print(f"[warn] duplicate tasks file name: {p.name} -> {p} (kept {tasks_by_name[p.name]})")
            continue
        tasks_by_name[p.name] = p

    keeps: Dict[Path, Dict[str, Dict[int, str]]] = {}
    for ann_path in sorted(annotated_logs_dir.rglob("*.json")):
        payload = _load_json(ann_path)
        if _is_failure(payload):
            continue
        parsed = _parse_task_ref(ann_path, payload)
        if not parsed:
            print(f"[warn] {ann_path.name}: cannot parse task reference")
            continue
        tasks_filename, branch, task_idx = parsed
        task_path = tasks_by_name.get(tasks_filename)
        if not task_path:
            print(f"[warn] {ann_path.name}: tasks file '{tasks_filename}' not found under {tasks_dir}")
            continue
        per_file = keeps.setdefault(task_path, {})
        per_branch = per_file.setdefault(branch, {})
        if task_idx in per_branch:
            print(f"[warn] {ann_path.name}: duplicate annotated log for {tasks_filename} {branch}[{task_idx}]")
            continue
        per_branch[task_idx] = str(ann_path)

    # write all tasks files; include only tasks with annotated logs
    for task_path in task_files:
        payload = _load_json(task_path)
        payload = _filter_tasks(payload, keeps.get(task_path, {}))

        rel_path = task_path.relative_to(tasks_dir)
        out_path = tasks_saved_dir / rel_path
        _save_json(out_path, payload)
        print(f"[saved] {task_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
