"""
Evaluate each trajectory with LLM-as-evaluator.

Reads task files (user+task JSON with annotated_log_path), finds corresponding raw logs,
and writes evaluated results to evaluated_tasks_dir/<same_filename>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bench.backend.utils.clients import call_llm, get_ann_route, is_gemini_client
from bench.backend.utils.misc import strip_code_fences
from bench.prompts.generation_prompt import base_evaluate_prompt, platform_overview
import ast


def _render_tools(tools: List[Dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, indent=2)


def _llm_temperature(model: str) -> Optional[float]:
    return None if str(model).startswith("gpt-5") else 0


def build_evaluator_prompt(
    *,
    branch: str,
    tools: List[Dict[str, Any]],
    solution_summary: str,
    targets: List[Any],
    full_messages: List[Any],
) -> str:
    sections = [
        platform_overview,
        base_evaluate_prompt,
        "### Available Tools\n" + _render_tools(tools),
        "### Solution Summary (for reference)\n" + (solution_summary or ""),
        "### Task Targets\n" + json.dumps(targets, ensure_ascii=False, indent=2),
        "### Conversation Log (full_messages)\n" + json.dumps(full_messages, ensure_ascii=False, indent=2),
    ]
    return "\n\n".join([s for s in sections if s is not None and str(s).strip() != ""])


def build_evaluator_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _call_with_retry(fn, *args, **kwargs):
    last_exc: Optional[Exception] = None
    for _ in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            import time

            time.sleep(60)
    if last_exc:
        raise last_exc


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_tool_schemas(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _tool_schema_name(item: Dict[str, Any]) -> str:
    func = item.get("function")
    if isinstance(func, dict):
        return str(func.get("name", ""))
    return str(item.get("name", ""))


def _expand_tool_schemas_for_names(
    tool_names: List[str],
    platform_schemas: List[Dict[str, Any]],
    med_schemas: List[Dict[str, Any]],
    source_schemas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    platform_map = {_tool_schema_name(s): s for s in platform_schemas}
    med_map = {_tool_schema_name(s): s for s in med_schemas}
    source_map = {_tool_schema_name(s): s for s in source_schemas}

    prefixed_sources = {name.split(".")[0] for name in tool_names if "." in name and not name.startswith("med.")}

    expanded: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(schema: Dict[str, Any]) -> None:
        name = _tool_schema_name(schema)
        if name and name not in seen:
            expanded.append(schema)
            seen.add(name)

    for name in tool_names:
        if name in platform_map:
            _add(platform_map[name])
        elif name in med_map:
            _add(med_map[name])

    for source in prefixed_sources:
        for base_name, schema in source_map.items():
            # Deep copy to avoid mutating base schema
            schema_copy = json.loads(json.dumps(schema))
            func = schema_copy.get("function", {})
            if func:
                func["name"] = f"{source}.{base_name}"
            else:
                schema_copy["name"] = f"{source}.{base_name}"
            _add(schema_copy)

    return expanded


def _tool_names_from_assistant_traj(traj: Dict[str, Any] | None) -> List[str]:
    if not isinstance(traj, dict):
        return []
    messages = traj.get("assistant_traj")
    if not isinstance(messages, list):
        return []
    names: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        if isinstance(calls, dict):
            calls = [calls]
        for call in calls:
            if not isinstance(call, dict):
                continue
            func = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = call.get("name") or func.get("name") or ""
            if name:
                names.append(str(name))
    return names


def _extract_tool_call_multiset(traj: Dict[str, Any] | None) -> Dict[Tuple[str, str], int]:
    if not isinstance(traj, dict):
        return {}
    messages = traj.get("assistant_traj")
    if not isinstance(messages, list):
        return {}
    counts: Dict[Tuple[str, str], int] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        if isinstance(calls, dict):
            calls = [calls]
        for call in calls:
            if not isinstance(call, dict):
                continue
            func = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = call.get("name") or func.get("name") or ""
            args = call.get("arguments", func.get("arguments", "")) or ""
            key = (str(name), str(args))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _multiset_metrics(
    reference: Dict[Tuple[str, str], int],
    candidate: Dict[Tuple[str, str], int],
) -> Dict[str, float]:
    def canonicalize_counter(counter: dict):
        def parse_args(s: str) -> dict:
            if s:
                try:
                    return json.loads(s)  # JSON: {"date": "2024-05-21"}
                except json.JSONDecodeError:
                    return ast.literal_eval(s)
            else:
                return {}


        def canonical_key(tool: str, arg_str: str):
            args = parse_args(arg_str)
            return (tool, json.dumps(args, sort_keys=True, separators=(",", ":")))

        new = {}
        for (tool, arg_str), v in counter.items():
            key = canonical_key(tool, arg_str)
            new[key] = new.get(key, 0) + v
        return new

    reference = canonicalize_counter(reference)
    candidate = canonicalize_counter(candidate)

    if not reference and not candidate:
        return {"jaccard": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = 0
    for key, ref_count in reference.items():
        cand_count = candidate.get(key, 0)
        overlap += min(ref_count, cand_count)
    ref_total = sum(reference.values())
    cand_total = sum(candidate.values())
    union_total = 0
    all_keys = set(reference.keys()) | set(candidate.keys())
    for key in all_keys:
        union_total += max(reference.get(key, 0), candidate.get(key, 0))
    precision = (overlap / cand_total) if cand_total else 0.0
    recall = (overlap / ref_total) if ref_total else 0.0
    jaccard = (overlap / union_total) if union_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _select_summary_and_targets(annotated: Dict[str, Any]) -> Tuple[str, List[Any]]:
    final_sel = annotated.get("final_selection") if isinstance(annotated.get("final_selection"), dict) else {}
    selected_model = str(final_sel.get("select_assistant_model") or "")
    targets = []
    if isinstance(annotated.get("aligned_targets"), list) and annotated.get("aligned_targets"):
        targets = annotated.get("aligned_targets", [])
    elif isinstance(annotated.get("targets"), list):
        targets = annotated.get("targets", [])

    def _find_summary(pool: List[Dict[str, Any]]) -> str:
        for t in pool:
            if not isinstance(t, dict):
                continue
            if selected_model and str(t.get("assistant_model") or "") != selected_model:
                continue
            ann = t.get("annotation") if isinstance(t.get("annotation"), dict) else {}
            return str(ann.get("trajectory_summary") or "")
        return ""

    summary = ""
    if selected_model:
        summary = _find_summary(annotated.get("rerun_trajs", [])) or _find_summary(annotated.get("trajs", []))
    if not summary:
        summary = _find_summary(annotated.get("trajs", []))
    return summary, targets


def _select_reference_traj(annotated: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    final_sel = annotated.get("final_selection") if isinstance(annotated.get("final_selection"), dict) else {}
    selected_model = str(final_sel.get("select_assistant_model") or "")
    if not selected_model:
        return None
    pools = [
        annotated.get("rerun_trajs", []),
        annotated.get("trajs", []),
    ]
    for pool in pools:
        if not isinstance(pool, list):
            continue
        for t in pool:
            if not isinstance(t, dict):
                continue
            if str(t.get("assistant_model") or "") == selected_model:
                return t
    return None


def _find_inference_log_for_task(tasks_path: Path, branch: str, task_idx: int) -> Optional[Path]:
    if not tasks_path or not branch:
        return None

    # tasks_path = .../runs/<run_id>/tasks_saved/<file>.json
    run_root = tasks_path.parent.parent
    if not run_root.is_dir():
        return None

    inference_dir = run_root / "inference_logs"
    if not inference_dir.is_dir():
        return None

    tasks_stem = tasks_path.stem
    candidate = inference_dir / f"{tasks_stem}_{branch}_{task_idx}.json"
    return candidate if candidate.is_file() else None



def evaluate_traj(
    *,
    ann_client,
    ann_model: str,
    branch: str,
    tools: List[Dict[str, Any]],
    solution_summary: str,
    targets: List[Any],
    full_messages: List[Any],
) -> Dict[str, Any]:
    messages = build_evaluator_messages(
        build_evaluator_prompt(
            branch=branch,
            tools=tools,
            solution_summary=solution_summary,
            targets=targets,
            full_messages=full_messages,
        )
    )

    content: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            content = _call_with_retry(
                call_llm,
                ann_client,
                model=ann_model,
                messages=messages,
                temperature=_llm_temperature(ann_model),
                response_mime_type="application/json" if is_gemini_client(ann_client) else None,
            )
        except Exception:
            continue
        try:
            parsed = json.loads(strip_code_fences(content or ""))
            break
        except Exception:
            continue

    if parsed is None or not isinstance(parsed, dict):
        return {
            "ok": False,
            "failure_category": "parse_error",
            "reason": "Evaluator parse failed after 3 attempts.",
            "llm_raw": content or "",
            "evaluator_model": ann_model,
        }

    parsed_out = dict(parsed)
    parsed_out.setdefault("evaluator_model", ann_model)
    parsed_out.setdefault("ok", True)
    return parsed_out


def _write_payload(path: Path, payload: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[evaluated] {path.name} -> {out_path}")


def evaluate_task(
    *,
    task: Dict[str, Any],
    branch: str,
    platform_schemas: List[Dict[str, Any]],
    med_schemas: List[Dict[str, Any]],
    source_schemas: List[Dict[str, Any]],
    ann_client,
    ann_model: str,
    tasks_path: Path,
    task_idx: int,
) -> Dict[str, Any]:
    annotated_path = Path(str(task.get("annotated_log_path") or "")).expanduser()
    if not annotated_path.is_file():
        return {
            "task": task,
            "ok": False,
            "failure_category": "missing_annotated",
            "reason": "annotated_file_not_found",
        }

    log_path = _find_inference_log_for_task(tasks_path, branch, task_idx)
    if not log_path or not log_path.is_file():
        return {
            "task": task,
            "ok": False,
            "failure_category": "missing_log",
            "reason": f"log_file_not_found:{tasks_path.stem}_{branch}_{task_idx}.json",
            "annotated_log_path": str(annotated_path),
        }

    annotated_payload = _load_json(annotated_path)
    log_payload = _load_json(log_path)
    if annotated_payload is None or log_payload is None:
        return {
            "task": task,
            "ok": False,
            "failure_category": "read_error",
            "reason": "read_payload_error",
            "annotated_log_path": str(annotated_path),
            "log_path": str(log_path),
        }

    branch_name = str(branch or log_payload.get("branch") or annotated_payload.get("branch") or "")

    final_sel = (
        annotated_payload.get("final_selection")
        if isinstance(annotated_payload.get("final_selection"), dict)
        else {}
    )
    if final_sel.get("status") == "failed":
        trajs = log_payload.get("trajs", [])
        if not isinstance(trajs, list):
            trajs = []
        results = []
        for idx, traj in enumerate(trajs):
            if not isinstance(traj, dict):
                continue
            results.append(
                {
                    "traj_idx": idx,
                    "assistant_model": traj.get("assistant_model", ""),
                    "evaluation": {
                        "ok": False,
                        "failure_category": "not_evaluable",
                        "reason": "final_selection_failed",
                        "final_selection": final_sel,
                        "evaluator_model": ann_model,
                    },
                }
            )
        if not results:
            results = [
                {
                    "ok": False,
                    "failure_category": "not_evaluable",
                    "reason": "final_selection_failed",
                    "final_selection": final_sel,
                    "evaluator_model": ann_model,
                }
            ]
        return {
            "task": task,
            "branch": branch_name,
            "annotated_log_path": str(annotated_path),
            "log_path": str(log_path),
            "evaluations": results,
        }

    solution_summary, targets = _select_summary_and_targets(annotated_payload)
    ref_traj = _select_reference_traj(annotated_payload) or {}
    ref_multiset = _extract_tool_call_multiset(ref_traj)
    ref_state_check = ref_traj.get("state_check")
    trajs = log_payload.get("trajs", [])
    if not isinstance(trajs, list):
        trajs = []

    results: List[Dict[str, Any]] = []
    for idx, traj in enumerate(trajs):
        if not isinstance(traj, dict):
            continue
        full_messages = traj.get("full_messages", [])
        if not isinstance(full_messages, list):
            full_messages = []
        tool_names = set(_tool_names_from_assistant_traj(traj))
        tool_names.update(_tool_names_from_assistant_traj(ref_traj))
        tools = _expand_tool_schemas_for_names(
            list(tool_names),
            platform_schemas=platform_schemas,
            med_schemas=med_schemas,
            source_schemas=source_schemas,
        )
        candidate_multiset = _extract_tool_call_multiset(traj)
        tool_call_metrics = _multiset_metrics(ref_multiset, candidate_multiset)
        # Even if terminated, still evaluate with LLM evaluator
        try:
            ev = evaluate_traj(
                ann_client=ann_client,
                ann_model=ann_model,
                branch=branch_name,
                tools=tools,
                solution_summary=solution_summary,
                targets=targets,
                full_messages=full_messages,
            )
            # Mark as terminated if the trajectory was terminated
            if traj.get("terminated") is True:
                ev["terminated"] = True
        except Exception as e:  # noqa: BLE001
            ev = {
                "ok": False,
                "failure_category": "error",
                "reason": f"evaluator_error:{e.__class__.__name__}:{str(e).strip().replace(chr(10), ' ')}",
                "evaluator_model": ann_model,
            }
            if traj.get("terminated") is True:
                ev["terminated"] = True
        ev["tool_call_metrics"] = tool_call_metrics
        ev["state_check"] = bool(ref_state_check == traj.get("state_check"))
        results.append(
            {
                "traj_idx": idx,
                "assistant_model": traj.get("assistant_model", ""),
                "evaluation": ev,
            }
        )

    return {
        "task": task,
        "branch": branch_name,
        "annotated_log_path": str(annotated_path),
        "log_path": str(log_path),
        "evaluations": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectories with LLM-as-evaluator.")
    default_base = Path(__file__).resolve().parent
    parser.add_argument(
        "--tasks_dir",
        default="/home/chy/state_aware_bench/bench/runs/testrun2_inference_rerun/saved_tasks",
        help="Directory containing task files (user+task JSON).",
    )
    parser.add_argument(
        "--branch",
        default="all",
        help="Branch name to evaluate, or 'all' for every branch.",
    )
    parser.add_argument(
        "--evaluated_tasks_dir",
        default="/home/chy/state_aware_bench/bench/runs/testrun2_inference_rerun/evaluated_tasks1",
        help="Where to save evaluated task files.",
    )
    args = parser.parse_args()

    ann_client, ann_model = get_ann_route()

    tool_schema_path = default_base / "backend" / "tool_schemas" / "platform_tools.json"
    med_schema_path = default_base / "backend" / "tool_schemas" / "med_tools.json"
    source_schema_path = default_base / "backend" / "tool_schemas" / "source_tools.json"
    tool_schemas = _load_tool_schemas(tool_schema_path)
    med_schemas = _load_tool_schemas(med_schema_path)
    source_schemas = _load_tool_schemas(source_schema_path)

    out_dir = Path(args.evaluated_tasks_dir)
    tasks_root = Path(args.tasks_dir)
    task_files = sorted(p for p in tasks_root.rglob("*.json") if p.is_file())
    if not task_files:
        print("No task files matched.")

    for task_path in task_files:
        payload = _load_json(task_path)
        if payload is None:
            continue
        tasks_by_branch = payload.get("tasks_by_branch") if isinstance(payload.get("tasks_by_branch"), dict) else {}
        evaluations_by_branch: Dict[str, List[Dict[str, Any]]] = {}
        for branch, tasks in tasks_by_branch.items():
            if args.branch != "all" and str(branch) != args.branch:
                continue
            if not isinstance(tasks, list):
                continue
            results = []
            for task_idx, task in enumerate(tasks):
                if not isinstance(task, dict):
                    continue
                results.append(
                    evaluate_task(
                        task=task,
                        branch=str(branch),
                        platform_schemas=tool_schemas,
                        med_schemas=med_schemas,
                        source_schemas=source_schemas,
                        ann_client=ann_client,
                        ann_model=ann_model,
                        tasks_path=task_path,
                        task_idx=task_idx,
                    )
                )
            evaluations_by_branch[str(branch)] = results

        out_payload = {
            "store_meta": payload.get("store_meta", {}),
            "profile": payload.get("profile", {}),
            "tasks_by_branch": tasks_by_branch,
            "evaluations_by_branch": evaluations_by_branch,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / task_path.name
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[evaluated] {task_path.name} -> {out_path}")


