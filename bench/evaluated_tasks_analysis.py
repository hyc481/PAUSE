from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from bench.backend.utils.clients import call_llm, get_ann_routes, is_gemini_client
from bench.backend.utils.misc import load_branch_tool_allowlist, load_tool_summaries, strip_code_fences
from bench.prompts.generation_prompt import base_classify_prompt, platform_overview


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
    return False


def _targets_ratio(ev: Dict[str, Any]) -> Tuple[int, int]:
    achieved = ev.get("targets_achieved")
    if not isinstance(achieved, list):
        return 0, 0
    total = len(achieved)
    ok = sum(1 for v in achieved if _safe_bool(v))
    return ok, total


def _task_completion(ev: Dict[str, Any]) -> bool:
    """Check if all targets are achieved (task completion)."""
    ok, total = _targets_ratio(ev)
    return total > 0 and ok == total


def _fmt_ratio(ok: int, total: int) -> str:
    return f"{ok}/{total}" if total else "0/0"


def _collect_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    evaluations_by_branch = payload.get("evaluations_by_branch")
    if not isinstance(evaluations_by_branch, dict):
        return rows
    for branch, tasks in evaluations_by_branch.items():
        if not isinstance(tasks, list):
            continue
        for task_idx, task_entry in enumerate(tasks):
            if not isinstance(task_entry, dict):
                continue
            task = task_entry.get("task") if isinstance(task_entry.get("task"), dict) else {}
            evaluations = task_entry.get("evaluations")
            if not isinstance(evaluations, list):
                continue
            for ev_entry in evaluations:
                if not isinstance(ev_entry, dict):
                    continue
                ev = ev_entry.get("evaluation") if isinstance(ev_entry.get("evaluation"), dict) else {}
                model = str(ev_entry.get("assistant_model") or "")
                ok, total = _targets_ratio(ev)
                # Support both new format (tool_calling_correct) and old format (tool_calling_logic_correct + tool_calling_parameter_correct)
                tool_call_correct_new = _safe_bool(ev.get("tool_calling_correct"))
                tool_call_correct_old = _safe_bool(ev.get("tool_calling_logic_correct")) and _safe_bool(
                    ev.get("tool_calling_parameter_correct")
                )
                tool_call_correct = tool_call_correct_new if ev.get("tool_calling_correct") is not None else tool_call_correct_old
                # For backward compatibility, also extract logic and param if available
                logic = _safe_bool(ev.get("tool_calling_logic_correct")) if ev.get("tool_calling_logic_correct") is not None else tool_call_correct
                param = _safe_bool(ev.get("tool_calling_parameter_correct")) if ev.get("tool_calling_parameter_correct") is not None else tool_call_correct
                metrics = ev.get("tool_call_metrics") if isinstance(ev.get("tool_call_metrics"), dict) else {}
                rows.append(
                    {
                        "branch": str(branch),
                        "task_idx": task_idx,
                        "label": task.get("label", ""),
                        "assistant_model": model,
                        "logic": logic,
                        "param": param,
                        "reason_policy": _safe_bool(ev.get("reason_policy_correct")),
                        "tool_call_correct": tool_call_correct,
                        "targets_ok": ok,
                        "targets_total": total,
                        "jaccard": float(metrics.get("jaccard", 0.0) or 0.0),
                        "precision": float(metrics.get("precision", 0.0) or 0.0),
                        "recall": float(metrics.get("recall", 0.0) or 0.0),
                        "f1": float(metrics.get("f1", 0.0) or 0.0),
                        "task_completion": _task_completion(ev),
                        "targets_achieved_ratio": (ok / total) if total > 0 else 0.0,
                        # Store evaluation data for failure classification
                        "evaluation": ev,
                        "task": task,
                        "traj_idx": ev_entry.get("traj_idx"),
                    }
                )
    return rows


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r.get("assistant_model", ""), []).append(r)

    for model, items in grouped.items():
        if not items:
            continue
        count = len(items)
        logic_avg = sum(1 for r in items if r["logic"]) / count
        param_avg = sum(1 for r in items if r["param"]) / count
        reason_avg = sum(1 for r in items if r["reason_policy"]) / count
        tool_call_correct_avg = sum(1 for r in items if r["tool_call_correct"]) / count
        task_completion_avg = sum(1 for r in items if r["task_completion"]) / count
        targets_achieved_avg = sum(r["targets_achieved_ratio"] for r in items) / count
        jaccard_avg = sum(r["jaccard"] for r in items) / count
        precision_avg = sum(r["precision"] for r in items) / count
        recall_avg = sum(r["recall"] for r in items) / count
        f1_avg = sum(r["f1"] for r in items) / count
        
        # Failure classification statistics (only for failed trajs with error_analysis)
        failure_categories: Dict[str, int] = {}
        failed_with_error_analysis_count = 0
        for r in items:
            # Count failed trajs with error_analysis
            if not r.get("task_completion", True):
                ev = r.get("evaluation", {})
                error_analysis = ev.get("error_analysis", "")
                if error_analysis and error_analysis.strip():
                    failed_with_error_analysis_count += 1
            
            # Count classification results
            category = r.get("failure_category")
            if category:
                failure_categories[category] = failure_categories.get(category, 0) + 1
        
        summary[model] = {
            "count": count,
            "logic_avg": logic_avg,
            "param_avg": param_avg,
            "reason_policy_avg": reason_avg,
            "tool_call_correct_avg": tool_call_correct_avg,
            "task_completion_avg": task_completion_avg,
            "targets_achieved_avg": targets_achieved_avg,
            "jaccard_avg": jaccard_avg,
            "precision_avg": precision_avg,
            "recall_avg": recall_avg,
            "f1_avg": f1_avg,
            "failure_categories": failure_categories,
            "failed_with_error_analysis_count": failed_with_error_analysis_count,
        }
    return summary


def _load_tools_for_branch(branch: str) -> List[Dict[str, str]]:
    """Load tool summaries for a given branch."""
    base_path = Path(__file__).parent.parent / "backend" / "tool_schemas"
    platform_path = base_path / "platform_tools.json"
    med_path = base_path / "med_tools.json"
    source_path = base_path / "source_tools.json"
    
    all_tools: List[Dict[str, str]] = []
    
    # Load platform tools
    if platform_path.exists():
        all_tools.extend(load_tool_summaries(platform_path))
    
    # Load med tools
    if med_path.exists():
        all_tools.extend(load_tool_summaries(med_path))
    
    # Load source tools
    if source_path.exists():
        all_tools.extend(load_tool_summaries(source_path))
    
    # Filter by branch allowlist if available
    allowlist = load_branch_tool_allowlist(branch)
    if allowlist:
        allow = set(allowlist)
        all_tools = [t for t in all_tools if t.get("name") in allow]
    
    return all_tools


def _build_classification_prompt(
    tools: List[Dict[str, str]],
    targets: List[str],
    trajectory_summary: str,
    error_analysis: str,
) -> str:
    """Build the prompt for failure classification."""
    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    targets_str = json.dumps(targets, ensure_ascii=False, indent=2)
    
    sections = [
        base_classify_prompt,
        platform_overview,
        # "### Available Tools\n" + tools_str,
        "### User Targets\n" + targets_str,
        "### Agent Trajectory Summary\n" + trajectory_summary,
        "### Error Analysis\n" + error_analysis,
    ]
    return "\n\n".join(sections)


def _classify_failure(
    client_obj,
    model: str,
    tools: List[Dict[str, str]],
    targets: List[str],
    trajectory_summary: str,
    error_analysis: str,
) -> Optional[Dict[str, str]]:
    """Classify failure reason using LLM."""
    prompt = _build_classification_prompt(tools, targets, trajectory_summary, error_analysis)
    messages = [{"role": "user", "content": prompt}]
    
    try:
        temperature = 0 if not model.startswith("gpt-5") else None
        response_mime_type = "application/json" if is_gemini_client(client_obj) else None
        content = call_llm(
            client_obj,
            model=model,
            messages=messages,
            temperature=temperature,
            response_mime_type=response_mime_type,
        )
        
        # Parse JSON response
        cleaned = strip_code_fences(content)
        parsed = json.loads(cleaned)
        
        if isinstance(parsed, dict):
            category = parsed.get("category", "").strip().upper()
            reason = parsed.get("reason", "").strip()
            
            # Validate category
            if category in ["A", "B", "C", "D", "E", "F"]:
                return {"category": category, "reason": reason}
        
        return None
    except Exception as e:
        print(f"[WARN] Failed to classify failure: {e}")
        return None




def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluated_tasks_3_all outputs.")
    parser.add_argument(
        "--evaluated_tasks_dir",
        default="/home/chy/state_aware_bench/bench/bench_tasks/abalation_run4/evaluated_tasks_pro",
        help="Directory containing evaluated task files.",
    )
    parser.add_argument(
        "--enable_failure_classification",
        action="store_true",
        help="Enable LLM-based failure classification for failed trajectories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result.json",
        help="Output JSON file path to save analysis results.",
    )
    args = parser.parse_args()

    root = Path(args.evaluated_tasks_dir)
    files = sorted(p for p in root.rglob("*.json") if p.is_file())
    if not files:
        print("No evaluated task files matched.")
        return

    all_rows: List[Dict[str, Any]] = []
    for path in files:
        payload = _load_json(path)
        if payload is None:
            print(f"[WARN] failed to load: {path.name}")
            continue
        rows = _collect_rows(payload)
        all_rows.extend(rows)
        for r in rows:
            print(
                f"{path.name} | {r['branch']} | task_idx={r['task_idx']} | "
                f"{r['assistant_model']} | logic={r['logic']} param={r['param']} "
                f"tool_call_correct={r['tool_call_correct']} reason_policy={r['reason_policy']} "
                f"targets={_fmt_ratio(r['targets_ok'], r['targets_total'])} "
                f"jaccard={r['jaccard']:.3f} precision={r['precision']:.3f} "
                f"recall={r['recall']:.3f} f1={r['f1']:.3f}"
            )

    # Perform failure classification if enabled
    if args.enable_failure_classification:
        print("\n[INFO] Starting failure classification...")
        ann_routes = get_ann_routes()
        if not ann_routes:
            print("[WARN] No annotator routes available. Skipping failure classification.")
        else:
            # Use first annotator route
            ann_route = ann_routes[0]
            client_obj = ann_route["client"]
            model = ann_route["model"]
            
            # Group rows by branch to load tools efficiently
            rows_by_branch: Dict[str, List[Dict[str, Any]]] = {}
            for r in all_rows:
                branch = r.get("branch", "")
                rows_by_branch.setdefault(branch, []).append(r)
            
            # Classify failures
            classified_count = 0
            failed_count = 0
            for branch, branch_rows in rows_by_branch.items():
                # Load tools for this branch
                tools = _load_tools_for_branch(branch)
                
                for r in branch_rows:
                    # Only classify failed trajectories (where task_completion is False)
                    if r.get("task_completion", False):
                        continue
                    
                    ev = r.get("evaluation", {})
                    task = r.get("task", {})
                    
                    trajectory_summary = ev.get("trajectory_summary", "")
                    error_analysis = ev.get("error_analysis", "")
                    targets = task.get("targets", [])
                    
                    # Only classify trajectories with error_analysis
                    if not error_analysis or not error_analysis.strip():
                        continue
                    
                    failed_count += 1
                    
                    # Skip if essential data is missing
                    if not trajectory_summary:
                        print(f"[WARN] Missing trajectory_summary for {r['assistant_model']} | branch={branch} | task_idx={r['task_idx']}")
                        continue
                    if not targets:
                        print(f"[WARN] Missing targets for {r['assistant_model']} | branch={branch} | task_idx={r['task_idx']}")
                        continue
                    
                    classification = _classify_failure(
                        client_obj,
                        model,
                        tools,
                        targets,
                        trajectory_summary,
                        error_analysis,
                    )
                    
                    if classification:
                        r["failure_category"] = classification["category"]
                        r["failure_reason"] = classification["reason"]
                        classified_count += 1
                        print(
                            f"[CLASSIFIED] {r['assistant_model']} | branch={branch} | "
                            f"task_idx={r['task_idx']} | category={classification['category']}"
                        )
            
            print(f"[INFO] Classified {classified_count}/{failed_count} failed trajectories with error_analysis.")

    summary = _summarize(all_rows)
    print("\nSUMMARY")
    for model, stats in summary.items():
        print(
            f"{model}: count={stats['count']} logic={stats['logic_avg']:.3f} "
            f"param={stats['param_avg']:.3f} tool_call_correct={stats['tool_call_correct_avg']:.3f} "
            f"reason_policy={stats['reason_policy_avg']:.3f} "
            f"task_completion={stats['task_completion_avg']:.3f} targets_achieved_avg={stats['targets_achieved_avg']:.3f} "
            f"jaccard={stats['jaccard_avg']:.3f} precision={stats['precision_avg']:.3f} "
            f"recall={stats['recall_avg']:.3f} f1={stats['f1_avg']:.3f}"
        )
        
        # Print failure classification statistics
        if args.enable_failure_classification:
            failed_with_error_analysis = stats.get("failed_with_error_analysis_count", 0)
            if failed_with_error_analysis > 0:
                print(f"  Failed trajectories with error_analysis: {failed_with_error_analysis}")
            if stats.get("failure_categories"):
                print(f"  Failure Categories (classified):")
                total_classified = sum(stats["failure_categories"].values())
                for category, count in sorted(stats["failure_categories"].items()):
                    category_names = {
                        "A": "Tool Use Failure",
                        "B": "Tool Discovery Failure",
                        "C": "Environment & Policy Misunderstanding",
                        "D": "Reasoning / Planning Failure",
                        "E": "Premature Task Completion",
                        "F": "Non-Assistant Failure",
                    }
                    name = category_names.get(category, category)
                    percentage = (count / total_classified * 100) if total_classified > 0 else 0
                    print(f"    {category} ({name}): {count} ({percentage:.1f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect failure classifications if enabled
    failure_classifications = []
    failure_category_summary = {}
    if args.enable_failure_classification:
        category_counts: Dict[str, int] = {}
        category_by_model: Dict[str, Dict[str, int]] = {}
        
        for row in all_rows:
            if row.get("failure_category"):
                category = row.get("failure_category")
                model = row.get("assistant_model", "")
                
                failure_classifications.append({
                    "branch": row.get("branch", ""),
                    "task_idx": row.get("task_idx"),
                    "label": row.get("label", ""),
                    "assistant_model": model,
                    "failure_category": category,
                    "failure_reason": row.get("failure_reason", ""),
                })
                
                # Count by category
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count by model and category
                if model not in category_by_model:
                    category_by_model[model] = {}
                category_by_model[model][category] = category_by_model[model].get(category, 0) + 1
        
        # Calculate percentages
        total_classified = len(failure_classifications)
        category_names = {
            "A": "Tool Use Failure",
            "B": "Tool Discovery Failure",
            "C": "Environment & Policy Misunderstanding",
            "D": "Reasoning / Planning Failure",
            "E": "Premature Task Completion",
            "F": "Non-Assistant Failure",
        }
        
        if total_classified > 0:
            failure_category_summary = {
                "total_classified": total_classified,
                "by_category": {
                    category: {
                        "count": category_counts.get(category, 0),
                        "percentage": round(category_counts.get(category, 0) / total_classified * 100, 1),
                        "name": category_names.get(category, category),
                    }
                    for category in ["A", "B", "C", "D", "E", "F"]
                },
                "by_model": {
                    model: {
                        category: {
                            "count": category_by_model[model].get(category, 0),
                            "percentage": round(
                                category_by_model[model].get(category, 0) / sum(category_by_model[model].values()) * 100, 
                                1
                            ) if sum(category_by_model[model].values()) > 0 else 0.0,
                            "name": category_names.get(category, category),
                        }
                        for category in ["A", "B", "C", "D", "E", "F"]
                    }
                    for model in category_by_model.keys()
                },
            }
    
    output_data = {
        "summary": summary,
        "total_rows": len(all_rows),
        "evaluated_tasks_dir": str(args.evaluated_tasks_dir),
        "failure_classification_enabled": args.enable_failure_classification,
    }
    
    if args.enable_failure_classification:
        output_data["failure_classifications"] = failure_classifications
        output_data["failure_classification_count"] = len(failure_classifications)
        output_data["failure_category_summary"] = failure_category_summary
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] Analysis results saved to: {output_path}")
    print(f"  - Summary statistics for {len(summary)} model(s)")
    if args.enable_failure_classification:
        print(f"  - Failure classifications: {len(failure_classifications)} entries")
        if failure_category_summary:
            print(f"\n  Failure Category Summary (Overall):")
            for category in ["A", "B", "C", "D", "E", "F"]:
                cat_info = failure_category_summary.get("by_category", {}).get(category, {})
                if cat_info.get("count", 0) > 0:
                    print(f"    {category} ({cat_info.get('name', category)}): {cat_info.get('count', 0)} ({cat_info.get('percentage', 0.0)}%)")


if __name__ == "__main__":
    main()

