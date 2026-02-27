"""
Error classification script using LLM-as-classifier.

Reads evaluated task files, filters evaluations with failures (targets_achieved contains False
and error_analysis is not empty), classifies error types using LLM, and summarizes counts by model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

from bench.utils.clients import call_llm, get_ann_route, is_gemini_client
from bench.utils.misc import strip_code_fences
from bench.prompts.generation_prompt import base_classify_prompt, platform_overview


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None on error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def build_classify_prompt(
    *,
    trajectory_summary: str,
    error_analysis: str,
    targets: List[Any],
) -> str:
    """Build the classification prompt from trajectory_summary, error_analysis, and targets."""
    sections = [
        platform_overview,
        base_classify_prompt,
        "### Task Targets\n" + json.dumps(targets, ensure_ascii=False, indent=2),
        "### trajectory_summary\n" + (trajectory_summary or ""),
        "### error_analysis\n" + (error_analysis or ""),
    ]
    return "\n\n".join([s for s in sections if s is not None and str(s).strip() != ""])


def build_classify_messages(prompt: str) -> List[Dict[str, str]]:
    """Build messages list for LLM call."""
    return [{"role": "user", "content": prompt}]


def _call_with_retry(fn, *args, **kwargs):
    """Retry function call up to 3 times with 60s delay."""
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


def classify_error(
    *,
    ann_client,
    ann_model: str,
    trajectory_summary: str,
    error_analysis: str,
    targets: List[Any],
) -> Dict[str, Any]:
    """Classify error type using LLM."""
    messages = build_classify_messages(
        build_classify_prompt(
            trajectory_summary=trajectory_summary,
            error_analysis=error_analysis,
            targets=targets,
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
                temperature=0,
                response_mime_type="application/json" if is_gemini_client(ann_client) else None,
            )
        except Exception:
            continue
        try:
            cleaned_content = strip_code_fences(content or "")
            parsed = json.loads(cleaned_content)
            break
        except Exception:
            continue

    if parsed is None or not isinstance(parsed, dict):
        return {
            "ok": False,
            "error_type": "PARSE_ERROR",
            "reason": "Classification parse failed after 3 attempts.",
            "llm_raw": content or "",
            "classifier_model": ann_model,
        }

    error_type = str(parsed.get("error_type", "OTHER")).strip().upper()
    valid_types = {"DCE", "TBE", "RNE", "SCE", "OTHER"}
    if error_type not in valid_types:
        error_type = "OTHER"

    return {
        "ok": True,
        "error_type": error_type,
        "classifier_model": ann_model,
        "raw_response": parsed,
    }


def should_classify(evaluation: Dict[str, Any]) -> bool:
    """Check if evaluation should be classified (has failures and error_analysis)."""
    if not isinstance(evaluation, dict):
        return False
    
    # Check if targets_achieved contains False
    targets_achieved = evaluation.get("targets_achieved")
    if not isinstance(targets_achieved, list):
        return False
    has_failure = any(ta is False for ta in targets_achieved)
    if not has_failure:
        return False
    
    # Check if error_analysis is not empty
    error_analysis = evaluation.get("error_analysis")
    if not error_analysis or not str(error_analysis).strip():
        return False
    
    return True


def process_task_file(
    *,
    task_path: Path,
    model_name: str,
    ann_client,
    ann_model: str,
) -> List[Dict[str, Any]]:
    """Process a single task file for a specific model and return classification results."""
    payload = _load_json(task_path)
    if payload is None:
        return []
    
    evaluations_by_branch = payload.get("evaluations_by_branch")
    if not isinstance(evaluations_by_branch, dict):
        return []
    
    results: List[Dict[str, Any]] = []
    
    for branch, evaluations in evaluations_by_branch.items():
        if not isinstance(evaluations, list):
            continue
        
        for eval_item in evaluations:
            if not isinstance(eval_item, dict):
                continue
            
            # Get task to extract targets
            task = eval_item.get("task")
            if not isinstance(task, dict):
                task = {}
            targets = task.get("targets", [])
            if not isinstance(targets, list):
                targets = []
            
            # Get evaluations array from this item
            evals = eval_item.get("evaluations")
            if not isinstance(evals, list):
                continue
            
            for eval_data in evals:
                if not isinstance(eval_data, dict):
                    continue
                
                assistant_model = str(eval_data.get("assistant_model", ""))
                if assistant_model != model_name:
                    continue
                
                evaluation = eval_data.get("evaluation")
                if not isinstance(evaluation, dict):
                    continue
                
                # Check filtering conditions
                if not should_classify(evaluation):
                    continue
                
                trajectory_summary = str(evaluation.get("trajectory_summary", ""))
                error_analysis = str(evaluation.get("error_analysis", ""))
                
                # Classify error
                classification = classify_error(
                    ann_client=ann_client,
                    ann_model=ann_model,
                    trajectory_summary=trajectory_summary,
                    error_analysis=error_analysis,
                    targets=targets,
                )
                
                results.append({
                    "task_file": task_path.name,
                    "branch": str(branch),
                    "assistant_model": assistant_model,
                    "trajectory_summary": trajectory_summary,
                    "error_analysis": error_analysis,
                    "classification": classification,
                })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify errors in evaluated trajectories using LLM.")
    default_base = Path(__file__).resolve().parent
    parser.add_argument(
        "--tasks_dir",
        required=True,
        help="Directory containing evaluated task files (JSON).",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        nargs="+",
        help="Model name(s) to filter evaluations (e.g., 'gemini-2.5-pro' or 'gemini-2.5-pro' 'gpt-4.1').",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output file path for classification results (JSON). If not specified, prints to stdout.",
    )
    args = parser.parse_args()

    ann_client, ann_model = get_ann_route()

    tasks_root = Path(args.tasks_dir)
    task_files = sorted(p for p in tasks_root.rglob("*.json") if p.is_file())
    if not task_files:
        print("No task files matched.")
        exit(0)

    model_names = args.model_name
    print(f"Processing {len(task_files)} task files for {len(model_names)} model(s): {', '.join(model_names)}")
    
    # Process each model separately
    all_results_by_model: Dict[str, List[Dict[str, Any]]] = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        model_results: List[Dict[str, Any]] = []
        for task_path in task_files:
            print(f"  Processing: {task_path.name}")
            results = process_task_file(
                task_path=task_path,
                model_name=model_name,
                ann_client=ann_client,
                ann_model=ann_model,
            )
            model_results.extend(results)
            if results:
                print(f"    Found {len(results)} evaluations to classify")
        
        all_results_by_model[model_name] = model_results
        print(f"  Total evaluations for {model_name}: {len(model_results)}")

    # Count error types by model
    error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_results: List[Dict[str, Any]] = []
    
    for model_name, results in all_results_by_model.items():
        all_results.extend(results)
        for result in results:
            classification = result["classification"]
            if classification.get("ok"):
                error_type = classification.get("error_type", "OTHER")
                error_counts[model_name][error_type] += 1
            else:
                error_counts[model_name]["PARSE_ERROR"] += 1

    # Prepare output
    output_data = {
        "model_names": model_names,
        "classifier_model": ann_model,
        "total_evaluations": len(all_results),
        "error_counts_by_model": dict(error_counts),
        "detailed_results_by_model": all_results_by_model,
        "detailed_results": all_results,
    }

    # Output results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n[Classification complete] Results saved to: {output_path}")
    else:
        print("\n" + json.dumps(output_data, ensure_ascii=False, indent=2))

    # Print summary
    print("\n" + "="*60)
    print("=== Error Type Summary ===")
    print("="*60)
    for model_name in model_names:
        counts = error_counts[model_name]
        if not counts:
            print(f"\nModel: {model_name} - No evaluations found")
            continue
        print(f"\nModel: {model_name}")
        total = sum(counts.values())
        for error_type, count in sorted(counts.items()):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {error_type}: {count} ({percentage:.1f}%)")
        print(f"  Total: {total}")

