"""
Rule-based evaluator for shopping trajectories.

Checks:
- Single completed order transaction
- Purchased items (product_id + quantity)
- Voucher usage (all required vouchers)
- Wallet balance <= max
- VIP status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _find_inference_log_for_task(tasks_path: Path, branch: str, task_idx: int) -> Optional[Path]:
    if not tasks_path or not branch:
        return None
    run_root = tasks_path.parent.parent
    if not run_root.is_dir():
        return None
    inference_dir = run_root / "inference_logs"
    if not inference_dir.is_dir():
        return None
    tasks_stem = tasks_path.stem
    candidate = inference_dir / f"{tasks_stem}_{branch}_{task_idx}.json"
    return candidate if candidate.is_file() else None


def _parse_tool_result(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(msg, dict):
        return None
    result = msg.get("result")
    if isinstance(result, dict):
        return result
    content = msg.get("content")
    if isinstance(content, str) and content:
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _last_tool_result(messages: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    last = None
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool":
            continue
        if msg.get("name") != tool_name:
            continue
        parsed = _parse_tool_result(msg)
        if isinstance(parsed, dict):
            last = parsed
    return last


def _sum_items(items: List[Dict[str, Any]]) -> Dict[tuple, int]:
    out: Dict[tuple, int] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        pid = it.get("product_id")
        size = it.get("size", "")
        qty = it.get("quantity", 0)
        if pid is None:
            continue
        try:
            qty_int = int(qty)
        except (TypeError, ValueError):
            qty_int = 0
        key = (str(pid), str(size or ""))
        out[key] = out.get(key, 0) + qty_int
    return out


def evaluate_traj(traj: Dict[str, Any], expected_tx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate shopping trajectory based on final shopping_state.
    
    Scoring:
    1. Product IDs match (1 point per expected product_id)
    2. Quantity and size match exactly (1 point)
    3. Voucher usage matches expected (1 point)
    4. Only one order transaction (excluding VIP transactions) (1 point)
    5. Cart is empty (1 point)
    6. Balance <= 5.0 (1 point)
    """
    shopping_state = traj.get("shopping_state", {}) or {}
    
    if not shopping_state:
        return {
            "ok": False,
            "score": 0.0,
            "max_score": 0.0,
            "issues": ["missing_shopping_state"],
            "expected_transaction": expected_tx,
            "observed": {"shopping_state": None},
        }
    
    wallet = shopping_state.get("wallet", {}) or {}
    cart = shopping_state.get("cart", []) or []
    transactions = shopping_state.get("transactions", []) or []
    
    expected_items = expected_tx.get("items", []) or []
    balance_max = float(expected_tx.get("wallet_balance_max", 5.0))
    
    issues: List[str] = []
    score = 0.0
    max_score = 0.0
    details: Dict[str, Any] = {}
    
    # Find completed order transaction
    completed_order_tx = None
    for tx in transactions:
        if isinstance(tx, dict) and tx.get("type") == "order" and tx.get("status") == "completed":
            completed_order_tx = tx
            break
    
    product_id_ok = False
    quantity_size_ok = False

    # 1. Check product IDs (1 point per expected product_id)
    if expected_items:
        expected_pids = {item.get("product_id") for item in expected_items if item.get("product_id")}
        max_score += len(expected_pids)

        actual_pids: set[str] = set()
        for tx in transactions:
            if not isinstance(tx, dict):
                continue
            for item in tx.get("items", []) or []:
                pid = item.get("product_id")
                if pid:
                    actual_pids.add(str(pid))

        matched_pids = expected_pids & actual_pids
        score += len(matched_pids)
        details["product_id_match"] = {
            "expected": list(expected_pids),
            "actual": list(actual_pids),
            "matched": list(matched_pids),
            "score": len(matched_pids),
        }

        if len(matched_pids) < len(expected_pids):
            issues.append(f"missing_product_ids: {list(expected_pids - actual_pids)}")
        else:
            product_id_ok = True
    else:
        issues.append("no_expected_items")
    
    # 2. Check quantity and size match exactly (1 point)
    # Must match on completed order transactions, and each target item must be found.
    if expected_items:
        max_score += 1

        def _normalize_item(it: Dict[str, Any]) -> Optional[tuple]:
            pid = it.get("product_id")
            size = str(it.get("size", "") or "")
            qty = it.get("quantity", 0)
            try:
                qty_int = int(qty)
            except (TypeError, ValueError):
                qty_int = 0
            if not pid or not size:
                return None
            return (str(pid), size, qty_int)

        expected_multiset: Dict[tuple, int] = {}
        for it in expected_items:
            if not isinstance(it, dict):
                continue
            key = _normalize_item(it)
            if key:
                expected_multiset[key] = expected_multiset.get(key, 0) + 1

        actual_multiset: Dict[tuple, int] = {}
        for tx in transactions:
            if not isinstance(tx, dict):
                continue
            if tx.get("type") != "order" or tx.get("status") != "completed":
                continue
            for it in tx.get("items", []) or []:
                if not isinstance(it, dict):
                    continue
                key = _normalize_item(it)
                if key:
                    actual_multiset[key] = actual_multiset.get(key, 0) + 1

        match_ok = True
        missing: Dict[tuple, int] = {}
        for key, needed in expected_multiset.items():
            have = actual_multiset.get(key, 0)
            if have < needed:
                match_ok = False
                missing[key] = needed - have

        if match_ok and expected_multiset:
            score += 1
            quantity_size_ok = True
            details["quantity_size_match"] = {"match": True, "score": 1}
        else:
            issues.append("quantity_size_mismatch")
            details["quantity_size_match"] = {
                "match": False,
                "expected": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in expected_multiset.items()},
                "actual": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in actual_multiset.items()},
                "missing": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in missing.items()},
                "score": 0,
            }
    else:
        issues.append("no_expected_items")
    
    # 3. Check voucher_ids match exactly (1 point)
    max_score += 1
    expected_voucher_ids = set(str(vid) for vid in (expected_tx.get("voucher_ids", []) or []))
    best_voucher_match = False
    best_actual_voucher_ids: set[str] = set()

    for tx in transactions:
        if not isinstance(tx, dict):
            continue
        if tx.get("type") != "order" or tx.get("status") != "completed":
            continue
        actual_voucher_id = tx.get("voucher_id")
        actual_voucher_ids = {str(actual_voucher_id)} if actual_voucher_id else set()
        if actual_voucher_ids == expected_voucher_ids:
            best_voucher_match = True
            best_actual_voucher_ids = actual_voucher_ids
            break
        if not best_actual_voucher_ids and actual_voucher_ids:
            best_actual_voucher_ids = actual_voucher_ids

    if best_voucher_match:
        score += 1
        details["voucher_ids_match"] = {
            "match": True,
            "expected": list(expected_voucher_ids),
            "actual": list(best_actual_voucher_ids),
            "score": 1,
        }
    else:
        issues.append(f"voucher_ids_mismatch: expected {list(expected_voucher_ids)}, got {list(best_actual_voucher_ids)}")
        details["voucher_ids_match"] = {
            "match": False,
            "expected": list(expected_voucher_ids),
            "actual": list(best_actual_voucher_ids),
            "score": 0,
        }
    
    # 4. Check balance <= 5.0 (1 point)
    max_score += 1
    balance = wallet.get("balance")
    if product_id_ok and quantity_size_ok:
        if balance is not None:
            try:
                balance_float = float(balance)
                if balance_float <= balance_max:
                    score += 1
                    details["balance_check"] = {"match": True, "balance": balance_float, "max": balance_max, "score": 1}
                else:
                    issues.append(f"balance_exceeded: {balance_float} > {balance_max}")
                    details["balance_check"] = {
                        "match": False,
                        "balance": balance_float,
                        "max": balance_max,
                        "score": 0,
                    }
            except (TypeError, ValueError):
                issues.append("balance_invalid")
                details["balance_check"] = {"match": False, "balance": balance, "score": 0}
        else:
            issues.append("balance_missing")
            details["balance_check"] = {"match": False, "balance": None, "score": 0}
    else:
        issues.append("balance_skipped_prereq")
        details["balance_check"] = {"match": False, "balance": balance, "score": 0, "skipped": True}
    
    ok = (score == max_score) and max_score > 0
    
    return {
        "ok": ok,
        "score": score,
        "max_score": max_score,
        "score_ratio": score / max_score if max_score > 0 else 0.0,
        "issues": issues,
        "details": details,
        "expected_transaction": expected_tx,
        "observed": {
            "shopping_state": {
                "wallet": wallet,
                "cart": cart,
                "transactions": transactions,
            },
        },
    }


def evaluate_task(task: Dict[str, Any], branch: str, tasks_path: Path, task_idx: int) -> Dict[str, Any]:
    log_path = _find_inference_log_for_task(tasks_path, branch, task_idx)
    if not log_path or not log_path.is_file():
        return {
            "task": task,
            "ok": False,
            "failure_category": "missing_log",
            "reason": f"log_file_not_found:{tasks_path.stem}_{branch}_{task_idx}.json",
        }

    log_payload = _load_json(log_path)
    if log_payload is None:
        return {
            "task": task,
            "ok": False,
            "failure_category": "read_error",
            "reason": "read_payload_error",
            "log_path": str(log_path),
        }

    targets = task.get("targets", [])
    expected_tx = {}
    if isinstance(targets, list) and targets:
        tx = targets[0].get("transaction") if isinstance(targets[0], dict) else None
    else:
        tx = None
    if isinstance(tx, dict):
        expected_tx = tx
    if not expected_tx:
        return {
            "task": task,
            "ok": False,
            "failure_category": "missing_targets",
            "reason": "transaction_targets_not_found",
            "log_path": str(log_path),
        }

    evaluations = []
    for idx, traj in enumerate(log_payload.get("trajs", []) or []):
        if not isinstance(traj, dict):
            continue
        eval_result = evaluate_traj(traj, expected_tx)
        evaluations.append({
            "traj_idx": idx,
            "assistant_model": traj.get("assistant_model", ""),
            "evaluation": eval_result,
        })
    
    # Calculate summary statistics
    successful_evals = [e for e in evaluations if e.get("evaluation", {}).get("ok") is not None]
    if successful_evals:
        scores = [e["evaluation"].get("score", 0.0) for e in successful_evals]
        max_scores = [e["evaluation"].get("max_score", 0.0) for e in successful_evals]
        score_ratios = [e["evaluation"].get("score_ratio", 0.0) for e in successful_evals]
        
        summary = {
            "total_trajs": len(evaluations),
            "successful_evals": len(successful_evals),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_max_score": sum(max_scores) / len(max_scores) if max_scores else 0.0,
            "avg_score_ratio": sum(score_ratios) / len(score_ratios) if score_ratios else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }
    else:
        summary = {
            "total_trajs": len(evaluations),
            "successful_evals": 0,
            "avg_score": 0.0,
            "avg_max_score": 0.0,
            "avg_score_ratio": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
        }

    return {
        "task": task,
        "branch": branch,
        "log_path": str(log_path),
        "evaluations": evaluations,
        "summary": summary,
    }


def _write_evaluated(path: Path, evaluated_payload: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    out_path.write_text(json.dumps(evaluated_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluated] {path.name} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based evaluator for shopping trajectories.")
    parser.add_argument(
        "--tasks_dir",
        default="/home/chy/state_aware_bench/bench/runs/testrun5_inference5/tasks",
        help="Directory containing task files (user+task JSON).",
    )
    parser.add_argument(
        "--branch",
        default="shopping",
        help="Branch name to evaluate, or 'all' for every branch.",
    )
    parser.add_argument(
        "--evaluated_tasks_dir",
        default="/home/chy/state_aware_bench/bench/runs/testrun5_inference5/evaluated_tasks_3_all",
        help="Where to save evaluated task files.",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_dir)
    task_files = sorted(p for p in tasks_root.rglob("*.json") if p.is_file())
    if not task_files:
        print("No task files matched.")

    out_dir = Path(args.evaluated_tasks_dir)
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
                        tasks_path=task_path,
                        task_idx=task_idx,
                    )
                )
            evaluations_by_branch[str(branch)] = results

        _write_evaluated(task_path, {"evaluations_by_branch": evaluations_by_branch}, out_dir)

