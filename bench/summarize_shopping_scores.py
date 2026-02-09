"""
Summarize shopping evaluation scores per model.

Reads evaluated task files (from evaluate_trajs_shopping.py output) and
aggregates total and per-component scores by assistant model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_detail_score(details: Dict[str, Any], key: str) -> float:
    section = details.get(key)
    if isinstance(section, dict):
        return _safe_float(section.get("score", 0.0), 0.0)
    return 0.0


def _get_pid_rate(details: Dict[str, Any]) -> float:
    section = details.get("product_id_match")
    if not isinstance(section, dict):
        return 0.0
    expected = section.get("expected", [])
    matched = section.get("matched", [])
    if not isinstance(expected, list) or not expected:
        return 0.0
    if not isinstance(matched, list):
        return 0.0
    return len(matched) / len(expected)


def _print_table(rows: list[Dict[str, Any]]) -> None:
    headers = [
        "model",
        "n",
        "avg_score",
        "avg_max",
        "avg_ratio",
        "ok_rate",
        "avg_pid",
        "avg_qty_size",
        "avg_voucher",
        "avg_balance",
    ]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(fmt(row.get(h, ""))))

    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        line = "  ".join(fmt(row.get(h, "")).ljust(col_widths[h]) for h in headers)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize shopping scores by model.")
    parser.add_argument(
        "--evaluated_dir",
        default="/home/chy/state_aware_bench/bench/runs/testrun5_inference2/evaluated_tasks_all",
        help="Directory containing evaluated task files.",
    )
    parser.add_argument(
        "--branch",
        default="shopping",
        help="Branch name to summarize (default: shopping).",
    )
    args = parser.parse_args()

    eval_root = Path(args.evaluated_dir)
    eval_files = sorted(p for p in eval_root.rglob("*.json") if p.is_file())
    if not eval_files:
        print("No evaluated task files matched.")
        return

    by_model: Dict[str, Dict[str, float]] = {}

    for path in eval_files:
        payload = _load_json(path)
        if payload is None:
            continue
        by_branch = payload.get("evaluations_by_branch", {})
        if not isinstance(by_branch, dict):
            continue
        tasks = by_branch.get(args.branch, [])
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            evaluations = task.get("evaluations", []) if isinstance(task, dict) else []
            if not isinstance(evaluations, list):
                continue
            for entry in evaluations:
                if not isinstance(entry, dict):
                    continue
                model = str(entry.get("assistant_model", "") or "unknown")
                evaluation = entry.get("evaluation", {})
                if not isinstance(evaluation, dict):
                    continue
                details = evaluation.get("details", {})
                if not isinstance(details, dict):
                    details = {}

                model_stats = by_model.setdefault(
                    model,
                    {
                        "n": 0.0,
                        "ok": 0.0,
                        "score": 0.0,
                        "max_score": 0.0,
                        "score_ratio": 0.0,
                        "pid_rate": 0.0,
                        "qty_size": 0.0,
                        "voucher": 0.0,
                        "balance": 0.0,
                    },
                )

                model_stats["n"] += 1.0
                if evaluation.get("ok") is True:
                    model_stats["ok"] += 1.0
                model_stats["score"] += _safe_float(evaluation.get("score", 0.0), 0.0)
                model_stats["max_score"] += _safe_float(evaluation.get("max_score", 0.0), 0.0)
                model_stats["score_ratio"] += _safe_float(evaluation.get("score_ratio", 0.0), 0.0)
                model_stats["pid_rate"] += _get_pid_rate(details)
                model_stats["qty_size"] += _get_detail_score(details, "quantity_size_match")
                model_stats["voucher"] += _get_detail_score(details, "voucher_ids_match")
                model_stats["balance"] += _get_detail_score(details, "balance_check")

    rows = []
    for model, s in sorted(by_model.items()):
        n = s["n"] or 1.0
        rows.append(
            {
                "model": model,
                "n": int(s["n"]),
                "avg_score": s["score"] / n,
                "avg_max": s["max_score"] / n,
                "avg_ratio": s["score_ratio"] / n,
                "ok_rate": s["ok"] / n,
                "avg_pid": s["pid_rate"] / n,
                "avg_qty_size": s["qty_size"] / n,
                "avg_voucher": s["voucher"] / n,
                "avg_balance": s["balance"] / n,
            }
        )

    _print_table(rows)


if __name__ == "__main__":
    main()

