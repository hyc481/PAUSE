from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _tool_call_count(traj: Dict[str, Any]) -> int:
    messages = traj.get("assistant_traj")
    if not isinstance(messages, list):
        messages = traj.get("full_messages")
    if not isinstance(messages, list):
        return 0
    count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        if isinstance(calls, dict):
            count += 1
        elif isinstance(calls, list):
            count += len(calls)
    return count


def _extract_traj_judges(trajs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, traj in enumerate(trajs):
        annotations = traj.get("annotations") if isinstance(traj.get("annotations"), list) else []
        out.append(
            {
                "traj_idx": idx,
                "assistant_model": traj.get("assistant_model", ""),
                "tool_call_count": _tool_call_count(traj),
                "annotations": annotations,
            }
        )
    return out


def _print_traj_judges(label: str, trajs: List[Dict[str, Any]]) -> None:
    print(f"{label}: n={len(trajs)}")
    for idx, traj in enumerate(trajs):
        model = traj.get("assistant_model", "")
        tool_calls = _tool_call_count(traj)
        print(f"  traj[{idx}] model={model} tool_calls={tool_calls}")
        annotations = traj.get("annotations") if isinstance(traj.get("annotations"), list) else []
        for a_idx, ann in enumerate(annotations):
            ann_model = ann.get("annotator_model") or ann.get("annotator_key") or ""
            logic = ann.get("tool_calling_logic_correct")
            params = ann.get("tool_calling_parameter_correct")
            reason_policy = ann.get("reason_policy_correct")
            error_analysis = ann.get("error_analysis")
            print(
                f"    annotator[{a_idx}] model={ann_model} logic={logic} "
                f"params={params} reason_policy={reason_policy}"
            )
            print(f"      error_analysis={error_analysis}")


def _print_selection(label: str, sel: Any) -> None:
    if not isinstance(sel, dict):
        print(f"{label}: (none)")
        return
    print(f"{label}: {json.dumps(sel, ensure_ascii=False)}")


def analyze_file(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        print(f"[WARN] failed to load: {path.name}")
        return {"file": path.name, "load_error": True}

    trajs = payload.get("trajs", []) if isinstance(payload.get("trajs"), list) else []
    rerun_trajs = payload.get("rerun_trajs", []) if isinstance(payload.get("rerun_trajs"), list) else []

    print("\n" + "=" * 80)
    print(f"FILE: {path.name}")
    _print_traj_judges("trajs (initial)", trajs)

    selection = payload.get("selection")
    _print_selection("selection", selection)

    if rerun_trajs:
        _print_traj_judges("trajs (rerun)", rerun_trajs)
        _print_selection("selection (post-rerun)", selection)

    align_targets = payload.get("align_targets")
    aligned_targets = payload.get("aligned_targets")
    targets_adjust_reason = payload.get("targets_adjust_reason")
    print(f"align_targets: {json.dumps(align_targets, ensure_ascii=False)}")
    print(f"aligned_targets: {json.dumps(aligned_targets, ensure_ascii=False)}")
    print(f"targets_adjust_reason: {targets_adjust_reason}")

    final_selection = payload.get("final_selection")
    _print_selection("final_selection", final_selection)

    return {
        "file": path.name,
        "trajs": _extract_traj_judges(trajs),
        "selection": selection,
        "rerun_trajs": _extract_traj_judges(rerun_trajs),
        "align_targets": align_targets,
        "aligned_targets": aligned_targets,
        "targets_adjust_reason": targets_adjust_reason,
        "final_selection": final_selection,
    }


def build_summary(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_by_model: Dict[str, List[str]] = {}
    discarded: List[Dict[str, Any]] = []
    for item in analyses:
        if item.get("load_error"):
            discarded.append({"file": item.get("file"), "reason": "load_error"})
            continue
        final_sel = item.get("final_selection") if isinstance(item.get("final_selection"), dict) else {}
        model = str(final_sel.get("select_assistant_model") or "")
        tie = final_sel.get("tie", True)
        if model and not tie:
            selected_by_model.setdefault(model, []).append(item.get("file"))
        else:
            discarded.append(
                {
                    "file": item.get("file"),
                    "reason": final_sel.get("failure_category") or final_sel.get("select_reason") or "no_selection",
                }
            )
    return {
        "total_files": len(analyses),
        "selected_by_model": selected_by_model,
        "discarded": discarded,
    }


def _classify_file(item: Dict[str, Any], *, model_1: str, model_2: str) -> str:
    if item.get("load_error"):
        return "missing"
    final_sel = item.get("final_selection") if isinstance(item.get("final_selection"), dict) else {}
    if not final_sel:
        return "missing"
    tie = bool(final_sel.get("tie", True))
    if tie:
        return "discarded"
    selected = str(final_sel.get("select_assistant_model") or "")
    if selected and selected == model_1:
        return "model1"
    if selected and selected == model_2:
        return "model2"
    return "discarded"


def _build_confusion_matrix(
    current: List[Dict[str, Any]],
    compare: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    labels = ["model1", "model2", "discarded", "missing"]
    matrix: Dict[str, Dict[str, int]] = {a: {b: 0 for b in labels} for a in labels}
    cur_map = {str(item.get("file")): item for item in current}
    cmp_map = {str(item.get("file")): item for item in compare}
    def _first_models(items: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        for item in items.values():
            if not isinstance(item, dict) or item.get("load_error"):
                continue
            trajs = item.get("trajs") if isinstance(item.get("trajs"), list) else []
            model_1 = str((trajs[0].get("assistant_model") if len(trajs) > 0 else "") or "")
            model_2 = str((trajs[1].get("assistant_model") if len(trajs) > 1 else "") or "")
            if model_1 and model_2:
                return model_1, model_2
        return "", ""

    cur_model_1, cur_model_2 = _first_models(cur_map)
    cmp_model_1, cmp_model_2 = _first_models(cmp_map)
    all_files = sorted(set(cur_map.keys()) | set(cmp_map.keys()))
    for name in all_files:
        cur_label = _classify_file(
            cur_map.get(name, {"file": name, "load_error": True}),
            model_1=cur_model_1,
            model_2=cur_model_2,
        )
        cmp_label = _classify_file(
            cmp_map.get(name, {"file": name, "load_error": True}),
            model_1=cmp_model_1,
            model_2=cmp_model_2,
        )
        matrix[cur_label][cmp_label] += 1
    return labels, matrix


def _print_confusion_matrix(labels: List[str], matrix: Dict[str, Dict[str, int]]) -> None:
    if not labels:
        print("CONFUSION_MATRIX: (empty)")
        return
    header = ["current\\compare"] + labels
    col_widths = [max(len(h), 7) for h in header]
    for i, col in enumerate(labels, start=1):
        max_cell = max(len(str(matrix[row][col])) for row in labels)
        col_widths[i] = max(col_widths[i], max_cell)

    def _fmt_row(cols: List[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))

    print(_fmt_row(header))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    for row in labels:
        row_vals = [row] + [str(matrix[row][col]) for col in labels]
        print(_fmt_row(row_vals))


def _is_task_file(path: Path) -> bool:
    return path.name.startswith("tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze annotated_logs and print detailed judgments.")
    parser.add_argument(
        "--annotated_dir",
        default="/home/chy/state_aware_bench/bench/bench_tasks/testrun2_inference1/inference_logs",
        help="Path to annotated_logs directory.",
    )
    parser.add_argument(
        "--compare_dir",
        default=None,
        help="Optional path to another annotated_logs directory to compare against.",
    )
    args = parser.parse_args()

    root = Path(args.annotated_dir)
    files = sorted(p for p in root.rglob("*.json") if p.is_file() and _is_task_file(p))
    if not files:
        print("No annotated logs found.")
        return

    analyses: List[Dict[str, Any]] = []
    for f in files:
        analyses.append(analyze_file(f))

    summary = build_summary(analyses)
    print("\n" + "#" * 80)
    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    compare_summary = None
    if args.compare_dir:
        compare_path = Path(args.compare_dir) / "annotated_logs_analysis.json"
        compare_payload = _load_json(compare_path) if compare_path.is_file() else None
        compare_files = []
        if compare_payload and isinstance(compare_payload.get("files"), list):
            compare_files = [
                f
                for f in compare_payload.get("files")
                if isinstance(f, dict) and str(f.get("file", "")).startswith("tasks")
            ]
        labels, matrix = _build_confusion_matrix(analyses, compare_files)
        print("\n" + "#" * 80)
        print("CONFUSION_MATRIX (current vs compare)")
        _print_confusion_matrix(labels, matrix)
        compare_summary = compare_payload.get("summary") if isinstance(compare_payload, dict) else None

    out_path = root / "annotated_logs_analysis.json"
    out_path.write_text(
        json.dumps(
            {
                "files": analyses,
                "summary": summary,
                "compare_summary": compare_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved analysis JSON to: {out_path}")


if __name__ == "__main__":
    main()
