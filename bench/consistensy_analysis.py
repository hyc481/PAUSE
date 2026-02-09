"""
File-wise comparison for two annotated_logs directories.

For each matched filename, print:
- Whether rerun happened
- final_selection: tie + selected assistant model
- The judge booleans for the trajectories that were fed into the final select step:
  - tool_calling_logic_correct
  - tool_calling_parameter_correct
  - reason_policy_correct
  - targets_fully_fulfillment
  - unnecessary_tool_calls

If rerun exists, we treat `rerun_trajs` as the final select input; otherwise `trajs`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


JUDGE_FIELDS = [
    "tool_calling_logic_correct",
    "tool_calling_parameter_correct",
    "reason_policy_correct",
    "targets_fully_fulfillment",
]

TOOL_CALL_RE = re.compile(r"\[ASSISTANT TOOL CALL\]\s*([^\s(]+)\s*\(")

def _normalize_bool(v: Any) -> Any:
    """
    Accept bools or string bools like "True"/"False" (any casing/whitespace) and
    normalize them to Python booleans. Otherwise return original value.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
    return v


def _load_json(p: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None, "not_a_dict"
        return data, ""
    except Exception as e:  # noqa: BLE001
        return None, f"{e.__class__.__name__}:{str(e).strip().replace(chr(10), ' ')}"


def _has_rerun(payload: Dict[str, Any]) -> bool:
    rt = payload.get("rerun_trajs")
    if isinstance(rt, list) and len(rt) > 0:
        return True
    if isinstance(payload.get("rerun_selection"), dict):
        return True
    # Some older logs might only carry final_selection; we keep this conservative.
    return False


def _pick_select_pool(payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (pool_name, traj_list) where pool_name in {'rerun_trajs','trajs'}.
    """
    if _has_rerun(payload):
        rt = payload.get("rerun_trajs")
        if isinstance(rt, list):
            return "rerun_trajs", [t for t in rt if isinstance(t, dict)]
    tr = payload.get("trajs")
    if isinstance(tr, list):
        return "trajs", [t for t in tr if isinstance(t, dict)]
    return "trajs", []


def _fmt_bool(v: Any) -> str:
    v = _normalize_bool(v)
    if v is True:
        return "T"
    if v is False:
        return "F"
    return "?"


def _extract_candidates(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in pool:
        ann = t.get("annotation") if isinstance(t.get("annotation"), dict) else {}
        row = {
            "assistant_model": str(t.get("assistant_model") or ""),
        }
        for k in JUDGE_FIELDS:
            row[k] = _normalize_bool(ann.get(k, None))
        out.append(row)
    return out


def _extract_tool_calls_from_messages(msgs: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in msgs:
        for name in TOOL_CALL_RE.findall(m):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _extract_tool_calls_from_traj(traj: Dict[str, Any]) -> Dict[str, int]:
    msgs = traj.get("full_messages")
    if not isinstance(msgs, list):
        return {}
    msg_list = [m for m in msgs if isinstance(m, str)]
    return _extract_tool_calls_from_messages(msg_list)


def _extract_pool_tool_calls(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return a list of per-traj tool call dicts in pool order.
    Each entry: {"assistant_model": str, "tool_calls": {name: count}}
    """
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(pool):
        if not isinstance(t, dict):
            continue
        am = str(t.get("assistant_model") or f"idx_{i}")
        out.append({"assistant_model": am, "tool_calls": _extract_tool_calls_from_traj(t)})
    return out


def _normalize_for_compare(payload: Optional[Dict[str, Any]], err: str) -> Dict[str, Any]:
    """
    Build a stable representation of what we print, so we can count "fully identical" files.
    """
    if payload is None:
        return {"load_error": err}

    has_rerun = _has_rerun(payload)
    pool_name, pool = _pick_select_pool(payload)
    final_sel = payload.get("final_selection") if isinstance(payload.get("final_selection"), dict) else {}
    tie_raw = final_sel.get("tie", True) if isinstance(final_sel, dict) else True
    tie = bool(_normalize_bool(tie_raw))
    chosen = str(final_sel.get("select_assistant_model") or "") if isinstance(final_sel, dict) else ""

    cands = _extract_candidates(pool)
    # Normalize candidates into mapping keyed by assistant_model for deterministic equality.
    cand_map: Dict[str, Dict[str, Any]] = {}
    for i, c in enumerate(cands):
        am = str(c.get("assistant_model") or f"idx_{i}")
        cand_map[am] = {k: c.get(k, None) for k in JUDGE_FIELDS}

    return {
        "has_rerun": has_rerun,
        "pool": pool_name,
        "final_selection": {
            "tie": tie,
            "select_assistant_model": chosen,
        },
        "candidates": cand_map,
    }


def _print_one_side(tag: str, payload: Optional[Dict[str, Any]], err: str) -> None:
    print(f"  [{tag}]")
    if payload is None:
        print(f"    load_error: {err}")
        return

    has_rerun = _has_rerun(payload)
    final_sel = payload.get("final_selection") if isinstance(payload.get("final_selection"), dict) else {}
    tie = final_sel.get("tie", True) if isinstance(final_sel, dict) else True
    chosen = ""
    if isinstance(final_sel, dict):
        chosen = str(final_sel.get("select_assistant_model") or "")

    pool_name, pool = _pick_select_pool(payload)
    cands = _extract_candidates(pool)
    tool_calls = _extract_pool_tool_calls(pool)

    print(f"    has_rerun: {has_rerun} (pool={pool_name}, n={len(cands)})")
    print(f"    final_selection: tie={bool(tie)} select_assistant_model={chosen or '∅'}")
    print(f"    candidates (select-input):")
    if not cands:
        print("      (none)")
    else:
        # Stable print order: by assistant_model then original index
        for i, c in enumerate(cands):
            am = c.get("assistant_model") or f"idx_{i}"
            bits = " ".join([f"{k}={_fmt_bool(c.get(k))}" for k in JUDGE_FIELDS])
            mark = " <== SELECTED" if (chosen and am == chosen) else ""
            print(f"      - {am}: {bits}{mark}")

    print("    tool_calls (per-traj full_messages, order ignored):")
    if not tool_calls:
        print("      (none)")
    else:
        for entry in tool_calls:
            am = entry.get("assistant_model") or "unknown"
            calls = entry.get("tool_calls") or {}
            if not calls:
                print(f"      - {am}: (none)")
            else:
                items = " ".join([f"{k}={v}" for k, v in sorted(calls.items())])
                print(f"      - {am}: {items}")


def compare_dirs(dir_a: Path, dir_b: Path, limit: Optional[int] = None) -> None:
    files_a = {p.name: p for p in dir_a.rglob("*.json") if p.is_file()}
    files_b = {p.name: p for p in dir_b.rglob("*.json") if p.is_file()}
    all_names = sorted(set(files_a.keys()) | set(files_b.keys()))
    if limit is not None:
        all_names = all_names[: max(0, int(limit))]

    identical: List[str] = []
    different: List[str] = []

    # Field-wise agreement stats
    top_denom: Dict[str, int] = {}
    top_match: Dict[str, int] = {}
    cand_denom: Dict[str, int] = {}
    cand_match: Dict[str, int] = {}
    # Tool-call agreement within a single log (between two trajs)
    tool_intra_denom = 0
    tool_intra_match = 0
    # When select_assistant_model matches, track agreement on selected traj's judge fields
    selected_denom: Dict[str, int] = {}
    selected_match: Dict[str, int] = {}
    # When select_assistant_model differs, compare tuple of both selected models' judge fields
    selected_diff_total = 0
    selected_diff_match: Dict[str, int] = {}
    selected_diff_mismatch: Dict[str, int] = {}

    def _bump(d: Dict[str, int], k: str, n: int = 1) -> None:
        d[k] = d.get(k, 0) + n

    def _pct(m: int, d: int) -> str:
        if d <= 0:
            return "n/a"
        return f"{(100.0 * m / d):.1f}%"

    for name in all_names:
        pa = files_a.get(name)
        pb = files_b.get(name)
        print("\n" + "=" * 100)
        print(f"FILE: {name}")
        print(f"  A: {str(pa) if pa else '(missing)'}")
        print(f"  B: {str(pb) if pb else '(missing)'}")

        payload_a, err_a = _load_json(pa) if pa else (None, "missing")
        payload_b, err_b = _load_json(pb) if pb else (None, "missing")

        _print_one_side("A", payload_a, err_a)
        _print_one_side("B", payload_b, err_b)

        na = _normalize_for_compare(payload_a, err_a)
        nb = _normalize_for_compare(payload_b, err_b)
        if na == nb:
            identical.append(name)
        else:
            different.append(name)

        # Field-wise agreement (only when both payloads are present and parsed as dict)
        comparable = ("load_error" not in na) and ("load_error" not in nb)
        if comparable:
            # Top-level fields we print
            for k in ["has_rerun", "pool"]:
                _bump(top_denom, k)
                if na.get(k) == nb.get(k):
                    _bump(top_match, k)

            _bump(top_denom, "final_selection.tie")
            if (na.get("final_selection", {}) or {}).get("tie") == (nb.get("final_selection", {}) or {}).get("tie"):
                _bump(top_match, "final_selection.tie")

            _bump(top_denom, "final_selection.select_assistant_model")
            if (na.get("final_selection", {}) or {}).get("select_assistant_model") == (nb.get("final_selection", {}) or {}).get("select_assistant_model"):
                _bump(top_match, "final_selection.select_assistant_model")

            # Candidate set agreement + per-field agreement over aligned assistant_model keys
            ca = na.get("candidates", {}) if isinstance(na.get("candidates"), dict) else {}
            cb = nb.get("candidates", {}) if isinstance(nb.get("candidates"), dict) else {}
            _bump(cand_denom, "candidates.models_set")
            if set(ca.keys()) == set(cb.keys()):
                _bump(cand_match, "candidates.models_set")

            common_models = sorted(set(ca.keys()) & set(cb.keys()))
            for am in common_models:
                va = ca.get(am, {}) if isinstance(ca.get(am), dict) else {}
                vb = cb.get(am, {}) if isinstance(cb.get(am), dict) else {}
                for f in JUDGE_FIELDS:
                    key = f"candidates.{f}"
                    _bump(cand_denom, key)
                    if va.get(f, None) == vb.get(f, None):
                        _bump(cand_match, key)

        # Tool call comparison within a single log (use A if available, else B)
        payload_for_tool = payload_a if payload_a is not None else payload_b
        if isinstance(payload_for_tool, dict):
            _, pool_for_tool = _pick_select_pool(payload_for_tool)
            tool_list = _extract_pool_tool_calls(pool_for_tool)
            if len(tool_list) >= 2:
                tool_intra_denom += 1
                t0 = tool_list[0].get("tool_calls", {})
                t1 = tool_list[1].get("tool_calls", {})
                if t0 == t1:
                    tool_intra_match += 1

            # When select_assistant_model matches, check agreement on selected traj's judge fields
            sel_a = (na.get("final_selection", {}) or {}).get("select_assistant_model", "")
            sel_b = (nb.get("final_selection", {}) or {}).get("select_assistant_model", "")
            if sel_a and sel_b and sel_a == sel_b:
                # Both selected the same model; compare that traj's judge fields
                sel_am = str(sel_a)
                sel_va = ca.get(sel_am, {}) if isinstance(ca.get(sel_am), dict) else {}
                sel_vb = cb.get(sel_am, {}) if isinstance(cb.get(sel_am), dict) else {}
                for f in JUDGE_FIELDS:
                    key = f"selected_traj.{f}"
                    _bump(selected_denom, key)
                    if sel_va.get(f, None) == sel_vb.get(f, None):
                        _bump(selected_match, key)
            elif sel_a and sel_b and sel_a != sel_b:
                # Selected different models; compare tuple(sel_a, sel_b) values per field
                selected_diff_total += 1
                sel_a_key = str(sel_a)
                sel_b_key = str(sel_b)
                sel_a_va = ca.get(sel_a_key, {}) if isinstance(ca.get(sel_a_key), dict) else {}
                sel_b_va = ca.get(sel_b_key, {}) if isinstance(ca.get(sel_b_key), dict) else {}
                sel_a_vb = cb.get(sel_a_key, {}) if isinstance(cb.get(sel_a_key), dict) else {}
                sel_b_vb = cb.get(sel_b_key, {}) if isinstance(cb.get(sel_b_key), dict) else {}
                for f in JUDGE_FIELDS:
                    tuple_a = (sel_a_va.get(f, None), sel_b_va.get(f, None))
                    tuple_b = (sel_a_vb.get(f, None), sel_b_vb.get(f, None))
                    if tuple_a == tuple_b:
                        selected_diff_match[f] = selected_diff_match.get(f, 0) + 1
                    else:
                        selected_diff_mismatch[f] = selected_diff_mismatch.get(f, 0) + 1

    print("\n" + "#" * 100)
    print("SUMMARY")
    print(f"  total_files: {len(all_names)}")
    print(f"  identical_files: {len(identical)}")
    print(f"  different_files: {len(different)}")
    if identical:
        print("  identical_file_names:")
        for n in identical:
            print(f"    - {n}")

    print("\n" + "#" * 100)
    print("FIELD AGREEMENT (A vs B)")
    # Top-level
    for k in ["has_rerun", "pool", "final_selection.tie", "final_selection.select_assistant_model"]:
        d = top_denom.get(k, 0)
        m = top_match.get(k, 0)
        print(f"  {k}: {m}/{d} ({_pct(m, d)})")
    # Candidates
    for k in ["candidates.models_set"] + [f"candidates.{f}" for f in JUDGE_FIELDS]:
        d = cand_denom.get(k, 0)
        m = cand_match.get(k, 0)
        print(f"  {k}: {m}/{d} ({_pct(m, d)})")
    print("\n" + "#" * 100)
    print("TOOL CALL AGREEMENT (within log, two trajs)")
    print(f"  tool_calls.intra_traj: {tool_intra_match}/{tool_intra_denom} ({_pct(tool_intra_match, tool_intra_denom)})")

    # When select_assistant_model matches, agreement on selected traj's judge fields
    if selected_denom:
        print("\n" + "#" * 100)
        print("SELECTED TRAJ AGREEMENT (when select_assistant_model matches)")
        for f in JUDGE_FIELDS:
            key = f"selected_traj.{f}"
            d = selected_denom.get(key, 0)
            m = selected_match.get(key, 0)
            print(f"  {f}: {m}/{d} ({_pct(m, d)})")

    if selected_diff_total > 0:
        print("\n" + "#" * 100)
        print("SELECTED TRAJ TUPLE AGREEMENT (when select_assistant_model differs)")
        print(f"  diff_selection_cases: {selected_diff_total}")
        for f in JUDGE_FIELDS:
            m = selected_diff_match.get(f, 0)
            mm = selected_diff_mismatch.get(f, 0)
            d = m + mm
            print(f"  {f}: match={m} mismatch={mm} ({_pct(m, d)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two annotated_logs directories file-wise (final_selection + per-traj judge booleans)."
    )
    parser.add_argument("--dir_a", default="/home/chy/state_aware_bench/bench/runs/debugging1/annotated_logs", help="First annotated_logs directory.")
    parser.add_argument("--dir_b", default="/home/chy/state_aware_bench/bench/runs/debugging1/annotated_logs2", help="Second annotated_logs directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of files to print.")
    args = parser.parse_args()

    compare_dirs(Path(args.dir_a), Path(args.dir_b), limit=args.limit)


