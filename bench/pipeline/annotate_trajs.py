"""
Annotate trajectories with LLM-as-judge.

Reads traj JSON files (from run_user_tasks.py) and saves annotated results
to annotated_logs/<same_filename>.json with per-traj summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from bench.pipeline.generate_task.branches.wearable_data_casual import (
    wearable_data_casual_tool_selection_prompt,
)
from bench.utils.clients import (
    call_llm,
    get_ann_routes,
    get_traj_model_routes,
    is_gemini_client
)
from bench.pipeline.build_store import load_store_from_meta_profile
from bench.utils.agent_runner import AgentRunner
from bench.prompts.generation_prompt import (
    base_select_prompt,
    base_judge_prompt,
    base_rerun_prompt,
    base_align_targets_prompt,
    platform_overview,
)
from bench.utils.misc import (
    load_tool_summaries,
    strip_code_fences,
    load_branch_tool_allowlist,
    load_branch_assistant_guidance,
    group_routes_by_key,
)


# -----------------------------
# Unified pipeline error records
# -----------------------------

ErrorKind = Literal[
    "llm_api_error",
    "llm_parse_error",
    "io_error",
    "runner_error",
    "unknown_error",
]


@dataclass
class PipelineError:
    kind: ErrorKind
    stage: str  # e.g. "judge.api", "judge.parse", "select.api", "select.parse", "read", "write", "rerun.run"
    detail: str
    file: str = ""
    branch: str = ""
    label: str = ""
    traj_idx: Optional[int] = None
    assistant_model: str = ""
    raw: Optional[str] = None  # LLM raw output (optional)


class ErrorCollector:
    def __init__(self) -> None:
        self._errors: List[PipelineError] = []

    def add(
        self,
        *,
        kind: ErrorKind,
        stage: str,
        detail: str,
        file: str = "",
        branch: str = "",
        label: str = "",
        traj_idx: Optional[int] = None,
        assistant_model: str = "",
        raw: Optional[str] = None,
    ) -> None:
        self._errors.append(
            PipelineError(
                kind=kind,
                stage=stage,
                detail=detail,
                file=file,
                branch=branch,
                label=label,
                traj_idx=traj_idx,
                assistant_model=assistant_model,
                raw=raw,
            )
        )

    def dump(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self._errors]


# -----------------------------
# Prompt helpers
# -----------------------------

def _branch_annotator_prompt(branch: str) -> str:
    if branch == "wearable_data_casual":
        return wearable_data_casual_tool_selection_prompt
    return ""


def _render_tools(tools: List[Dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, indent=2)


def _load_tool_schemas(path: Path, max_tools: int | None = None) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if max_tools is not None:
        data = data[:max_tools]
    return [item for item in data if isinstance(item, dict)]


def _tool_schema_name(item: Dict[str, Any]) -> str:
    func = item.get("function")
    if isinstance(func, dict):
        return str(func.get("name", ""))
    return str(item.get("name", ""))


def _tool_summary_name(item: Dict[str, Any]) -> str:
    if isinstance(item, dict):
        return str(item.get("name", ""))
    return ""


def _tool_names_from_trajs(trajs: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for traj in trajs or []:
        if not isinstance(traj, dict):
            continue
        messages = traj.get("assistant_traj") or []
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
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


def _expand_tool_desc_for_names(
    tool_names: List[str],
    platform_desc: List[Dict[str, Any]],
    med_desc: List[Dict[str, Any]],
    source_desc: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    platform_map = {_tool_summary_name(s): s for s in platform_desc}
    med_map = {_tool_summary_name(s): s for s in med_desc}
    source_map = {_tool_summary_name(s): s for s in source_desc}

    prefixed_sources = {name.split("-")[0] for name in tool_names if "-" in name and not name.startswith("med-")}

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item: Dict[str, Any]) -> None:
        name = _tool_summary_name(item)
        if name and name not in seen:
            out.append(item)
            seen.add(name)

    for name in tool_names:
        if name in platform_map:
            _add(platform_map[name])
        elif name in med_map:
            _add(med_map[name])

    for source in prefixed_sources:
        for base_name, item in source_map.items():
            entry = json.loads(json.dumps(item))
            entry["name"] = f"{source}-{base_name}"
            _add(entry)

    return out


def _expand_tool_schemas_for_names(
    tool_names: List[str],
    platform_schemas: List[Dict[str, Any]],
    med_schemas: List[Dict[str, Any]],
    source_schemas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    platform_map = {_tool_schema_name(s): s for s in platform_schemas}
    med_map = {_tool_schema_name(s): s for s in med_schemas}
    source_map = {_tool_schema_name(s): s for s in source_schemas}

    prefixed_sources = {name.split("-")[0] for name in tool_names if "-" in name and not name.startswith("med-")}

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(schema: Dict[str, Any]) -> None:
        name = _tool_schema_name(schema)
        if name and name not in seen:
            out.append(schema)
            seen.add(name)

    for name in tool_names:
        if name in platform_map:
            _add(platform_map[name])
        elif name in med_map:
            _add(med_map[name])

    for source in prefixed_sources:
        for base_name, schema in source_map.items():
            schema_copy = json.loads(json.dumps(schema))
            func = schema_copy.get("function", {})
            if func:
                func["name"] = f"{source}-{base_name}"
            else:
                schema_copy["name"] = f"{source}-{base_name}"
            _add(schema_copy)

    return out



def _strip_llm_raw(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_llm_raw(v)
            for k, v in value.items()
            if k not in {"llm_raw", "llm_raws"}
        }
    if isinstance(value, list):
        return [_strip_llm_raw(v) for v in value]
    return value


def build_judge_prompt(
    tools: List[Dict[str, Any]],
    full_messages: List[Any],
    branch: str,
    now_str: str = "",
) -> str:
    branch_prompt = _branch_annotator_prompt(branch)
    sections = [
        base_judge_prompt,
        platform_overview,
        branch_prompt,
        "### Available Tools\n" + _render_tools(tools),
        "### Conversation Log (full_messages)\n" + json.dumps(full_messages, ensure_ascii=False, indent=2),
    ]
    if now_str:
        sections = [f"Current Time: {now_str}"] + sections
    return "\n\n".join(sections)


def build_select_prompt(
    targets: List[Any],
    branch: str,
    evaluations: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    now_str: str = "",
) -> str:
    branch_prompt = _branch_annotator_prompt(branch)
    safe_evaluations = _strip_llm_raw(evaluations)
    sections = [
        base_select_prompt,
        platform_overview,
        branch_prompt,
        "### Available Tools\n" + _render_tools(tools),
        "### Candidate Trajectory Evaluations\n" + json.dumps(safe_evaluations, ensure_ascii=False, indent=2),
    ]
    if now_str:
        sections = [f"Current Time: {now_str}"] + sections
    return "\n\n".join(sections)


def build_rerun_prompt(
    branch: str,
    evaluations: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    now_str: str = "",
) -> str:
    branch_prompt = _branch_annotator_prompt(branch)
    safe_evaluations = _strip_llm_raw(evaluations)
    sections = [
        base_rerun_prompt,
        platform_overview,
        branch_prompt,
        "### Available Tools\n" + _render_tools(tools),
        "### Candidate Trajectory Evaluations\n" + json.dumps(safe_evaluations, ensure_ascii=False, indent=2),
    ]
    if now_str:
        sections = [f"Current Time: {now_str}"] + sections
    return "\n\n".join(sections)


def build_align_targets_prompt(
    branch: str,
    targets: List[Any],
    tools: List[Dict[str, Any]],
    full_messages: List[Any],
    now_str: str = "",
) -> str:
    branch_prompt = _branch_annotator_prompt(branch)
    safe_messages = _strip_llm_raw(full_messages)
    sections = [
        base_align_targets_prompt,
        platform_overview,
        branch_prompt,
        "### Available Tools\n" + _render_tools(tools),
        "### Task Targets\n" + json.dumps(targets, ensure_ascii=False, indent=2),
        "### Conversation Log (full_messages)\n" + json.dumps(safe_messages, ensure_ascii=False, indent=2),
    ]
    if now_str:
        sections = [f"Current Time: {now_str}"] + sections
    return "\n\n".join(sections)


def build_judge_messages(
    tools: List[Dict[str, Any]],
    full_messages: List[Any],
    branch: str,
    now_str: str = "",
) -> List[Dict[str, str]]:
    user_content = build_judge_prompt(
        tools=tools,
        full_messages=full_messages,
        branch=branch,
        now_str=now_str,
    )
    return [{"role": "user", "content": user_content}]


def build_selection_messages(
    targets: List[Any],
    branch: str,
    evaluations: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    now_str: str = "",
) -> List[Dict[str, str]]:
    user_content = build_select_prompt(
        targets=targets,
        branch=branch,
        evaluations=evaluations,
        tools=tools,
        now_str=now_str,
    )
    return [{"role": "user", "content": user_content}]


def build_rerun_messages(
    branch: str,
    evaluations: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    now_str: str = "",
) -> List[Dict[str, str]]:
    user_content = build_rerun_prompt(
        branch=branch,
        evaluations=evaluations,
        tools=tools,
        now_str=now_str,
    )
    return [{"role": "user", "content": user_content}]


def build_align_targets_messages(
    branch: str,
    targets: List[Any],
    tools: List[Dict[str, Any]],
    full_messages: List[Any],
    now_str: str = "",
) -> List[Dict[str, str]]:
    user_content = build_align_targets_prompt(
        branch=branch,
        targets=targets,
        tools=tools,
        full_messages=full_messages,
        now_str=now_str,
    )
    return [{"role": "user", "content": user_content}]


# -----------------------------
# LLM call helpers
# -----------------------------

def _exc_reason(prefix: str, e: Exception) -> str:
    msg = str(e).strip().replace("\n", " ")
    return f"{prefix}:{e.__class__.__name__}:{msg}"


def _weekday_of_iso(iso_dt: str) -> str:
    try:
        return datetime.fromisoformat(iso_dt).strftime("%A")
    except Exception:
        return ""


def _annotator_detail(annotator_tag: str, detail: str) -> str:
    if annotator_tag:
        return f"[{annotator_tag}] {detail}"
    return detail


def call_with_retry(fn, *args, **kwargs):
    last_exc = None
    for _ in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            import time
            time.sleep(60)
    if last_exc:
        raise last_exc


def _llm_temperature(model: str) -> Optional[float]:
    if model.startswith("gpt-5"):
        return None
    return 0


# -----------------------------
# Judge / Select
# -----------------------------

def judge_traj(
    traj: Dict[str, Any],
    tools: List[Dict[str, Any]],
    model: str,
    client_obj,
    branch: str,
    now_str: str = "",
    *,
    errors: Optional[ErrorCollector] = None,
    ctx_file: str = "",
    ctx_label: str = "",
    traj_idx: Optional[int] = None,
    annotator_tag: str = "",
) -> Dict[str, Any]:
    messages = build_judge_messages(
        tools=tools,
        full_messages=traj.get("full_messages", []),
        branch=branch,
        now_str=now_str,
    )

    content: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    attempts = 0

    while attempts < 3:
        attempts += 1
        try:
            content = call_with_retry(
                call_llm,
                client_obj,
                model=model,
                messages=messages,
                temperature=_llm_temperature(model),
                response_mime_type="application/json" if is_gemini_client(client_obj) else None,
            )
        except Exception as e:
            if errors is not None:
                errors.add(
                    kind="llm_api_error",
                    stage="judge.api",
                    detail=_annotator_detail(annotator_tag, _exc_reason(f"judge_api_attempt_{attempts}", e)),
                    file=ctx_file,
                    branch=branch,
                    label=ctx_label,
                    traj_idx=traj_idx,
                    assistant_model=traj.get("assistant_model", ""),
                )
            continue

        try:
            cleaned = strip_code_fences(content)
            parsed = json.loads(cleaned)
            break
        except Exception as e:
            if errors is not None:
                errors.add(
                    kind="llm_parse_error",
                    stage="judge.parse",
                    detail=_annotator_detail(annotator_tag, _exc_reason(f"judge_parse_attempt_{attempts}", e)),
                    file=ctx_file,
                    branch=branch,
                    label=ctx_label,
                    traj_idx=traj_idx,
                    assistant_model=traj.get("assistant_model", ""),
                    raw=content,
                )
            continue

    if parsed is None:
        parsed = {}

    error_analysis = parsed.get("error_analysis", "") if parsed else "Judge parse failed after 3 attempts."
    reason_policy_correct = _normalize_bool(parsed.get("reason_policy_correct")) is True
    if error_analysis == "":
        reason_policy_correct = True

    return {
        "llm_raw": content,
        "trajectory_summary": parsed.get("trajectory_summary", ""),
        "tool_calling_logic_correct": _normalize_bool(parsed.get("tool_calling_logic_correct")) is True,
        "tool_calling_parameter_correct": _normalize_bool(parsed.get("tool_calling_parameter_correct")) is True,
        "reason_policy_correct": reason_policy_correct,
        "error_analysis": error_analysis,
    }


def _majority_bool(values: List[Any]) -> bool:
    normalized = [_normalize_bool(v) for v in values]
    true_count = sum(v is True for v in normalized)
    false_count = sum(v is False for v in normalized)
    if true_count == 0 and false_count == 0:
        return False
    return true_count > false_count


def _aggregate_annotations(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not annotations:
        return {
            "llm_raw": "",
            "trajectory_summary": "",
            "tool_calling_logic_correct": False,
            "tool_calling_parameter_correct": False,
            "reason_policy_correct": False,
            "error_analysis": [],
        }

    llm_raw = next((a.get("llm_raw") for a in annotations if a.get("llm_raw")), "")
    summary = next((a.get("trajectory_summary") for a in annotations if a.get("trajectory_summary")), "")
    logic = _majority_bool([a.get("tool_calling_logic_correct") for a in annotations])
    params = _majority_bool([a.get("tool_calling_parameter_correct") for a in annotations])
    reason_policy = (sum([a.get("reason_policy_correct") for a in annotations]) > 0)
    if (not logic) or (not params):
        error_analysis = [a.get("error_analysis") for a in annotations if a.get("error_analysis")]
    else:
        error_analysis = []
    return {
        "llm_raw": llm_raw,
        "trajectory_summary": summary,
        "tool_calling_logic_correct": logic,
        "tool_calling_parameter_correct": params,
        "reason_policy_correct": reason_policy,
        "error_analysis": error_analysis,
    }


def _assistant_message_count(traj: Dict[str, Any]) -> int:
    return sum(1 for m in traj.get("assistant_traj", []) if m.get("role") == "assistant")


def _tool_call_signatures(traj: Dict[str, Any]) -> List[str]:
    signatures: List[str] = []
    for msg in traj.get("assistant_traj", []):
        if msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        if isinstance(calls, dict):
            calls = [calls]
        if msg.get("function_call"):
            calls = calls + [msg["function_call"]]
        for call in calls:
            if not isinstance(call, dict):
                continue
            func = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = call.get("name") or func.get("name") or ""
            args = call.get("arguments", func.get("arguments", "")) or ""
            args_str = _normalize_tool_args(args)
            signatures.append(f"{name}:{args_str}")
    return signatures


def _normalize_tool_args(args: Any) -> str:
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    if isinstance(args, str):
        raw = args.strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except Exception:
            return "".join(raw.split())
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        return json.dumps(parsed, ensure_ascii=False)
    return str(args)


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _normalize_bool_list(values: Any) -> Optional[List[bool]]:
    if not isinstance(values, list):
        return None
    out: List[bool] = []
    for v in values:
        normalized = _normalize_bool(v)
        if normalized is None:
            return None
        out.append(normalized)
    return out




def judge_traj_multi(
    traj: Dict[str, Any],
    tools_desc: List[Dict[str, str]],
    ann_routes: List[Dict[str, Any]],
    branch: str,
    now_str: str = "",
    *,
    errors: Optional[ErrorCollector] = None,
    ctx_file: str = "",
    ctx_label: str = "",
    traj_idx: Optional[int] = None,
) -> Dict[str, Any]:
    if not ann_routes:
        return {
            "annotations": [],
            "annotation": _aggregate_annotations([]),
        }

    annotations: List[Dict[str, Any]] = []

    def _run_one(route: Dict[str, Any]) -> Dict[str, Any]:
        tools = tools_desc
        tag = route.get("model") or route.get("key") or ""
        ann = judge_traj(
            traj=traj,
            tools=tools,
            model=route.get("model", ""),
            client_obj=route.get("client"),
            branch=branch,
            now_str=now_str,
            errors=errors,
            ctx_file=ctx_file,
            ctx_label=ctx_label,
            traj_idx=traj_idx,
            annotator_tag=tag,
        )
        ann["annotator_model"] = route.get("model", "")
        ann["annotator_key"] = route.get("key", "")
        return ann

    with ThreadPoolExecutor(max_workers=len(ann_routes) or 1) as ex:
        futures = [ex.submit(_run_one, route) for route in ann_routes]
        for fut in as_completed(futures):
            try:
                annotations.append(fut.result())
            except Exception as e:
                if errors is not None:
                    errors.add(
                        kind="unknown_error",
                        stage="judge.unknown",
                        detail=_annotator_detail("multi", _exc_reason("judge_unknown", e)),
                        file=ctx_file,
                        branch=branch,
                        label=ctx_label,
                        traj_idx=traj_idx,
                        assistant_model=traj.get("assistant_model", ""),
                    )
                annotations.append(
                    {
                        "llm_raw": "",
                        "trajectory_summary": "",
                        "tool_calling_logic_correct": False,
                        "tool_calling_parameter_correct": False,
                        "error_analysis": _exc_reason("judge_unknown", e),
                        "annotator_model": "",
                        "annotator_key": "",
                    }
                )

    return {
        "annotations": annotations,
        "annotation": _aggregate_annotations(annotations),
    }


def select_best_traj(
    annotations: List[Dict[str, Any]],
    targets: List[Any],
    branch: str,
    model: str,
    client_obj,
    tools: List[Dict[str, Any]],
    now_str: str = "",
    *,
    errors: Optional[ErrorCollector] = None,
    ctx_file: str = "",
    ctx_label: str = "",
) -> Dict[str, Any]:
    # For select, per-traj info should ONLY include judge fields (see judge_traj return).
    allowed_fields = [
        "trajectory_summary",
        "tool_calling_logic_correct",
        "tool_calling_parameter_correct",
        "reason_policy_correct",
        "error_analysis",
        "assistant_model",
    ]

    evaluations: List[Dict[str, Any]] = []
    assistant_models: List[str] = []

    for a in annotations or []:
        if not isinstance(a, dict):
            continue
        evaluations.append({k: a.get(k) for k in allowed_fields})
        assistant_models.append(a.get("assistant_model", ""))

    messages = build_selection_messages(targets, branch, evaluations, tools, now_str=now_str)

    content: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    attempts = 0

    while attempts < 3:
        attempts += 1
        try:
            content = call_with_retry(
                call_llm,
                client_obj,
                model=model,
                messages=messages,
                temperature=_llm_temperature(model),
                response_mime_type="application/json" if is_gemini_client(client_obj) else None,
            )
        except Exception as e:
            if errors is not None:
                errors.add(
                    kind="llm_api_error",
                    stage="select.api",
                    detail=_exc_reason(f"select_api_attempt_{attempts}", e),
                    file=ctx_file,
                    branch=branch,
                    label=ctx_label,
                )
            continue

        try:
            cleaned = strip_code_fences(content)
            parsed = json.loads(cleaned)
            break
        except Exception as e:
            if errors is not None:
                errors.add(
                    kind="llm_parse_error",
                    stage="select.parse",
                    detail=_exc_reason(f"select_parse_attempt_{attempts}", e),
                    file=ctx_file,
                    branch=branch,
                    label=ctx_label,
                    raw=content,
                )
            continue

    if parsed is None:
        return {
            "tie": True,
            "status": "tie",
            "failure_category": "select_parse_error",
            "select_assistant_model": "",
            "select_reason": "Selection parse failed after 3 attempts.",
            "llm_raw": content or "",
        }

    sel = parsed.get("select_assistant_model", parsed.get("select_index", parsed.get("select_idx", None)))
    if isinstance(sel, int):
        select_assistant_model = assistant_models[sel] if 0 <= sel < len(assistant_models) else ""
    else:
        select_assistant_model = sel or ""
    tie_raw = parsed.get("tie", None)
    tie = _normalize_bool(tie_raw)
    if tie is None:
        tie = not bool(select_assistant_model)
    if tie:
        select_assistant_model = ""

    return {
        "tie": tie,
        "status": "tie" if tie else "selected",
        "failure_category": "tie" if tie else "",
        "select_assistant_model": select_assistant_model,
        "select_reason": parsed.get("select_reason", parsed.get("reason", "")),
        "llm_raw": content,
    }


def rerun_guidance_from_annotations(
    annotations: List[Dict[str, Any]],
    branch: str,
    model: str,
    client_obj,
    tools: List[Dict[str, Any]],
    now_str: str = "",
    *,
    errors: Optional[ErrorCollector] = None,
    ctx_file: str = "",
    ctx_label: str = "",
) -> Dict[str, Any]:
    messages = build_rerun_messages(branch, annotations, tools, now_str=now_str)
    content: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    attempts = 0

    while attempts < 3:
        attempts += 1
        try:
            content = call_with_retry(
                call_llm,
                client_obj,
                model=model,
                messages=messages,
                temperature=_llm_temperature(model),
                response_mime_type="application/json" if is_gemini_client(client_obj) else None,
            )
        except Exception as e:
            if errors is not None:
                errors.add(
                    kind="llm_api_error",
                    stage="rerun.api",
                    detail=_exc_reason(f"rerun_api_attempt_{attempts}", e),
                    file=ctx_file,
                    branch=branch,
                    label=ctx_label,
                )
            continue

        try:
            cleaned = strip_code_fences(content)
            parsed = json.loads(cleaned)
            break
        except Exception as e:
            if errors is not None:
                errors.add(
                    kind="llm_parse_error",
                    stage="rerun.parse",
                    detail=_exc_reason(f"rerun_parse_attempt_{attempts}", e),
                    file=ctx_file,
                    branch=branch,
                    label=ctx_label,
                    raw=content,
                )
            continue

    if parsed is None:
        return {
            "rerun_guidance": "",
            "llm_raw": content or "",
        }

    return {
        "rerun_guidance": parsed.get("rerun_guidance", ""),
        "llm_raw": content or "",
    }


def align_targets_from_traj(
    traj: Dict[str, Any],
    targets: List[Any],
    branch: str,
    tools_desc: List[Dict[str, str]],
    ann_routes: List[Dict[str, Any]],
    now_str: str = "",
    *,
    errors: Optional[ErrorCollector] = None,
    ctx_file: str = "",
    ctx_label: str = "",
    traj_idx: Optional[int] = None,
) -> Dict[str, Any]:
    if len(ann_routes) < 2:
        return {
            "ok": False,
            "reason": "align_targets requires two annotators.",
            "analysis": "",
            "achieved": [],
            "llm_raws": [],
        }

    routes = ann_routes[:2]
    results: List[Dict[str, Any]] = []

    for route in routes:
        tools = tools_desc
        messages = build_align_targets_messages(
            branch=branch,
            targets=targets,
            tools=tools,
            full_messages=traj.get("full_messages", []),
            now_str=now_str,
        )
        content: Optional[str] = None
        parsed: Optional[Dict[str, Any]] = None
        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                content = call_with_retry(
                    call_llm,
                    route["client"],
                    model=route["model"],
                    messages=messages,
                    temperature=_llm_temperature(route["model"]),
                    response_mime_type="application/json" if is_gemini_client(route.get("client")) else None,
                )
            except Exception as e:
                if errors is not None:
                    errors.add(
                        kind="llm_api_error",
                        stage="align.api",
                        detail=_exc_reason(f"align_api_attempt_{attempts}", e),
                        file=ctx_file,
                        branch=branch,
                        label=ctx_label,
                        traj_idx=traj_idx,
                    )
                continue

            try:
                cleaned = strip_code_fences(content)
                parsed = json.loads(cleaned)
                break
            except Exception as e:
                if errors is not None:
                    errors.add(
                        kind="llm_parse_error",
                        stage="align.parse",
                        detail=_exc_reason(f"align_parse_attempt_{attempts}", e),
                        file=ctx_file,
                        branch=branch,
                        label=ctx_label,
                        traj_idx=traj_idx,
                        raw=content,
                    )
                continue

        results.append(
            {
                "analysis": (parsed or {}).get("analysis", ""),
                "achieved": (parsed or {}).get("achieved", []),
                "llm_raw": content or "",
            }
        )

    achieved_a = _normalize_bool_list(results[0].get("achieved", []))
    achieved_b = _normalize_bool_list(results[1].get("achieved", []))
    if (
        achieved_a is None
        or achieved_b is None
        or len(achieved_a) != len(targets)
        or len(achieved_b) != len(targets)
    ):
        return {
            "ok": False,
            "reason": "align_targets invalid achieved length.",
            "analysis": results[0].get("analysis", ""),
            "achieved": [],
            "llm_raws": [r.get("llm_raw", "") for r in results],
        }

    achieved = [a and b for a, b in zip(achieved_a, achieved_b)]
    failed = sum(1 for v in achieved if not v)
    failed_targets_num_limit = 1 if len(targets) <= 5 else 2
    ok = failed <= failed_targets_num_limit

    return {
        "ok": ok,
        "reason": "" if ok else "More than one/two target not achieved.",
        "analysis": results[0].get("analysis", ""),
        "achieved": achieved,
        "llm_raws": [r.get("llm_raw", "") for r in results],
    }


# -----------------------------
# File annotate
# -----------------------------

def _write_payload(path: Path, payload: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[annotated] {path.name} -> {out_path}")


def annotate_file(
    path: Path,
    tool_desc: List[Dict[str, str]],
    tool_schemas_openai: List[Dict[str, Any]],
    med_tool_desc: List[Dict[str, str]],
    med_tool_schemas: List[Dict[str, Any]],
    source_tool_desc: List[Dict[str, str]],
    source_tool_schemas: List[Dict[str, Any]],
    ann_routes: List[Dict[str, Any]],
    traj_routes: List[tuple],
    max_trajs: int | None,
    out_dir: Path,
    rerun_enabled: bool,
) -> None:
    errors = ErrorCollector()

    def _selection_fail(
        reason: str,
        *,
        rerun_guidance: str = "",
        llm_raw: str = "",
        failure_category: str = "error",
    ) -> Dict[str, Any]:
        return {
            "tie": True,
            "status": "failed",
            "failure_category": failure_category,
            "select_assistant_model": "",
            "select_reason": reason,
            "rerun_guidance": rerun_guidance,
            "llm_raw": llm_raw,
        }

    def _finalize_write() -> None:
        # always attach pipeline errors before writing
        payload["pipeline_errors"] = errors.dump()
        try:
            _write_payload(path, payload, out_dir)
        except Exception as e:
            errors.add(
                kind="io_error",
                stage="write",
                detail=_exc_reason("write_error", e),
                file=path.name,
                branch=payload.get("branch", "") or "",
                label=payload.get("label", "") or "",
            )
            # last-ditch: print and move on
            print(f"[annotate][FATAL] failed to write payload for {path.name}: {_exc_reason('write_error', e)}")

    def _set_final(sel: Dict[str, Any]) -> None:
        payload["final_selection"] = sel

    def _finalize(sel: Dict[str, Any], *, key: str = "selection") -> None:
        payload[key] = sel
        _set_final(sel)
        _finalize_write()

    def _single_pass_select(passing: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "tie": False,
            "status": "selected",
            "failure_category": "",
            "select_assistant_model": passing[0].get("assistant_model", ""),
            "select_reason": "Exactly one trajectory satisfies judge criteria (logic and parameters).",
            "rerun_guidance": "",
            "llm_raw": "",
        }

    def _selection_for_traj(traj: Dict[str, Any], reason: str, *, llm_raw: str = "") -> Dict[str, Any]:
        return {
            "tie": False,
            "status": "selected",
            "failure_category": "",
            "select_assistant_model": traj.get("annotation", {}).get("assistant_model", traj.get("assistant_model", "")),
            "select_reason": reason,
            "rerun_guidance": "",
            "llm_raw": llm_raw,
        }

    def _finalize_with_align(sel: Dict[str, Any], selected_traj: Dict[str, Any]) -> None:
        targets = payload.get("targets", [])
        align = align_targets_from_traj(
            traj=selected_traj,
            targets=targets,
            branch=branch_name,
            tools_desc=tool_desc,
            ann_routes=ann_routes,
            now_str=now_str,
            errors=errors,
            ctx_file=path.name,
            ctx_label=label,
        )
        payload["align_targets"] = align

        if not align.get("ok", False):
            selection = _selection_fail(
                align.get("reason", "align_targets failed."),
                failure_category="align_targets_failed",
            )
            _finalize(selection, key="selection")
            return

        achieved = align.get("achieved", [])
        if isinstance(achieved, list) and targets:
            aligned_targets = [t for t, ok in zip(targets, achieved) if ok]
            payload["aligned_targets"] = aligned_targets
            payload["targets"] = aligned_targets
            payload["targets_adjust_reason"] = (
                "align_targets: dropped unmet targets; allowed up to 1 unmet target."
            )

        _finalize(sel, key="selection")

    def _select_from_trajs(
        candidate_trajs: List[Dict[str, Any]],
        *,
        stage_label: str,
    ) -> tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        passing_tt = [t for t in candidate_trajs if _passes(t.get("annotation", {}))]

        if len(passing_tt) == 1:
            selection = _single_pass_select([passing_tt[0].get("annotation", {})])
            print(
                f"[annotate][SUCCESS] {path.name} single-pass select "
                f"{selection.get('select_assistant_model')} ({stage_label})."
            )
            return "selected", selection, passing_tt[0]

        if len(passing_tt) == 2:
            tool_calls_a = _tool_call_signatures(passing_tt[0])
            tool_calls_b = _tool_call_signatures(passing_tt[1])
            if tool_calls_a == tool_calls_b:
                chosen = min(passing_tt, key=_assistant_message_count)
                selection = _selection_for_traj(
                    chosen,
                    "Tool-call sequences are identical; selected fewer assistant turns.",
                )
                print(
                    f"[annotate][SUCCESS] {path.name} rule-based select "
                    f"{selection.get('select_assistant_model')} ({stage_label})."
                )
                return "selected", selection, chosen

            annotations_tt = [t.get("annotation", {}) for t in passing_tt]
            tools_for_select = tool_desc
            selection = select_best_traj(
                annotations=annotations_tt,
                targets=[{"task_instruction": payload.get("task_instruction", "")}],
                branch=branch_name,
                model=select_route.get("model", ""),
                client_obj=select_route.get("client"),
                tools=tools_for_select,
                now_str=now_str,
                errors=errors,
                ctx_file=path.name,
                ctx_label=label,
            )

            if selection.get("tie", True) or not selection.get("select_assistant_model"):
                chosen = min(passing_tt, key=_assistant_message_count)
                selection = _selection_for_traj(
                    chosen,
                    "Tie or inconclusive LLM select; selected fewer assistant turns.",
                    llm_raw=selection.get("llm_raw", "") if isinstance(selection, dict) else "",
                )
                print(
                    f"[annotate][SUCCESS] {path.name} tie-break select "
                    f"{selection.get('select_assistant_model')} ({stage_label})."
                )
                return "selected", selection, chosen

            selected = next(
                (
                    t
                    for t in passing_tt
                    if t.get("annotation", {}).get("assistant_model") == selection.get("select_assistant_model")
                ),
                min(passing_tt, key=_assistant_message_count),
            )
            print(
                f"[annotate][SUCCESS] {path.name} select assistant model "
                f"{selection.get('select_assistant_model')} ({stage_label})."
            )
            return "selected", selection, selected

        selection = _selection_fail(
            "No (T,T) trajectory; rerun required.",
            failure_category="rerun_pending",
        )
        print(f"[annotate][RERUN] {path.name} no (T,T) trajectory ({stage_label}).")
        return "rerun", selection, None

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        payload = {"selection": _selection_fail(_exc_reason("read_payload_error", e), failure_category="error")}
        errors.add(
            kind="io_error",
            stage="read",
            detail=_exc_reason("read_payload_error", e),
            file=path.name,
        )
        payload["final_selection"] = payload["selection"]
        _finalize_write()
        return

    branch_name = payload.get("branch", "") or ""
    label = payload.get("label", "") or ""
    now_str = ""
    if payload.get("profile") and payload.get("meta"):
        try:
            store = load_store_from_meta_profile(payload["profile"], payload["meta"])
            now_str = store.get("profile", {}).get("now", "") if isinstance(store, dict) else ""
        except Exception as e:
            errors.add(
                kind="runner_error",
                stage="now_str",
                detail=_exc_reason("load_store_now_str", e),
                file=path.name,
                branch=branch_name,
                label=label,
            )
    if now_str:
        weekday = _weekday_of_iso(now_str)
        if weekday:
            now_str = f"{now_str}, {weekday}"

    # Sync platform tool summaries to branch tool allowlist if available.
    allowlist = load_branch_tool_allowlist(branch_name) if branch_name else None
    if allowlist:
        allow = set(allowlist)
        tool_desc = [t for t in tool_desc if t.get("name") in allow]
        tool_schemas_openai = [t for t in tool_schemas_openai if _tool_schema_name(t) in allow]

    base_platform_desc = list(tool_desc)
    base_platform_schemas = list(tool_schemas_openai)

    trajs: List[Dict[str, Any]] = payload.get("trajs", [])
    if max_trajs is not None:
        trajs = trajs[:max_trajs]

    tool_names = _tool_names_from_trajs(trajs)
    tool_desc = _expand_tool_desc_for_names(
        tool_names,
        platform_desc=base_platform_desc,
        med_desc=med_tool_desc,
        source_desc=source_tool_desc,
    )
    tool_schemas_openai = _expand_tool_schemas_for_names(
        tool_names,
        platform_schemas=base_platform_schemas,
        med_schemas=med_tool_schemas,
        source_schemas=source_tool_schemas,
    )

    if not ann_routes:
        errors.add(
            kind="runner_error",
            stage="judge.config",
            detail="No annotator routes configured.",
            file=path.name,
            branch=branch_name,
            label=label,
        )
        _finalize(_selection_fail("No annotator routes configured.", failure_category="error"), key="selection")
        return

    select_route = ann_routes[0]

    # Judge all trajs (never crash the whole file on per-traj failures)
    for i, traj in enumerate(trajs):
        if bool(traj.get("terminated")):
            errors.add(
                kind="runner_error",
                stage="judge.skip_terminated",
                detail="terminated=True; skipped judge.",
                file=path.name,
                branch=branch_name,
                label=label,
                traj_idx=i,
                assistant_model=traj.get("assistant_model", ""),
            )
            traj["annotations"] = []
            traj["annotation"] = {
                "llm_raw": "",
                "trajectory_summary": "",
                "tool_calling_logic_correct": False,
                "tool_calling_parameter_correct": False,
                "reason_policy_correct": False,
                "error_analysis": True,
                "assistant_model": traj.get("assistant_model", ""),
            }
            continue
        try:
            result = judge_traj_multi(
                traj=traj,
                tools_desc=tool_desc,
                ann_routes=ann_routes,
                branch=branch_name,
                now_str=now_str,
                errors=errors,
                ctx_file=path.name,
                ctx_label=label,
                traj_idx=i,
            )
        except Exception as e:
            errors.add(
                kind="unknown_error",
                stage="judge.unknown",
                detail=_exc_reason("judge_unknown", e),
                file=path.name,
                branch=branch_name,
                label=label,
                traj_idx=i,
                assistant_model=traj.get("assistant_model", ""),
            )
            result = {
                "annotations": [],
                "annotation": {
                "llm_raw": "",
                "trajectory_summary": "",
                "tool_calling_logic_correct": False,
                "tool_calling_parameter_correct": False,
                "error_analysis": _exc_reason("judge_unknown", e),
                },
            }

        result["annotation"]["assistant_model"] = traj.get("assistant_model", "")
        traj["annotations"] = result.get("annotations", [])
        traj["annotation"] = result.get("annotation", {})

    payload["trajs"] = trajs

    def _passes(ann: Dict[str, Any]) -> bool:
        return (
            _normalize_bool(ann.get("tool_calling_logic_correct")) is True
            and _normalize_bool(ann.get("tool_calling_parameter_correct")) is True
            and _normalize_bool(ann.get("reason_policy_correct")) is True
        )

    if len(trajs) != 2:
        selection = _selection_fail(
            "Only supports traj_num=2.",
            failure_category="invalid",
        )
        print(f"[annotate][INVALID] {path.name} expected 2 trajectories.")
        _finalize(selection, key="selection")
        return

    status, selection, selected_traj = _select_from_trajs(trajs, stage_label="initial")
    if status == "selected" and selected_traj is not None:
        _finalize_with_align(selection, selected_traj)
        return

    # rerun: generate guidance and re-run trajectories, up to 2 extra attempts
    if not rerun_enabled:
        _finalize(selection, key="selection")
        return
    if not payload.get("meta") or not payload.get("profile"):
        selection["failure_category"] = "rerun_missing_profile"
        _finalize(selection, key="selection")
        return
    if not traj_routes:
        selection["failure_category"] = "rerun_no_routes"
        _finalize(selection, key="selection")
        return

    allowlist = load_branch_tool_allowlist(branch_name)
    if not allowlist:
        selection = _selection_fail(
            f"Branch {branch_name} does not expose tool allowlist.",
            failure_category="rerun_missing_allowlist",
        )
        _finalize(selection, key="selection")
        return
    assistant_policy = load_branch_assistant_guidance(branch_name)
    for attempt in range(1, 3):
        rerun_tool_names = _tool_names_from_trajs(trajs)
        rerun_tool_schemas = _expand_tool_schemas_for_names(
            rerun_tool_names,
            platform_schemas=base_platform_schemas,
            med_schemas=med_tool_schemas,
            source_schemas=source_tool_schemas,
        )
        if select_route["model"].startswith("gemini"):
            rerun_tools = [schema["function"] for schema in rerun_tool_schemas]
        else:
            rerun_tools = rerun_tool_schemas

        evals = [t.get("annotation", {}) for t in trajs]
        guidance = rerun_guidance_from_annotations(
            annotations=evals,
            branch=branch_name,
            model=select_route.get("model", ""),
            client_obj=select_route.get("client"),
            tools=rerun_tools,
            now_str=now_str,
            errors=errors,
            ctx_file=path.name,
            ctx_label=label,
        )
        rerun_guidance = guidance.get("rerun_guidance", "")
        if not rerun_guidance:
            selection = _selection_fail(
                "Failed to generate rerun guidance.",
                failure_category="rerun_guidance_error",
            )
            selection["llm_raw"] = guidance.get("llm_raw", "")
            _finalize(selection, key="selection")
            return

        runner = AgentRunner()
        rerun_trajs: List[Dict[str, Any]] = []
        groups = group_routes_by_key(traj_routes, "key_assistant")

        def _run_group(key, routes):
            out: List[Dict[str, Any]] = []
            for r in routes:
                store = load_store_from_meta_profile(payload["profile"], payload["meta"])
                out.append(
                    runner.run(
                        store=store,
                        task_instruction_text=payload.get("task_instruction", ""),
                        targets=json.dumps(payload.get("targets", []), ensure_ascii=False),
                        label=payload.get("label", ""),
                        rerun_guidance=rerun_guidance,
                        assistant_tool_allowlist=allowlist,
                        assistant_policy=assistant_policy,
                        traj_route=r,
                    )
                )
            return out

        try:
            with ThreadPoolExecutor(max_workers=len(groups) or 1) as ex:
                futures = [ex.submit(_run_group, key, routes) for key, routes in groups.items()]
                for fut in as_completed(futures):
                    rerun_trajs.extend(fut.result())
        except Exception as e:
            errors.add(
                kind="runner_error",
                stage="rerun.run",
                detail=_exc_reason(f"rerun_attempt_{attempt}_run_error", e),
                file=path.name,
                branch=branch_name,
                label=label,
                raw=None,
            )
            selection = _selection_fail(
                _exc_reason(f"rerun_attempt_{attempt}_run_error", e),
                        rerun_guidance=rerun_guidance,
                failure_category="rerun_run_error",
                )
            _finalize(selection, key="selection")
            return

        rerun_trajs = rerun_trajs[:2]
        # Refresh tool lists using rerun trajectories to include any new tools.
        rerun_tool_names = _tool_names_from_trajs(rerun_trajs)
        tool_desc = _expand_tool_desc_for_names(
            rerun_tool_names,
            platform_desc=base_platform_desc,
            med_desc=med_tool_desc,
            source_desc=source_tool_desc,
        )
        tool_schemas_openai = _expand_tool_schemas_for_names(
            rerun_tool_names,
            platform_schemas=base_platform_schemas,
            med_schemas=med_tool_schemas,
            source_schemas=source_tool_schemas,
        )
        for i, t in enumerate(rerun_trajs):
            result = judge_traj_multi(
                    traj=t,
                tools_desc=tool_desc,
                ann_routes=ann_routes,
                    branch=branch_name,
                now_str=now_str,
                    errors=errors,
                    ctx_file=path.name,
                    ctx_label=label,
                    traj_idx=i,
                )
            result["annotation"]["assistant_model"] = t.get("assistant_model", "")
            t["annotations"] = result.get("annotations", [])
            t["annotation"] = result.get("annotation", {})

        payload.setdefault("rerun_trajs", []).extend(rerun_trajs)

        status, selection, selected_traj = _select_from_trajs(rerun_trajs, stage_label=f"rerun_attempt_{attempt}")
        if status == "selected" and selected_traj is not None:
            payload["selection"] = selection
            _finalize_with_align(selection, selected_traj)
            return

        trajs = rerun_trajs

    selection["failure_category"] = "rerun_exhausted"
    _finalize(selection, key="selection")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate trajectories with LLM-as-judge.")
    default_base = Path(__file__).resolve().parent

    parser.add_argument(
        "--logs_glob",
        default="",
        help="Root dir for traj files (we will rglob '*.json').",
    )
    parser.add_argument(
        "--max_trajs_per_file",
        type=int,
        default=None,
        help="Optional cap on number of trajs per file.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Where to save annotated files.",
    )
    # keep default True, allow --rerun / --no-rerun
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Allow reruns when refine returns rerun_guidance.",
    )

    args = parser.parse_args()

    ann_routes = get_ann_routes()
    traj_routes = get_traj_model_routes(inference=False)

    tool_schema_path = default_base / "backend" / "tool_schemas" / "platform_tools.json"
    med_schema_path = default_base / "backend" / "tool_schemas" / "med_tools.json"
    source_schema_path = default_base / "backend" / "tool_schemas" / "source_tools.json"
    tool_desc = load_tool_summaries(tool_schema_path)
    tool_schemas_openai = _load_tool_schemas(tool_schema_path)
    med_tool_desc = load_tool_summaries(med_schema_path)
    med_tool_schemas = _load_tool_schemas(med_schema_path)
    source_tool_desc = load_tool_summaries(source_schema_path)
    source_tool_schemas = _load_tool_schemas(source_schema_path)

    out_dir = Path(args.output_dir) if args.output_dir else (default_base / "generations" / "annotated_logs")
    log_root = Path(args.logs_glob)

    files = sorted(p for p in log_root.rglob("*.json") if p.is_file())
    if not files:
        print("No traj files matched.")

    for file_path in files:
        annotate_file(
            path=Path(file_path),
            tool_desc=tool_desc,
            tool_schemas_openai=tool_schemas_openai,
            med_tool_desc=med_tool_desc,
            med_tool_schemas=med_tool_schemas,
            source_tool_desc=source_tool_desc,
            source_tool_schemas=source_tool_schemas,
            ann_routes=ann_routes,
            traj_routes=traj_routes,
            max_trajs=args.max_trajs_per_file,
            out_dir=out_dir,
            rerun_enabled=args.rerun,
        )
