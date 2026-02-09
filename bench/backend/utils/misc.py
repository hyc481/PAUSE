import json
from pathlib import Path
from typing import Dict, List, Any
import importlib

def load_tool_summaries(path: Path, max_tools: int = 50) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data[:max_tools]:
        func = item.get("function", {})
        out.append(
            {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
        )
    return out


def load_branch_tool_allowlist(branch_name: str) -> List[str] | None:
    """
    Load branch tool allowlist from `bench.backend.generate_task.branches.<branch_name>`.
    Convention:
    - Prefer `get_involved_platform_tool_names() -> list[str]`
    - Fallback to module-level `involved_tool_names`
    Returns None if branch module can't be loaded or doesn't expose allowlist.
    """
    try:
        mod = importlib.import_module(f"bench.backend.generate_task.branches.{branch_name}")
    except Exception:
        return None
    if hasattr(mod, "get_involved_platform_tool_names"):
        try:
            return list(getattr(mod, "get_involved_platform_tool_names")())
        except Exception:
            return None
    if hasattr(mod, "involved_tool_names"):
        try:
            return list(getattr(mod, "involved_tool_names"))
        except Exception:
            return None
    return None


def load_branch_assistant_guidance(branch_name: str) -> str | None:
    """
    Load a branch-specific assistant guidance prompt for 'standard' (non-inference) runs.

    Conventions (checked in order):
    - get_branch_assistant_guidance() -> str
    - <branch_name>_tool_selection_prompt (e.g. wearable_data_casual_tool_selection_prompt)
    - branch_tool_selection_prompt
    """
    try:
        mod = importlib.import_module(f"bench.backend.generate_task.branches.{branch_name}")
    except Exception:
        return None

    if hasattr(mod, "get_branch_assistant_guidance"):
        try:
            s = getattr(mod, "get_branch_assistant_guidance")()
            return str(s) if s else None
        except Exception:
            return None

    key = f"{branch_name}_tool_selection_prompt"
    if hasattr(mod, key):
        try:
            s = getattr(mod, key)
            return str(s) if s else None
        except Exception:
            return None

    if hasattr(mod, "branch_tool_selection_prompt"):
        try:
            s = getattr(mod, "branch_tool_selection_prompt")
            return str(s) if s else None
        except Exception:
            return None

    return None


def group_routes_by_key(routes: List[Dict[str, Any]], key_field: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in routes:
        k = str(r.get(key_field, ""))
        groups.setdefault(k, []).append(r)
    return groups

def strip_code_fences(s: str) -> str:
    """
    Remove markdown code fences like ```json ... ``` if present.
    """
    if s is None:
        return ""
    txt = s.strip()
    if txt.startswith("```"):
        txt = txt.lstrip("`")
        if "\n" in txt:
            txt = txt.split("\n", 1)[1]
        if txt.rstrip().endswith("```"):
            txt = txt.rstrip().rsplit("```", 1)[0]
    return txt.strip()