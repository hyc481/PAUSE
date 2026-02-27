from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Iterable, Optional

def parse_tool_call(
    tool_call: Any,
    model_name: Optional[str] = None,
    call_counters: Optional[Dict[str, int]] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize an OpenAI tool_call object to (name, args_dict).
    Safely JSON-parses arguments; falls back to an empty dict on failure.
    """

    is_gemini = bool(model_name) and "gemini" in str(model_name).lower()
    needs_schema_format = {"get_source_features", "get_med_features"}
    id_tools = {
        "add_note",
        "create_session_record",
        "create_meal_record",
        "create_daily_reminder",
        "plot_time_series",
        "med-create_appointment",
        "add_to_cart",
        "prepare_order",
        "upgrade_membership_request",
    }

    # Standard OpenAI shape: tool_call.function.name / .arguments
    func = None
    if isinstance(tool_call, dict):
        func = tool_call.get("function")
        if func is not None:
            name = func.get("name")
            raw_args = func.get("arguments")
        else:
            name = tool_call.get("name")
            raw_args = tool_call.get("arguments")
    else:
        func = getattr(tool_call, "function", None)
        if func is not None:
            name = getattr(func, "name", None)
            raw_args = getattr(func, "arguments", None)
        else:
            # Fallbacks
            name = getattr(tool_call, "name", None)
            raw_args = getattr(tool_call, "arguments", None)

    if not name:
        raise ValueError("Tool call missing name")

    if name.startswith("med-"):
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        elif raw_args is None:
            args = {}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
        else:
            raise TypeError("Invalid tool arguments type")

        if is_gemini and name in needs_schema_format:
            args["_tool_schema_format"] = "gemini"

        if name in id_tools and call_counters is not None:
            call_counters[name] = call_counters.get(name, 0) + 1
            args["_tool_call_name"] = name
            args["_tool_call_index"] = call_counters[name]
            if rng_seed is not None:
                args["_id_seed"] = rng_seed

        return name, args

    if "-" in name:
        source, name = name.split("-", 1)
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
            args["source_name"] = source
        elif raw_args is None:
            args = {}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
            args["source_name"] = source
        else:
            raise TypeError("Invalid tool arguments type")

        # Preserve full tool name for allowlist checks (e.g., "fitbit-get_intraday_steps")
        args["_tool_full_name"] = f"{source}-{name}"

        if is_gemini and name in needs_schema_format:
            args["_tool_schema_format"] = "gemini"

        if name in id_tools and call_counters is not None:
            call_counters[name] = call_counters.get(name, 0) + 1
            args["_tool_call_name"] = name
            args["_tool_call_index"] = call_counters[name]
            if rng_seed is not None:
                args["_id_seed"] = rng_seed

        return name, args

    if "." in name:
        source, name = name.split(".")[0], name.split(".")[1]
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
            args["source_name"] = source
        elif raw_args is None:
            args = {}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
            args["source_name"] = source
        else:
            raise TypeError("Invalid tool arguments type")

        # Preserve full tool name for allowlist checks (e.g., "fitbit.get_intraday_steps")
        args["_tool_full_name"] = f"{source}.{name}"

        if is_gemini and name in needs_schema_format:
            args["_tool_schema_format"] = "gemini"

        if name in id_tools and call_counters is not None:
            call_counters[name] = call_counters.get(name, 0) + 1
            args["_tool_call_name"] = name
            args["_tool_call_index"] = call_counters[name]
            if rng_seed is not None:
                args["_id_seed"] = rng_seed

        return name, args

    else:
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        elif raw_args is None:
            args = {}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
        else:
            raise TypeError("Invalid tool arguments type")

        if is_gemini and name in needs_schema_format:
            args["_tool_schema_format"] = "gemini"

        if name in id_tools and call_counters is not None:
            call_counters[name] = call_counters.get(name, 0) + 1
            args["_tool_call_name"] = name
            args["_tool_call_index"] = call_counters[name]
            if rng_seed is not None:
                args["_id_seed"] = rng_seed

        return name, args


def validate_tool_call(tool_name: str, actor: str, user_only_tools: Iterable[str]) -> None:
    """
    Ensure user-only tools are only invoked by the user agent.
    actor: "assistant" or "user_agent"
    """
    if tool_name in user_only_tools and actor != "user_agent":
        raise PermissionError(f"Tool '{tool_name}' is user-only and cannot be called by assistant.")
