import os
import json
from typing import Any
from openai import OpenAI
from google import genai
from google.genai import types


# Base clients (extend as needed)
endpoint = ""
endpoint_siri = ""
endpoint_siri_200 = ""
endpoint_jerry = ""
end_point_comet = ""


api_key = ""
api_key_advanced = ""
api_key_comet1 = ""
api_key_comet2 = ""
api_key_siri=""
api_key_jerry = ""
api_key_siri_200 = ""
api_key_gemini1 = ""
api_key_gemini2 = ""
api_key_gemini3 = ""

api_key_deepseek1 = ""
api_key_deepseek2 = ""
api_key_deepseek3 = ""
api_key_deepseek4 = ""
os.environ["GEMINI_API_KEY"] = ""



client = OpenAI(base_url=endpoint, api_key=api_key)
client_siri = OpenAI(base_url=endpoint_siri, api_key=api_key_siri)
client_siri_200 = OpenAI(base_url=endpoint_siri_200, api_key=api_key_siri_200)
client_jerry = OpenAI(base_url=endpoint_jerry, api_key=api_key_jerry)
client_gemini1 = genai.Client(api_key=api_key_gemini1)
client_gemini2 = genai.Client(api_key=api_key_gemini2)
client_gemini3 = genai.Client(api_key=api_key_gemini3)

client_comet1 = OpenAI(base_url=end_point_comet, api_key=api_key_comet1)
client_comet2 = OpenAI(base_url=end_point_comet, api_key=api_key_comet2)
client_advanced = OpenAI(api_key=api_key_advanced)
client_deepseek1 = OpenAI(base_url="https://api.deepseek.com", api_key=api_key_deepseek1)
client_deepseek2 = OpenAI(base_url="https://api.deepseek.com", api_key=api_key_deepseek2)
client_deepseek3 = OpenAI(base_url="https://api.deepseek.com", api_key=api_key_deepseek3)
client_deepseek4 = OpenAI(base_url="https://api.deepseek.com", api_key=api_key_deepseek4)

# Client registry (key -> client)
CLIENTS = {
    # azure
    "default": client,
    "advanced": client_advanced,
    "siri": client_siri,
    "jerry": client_jerry,
    # comet: gpt-4.1, o4-mini, Llama3.3-70B, qwen,
    "siri_200": client_siri_200,
    "comet1": client_comet1,
    "comet2": client_comet2,
    # deepseek
    "deepseek1": client_deepseek1,
    "deepseek2": client_deepseek2,
    "deepseek3": client_deepseek3,
    "deepseek4": client_deepseek4,
    # gemini
    "gemini1": client_gemini1,
    "gemini2": client_gemini2,
    "gemini3": client_gemini3
}

# Route definitions by role (client key + model name)
# GEN_ROUTE = {"client": "gemini1", "model": "gemini-3-pro-preview"}
GEN_ROUTE = {"client": "default", "model": "gpt-5.2"}
ANN_ROUTES = [
    {"client": "gemini1", "model": "gemini-3-flash-preview"},
    {"client": "default", "model": "gpt-5.2"},
    ]
# ANN_ROUTE = {"client": "default", "model": "gpt-5"}

# Trajectory collection routes: list of {client, assistant_model, user_model}
TRAJ_MODEL_ROUTES = [
    # {"client_assistant": "default", "assistant_model": "gpt-5", "client_user": "deepseek1", "user_model": "deepseek-chat"},
    {
        "client_assistant": "deepseek1",
        "assistant_model": "deepseek-reasoner",
        "client_user": "gemini1",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini1",
        "user_valid_model": "gemini-3-flash-preview",
    },
    {
        "client_assistant": "siri_200",
        "assistant_model": "gpt-4.1",
        "client_user": "gemini2",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini2",
        "user_valid_model": "gemini-3-flash-preview",
    },
    {
        "client_assistant": "default",
        "assistant_model": "gpt-5",
        "client_user": "gemini2",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini2",
        "user_valid_model": "gemini-3-flash-preview",
    },
]

INFERENCE_TRAJ_MODEL_ROUTES = [
    {
        "client_assistant": "deepseek1",
        "assistant_model": "deepseek-reasoner",
        "client_user": "gemini1",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini1",
        "user_valid_model": "gemini-3-flash-preview",
    },
    {
        "client_assistant": "siri_200",
        "assistant_model": "gpt-4.1",
        "client_user": "gemini2",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini2",
        "user_valid_model": "gemini-3-flash-preview",
    },
    {
        "client_assistant": "gemini3",
        "assistant_model": "gemini-3-pro-preview",
        "client_user": "gemini3",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini3",
        "user_valid_model": "gemini-3-flash-preview",
    },
]

"""
INFERENCE_TRAJ_MODEL_ROUTES = [
    {
        "client_assistant": "gemini2",
        "assistant_model": "gemini-2.5-pro",
        "client_user": "gemini2",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini2",
        "user_valid_model": "gemini-3-flash-preview",
    },
    {
        "client_assistant": "default",
        "assistant_model": "gpt-5",
        "client_user": "gemini1",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini1",
        "user_valid_model": "gemini-3-flash-preview",
    },
    {
        "client_assistant": "gemini3",
        "assistant_model": "gemini-3-pro-preview",
        "client_user": "gemini3",
        "user_model": "gemini-3-flash-preview",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "gemini3",
        "user_valid_model": "gemini-3-flash-preview",
    },
]
"""

"""
    {
        "client_assistant": "siri_200",
        "assistant_model": "gpt-4.1",
        "client_user": "advanced",
        "user_model": "gpt-5",
        # user roleplay validation (user_valid) route for this trajectory
        "valid_client": "advanced",
        "user_valid_model": "gpt-5-mini",
    },
{
    "client_assistant": "default",
    "assistant_model": "gpt-4.1-mini",
    "client_user": "deepseek1",
    "user_model": "deepseek-chat",
    # user roleplay validation (user_valid) route for this trajectory
    "valid_client": "deepseek1",
    "user_valid_model": "deepseek-chat",
},
# Example additional route (uncomment to use):
{
    "client_assistant": "gemini1",
    "assistant_model": "gemini-3-flash-preview",
    "client_user": "deepseek2",
    "user_model": "deepseek-chat",
    "valid_client": "deepseek2",
    "user_valid_model": "deepseek-chat",
},
"""

def _pick_client(key: str) -> Any:
    if key not in CLIENTS:
        raise ValueError(f"Unknown client key: {key}")
    return CLIENTS[key]


def gemini_tools_from_openai(tools: list[dict]):
    """
    Placeholder for Gemini tool schema conversion.
    TODO: map OpenAI tool schema to Gemini format when tool calling is enabled.
    """
    if not tools:
        return []
    decls: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t:
            func = t.get("function") or {}
            decls.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
            )
        elif "name" in t:
            decls.append(
                {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                }
            )
    if not decls:
        return []
    return [types.Tool(function_declarations=decls)]


def call_llm(
    client_obj: Any,
    *,
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    response_mime_type: str | None = None,
) -> str:
    """
    Unified LLM call. Returns text content.
    - OpenAI-compatible clients: client_obj.chat.completions.create
    - Gemini client: client_obj.models.generate_content
    Note: For DeepSeek reasoner, this function does not return reasoning_content.
    Use call_llm_with_tools if you need reasoning_content.
    """
    if hasattr(client_obj, "models") and hasattr(client_obj.models, "generate_content"):
        # Gemini style
        tools_cfg = gemini_tools_from_openai(tools or [])
        config = types.GenerateContentConfig(
            tools=tools_cfg or None,
            temperature=temperature,
            response_mime_type=response_mime_type,
        )
        resp = client_obj.models.generate_content(
            model=model,
            contents=_gemini_contents(messages),
            config=config,
        )
        return getattr(resp, "text", "") or ""

    # OpenAI style
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    
    # Enable thinking mode for DeepSeek reasoner
    if is_deepseek_reasoner(model, client_obj):
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    
    resp = client_obj.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def is_gemini_client(client_obj: Any) -> bool:
    return hasattr(client_obj, "models") and hasattr(client_obj.models, "generate_content")


def is_deepseek_reasoner(model: str, client_obj: Any) -> bool:
    """
    Check if the model is DeepSeek reasoner (thinking mode enabled).
    Only returns True if model name explicitly contains 'reasoner'.
    This ensures we don't accidentally enable thinking mode for regular DeepSeek models.
    """
    return "reasoner" in model.lower()


def _gemini_contents(messages: list[dict]) -> list[Any]:
    """
    Convert OpenAI-style messages to Gemini contents (text-only).
    """
    contents: list[Any] = []
    for m in messages or []:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            # If we have raw Gemini content, reuse it to preserve thought_signature.
            raw_content = m.get("gemini_content")
            if raw_content is not None:
                contents.append(raw_content)
                continue
            # Gemini expects function_call parts to come from the model response directly.
            # Do NOT reconstruct function_call parts from tool_calls to avoid thought_signature errors.
            role = "model"
            parts: list[Any] = []
            if content:
                parts.append({"text": str(content)})
            contents.append(types.Content(role=role, parts=parts))
            continue
        if role == "system":
            role = "user"
            content = f"[SYSTEM]\n{content}"
            contents.append({"role": role, "parts": [{"text": str(content)}]})
            continue
        if role == "tool":
            tool_name = m.get("name", "") or m.get("tool_name", "")
            response_payload = m.get("result")
            if response_payload is None:
                raw = m.get("content", "") or ""
                try:
                    response_payload = json.loads(raw)
                except Exception:
                    response_payload = {"result": raw}
            if not isinstance(response_payload, dict):
                response_payload = {"result": response_payload}
            part = types.Part.from_function_response(name=tool_name, response=response_payload)
            contents.append(types.Content(role="user", parts=[part]))
            continue
        contents.append({"role": role, "parts": [{"text": str(content)}]})
    return contents


def call_llm_with_tools(
    client_obj: Any,
    *,
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    response_mime_type: str | None = None,
) -> tuple[str, list[dict], Any, str | None]:
    """
    Unified LLM call that returns (content, tool_calls, raw_content, reasoning_content).
    tool_calls are normalized to OpenAI-like dicts.
    reasoning_content is only returned for DeepSeek reasoner models.
    """
    if hasattr(client_obj, "models") and hasattr(client_obj.models, "generate_content"):
        tools_cfg = gemini_tools_from_openai(tools or [])
        config = types.GenerateContentConfig(
            tools=tools_cfg or None,
            temperature=temperature,
            response_mime_type=response_mime_type,
        )
        resp = client_obj.models.generate_content(
            model=model,
            contents=_gemini_contents(messages),
            config=config,
        )
        content = getattr(resp, "text", "") or ""
        tool_calls: list[dict] = []
        candidates = getattr(resp, "candidates", []) or []
        if candidates:
            parts = getattr(candidates[0], "content", None)
            parts = getattr(parts, "parts", []) if parts is not None else []
            for i, part in enumerate(parts or []):
                fn_call = getattr(part, "function_call", None)
                if fn_call is None:
                    continue
                name = getattr(fn_call, "name", "") or ""
                args = getattr(fn_call, "args", {}) or {}
                tool_calls.append(
                    {
                        "id": f"gemini_{i}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                )
        raw_content = getattr(candidates[0], "content", None) if candidates else None
        return content, tool_calls, raw_content, None

    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    
    # Enable thinking mode for DeepSeek reasoner
    if is_deepseek_reasoner(model, client_obj):
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    
    resp = client_obj.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or None
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    tool_calls: list[dict] = []
    for tc in tool_calls_raw:
        func = getattr(tc, "function", None)
        tool_calls.append(
            {
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(func, "name", ""),
                    "arguments": getattr(func, "arguments", ""),
                },
            }
        )
    return content, tool_calls, None, reasoning_content


def get_gen_route():
    return _pick_client(GEN_ROUTE["client"]), GEN_ROUTE["model"]


def get_ann_route():
    """
    Backward-compatible single annotator route.
    Returns the first entry in ANN_ROUTES.
    """
    if not ANN_ROUTES:
        raise ValueError("ANN_ROUTES is empty.")
    first = ANN_ROUTES[0]
    return _pick_client(first["client"]), first["model"]


def get_ann_routes():
    """
    Returns list of annotator routes with client objects and model names.
    """
    routes: list[dict] = []
    for r in ANN_ROUTES:
        if not isinstance(r, dict):
            continue
        key = r.get("client")
        routes.append({
            "client": _pick_client(key),
            "model": r.get("model", ""),
            "key": key,
        })
    return routes


def get_traj_model_routes(inference:bool):
    """
    Returns list of dict routes with both client objects and model names, so callers can pass a single
    `traj_route` object around.

    Each route contains:
      - assistant_client, assistant_model
      - user_client, user_model
      - valid_client, user_valid_model
      - key_assistant, key_user, key_valid (client registry keys)
    """
    routes: list[dict] = []
    if inference:
        selected_routes = INFERENCE_TRAJ_MODEL_ROUTES
    else:
        selected_routes = TRAJ_MODEL_ROUTES
    for r in selected_routes:
        if not isinstance(r, dict):
            # Allow comments / accidental non-dict entries without breaking runtime.
            continue
        key_a = r.get("client_assistant")
        key_u = r.get("client_user")
        key_v = r.get("valid_client") or key_u  # default to user client if unspecified
        routes.append({
            "assistant_client": _pick_client(key_a),
            "assistant_model": r["assistant_model"],
            "user_client": _pick_client(key_u),
            "user_model": r["user_model"],
            "valid_client": _pick_client(key_v),
            "user_valid_model": r.get("user_valid_model", r["user_model"]),
            "key_assistant": key_a,
            "key_user": key_u,
            "key_valid": key_v,
        })
    return routes
