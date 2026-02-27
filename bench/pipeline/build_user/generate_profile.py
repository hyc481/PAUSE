from __future__ import annotations

"""
Draft helpers to prepare LLM-ready inputs for user profile generation.
Inputs:
- 30-day daily totals (steps, mets, calories, AZM) computed from wearable tables.
- Habit templates from inject_sports.HABIT_TEMPLATES.
- Meal calories synthesized by inject_meals.generate_meals.
- Common product tags from shopping_bench/product_generation/generate_product_log.py.

Outputs:
- A structured payload that can be fed to prompts in bench/prompts/generation_prompt.py
  to generate high-level NLP descriptions and user_profile / med_user_profile.

Note: This is a scaffold; actual LLM calls live elsewhere.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import json
import random
from bench.utils.clients import GEN_ROUTE, CLIENTS, call_llm
from bench.prompts.generation_prompt import user_generation_prompt
from bench.utils.generate_ids import generate_health_provider_id

# --------------------------------------------------------------------------- #
# Wearable aggregation
# --------------------------------------------------------------------------- #

def _daily_sum(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "ActivityMinute",
    *,
    as_int: bool = False,
    round_ndigits: int | None = None,
) -> Dict[str, float | int]:
    if df is None or df.empty or value_col not in df.columns:
        return {}

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    grouped = df.groupby(df[date_col].dt.strftime("%Y-%m-%d"))[value_col].sum()

    out = {}
    for k, v in grouped.items():
        val = float(v)
        if round_ndigits is not None:
            val = round(val, round_ndigits)
        if as_int:
            val = int(round(val))
        out[k] = val

    return out



def extract_30day_activity_totals(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float | int]]:
    steps = _daily_sum(
        tables.get("minute_steps"),
        "Steps",
        as_int=True,               # ✅ 整数
    )

    mets = _daily_sum(
        tables.get("minute_mets"),
        "METs",
        as_int=True,               # ✅ 整数
    )

    calories = _daily_sum(
        tables.get("minute_calories"),
        "Calories",
        round_ndigits=2,           # ✅ 保留两位小数
    )

    azm = {}
    int_df = tables.get("minute_intensities")
    if int_df is not None and not int_df.empty and "Intensity" in int_df.columns:
        df = int_df.copy()
        df["ActivityMinute"] = pd.to_datetime(df["ActivityMinute"])
        df["is_azm"] = (df["Intensity"] >= 2).astype(int)
        grouped = df.groupby(df["ActivityMinute"].dt.strftime("%Y-%m-%d"))["is_azm"].sum()
        for k, v in grouped.items():
            azm[k] = int(v)         # 已经是你想要的整数

    return {
        "steps": steps,
        "mets": mets,
        "calories": calories,
        "azm": azm,
    }



# --------------------------------------------------------------------------- #
# Meals summary
# --------------------------------------------------------------------------- #

def summarize_meal_calories(meals: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Sum calories per day from meal records (expects nutrition.calories).
    """
    out: Dict[str, float] = {}
    for m in meals or []:
        ts = m.get("time") or m.get("timestamp")
        if not ts:
            continue
        day = ts.split("T")[0]
        cals = m.get("nutrition", {}).get("calories", 0.0)
        out[day] = out.get(day, 0.0) + float(cals)

    return {day: round(total, 2) for day, total in out.items()}


# --------------------------------------------------------------------------- #
# Source profile synthesis (light jitter from user profile)
# --------------------------------------------------------------------------- #

def _jitter_int(val: int, rng: random.Random, pct: float = 0.2) -> int:
    if val is None:
        return 0
    delta = rng.uniform(-pct, pct) * val
    return max(0, int(round(val + delta)))

def _human_round(value, scale):
    value = int(round(value/scale)*scale)
    return value

def _make_source_profile_entry(source: str, profile: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    user_profile = profile.get("user_profile", {}) or {}
    basic = user_profile.get("basic_info", {}) or {}
    daily_goal = user_profile.get("daily_goal", {}) or {}

    # jitter daily goals; fallback to 0
    steps = _human_round(_jitter_int(int(daily_goal.get("steps", 0)), rng), 100)
    mets = _human_round(_jitter_int(int(daily_goal.get("mets", 0)), rng), 100)
    calories_burn = _human_round(_jitter_int(int(daily_goal.get("calories_burn", 0)), rng), 100)
    azm = _human_round(_jitter_int(int(daily_goal.get("AZM", 0)), rng), 5)

    weekly_goal = {
        "steps": _human_round(steps*7, 1000),
        "mets": _human_round(mets*7, 1000),
        "calories_burn": _human_round(calories_burn*7, 1000),
        "AZM": _human_round(azm*7, 10),
    }

    return {
        "source": source,
        "basic_info": dict(basic),
        "daily_goal": {
            "steps": steps,
            "mets": mets,
            "calories_burn": calories_burn,
            "AZM": azm,
        },
        "weekly_goal": weekly_goal,
    }


def generate_source_profiles(profile: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """
    Build source_profile entries for each source appearing in source_assignment (excluding 'missing').
    Goals are lightly jittered from user_profile.daily_goal to avoid identical copies.
    """
    assignment = profile.get("source_assignment", {}) or {}
    sources = {s for s in assignment.values() if s and s != "missing"}
    result: List[Dict[str, Any]] = []
    for src in sorted(sources):
        result.append(_make_source_profile_entry(src, profile, rng))
    return result



# --------------------------------------------------------------------------- #
# Product tags
# --------------------------------------------------------------------------- #

COMMON_PRODUCT_TAGS = [
    # nutrition / diet
    "low_sugar", "high_fiber", "high_protein", "low_fat",
    "low_calorie", "diabetic_safe", "lactose_free", "gluten_free",
    # category markers
    "biscuit", "cookie", "cracker", "bar", "nut", "drink", "chip", "meal_replacement",
    # lifestyle / positioning
    "premium", "organic", "vegan",
]


# --------------------------------------------------------------------------- #
# Payload builder
# --------------------------------------------------------------------------- #

def build_user_generation_request(
    date_range: Tuple[str, str],
    profile: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    habit_template: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a single LLM-ready request for user persona synthesis.
    This includes:
    - compact structured payload (facts & summaries)
    - unified user generation prompt
    """

    # ---------- extract facts ----------

    activity_totals = extract_30day_activity_totals(tables)
    meal_calories = summarize_meal_calories(profile.get("meals", []))

    payload = {
        "meta": {
            "date_range": date_range,
            "now": profile.get("now"),
        },
        # used ONLY for daily_goal inference
        "activity_30d": activity_totals,
        "meals_30d_calories": meal_calories,
        # used ONLY for high-level lifestyle characterization
        "habit_templates": habit_template,
        # used ONLY to anchor health_risks / dietary_restrictions
        "common_tags": COMMON_PRODUCT_TAGS,
    }

    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    prompt = f"{user_generation_prompt}\n\nInput:\n{payload_json}"

    return {
        "name": "user_persona_generation",
        "prompt": prompt,
        "payload": payload,  # kept for traceability / debugging
    }


# --------------------------------------------------------------------------- #
# Simple sender (user supplies an OpenAI-compatible client)
# --------------------------------------------------------------------------- #

def _strip_json_fence(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences if present.
    """
    if not text:
        return text

    stripped = text.strip()

    # Handle ```json\n ... \n``` or ```\n ... \n```
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove first line (``` or ```json)
        if lines:
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    return stripped


def _validate_profile_schema(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False

    # ---------- Top-level ----------
    expected_top_keys = {
        "user_profile",
        "med_user_profile",
        "narrative_summary",
    }
    if set(data.keys()) != expected_top_keys:
        return False

    # ---------- user_profile ----------
    user = data["user_profile"]
    if not isinstance(user, dict):
        return False

    expected_user_keys = {
        "basic_info",
        "health_risks",
        "dietary_restrictions",
        "preferences",
    }
    if set(user.keys()) != expected_user_keys:
        return False

    # basic_info
    basic = user["basic_info"]
    if not isinstance(basic, dict):
        return False
    if set(basic.keys()) != {"age", "gender", "name"}:
        return False
    if not isinstance(basic["age"], int):
        return False
    if not isinstance(basic["gender"], str):
        return False
    if not isinstance(basic["name"], str):
        return False

    # health_risks
    if not isinstance(user["health_risks"], list):
        return False

    # dietary_restrictions
    if not isinstance(user["dietary_restrictions"], list):
        return False

    # preferences
    prefs = user["preferences"]
    if not isinstance(prefs, dict):
        return False
    if set(prefs.keys()) != {"sport", "food"}:
        return False
    if not all(isinstance(prefs[k], list) for k in prefs):
        return False

    # ---------- med_user_profile ----------
    med = data["med_user_profile"]
    if not isinstance(med, dict):
        return False

    expected_med_keys = {
        "basic_info",
        "health_risks",
        "dietary_restrictions",
    }
    if set(med.keys()) != expected_med_keys:
        return False

    if not isinstance(med["basic_info"], dict):
        return False
    if not isinstance(med["health_risks"], list):
        return False
    if not isinstance(med["dietary_restrictions"], list):
        return False

    # ---------- narrative_summary ----------
    summary = data["narrative_summary"]
    if not isinstance(summary, str):
        return False

    word_count = len(summary.split())
    if not (80 <= word_count <= 120):
        return False

    return True


def _sample_range(
    rng: random.Random,
    min_val: int,
    max_val: int,
    step: int,
) -> int:
    if step <= 0:
        raise ValueError("step must be positive")
    n_steps = ((max_val - min_val) // step) + 1
    return min_val + step * rng.randrange(n_steps)


def _sample_daily_goal(rng: random.Random) -> Dict[str, int]:
    """
    Sample daily goal values with fixed granularity to avoid LLM parsing.
    """
    return {
        "steps": _sample_range(rng, 8000, 10000, 100),
        "mets": _sample_range(rng, 20000, 25000, 1000),
        "calories_burn": _sample_range(rng, 2000, 2500, 100),
        "AZM": _sample_range(rng, 20, 60, 10),
        "calories_intake": _sample_range(rng, 2000, 2500, 100),
    }

def generate_profile(
    date_range: Tuple[str, str],
    profile: Dict[str, Any],
    rng: random.Random,
    tables: Dict[str, pd.DataFrame],
    habit_template: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Send a single combined profile-generation request to an OpenAI-compatible client.
    """
    request = build_user_generation_request(
        date_range, profile, tables, habit_template
    )
    client = CLIENTS[GEN_ROUTE["client"]]
    content = call_llm(
        client,
        model=GEN_ROUTE["model"],
        messages=[{"role": "user", "content": request["prompt"]}]
    )
    cleaned = _strip_json_fence(content)
    parsed = json.loads(cleaned)
    schema_valid = _validate_profile_schema(parsed)
    if not schema_valid:
        raise ValueError("Parsed JSON does not match expected schema")

    parsed["user_profile"]["daily_goal"] = _sample_daily_goal(rng)

    for k, v in parsed.items():
        profile[k] = v

    # Build source_profile using existing source_assignment (if present)
    profile["source_profile"] = generate_source_profiles(
        profile,
        rng=rng,
    )
    profile["health_providers"] = generate_health_providers(rng)
    return parsed


def generate_health_providers(
    rng: random.Random,
    min_n: int = 3,
    max_n: int = 5,
) -> List[Dict[str, str]]:
    doctor_pool = [
        "Dr. Emily Chen", "Dr. Michael Patel", "Dr. Sarah Johnson",
        "Dr. Daniel Kim", "Dr. Olivia Martinez", "Dr. James Wilson",
        "Dr. Aisha Ahmed", "Dr. Robert Nguyen",
    ]
    clinic_pool = [
        "River Valley Family Clinic", "Downtown Medical Centre",
        "Northgate Health Clinic", "Lakeside Primary Care",
        "Evergreen Wellness Clinic", "Maple Ridge Medical",
        "Aurora Community Health", "Cedar Grove Clinic",
    ]
    address_pool = [
        "101 Jasper Ave",
        "225 Whyte Ave",
        "4800 99 St NW",
        "12 Kingsway NW",
        "8500 112 St NW",
        "300 104 Ave NW",
        "6607 28 Ave NW",
        "170 St & 87 Ave",
    ]

    # -------- zip-style alignment --------
    rng.shuffle(doctor_pool)
    rng.shuffle(clinic_pool)
    rng.shuffle(address_pool)

    n = rng.randint(min_n, max_n)
    max_len = min(len(doctor_pool), len(clinic_pool), len(address_pool))
    n = min(n, max_len)

    providers: List[Dict[str, str]] = []
    for i in range(n):
        doctor = doctor_pool[i]
        clinic = clinic_pool[i]
        address = address_pool[i]

        providers.append({
            "provider_id": generate_health_provider_id(),
            "doctor": doctor,
            "clinic": clinic,
            "address": address
        })

    return providers


