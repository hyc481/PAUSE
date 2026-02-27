# bench/backend/tools/platform_tools.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import random

from bench.utils.paths import TOOL_SCHEMAS_DIR, DATA_DIR
from bench.utils.generate_ids import (
    generate_meal_id,
    generate_plot_id,
    generate_reminder_id,
    generate_session_id,
    generate_note_id,
)
from bench.pipeline.build_user.inject_meals import ATOM_MEALS

# ============================================================
# State checking: verify data availability for a given date
# ============================================================

def _check_data_availability(store: Dict[str, Any], date: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if data is available for a given date based on source_assignment and marketplaces.
    
    Returns:
        (is_available, source_name, device_id) or (False, None, None) if unavailable
    
    Logic:
        1. Check source_assignment[date] to see which source should provide data
        2. If "missing", data is not available
        3. If a source name, check if that source is connected in marketplaces
        4. If connected, return the source and corresponding device_id
    """
    profile = store.get("profile", {})
    source_assignment = profile.get("source_assignment", {})
    system_settings = profile.get("system_settings", {})
    marketplaces = system_settings.get("marketplaces", [])
    devices = system_settings.get("devices", [])

    assigned_source = source_assignment.get(date)

    if not assigned_source or assigned_source == "missing":
        return False, None, None

    # Check if the assigned source is connected
    source_connected = any(
        marketplace.get("source") == assigned_source and marketplace.get("connected", False)
        for marketplace in marketplaces
    )
    if not source_connected:
        return False, None, None

    # Find the device_id for this source
    device_id = next(
        (device.get("device_id") for device in devices if device.get("source") == assigned_source),
        None,
    )

    return True, assigned_source, device_id


def _check_vip_status(store: Dict[str, Any]) -> bool:
    """
    Check if user has VIP/subscription status.
    Returns True if user has VIP subscription active.
    """
    profile = store.get("profile", {})
    shopping = profile.get("shopping", {})
    wallet = shopping.get("wallet", {})
    
    # Check VIP status from wallet
    return wallet.get("vip", False)


def _filter_records_by_date_range(
    records: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    date_field: str,
) -> List[Dict[str, Any]]:
    """
    Filter records by date range.
    date_field can be 'start_time', 'end_time', 'timestamp', etc.
    """
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    
    filtered = []
    for record in records:
        date_str = record.get(date_field, "")
        if not date_str:
            continue
        
        # Parse the date from ISO format
        try:
            record_date = pd.to_datetime(date_str).date()
            if start <= record_date <= end:
                filtered.append(record)
        except (ValueError, TypeError):
            continue
    
    return filtered


# ============================================================
# Helper functions: extract minute-level data from store
# ============================================================

def _get_minute_df(store: Dict[str, Any], key: str, date: str) -> pd.DataFrame:
    """
    Extract minute-level data for a given date from store.
    Assumes column name is ActivityMinute (datetime64, already processed by build_store).
    """
    tables = store.get("wearable_tables", {})
    if key not in tables:
        raise KeyError(f"Minute-level table '{key}' not found in store.")

    df = tables[key].copy()
    if "ActivityMinute" not in df.columns:
        raise ValueError(f"Table '{key}' has no 'ActivityMinute' column.")

    # Ensure datetime format
    df["ActivityMinute"] = pd.to_datetime(df["ActivityMinute"])
    target_date = pd.to_datetime(date).date()
    df = df[df["ActivityMinute"].dt.date == target_date].copy()

    return df


def _get_daily_heartrate(store: Dict[str, Any], date: str) -> Optional[pd.Series]:
    """
    Extract daily heart rate data for a given date from store.
    Returns a pandas Series with heart rate metrics, or None if not available.
    """
    tables = store.get("wearable_tables", {})
    if "daily_heartrate" not in tables:
        return None

    df = tables["daily_heartrate"].copy()
    if "Date" not in df.columns:
        return None

    # Ensure datetime format
    df["Date"] = pd.to_datetime(df["Date"])
    target_date = pd.to_datetime(date).date()
    filtered = df[df["Date"].dt.date == target_date]

    if filtered.empty:
        return None

    # Return the first row as a Series
    return filtered.iloc[0]


def _get_daily_sleep(store: Dict[str, Any], date: str) -> Optional[pd.Series]:
    """
    Extract daily sleep data for a given date from store.
    Returns a pandas Series with sleep metrics, or None if not available.
    """
    tables = store.get("wearable_tables", {})
    if "daily_sleep" not in tables:
        return None

    df = tables["daily_sleep"].copy()
    if "Date" not in df.columns:
        return None

    # Ensure datetime format
    df["Date"] = pd.to_datetime(df["Date"])
    target_date = pd.to_datetime(date).date()
    filtered = df[df["Date"].dt.date == target_date]

    if filtered.empty:
        return None

    # Return the first row as a Series
    return filtered.iloc[0]


def _hourly_aggregate(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "ActivityMinute",
    start: int = 0,
    end: int = 24,
) -> List[Dict[str, Any]]:
    """
    Aggregate minute-level data by hour.
    Returns: [ { hour, time, value }, ... ]
    """
    if df.empty:
        return []

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["hour"] = df[date_col].dt.hour

    # Filter hour range
    # Half‑open interval [start, end)
    df = df[(df["hour"] >= start) & (df["hour"] < end)]

    grouped = df.groupby("hour")[value_col].sum().sort_index()

    result: List[Dict[str, Any]] = []
    for h, v in grouped.items():
        result.append(
            {
                "hour": int(h),
                "time": f"{h:02d}:00",
                "value": round(float(v), 2)
            }
        )
    return result


def _max_hour_entry(hours: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Given a list of hourly dicts with 'value', return the max one in:
      { "time": ..., "value": ... } format.
    """
    if not hours:
        return None
    best = max(hours, key=lambda x: x["value"])
    return {"time": best["time"], "value": round(best["value"],2)}


def _validate_hour_range(start: int, end: int) -> Optional[Dict[str, Any]]:
    """
    Validate hour range for half-open interval [start, end).
    start must be in [0, 23], end in (start, 24], and start < end.
    Returns None if valid.
    """
    if start is None or end is None:
        raise ValueError("Invalid range!")

    if not (0 <= start <= 23) or not (1 <= end <= 24) or start >= end:
        raise ValueError("Invalid range!")


# ============================================================
# Platform tool implementations
# ============================================================

def get_hourly_mets(
    store: Dict[str, Any],
    date: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    Retrieve hourly aggregated METs values for a given date.
    Checks state availability before accessing data.
    """
    # Validate hour range (half-open [start, end))
    _validate_hour_range(start, end)

    is_available, _, _ = _check_data_availability(store, date)
    if not is_available:
        return {
            "date": date,
            "metric": "mets",
            "hours": [],
        }

    df = _get_minute_df(store, "minute_mets", date)
    hours = _hourly_aggregate(df, value_col="METs", start=start, end=end)

    return {
        "date": date,
        "metric": "mets",
        "hours": hours,
    }


def get_hourly_steps(
    store: Dict[str, Any],
    date: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    Retrieve hourly aggregated step counts for a given date.
    Checks state availability before accessing data.
    """
    # Validate hour range (half-open [start, end))
    _validate_hour_range(start, end)

    is_available, _, _ = _check_data_availability(store, date)
    if not is_available:
        return {
            "date": date,
            "metric": "steps",
            "hours": [],
        }

    df = _get_minute_df(store, "minute_steps", date)
    hours = _hourly_aggregate(df, value_col="Steps", start=start, end=end)

    return {
        "date": date,
        "metric": "steps",
        "hours": hours,
    }


def get_hourly_calories(
    store: Dict[str, Any],
    date: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    Retrieve hourly aggregated calorie expenditure for a given date.
    Checks state availability before accessing data.
    """
    # Validate hour range (half-open [start, end))
    _validate_hour_range(start, end)

    is_available, _, _ = _check_data_availability(store, date)
    if not is_available:
        return {
            "date": date,
            "metric": "calories",
            "hours": [],
        }

    df = _get_minute_df(store, "minute_calories", date)
    hours = _hourly_aggregate(df, value_col="Calories", start=start, end=end)

    return {
        "date": date,
        "metric": "calories",
        "hours": hours,
    }


def get_hourly_activity(
    store: Dict[str, Any],
    date: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    Retrieve hourly activity intensity distribution for a given date.
    Uses minute_intensities table:
      Intensity=0 -> sedentary
      Intensity=1 -> lightly
      Intensity=2 -> fairly
      Intensity=3 -> very
    Active Zone Minutes (AZM) = Intensity >= 2 minutes.
    Checks state availability before accessing data.
    """
    # Validate hour range (half-open [start, end))
    _validate_hour_range(start, end)

    is_available, _, _ = _check_data_availability(store, date)
    if not is_available:
        return {
            "date": date,
            "hours": [],
        }

    df = _get_minute_df(store, "minute_intensities", date)
    if df.empty:
        return {
            "date": date,
            "hours": [],
        }

    df = df.copy()
    df["ActivityMinute"] = pd.to_datetime(df["ActivityMinute"])
    df["hour"] = df["ActivityMinute"].dt.hour

    # Classify intensities
    intensity = df["Intensity"]
    df["sedentary"] = (intensity == 0).astype(int)
    df["lightly"] = (intensity == 1).astype(int)
    df["fairly"] = (intensity == 2).astype(int)
    df["very"] = (intensity == 3).astype(int)
    df["azm"] = (intensity >= 2).astype(int)

    df = df[(df["hour"] >= start) & (df["hour"] < end)]

    grouped = df.groupby("hour")[["sedentary", "lightly", "fairly", "very", "azm"]].sum()

    hours: List[Dict[str, Any]] = []
    for h, row in grouped.sort_index().iterrows():
        hours.append(
            {
                "hour": int(h),
                "time": f"{h:02d}:00",
                "sedentary_minutes": int(row["sedentary"]),
                "lightly_active_minutes": int(row["lightly"]),
                "fairly_active_minutes": int(row["fairly"]),
                "very_active_minutes": int(row["very"]),
                "azm_total": int(row["azm"]),
            }
        )

    return {
        "date": date,
        "hours": hours,
    }


def get_daily_summary(
    store: Dict[str, Any],
    date: str,
) -> Dict[str, Any]:
    is_available, source, device_id = _check_data_availability(store, date)

    def _daytime_total(hourly_data, start=8, end=20):
        return round(sum(
            h["value"]
            for h in hourly_data
            if h["hour"] is not None and start <= h["hour"] < end
        ), 2)

    if not is_available:
        return {
            "source": source or "unknown",
            "device": device_id or "unknown",
            "date": date,
            "steps": {"total": 0, "daytime_total": 0, "max_hour": None},
            "calories": {"total": 0, "daytime_total": 0, "max_hour": None},
            "mets": {"total": 0, "daytime_total": 0, "max_hour": None},
            "activity": {},
            "heart_rate": None,
            "goal": {},
            "system_score": {},
        }

    minute_steps_df = _get_minute_df(store, "minute_steps", date)
    minute_calories_df = _get_minute_df(store, "minute_calories", date)
    minute_mets_df = _get_minute_df(store, "minute_mets", date)
    minute_intensities_df = _get_minute_df(store, "minute_intensities", date)

    total_steps = int(minute_steps_df["Steps"].sum()) if not minute_steps_df.empty else 0
    total_calories = round(float(minute_calories_df["Calories"].sum()), 2) if not minute_calories_df.empty else 0.0
    total_mets = round(float(minute_mets_df["METs"].sum()), 2) if not minute_mets_df.empty else 0.0

    if minute_intensities_df.empty:
        sedentary_minutes = lightly_active_minutes = fairly_active_minutes = very_active_minutes = 0
    else:
        intensity = minute_intensities_df["Intensity"]
        sedentary_minutes = int((intensity == 0).sum())
        lightly_active_minutes = int((intensity == 1).sum())
        fairly_active_minutes = int((intensity == 2).sum())
        very_active_minutes = int((intensity == 3).sum())

    activity_total = max(
        sedentary_minutes
        + lightly_active_minutes
        + fairly_active_minutes
        + very_active_minutes,
        1,
    )

    def _pct(p, t): return round(p / t * 100, 1)

    activity = {
        "sedentary_minutes": sedentary_minutes,
        "sedentary_percent": _pct(sedentary_minutes, activity_total),
        "lightly_active_minutes": lightly_active_minutes,
        "lightly_active_percent": _pct(lightly_active_minutes, activity_total),
        "fairly_active_minutes": fairly_active_minutes,
        "fairly_active_percent": _pct(fairly_active_minutes, activity_total),
        "very_active_minutes": very_active_minutes,
        "very_active_percent": _pct(very_active_minutes, activity_total),
        "azm_total": fairly_active_minutes + very_active_minutes,
    }

    hourly_steps = _hourly_aggregate(minute_steps_df, "Steps", start=0, end=24)
    hourly_calories = _hourly_aggregate(minute_calories_df, "Calories", start=0, end=24)
    hourly_mets = _hourly_aggregate(minute_mets_df, "METs", start=0, end=24)

    steps_block = {
        "total": total_steps,
        "daytime_total": _daytime_total(hourly_steps),
        "max_hour": _max_hour_entry(hourly_steps),
    }

    calories_block = {
        "total": total_calories,
        "daytime_total": _daytime_total(hourly_calories),
        "max_hour": _max_hour_entry(hourly_calories),
    }

    mets_block = {
        "total": total_mets,
        "daytime_total": _daytime_total(hourly_mets),
        "max_hour": _max_hour_entry(hourly_mets),
    }

    hr_data = _get_daily_heartrate(store, date)
    if hr_data is not None and "resting_hr" in hr_data.index and pd.notna(hr_data["resting_hr"]):
        custom_zones = []
        for name, lo, hi in [
            ("Below", 30, 90),
            ("Fat Burn", 90, 110),
            ("Cardio", 110, 140),
            ("Peak", 140, 220),
        ]:
            key = f"zone_{name}_minutes"
            if key in hr_data.index and pd.notna(hr_data[key]):
                custom_zones.append({
                    "name": name, "min": lo, "max": hi, "minutes": int(hr_data[key])
                })

        heart_rate = {
            "resting_hr": int(hr_data["resting_hr"]),
            "avg_hr": int(hr_data["avg_hr"]),
            "max_hr": int(hr_data["max_hr"]),
            "min_hr": int(hr_data["min_hr"]),
            "customHeartRateZones": custom_zones,
        }
    else:
        heart_rate = {
            "resting_hr": None,
            "avg_hr": None,
            "max_hr": None,
            "min_hr": None,
            "customHeartRateZones": [],
        }

    profile = store.get("profile", {}).get("user_profile", {})
    daily_goal = profile.get("daily_goal", {})

    goal = {
        "steps": total_steps >= daily_goal.get("steps", 10000),
        "mets": total_mets >= daily_goal.get("mets", 1000),
        "calories_burn": total_calories >= daily_goal.get("calories_burn", 2300),
        "AZM": activity["azm_total"] >= daily_goal.get("AZM", 30)
    }

    day_mod = int(date.split("-")[-1]) % 10
    activity_score = 80 + day_mod
    day_mod = int(date.split("-")[-1]) % 9
    hr_score = 90 + day_mod
    overall = int((activity_score + hr_score) / 2)

    system_score = {
        "activity_score": activity_score,
        "hr_score": hr_score,
        "overall": overall
    }

    return {
        "source": source,
        "device": device_id or f"{source}_device",
        "date": date,
        "steps": steps_block,
        "calories": calories_block,
        "mets": mets_block,
        "activity": activity,
        "heart_rate": heart_rate,
        "goal": goal,
        "system_score": system_score,
    }


def get_range_summary(
    store: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Retrieve aggregated metrics (max, min, mean) over a specified date range.
    Summarizes steps, calories, METs, activity intensity distribution, Active Zone Minutes,
    daily goal completion history, and a weighted system evaluation score.
    Only includes dates where data is available (based on source_assignment and marketplaces).
    Structure follows range_summary.json response_example.
    """
    # Generate date range
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    if start > end:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    # Collect daily summaries for all available dates
    daily_summaries: List[Dict[str, Any]] = []

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")

        # Check if data is available for this date
        is_available, _, _ = _check_data_availability(store, date_str)
        if is_available:
            daily_summary = get_daily_summary(store, date_str)
            daily_summaries.append(daily_summary)

        current_date += timedelta(days=1)

    days_count = len(daily_summaries)

    if days_count == 0:
        # Return empty structure if no data available
        return {
            "start_date": start_date,
            "end_date": end_date,
            "days_count": 0,
            "steps": {
                "total_sum": 0,
                "daily_avg": 0.0,
                "max_day": None,
                "min_day": None,
            },
            "calories": {
                "total_sum": 0,
                "daily_avg": 0.0,
                "max_day": None,
                "min_day": None,
            },
            "mets": {
                "total_sum": 0,
                "daily_avg": 0.0,
                "max_day": None,
                "min_day": None,
            },
            "activity": {
                "sedentary_percent_avg": 0.0,
                "lightly_active_percent_avg": 0.0,
                "fairly_active_percent_avg": 0.0,
                "very_active_percent_avg": 0.0,
                "azm_total_sum": 0,
            },
            "goal": {
                "steps": [],
                "mets": [],
                "calories_burn": [],
                "AZM": [],
            },
            "system_score": {
                "activity_score_avg": 0,
                "hr_score_avg": 0,
                "overall_avg": 0
            },
        }

    # Aggregate steps
    steps_values = [ds["steps"]["total"] for ds in daily_summaries]
    steps_total_sum = sum(steps_values)
    steps_daily_avg = steps_total_sum / days_count if days_count > 0 else 0.0

    steps_max_idx = max(range(len(steps_values)), key=lambda i: steps_values[i])
    steps_min_idx = min(range(len(steps_values)), key=lambda i: steps_values[i])

    steps_max_day = {
        "date": daily_summaries[steps_max_idx]["date"],
        "value": steps_values[steps_max_idx],
    }
    steps_min_day = {
        "date": daily_summaries[steps_min_idx]["date"],
        "value": steps_values[steps_min_idx],
    }

    # Aggregate calories
    calories_values = [ds["calories"]["total"] for ds in daily_summaries]
    calories_total_sum = sum(calories_values)
    calories_daily_avg = calories_total_sum / days_count if days_count > 0 else 0.0

    calories_max_idx = max(range(len(calories_values)), key=lambda i: calories_values[i])
    calories_min_idx = min(range(len(calories_values)), key=lambda i: calories_values[i])

    calories_max_day = {
        "date": daily_summaries[calories_max_idx]["date"],
        "value": calories_values[calories_max_idx],
    }
    calories_min_day = {
        "date": daily_summaries[calories_min_idx]["date"],
        "value": calories_values[calories_min_idx],
    }

    # Aggregate METs
    mets_values = [ds["mets"]["total"] for ds in daily_summaries]
    mets_total_sum = sum(mets_values)
    mets_daily_avg = mets_total_sum / days_count if days_count > 0 else 0.0

    mets_max_idx = max(range(len(mets_values)), key=lambda i: mets_values[i])
    mets_min_idx = min(range(len(mets_values)), key=lambda i: mets_values[i])

    mets_max_day = {
        "date": daily_summaries[mets_max_idx]["date"],
        "value": mets_values[mets_max_idx],
    }
    mets_min_day = {
        "date": daily_summaries[mets_min_idx]["date"],
        "value": mets_values[mets_min_idx],
    }

    # Aggregate activity percentages and AZM
    sedentary_percents = [ds["activity"]["sedentary_percent"] for ds in daily_summaries]
    lightly_percents = [ds["activity"]["lightly_active_percent"] for ds in daily_summaries]
    fairly_percents = [ds["activity"]["fairly_active_percent"] for ds in daily_summaries]
    very_percents = [ds["activity"]["very_active_percent"] for ds in daily_summaries]
    azm_totals = [ds["activity"]["azm_total"] for ds in daily_summaries]

    activity_agg = {
        "sedentary_percent_avg": round(sum(sedentary_percents) / days_count, 1) if days_count > 0 else 0.0,
        "lightly_active_percent_avg": round(sum(lightly_percents) / days_count, 1) if days_count > 0 else 0.0,
        "fairly_active_percent_avg": round(sum(fairly_percents) / days_count, 1) if days_count > 0 else 0.0,
        "very_active_percent_avg": round(sum(very_percents) / days_count, 1) if days_count > 0 else 0.0,
        "azm_total_sum": sum(azm_totals),
    }

    # Goal completion arrays (one boolean per day in order)
    goal_steps = [ds["goal"]["steps"] for ds in daily_summaries]
    goal_mets = [ds["goal"]["mets"] for ds in daily_summaries]
    goal_calories_burn = [ds["goal"]["calories_burn"] for ds in daily_summaries]
    goal_activity = [ds["goal"]["AZM"] for ds in daily_summaries]

    # System score averages
    activity_scores = [ds["system_score"]["activity_score"] for ds in daily_summaries]
    hr_scores = [ds["system_score"]["hr_score"] for ds in daily_summaries]
    overall_scores = [ds["system_score"]["overall"] for ds in daily_summaries]

    activity_score_avg = round(sum(activity_scores) / days_count) if days_count > 0 else 0
    hr_score_avg = round(sum(hr_scores) / days_count) if days_count > 0 else 0
    overall_avg = round(sum(overall_scores) / days_count) if days_count > 0 else 0

    return {
        "start_date": start_date,
        "end_date": end_date,
        "days_count": days_count,
        "steps": {
            "total_sum": steps_total_sum,
            "daily_avg": round(steps_daily_avg, 2),
            "max_day": steps_max_day,
            "min_day": steps_min_day,
        },
        "calories": {
            "total_sum": calories_total_sum,
            "daily_avg": round(calories_daily_avg, 2),
            "max_day": calories_max_day,
            "min_day": calories_min_day,
        },
        "mets": {
            "total_sum": mets_total_sum,
            "daily_avg": round(mets_daily_avg, 2),
            "max_day": mets_max_day,
            "min_day": mets_min_day,
        },
        "activity": activity_agg,
        "goal": {
            "steps": goal_steps,
            "mets": goal_mets,
            "calories_burn": goal_calories_burn,
            "AZM": goal_activity,
        },
        "system_score": {
            "activity_score_avg": activity_score_avg,
            "hr_score_avg": hr_score_avg,
            "overall_avg": overall_avg
        },
    }


def get_user_profile(
    store: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retrieve the user's demographic info, dietary_restrictions, daily goals,
    and long-term sport/food preferences.
    Returns exactly what's in debug_profile.user_profile.
    """
    profile = store.get("profile", {})
    user_profile = profile.get("user_profile", {})
    
    # Return user_profile exactly as stored (no mapping needed)
    return {
        "user_profile": {
            "basic_info": user_profile.get("basic_info", {}),
            "health_risks": user_profile.get("health_risks", []),
            "dietary_restrictions": user_profile.get("dietary_restrictions", []),
            "preferences": user_profile.get("preferences", {}),
            "daily_goal": user_profile.get("daily_goal", {}),
        }
    }


def update_profile(
    store: Dict[str, Any],
    basic_info: Optional[Dict[str, Any]] = None,
    health_risks: Optional[List[str]] = None,
    dietary_restrictions: Optional[List[str]] = None,
    preferences: Optional[Dict[str, Any]] = None,
    daily_goal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = store.get("profile", {})
    user_profile = profile.get("user_profile", {})

    updated = False  # 关键标记：是否真的修改了任何字段

    # basic_info（partial）
    if basic_info is not None:
        current = user_profile.get("basic_info", {})
        for k in ("name", "age", "gender"):
            if k in basic_info:
                current[k] = basic_info[k]
                updated = True
        user_profile["basic_info"] = current

    # health_risks（full replace）
    if health_risks is not None:
        user_profile["health_risks"] = health_risks
        updated = True

    # dietary_restrictions（full replace）
    if dietary_restrictions is not None:
        user_profile["dietary_restrictions"] = dietary_restrictions
        updated = True

    # preferences（partial）
    if preferences is not None:
        current = user_profile.get("preferences", {})
        for k in ("sport", "food"):
            if k in preferences:
                current[k] = preferences[k]
                updated = True
        user_profile["preferences"] = current

    # daily_goal（partial）
    if daily_goal is not None:
        current = user_profile.get("daily_goal", {})
        for k in ("steps", "calories_intake", "calories_burn", "AZM", "mets"):
            if k in daily_goal:
                current[k] = daily_goal[k]
                updated = True
        user_profile["daily_goal"] = current

    # 🚨 没有任何字段被命中 → 抛错
    if not updated:
        raise ValueError("No valid profile fields to update.")

    return {
        "updated_profile": {
            "basic_info": user_profile.get("basic_info", {}),
            "health_risks": user_profile.get("health_risks", []),
            "dietary_restrictions": user_profile.get("dietary_restrictions", []),
            "preferences": user_profile.get("preferences", {}),
            "daily_goal": user_profile.get("daily_goal", {}),
        }
    }


def get_system_settings(
    store: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retrieve platform-level device and source access information, permissions,
    notification settings, and upgraded features.
    Returns system_settings from debug_profile, mapping permissions to schema format.
    """
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    
    # Map permissions to schema format
    # debug_profile uses: allow_purchase, allow_med_assistant
    # schema expects: allow_purchase, allow_med_assistant
    mapped_permissions = {
        "allow_raw_data_access": permissions.get("allow_raw_data_access", False),
        "allow_user_notes_access": permissions.get("allow_user_notes_access", False),
        "allow_purchase": permissions.get("allow_purchase", False),
        "allow_med_assistant": permissions.get("allow_med_assistant", False),
    }
    
    # Return system_settings matching schema format
    return {
        "system_settings": {
            "marketplaces": system_settings.get("marketplaces", []),
            "devices": system_settings.get("devices", []),
            "permissions": mapped_permissions,
        }
    }


def get_source_features(
    store: Dict[str, Any],
    source_name: str,
    _tool_schema_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve the list of available agent tools for a given data source.
    Returns tool schemas from source_tools.json with source name prefix (e.g., fitbit.get_intraday_steps).
    Requires user authorization (allow_raw_data_access) and source existence.
    """
    # Check user authorization - verify allow_raw_data_access permission
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    allow_raw_data = permissions.get("allow_raw_data_access", False)
    
    if not allow_raw_data:
        return {
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "Raw data access permission is required to retrieve source tool features."
            }
        }
    
    # Check that source exists in marketplaces
    marketplaces = system_settings.get("marketplaces", [])
    source_exists = any(m.get("source") == source_name for m in marketplaces)
    if not source_exists:
        return {
            "error": {
                "code": "SOURCE_NOT_FOUND",
                "message": f"Source '{source_name}' is not configured or authorized."
            }
        }
    
    # Get the path to source_tools.json
    source_tools_path = TOOL_SCHEMAS_DIR / "source_tools.json"
    
    # Load all source tool schemas
    use_gemini = _tool_schema_format == "gemini"
    with open(source_tools_path, "r", encoding="utf-8") as f:
        all_source_schemas = json.load(f)
    
    # Add source prefix to function name
    result_schemas = []
    for schema in all_source_schemas:
        schema_copy = json.loads(json.dumps(schema))
        func = schema_copy.get("function", {})
        original_name = func.get("name") or schema_copy.get("name", "")
        prefixed_name = f"{source_name}-{original_name}"
        if func:
            func["name"] = prefixed_name
        else:
            schema_copy["name"] = prefixed_name
        result_schemas.append(schema_copy)

    # Gemini expects a list of function schemas only
    if use_gemini:
        result_schemas = [schema.get("function", {}) for schema in result_schemas]

    return {
        "source": source_name,
        "tools": result_schemas
    }


def get_med_features(
    store: Dict[str, Any],
    _tool_schema_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve the list of tools to retrieve information from health providers.
    Available to subscription users only.
    Requires VIP status and med_assistant permission.
    Returns all schemas from med_tools.json.
    """
    # Check VIP status
    if not _check_vip_status(store):
        return {
            "error": {
                "code": "SUBSCRIPTION_REQUIRED",
                "message": "VIP subscription is required to access medical assistant features."
            }
        }
    
    # Check med_assistant permission
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    allow_med = permissions.get("allow_med_assistant", False)
    
    if not allow_med:
        return {
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "Medical assistant permission is required to access medical features."
            }
        }
    
    # Get the path to med_tools.json
    med_tools_path = TOOL_SCHEMAS_DIR / "med_tools.json"
    
    # Load all medical tool schemas
    use_gemini = _tool_schema_format == "gemini"
    try:
        with open(med_tools_path, "r", encoding="utf-8") as f:
            med_schemas = json.load(f)
    except FileNotFoundError:
        return {
            "error": {
                "code": "SCHEMA_NOT_FOUND",
                "message": "Medical tools schema file not found."
            }
        }
    
    # Gemini expects a list of function schemas only
    if use_gemini:
        med_schemas = [schema.get("function", {}) for schema in med_schemas]

    # Return all medical tool schemas (no filtering needed)
    return {
        "tools": med_schemas
    }


def list_notes(
    store: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    List user-created notes within the given date range.
    Requires allow_user_notes_access permission.
    """
    # Check authorization
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    allow_notes = permissions.get("allow_user_notes_access", False)
    
    if not allow_notes:
        return {
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "User notes access permission is required to list notes."
            }
        }
    
    # Get notes from profile
    notes = profile.get("notes", [])
    
    # Filter by date range using the "time" field
    filtered_notes = _filter_records_by_date_range(notes, start_date, end_date, "time")
    
    return {
        "notes": filtered_notes
    }


def add_note(
    store: Dict[str, Any],
    note: str,
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Append a user note stamped with profile['now'].
    Requires allow_user_notes_access permission.
    """
    profile_root = store.get("profile", {})
    system_settings = profile_root.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    allow_notes = permissions.get("allow_user_notes_access", False)

    if not allow_notes:
        return {
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "User notes access permission is required to add notes."
            }
        }

    now_str = profile_root.get("now") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    notes = profile_root.get("notes", [])

    new_note = {
        "note_id": generate_note_id(_id_seed, _tool_call_name, _tool_call_index),
        "time": now_str,
        "content": note,
    }
    notes.append(new_note)
    profile_root["notes"] = notes

    return {"note": new_note}


def delete_note(
    store: Dict[str, Any],
    note_id: str,
) -> Dict[str, Any]:
    """
    Delete a user note by its note_id.
    Requires allow_user_notes_access permission.
    """
    profile_root = store.get("profile", {})
    system_settings = profile_root.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    allow_notes = permissions.get("allow_user_notes_access", False)

    if not allow_notes:
        return {
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "User notes access permission is required to delete notes."
            }
        }

    notes = profile_root.get("notes", [])
    for i, note in enumerate(notes):
        if note.get("note_id") == note_id:
            notes.pop(i)
            profile_root["notes"] = notes
            return {"deleted": True, "note_id": note_id}

    raise ValueError("Deletion failed!")


def plot_time_series(
    store: Dict[str, Any],
    start_time: str,
    granularity: str,
    time_series: List[float],
    unit: Optional[str] = None,
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Plot a time-series with explicit granularity and units.
    Returns an image URL (no actual plotting is performed).
    """
    # Validate granularity
    valid_granularities = ["minute", "hour", "day"]
    if granularity not in valid_granularities:
        raise ValueError("Invalid granularity!")
    # Validate time_series is not empty
    if not time_series or len(time_series) == 0:
        raise ValueError("Empty time series!")
    
    # Validate start_time format (ISO8601)
    try:
        datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError) as e:
        raise ValueError(
            "start_time must be in ISO8601 format (e.g., '2024-06-10T00:00:00').") from e

    # Generate a mock image URL
    # In a real implementation, this would be the URL to the generated plot
    # For now, we'll create a URL that represents the plot
    plot_id = generate_plot_id(_id_seed, _tool_call_name, _tool_call_index)
    image_url = f"https://api.example.com/plots/{plot_id}.png"
    
    return {
        "image_url": image_url,
        "plot_id": plot_id,
        "granularity": granularity,
        "unit": unit or "unknown",
        "data_points": len(time_series),
        "start_time": start_time,
    }


# ============================================================
# Tool registry for dispatcher
# ============================================================

def get_sport_records(
    store: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve sport records within a date range.
    Only includes records for dates where data source is available.
    """
    profile = store.get("profile", {})
    sports = profile.get("sports", [])
    
    # Filter by date range
    filtered_sports = _filter_records_by_date_range(sports, start_date, end_date, "start_time")
    
    # Filter by source availability: only include sports on dates where source is available
    available_records = []
    for sport in filtered_sports:
        start_time = sport.get("start_time", "")
        if start_time:
            try:
                sport_date = pd.to_datetime(start_time).strftime("%Y-%m-%d")
                is_available, _, _ = _check_data_availability(store, sport_date)
                if is_available:
                    available_records.append(sport)
            except (ValueError, TypeError):
                continue
    
    return available_records


def get_session_records(
    store: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve lifestyle session records (e.g., study, meditation, work) within a date range.
    """
    profile = store.get("profile", {})
    sessions = profile.get("sessions", [])
    
    # Filter by date range using start_time
    filtered_sessions = _filter_records_by_date_range(sessions, start_date, end_date, "start_time")
    
    return filtered_sessions


def get_meal_records(
    store: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve meal records within a date range.
    Returns detailed meal records including food items.
    """
    profile = store.get("profile", {})
    meals = profile.get("meals", [])
    
    # Filter by date range using timestamp
    filtered_meals = _filter_records_by_date_range(meals, start_date, end_date, "time")
    out = [
        {
            "record_id": m["record_id"],
            "time": m["time"],
            "meal_type": m["meal_type"],
            "items": m["items"],
            "total_calories": m["nutrition"]["calories"],
        }
        for m in filtered_meals
    ]
    return out

def analysis_meal(
    store: Dict[str, Any],
    record_id: str,
) -> Dict[str, Any]:
    """
    Analyze a specific meal record and return its numeric nutrition values.
    """
    profile = store.get("profile", {})
    meals = profile.get("meals", [])

    # Find the meal by record_id
    meal = next((m for m in meals if m.get("record_id") == record_id), None)
    if meal is None:
        raise ValueError(f"Meal record not found: {record_id}")

    nutrition = meal.get("nutrition", {})

    return {
        "record_id": meal["record_id"],
        "time": meal["time"],
        "meal_type": meal["meal_type"],
        "nutrition": {
            "calories": nutrition.get("calories", 0.0),
            "sugar_g": nutrition.get("sugar_g", 0.0),
            "fiber_g": nutrition.get("fiber_g", 0.0),
            "fat_g": nutrition.get("fat_g", 0.0),
        },
    }



def create_session_record(
    store: Dict[str, Any],
    start_time: str,
    end_time: str,
    session_type: str,
    user_note: Optional[str] = None,
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a lifestyle session record such as study, meditation, work, or other duration-based activities.
    Modifies the store's profile in-place.
    """

    try:
        datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError) as e:
        raise ValueError(
            "start_time must be in ISO8601 format (e.g., '2024-06-10T00:00:00').") from e

    try:
        datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError) as e:
        raise ValueError(
            "end_time must be in ISO8601 format (e.g., '2024-06-10T00:00:00').") from e


    profile = store.get("profile", {})
    sessions = profile.get("sessions", [])
    
    # Generate unique record_id
    record_id = generate_session_id(_id_seed, _tool_call_name, _tool_call_index)
    
    new_session = {
        "record_id": record_id,
        "session_type": session_type,  # Note: config has typo "session_type", but we use correct spelling
        "start_time": start_time,
        "end_time": end_time,
        "user_note": user_note or "",
    }
    
    sessions.append(new_session)
    profile["sessions"] = sessions
    
    return new_session


def create_meal_record(
    store: Dict[str, Any],
    timestamp: str,
    items: List[Dict[str, Any]],
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a meal record with backend-matched atom_meal nutrition.
    """
    # -------- validate timestamp --------
    try:
        parsed_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError) as e:
        raise ValueError(
            "timestamp must be in ISO8601 format (e.g., '2024-06-10T00:00:00')."
        ) from e

    # -------- validate and normalize items --------
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty list.")

    normalized_items: List[Dict[str, Any]] = []
    total_input_grams = 0.0
    grams_only = True
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid item at index {idx}: must be an object.")
        name = item.get("name")
        amount = item.get("amount")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid item name at index {idx}.")
        if not isinstance(amount, dict):
            raise ValueError(f"Invalid amount at index {idx}.")
        value = amount.get("value")
        unit = amount.get("unit")
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Invalid amount value at index {idx}.")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Invalid amount unit at index {idx}.")

        unit_lower = unit.strip().lower()
        if unit_lower in {"g", "gram", "grams"}:
            total_input_grams += float(value)
            normalized_items.append(
                {
                    "name": name.strip(),
                    "amount_g": int(round(float(value))),
                }
            )
        else:
            grams_only = False
            normalized_items.append(
                {
                    "name": name.strip(),
                    "amount": {"value": float(value), "unit": unit.strip()},
                }
            )

    profile = store.get("profile", {})
    meals = profile.get("meals", [])

    # -------- sample an atom_meal and scale nutrition --------
    def _infer_meal_type(dt: datetime) -> str:
        hour = dt.hour
        if 5 <= hour < 11:
            return "breakfast"
        if 11 <= hour < 16:
            return "lunch_dinner"
        if 16 <= hour < 21:
            return "lunch_dinner"
        return "snack"

    meal_type = _infer_meal_type(parsed_time)
    candidates = [m for m in ATOM_MEALS if m.get("meal_type") == meal_type]
    if not candidates:
        candidates = ATOM_MEALS

    rng = random.Random(_id_seed)
    atom = rng.choice(candidates)

    scale = 1.0
    if grams_only:
        atom_total = sum(float(it.get("amount_g", 0) or 0) for it in atom.get("items", []))
        if atom_total > 0 and total_input_grams > 0:
            scale = total_input_grams / atom_total
        # keep a reasonable range
        scale = max(0.5, min(scale, 1.8))

    scale *= rng.uniform(0.9, 1.1)
    nutrition = {
        k: round(float(v) * scale, 1)
        for k, v in atom.get("nutrition", {}).items()
    }

    # -------- create record --------
    record_id = generate_meal_id(_id_seed, _tool_call_name, _tool_call_index)

    new_meal = {
        "record_id": record_id,
        "time": timestamp,
        "meal_type": atom.get("meal_type", "snack"),
        "meal_id": atom.get("meal_id", "unknown_meal"),  # backend-only anchor
        "items": normalized_items,                       # user-visible
        "nutrition": nutrition,                          # truth
    }

    meals.append(new_meal)
    profile["meals"] = meals

    return {k: v for k, v in new_meal.items() if k != "nutrition"}


def delete_record(
    store: Dict[str, Any],
    record_id: str,
) -> Dict[str, Any]:
    """
    Delete a user record by its unique record_id.
    Can delete session and meal records, but NOT sport records.
    """
    profile = store.get("profile", {})

    # Search in sessions
    sessions = profile.get("sessions", [])
    for i, session in enumerate(sessions):
        if session.get("record_id") == record_id:
            deleted = sessions.pop(i)
            profile["sessions"] = sessions
            return {"deleted": True, "record_id": record_id, "type": "session"}
    
    # Search in meals
    meals = profile.get("meals", [])
    for i, meal in enumerate(meals):
        if meal.get("record_id") == record_id:
            deleted = meals.pop(i)
            profile["meals"] = meals
            return {"deleted": True, "record_id": record_id, "type": "meal"}
    
    # Record not found
    raise ValueError("Deletion failed!")


def recommend_sports(
        store: Dict[str, Any],
        recent_sports_records: List[Dict[str, Any]],
        preference: str,
        intensity: str,
) -> Dict[str, Any]:
    """
    Recommend sports activities based on past sports records, user preferences, and desired intensity.
    Subscription/VIP required.
    """
    # Check VIP status
    if not _check_vip_status(store):
        return {
            "error": {
                "code": "SUBSCRIPTION_REQUIRED",
                "message": "Subscription required to call this tool. Please upgrade to VIP to access personalized sports recommendations."
            }
        }

    # -----------------------------
    # Fake analysis from history
    # -----------------------------
    avg_calories = sum(r.get("calories", 0) for r in recent_sports_records) / len(recent_sports_records)
    avg_azm = sum(r.get("azm", 0) for r in recent_sports_records) / len(recent_sports_records)

    # Simple intensity scaling
    intensity_scale = {
        "low": 0.7,
        "medium": 1.0,
        "high": 1.3,
    }.get(intensity, 1.0)

    recommendations = []

    # -----------------------------
    # Preference-based recommendation
    # -----------------------------
    if preference == "cardio":
        sport_type = "cycling" if intensity != "high" else "running"
        duration = 40 if intensity != "low" else 50
        reason = "Based on your general cardio preference and recent activity level."

    elif preference == "endurance":
        sport_type = "steady_run"
        duration = 55
        reason = "Your recent activities suggest you can sustain longer-duration workouts."

    elif preference == "speed_training":
        sport_type = "interval_training"
        duration = 25
        reason = "Speed-focused training helps improve performance and matches your preference."

    elif preference == "strength":
        sport_type = "weight_training"
        duration = 45
        reason = "Strength training complements your recent exercise pattern."

    elif preference == "recovery":
        sport_type = "walking"
        duration = 35
        reason = "A recovery-focused session helps maintain activity while allowing rest."

    else:
        # Fallback (should rarely happen)
        sport_type = "mixed_cardio"
        duration = 40
        reason = "A balanced workout fits your recent activity profile."

    recommendations.append({
        "sport_type": sport_type,
        "intensity": intensity,
        "expected_duration_minutes": duration,
        "estimated_calories": int(avg_calories * intensity_scale),
        "estimated_azm": int(avg_azm * intensity_scale),
        "reason": reason
    })

    return {"recommendations": recommendations}


def recommend_health_food(
    store: Dict[str, Any],
    preference: str,
) -> Dict[str, Any]:
    """
    Recommend foods based on dietary preference only (ignoring recent_meals).
    Subscription/VIP required.
    """
    # Check VIP status
    if not _check_vip_status(store):
        return {
            "error": {
                "code": "SUBSCRIPTION_REQUIRED",
                "message": "Subscription required to call this tool. Please upgrade to VIP to access personalized food recommendations."
            }
        }
    
    # -----------------------------
    # Load product catalog (once)
    # -----------------------------
    product_path = DATA_DIR / "product_details.json"
    try:
        with open(product_path, "r", encoding="utf-8") as f:
            product_details = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("product_details.json not found at expected path.")

    products = list(product_details.values()) if isinstance(product_details, dict) else product_details

    # Map preference to tag + metric key/unit
    pref_map = {
        "high_protein": ("high_protein", "protein_g", "g"),
        "low_sugar": ("low_sugar", "sugar_g", "g"),
        "low_fat": ("low_fat", "fat_g", "g"),
        "high_fiber": ("high_fiber", "fiber_g", "g"),
        "low_calorie": ("low_calorie", "calories", "kcal"),
        "diabetic_safe": ("diabetic_safe", "sugar_g", "g"),
        "low_sodium": ("low_sodium", "sodium_mg", "mg"),
    }

    tag, metric_key, metric_unit = pref_map.get(preference,())

    # Filter products by tag
    tagged = [p for p in products if tag in p.get("tags", [])]
    pool = tagged if tagged else products

    def _clean_name(name: str) -> str:
        # not used for now for stability
        for brand in ("EverydayBite", "FamilyPack", "PureBalance", "EliteVital"):
            name = name.replace(brand, "").strip()
        return name or "Recommended item"

    def _metric_value(p: Dict[str, Any]) -> float:
        nutrition = p.get("packaging", {}).get("nutrition_per_serving", {})
        return nutrition.get(metric_key, 0)

    scenes = ["breakfast", "lunch", "dinner", "snack"]
    periods = ["for the next 3 days", "for the next 5 days", "for the next 7 days"]

    recommendations: List[Dict[str, Any]] = []
    sample_count = min(1, len(pool))
    for product in random.sample(pool, sample_count):
        name = product.get("name", "")
        metric_val = _metric_value(product)
        rounded_val = round(float(metric_val or 0), 1)
        scenario = f"{random.choice(scenes)} {random.choice(periods)}"


        recommendations.append({
            "item": name,
            "scenario": scenario,
            "trait": {
                "preference": preference,
                "metric": metric_key,
                "value": rounded_val,
                "unit": metric_unit,
            },
            "reason": f"{metric_key} ≈ {rounded_val}{metric_unit} per serving.",
        })

    return {"recommendations": recommendations}


def list_daily_reminders(
    store: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Return all daily reminders created by the user.
    """
    profile = store.get("profile", {})
    reminders = profile.get("reminders", [])
    
    return reminders


def create_daily_reminder(
    store: Dict[str, Any],
    title: str,
    time_of_day: str,
    repeat_days: List[str],
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a daily or weekly repeating reminder.
    Modifies the store's profile in-place.
    If repeat_days is empty/null, the reminder repeats daily.
    """
    profile = store.get("profile", {})
    reminders = profile.get("reminders", [])
    
    # Generate unique reminder_id
    reminder_id = generate_reminder_id(_id_seed, _tool_call_name, _tool_call_index)
    
    # If repeat_days is empty, it means repeat daily (all days)
    if not repeat_days:
        repeat_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    new_reminder = {
        "reminder_id": reminder_id,
        "title": title,
        "time_of_day": time_of_day,
        "repeat_days": repeat_days,
    }
    
    reminders.append(new_reminder)
    profile["reminders"] = reminders
    
    return new_reminder


def delete_daily_reminder(
    store: Dict[str, Any],
    reminder_id: str,
) -> Dict[str, Any]:
    """
    Delete a daily reminder by its ID.
    """
    profile = store.get("profile", {})
    reminders = profile.get("reminders", [])
    
    for i, reminder in enumerate(reminders):
        if reminder.get("reminder_id") == reminder_id:
            deleted = reminders.pop(i)
            profile["reminders"] = reminders
            return {"deleted": True, "reminder_id": reminder_id}

    raise ValueError("Deletion failed!")


def get_daily_sleep_summary(
    store: Dict[str, Any],
    date: str,
) -> Dict[str, Any]:
    """
    Return the info and summary of sleep segments in the designated date input (8:00 am to 8:00 am next day).
    Retrieves sleep data from daily_sleep table in the store.
    """
    sleep_data = _get_daily_sleep(store, date)

    if sleep_data is None or "date" not in sleep_data.index:
        # Return empty structure if no sleep data available
        return {
            "date": date,
            "total_sleep_minutes": 0,
            "total_sleep_hours": 0.0,
            "total_light_minutes": 0,
            "total_deep_minutes": 0,
            "total_rem_minutes": 0,
            "segment_count": 0,
            "segments": [],
        }

    # Extract basic metrics
    total_sleep_minutes = int(sleep_data["total_sleep_minutes"]) if pd.notna(sleep_data["total_sleep_minutes"]) else 0
    total_sleep_hours = float(sleep_data["total_sleep_hours"]) if pd.notna(sleep_data["total_sleep_hours"]) else 0.0
    total_light_minutes = int(sleep_data["total_light_minutes"]) if pd.notna(sleep_data["total_light_minutes"]) else 0
    total_deep_minutes = int(sleep_data["total_deep_minutes"]) if pd.notna(sleep_data["total_deep_minutes"]) else 0
    total_rem_minutes = int(sleep_data["total_rem_minutes"]) if pd.notna(sleep_data["total_rem_minutes"]) else 0
    segment_count = int(sleep_data["segment_count"]) if pd.notna(sleep_data["segment_count"]) else 0

    # Parse segments_json
    segments = []
    if "segments_json" in sleep_data.index and pd.notna(sleep_data["segments_json"]):
        try:
            segments_json_str = sleep_data["segments_json"]
            # The JSON string might have double quotes escaped, so we need to parse it
            segments = json.loads(segments_json_str)
        except (json.JSONDecodeError, TypeError):
            segments = []

    # Format the date from the sleep data (use the date from the data, not the input)
    sleep_date = sleep_data["date"]
    if isinstance(sleep_date, pd.Timestamp):
        formatted_date = sleep_date.strftime("%Y-%m-%d")
    else:
        formatted_date = pd.to_datetime(sleep_date).strftime("%Y-%m-%d")

    return {
        "date": formatted_date,
        "total_sleep_minutes": total_sleep_minutes,
        "total_sleep_hours": round(total_sleep_hours, 3),
        "total_light_minutes": total_light_minutes,
        "total_deep_minutes": total_deep_minutes,
        "total_rem_minutes": total_rem_minutes,
        "segment_count": segment_count,
        "segments": segments,
    }


registered_platform_tools = {
    "get_daily_summary": get_daily_summary,
    "get_range_summary": get_range_summary,
    "get_hourly_mets": get_hourly_mets,
    "get_hourly_steps": get_hourly_steps,
    "get_hourly_calories": get_hourly_calories,
    "get_hourly_activity": get_hourly_activity,
    "get_user_profile": get_user_profile,
    "update_profile": update_profile,
    "get_system_settings": get_system_settings,
    "get_sport_records": get_sport_records,
    "get_session_records": get_session_records,
    "get_meal_records": get_meal_records,
    "analysis_meal": analysis_meal,
    "create_session_record": create_session_record,
    "create_meal_record": create_meal_record,
    "delete_record": delete_record,
    "recommend_sports": recommend_sports,
    # "recommend_health_food": recommend_health_food,
    "list_daily_reminders": list_daily_reminders,
    "create_daily_reminder": create_daily_reminder,
    "delete_daily_reminder": delete_daily_reminder,
    "get_source_features": get_source_features,
    "get_med_features": get_med_features,
    "list_notes": list_notes,
    "add_note": add_note,
    "delete_note": delete_note,
    "plot_time_series": plot_time_series,
    "get_daily_sleep_summary": get_daily_sleep_summary,
}
