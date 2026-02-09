from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import pandas as pd

# Import helper from platform_tools
from .platform_tools import _get_minute_df


# ============================================================
# Helper functions for source tools
# ============================================================

def _check_raw_data_permission(store: Dict[str, Any]) -> bool:
    """Check if user has permission to access raw data (source tools)."""
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    return permissions.get("allow_raw_data_access", False)

def _filter_minute_data_by_time_window(
    df: pd.DataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter minute-level dataframe by time window (HH:mm format).
    If start_time or end_time is None, no filtering is applied for that boundary.
    """
    if df.empty:
        return df

    df = df.copy()
    df["ActivityMinute"] = pd.to_datetime(df["ActivityMinute"], errors="coerce")
    valid_times = df["ActivityMinute"].dropna()
    if valid_times.empty:
        return df.iloc[0:0]

    if start_time:
        try:
            h, m = map(int, start_time.split(":"))
            base_dt = valid_times.iloc[0].normalize()
            start_dt = base_dt + pd.Timedelta(hours=h, minutes=m)
            df = df[df["ActivityMinute"] >= start_dt]
        except ValueError:
            raise ValueError("Start time must be in HH:mm format")

    if end_time:
        try:
            h, m = map(int, end_time.split(":"))
            base_dt = valid_times.iloc[0].normalize()
            end_dt = base_dt + pd.Timedelta(hours=h, minutes=m)
            df = df[df["ActivityMinute"] <= end_dt]
        except ValueError:
            raise ValueError("End time must be in HH:mm format")

    return df


def _get_source_profile(store: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    """Get the source profile for a specific source from source_profile array."""
    profile = store.get("profile", {})
    source_profiles = profile.get("source_profile", [])
    for sp in source_profiles:
        if sp.get("source") == source:
            return sp
    return None


def _update_source_profile(store: Dict[str, Any], source: str, source_profile: Dict[str, Any]) -> None:
    """
    Update source profile for a specific source.
    Replaces the entire source_profile entry with the provided one.
    """
    profile = store.get("profile", {})
    source_profiles = profile.get("source_profile", [])

    found_index = None
    for i, sp in enumerate(source_profiles):
        if sp.get("source") == source:
            found_index = i
            break

    if found_index is not None:
        source_profiles[found_index] = source_profile
    else:
        source_profiles.append(source_profile)

    profile["source_profile"] = source_profiles


# ============================================================
# Generic intraday helper
# ============================================================

def _get_intraday_metric(
    store: Dict[str, Any],
    date: str,
    source_name: str,
    dataset_key: str,
    value_column: str,
    cast_fn: Callable,
):
    # Check permission
    if not _check_raw_data_permission(store):
        return {
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "Raw data access permission is required to use source tools."
            }
        }

    profile = store.get("profile", {})
    source_assignment = profile.get("source_assignment", {})
    assigned_source = source_assignment.get(date)

    # If missing or mismatch → treat as nonexistent data
    if assigned_source != source_name:
        return {
            "error": {
                "code": "DATA_NOT_AVAILABLE",
                "message": f"No data available for date {date}."
            }
        }

    # Check if source is connected
    system_settings = profile.get("system_settings", {})
    marketplaces = system_settings.get("marketplaces", [])
    source_connected = any(
        m.get("source") == source_name and m.get("connected", False)
        for m in marketplaces
    )
    if not source_connected:
        return {
            "error": {
                "code": "DATA_NOT_AVAILABLE",
                "message": f"No data available for date {date}."
            }
        }

    # Load dataframe safely
    try:
        df = _get_minute_df(store, dataset_key, date)
    except Exception:
        # Any internal loading issue → treat as missing data
        raise ValueError("Backend broken!")


    return df, source_name


def _build_intraday_response(
    df: pd.DataFrame,
    source: str,
    date: str,
    value_column: str,
    cast_fn: Callable,
    start: Optional[str],
    end: Optional[str],
) -> Dict[str, Any]:
    """Apply time-window filter and format minute-level data."""
    df = _filter_minute_data_by_time_window(df, start, end)

    data_points = [
        {
            "time": row["ActivityMinute"].strftime("%H:%M:%S"),
            "value": cast_fn(row[value_column])
        }
        for _, row in df.iterrows()
    ]

    return {
        "date": date,
        "source": source,
        "data": data_points
    }


# ============================================================
# Source tool implementations (fitbit tools from source_tools_1.json)
# ============================================================

def get_intraday_intensities(
    store: Dict[str, Any],
    date: str,
    source_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    result = _get_intraday_metric(store, date, source_name, "minute_intensities", "Intensity", int)
    if isinstance(result, dict) and "error" in result:
        return result
    df, source = result
    return _build_intraday_response(df, source, date, "Intensity", int, start, end)


def get_intraday_mets(
    store: Dict[str, Any],
    date: str,
    source_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    result = _get_intraday_metric(store, date, source_name, "minute_mets", "METs", float)
    if isinstance(result, dict) and "error" in result:
        return result
    df, source = result
    return _build_intraday_response(df, source, date, "METs", float, start, end)


def get_intraday_calories(
    store: Dict[str, Any],
    date: str,
    source_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    result = _get_intraday_metric(store, date, source_name, "minute_calories", "Calories", float)
    if isinstance(result, dict) and "error" in result:
        return result
    df, source = result
    return _build_intraday_response(df, source, date, "Calories", float, start, end)


def get_intraday_steps(
    store: Dict[str, Any],
    date: str,
    source_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    result = _get_intraday_metric(store, date, source_name, "minute_steps", "Steps", int)
    if isinstance(result, dict) and "error" in result:
        return result
    df, source = result
    return _build_intraday_response(df, source, date, "Steps", int, start, end)


# ============================================================
# Activity goals & plans
# ============================================================

def create_activity_goal(
    store: Dict[str, Any],
    period: str,
    goal_type: str,
    value: int,
    source_name: str,
) -> Dict[str, Any]:
    """
    Creates or updates a user's daily or weekly activity goal for a specific source.
    Steps unit: count; AZM unit: minutes; calories_burn unit: kilocalories.
    Updates source_profile for the specified source.
    Goal types match profile field names directly: steps, AZM, calories_burn.
    """
    profile = store.get("profile", {})
    source = source_name

    system_settings = profile.get("system_settings", {})
    marketplaces = system_settings.get("marketplaces", [])
    source_exists = any(m.get("source") == source for m in marketplaces)
    if not source_exists:
        return {
            "error": {
                "code": "SOURCE_NOT_FOUND",
                "message": f"Source '{source}' is not configured in marketplaces."
            }
        }

    # Get or create source profile
    source_profile = _get_source_profile(store, source)
    if not source_profile:
        user_profile = profile.get("user_profile", {})
        source_profile = {
            "source": source,
            "basic_info": user_profile.get("basic_info", {}).copy(),
            "daily_goal": {},
            "weekly_goal": {},
        }
        _update_source_profile(store, source, source_profile)

    valid_goal_types = ["steps", "AZM", "mets", "calories_burn"]
    if goal_type not in valid_goal_types:
        return {
            "error": {
                "code": "INVALID_GOAL_TYPE",
                "message": f"Unknown goal_type: {goal_type}. Must be one of {valid_goal_types}"
            }
        }

    if period == "daily":
        goal_dict = source_profile.get("daily_goal", {})
        goal_dict[goal_type] = value
        source_profile["daily_goal"] = goal_dict
    elif period == "weekly":
        goal_dict = source_profile.get("weekly_goal", {})
        goal_dict[goal_type] = value
        source_profile["weekly_goal"] = goal_dict
    else:
        return {
            "error": {
                "code": "INVALID_PERIOD",
                "message": f"Period must be 'daily' or 'weekly', got: {period}"
            }
        }

    _update_source_profile(store, source, source_profile)

    return {
        "period": period,
        "goal_type": goal_type,
        "value": value,
        "source": source
    }


def get_activity_goal(
    store: Dict[str, Any],
    period: str,
    source_name: str,
) -> Dict[str, Any]:
    """
    Retrieves a user's daily or weekly activity goal for a specific source.
    Steps unit: count; AZM unit: minutes; calories_burn unit: kilocalories.
    Returns goals from source_profile for the specified source.
    Goal types match profile field names directly: steps, AZM, calories_burn.
    """
    source = source_name

    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    marketplaces = system_settings.get("marketplaces", [])
    source_exists = any(m.get("source") == source for m in marketplaces)
    if not source_exists:
        raise ValueError("Wrong source name!")

    source_profile = _get_source_profile(store, source)

    if period == "daily":
        goals = source_profile.get("daily_goal", {})
    elif period == "weekly":
        goals = source_profile.get("weekly_goal", {})
    else:
        return {
            "error": {
                "code": "INVALID_PERIOD",
                "message": f"Period must be 'daily' or 'weekly', got: {period}"
            }
        }

    return {
        "period": period,
        "goals": goals,
        "source": source
    }


# ============================================================
# Tool registry for dispatcher
# ============================================================

def update_activity_target(
    store: Dict[str, Any],
    goal_type: str,
    value: int,
    source_name: str,
) -> Dict[str, Any]:
    """Daily convenience wrapper for create_activity_goal."""
    return create_activity_goal(store, period="daily", goal_type=goal_type, value=value, source_name=source_name)


def get_activity_target(
    store: Dict[str, Any],
    source_name: str,
) -> Dict[str, Any]:
    """Daily convenience wrapper for get_activity_goal."""
    result = get_activity_goal(store, period="daily", source_name=source_name)
    if "error" in result:
        return result
    return {
        "goals": result.get("goals", {}),
        "source": result.get("source")
    }


def create_activity_plan(
    store: Dict[str, Any],
    goal_type: str,
    value: int,
    source_name: str,
) -> Dict[str, Any]:
    """Weekly convenience wrapper for create_activity_goal."""
    return create_activity_goal(store, period="weekly", goal_type=goal_type, value=value, source_name=source_name)


def get_activity_plan(
    store: Dict[str, Any],
    source_name: str,
) -> Dict[str, Any]:
    """Weekly convenience wrapper for get_activity_goal."""
    result = get_activity_goal(store, period="weekly", source_name=source_name)
    if "error" in result:
        return result
    return {
        "goals": result.get("goals", {}),
        "source": result.get("source")
    }


registered_source_tools = {
    "get_intraday_intensities": get_intraday_intensities,
    "get_intraday_mets": get_intraday_mets,
    "get_intraday_calories": get_intraday_calories,
    "get_intraday_steps": get_intraday_steps,
    "create_activity_plan": create_activity_plan,
    "get_activity_plan": get_activity_plan,
}
