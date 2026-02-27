# bench/backend/tools/med_tools.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime

from bench.utils.generate_ids import (
    generate_appointment_id,
    generate_transaction_id,
)


# ============================================================
# Helper functions
# ============================================================

def _check_med_authorization(store: Dict[str, Any]):
    """
    Check if user has VIP status and med_assistant permission.
    Returns error dict if not authocreaterized, None if authorized.
    """
    profile = store.get("profile", {})

    # Check VIP status
    shopping = profile.get("shopping", {})
    wallet = shopping.get("wallet", {})
    is_vip = wallet.get("vip", False)

    if not is_vip:
        return {
            "error": {
                "code": "SUBSCRIPTION_REQUIRED",
                "message": "VIP subscription is required to access medical assistant features."
            }
        }

    # Check med_assistant permission
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

    return None


def _get_health_providers(store: Dict[str, Any]) -> List[Dict[str, Any]]:
    profile = store.get("profile", {})
    return profile.get("health_providers", []) or []


# ============================================================
# Medical tool implementations
# ============================================================

def med_get_user_profile(
        store: Dict[str, Any],
) -> Dict[str, Any]:
    """Get the user's medical profile information."""
    auth_error = _check_med_authorization(store)
    if auth_error:
        return auth_error

    profile = store.get("profile", {})
    med_user_profile = profile.get("med_user_profile", {})

    return {
        "user_profile": med_user_profile
    }


def med_get_provider_list(
        store: Dict[str, Any],
) -> Dict[str, Any]:
    """Get the list of available healthcare providers."""
    auth_error = _check_med_authorization(store)
    if auth_error:
        return auth_error

    providers = _get_health_providers(store)

    # 不做任何“聪明”的格式推断，直接返回
    return {
        "providers": [
            {
                "provider_id": p["provider_id"],
                "doctor": p.get("doctor"),
                "clinic": p.get("clinic"),
                "address": p.get("address"),
            }
            for p in providers
        ]
    }


def med_get_resources(
        store: Dict[str, Any],
        resource_type: str,
) -> Dict[str, Any]:
    """
    Get user-scoped medical resources such as appointments or care plans.
    Returns resources in the exact format from configuration.
    """
    auth_error = _check_med_authorization(store)
    if auth_error:
        return auth_error

    profile = store.get("profile", {})

    if resource_type == "appointments":
        # Return appointments in exact format from configuration
        appointments = profile.get("appointments", [])
        # Return a copy to avoid modifying the original
        return {
            "resource_type": "appointments",
            "resources": [dict(apt) for apt in appointments]
        }
    elif resource_type == "care_plans":
        # Return care_plans in exact format from configuration
        care_plans = profile.get("care_plans", [])
        # Return a copy to avoid modifying the original
        return {
            "resource_type": "care_plans",
            "resources": [dict(plan) for plan in care_plans]
        }
    else:
        raise ValueError("Unknown resource type")


def med_create_appointment(
        store: Dict[str, Any],
        provider_id: str,
        date: str,
        time: str,
        duration: int,
        reason: str,
        _tool_call_name: Optional[str] = None,
        _tool_call_index: Optional[int] = None,
        _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a new appointment."""
    auth_error = _check_med_authorization(store)
    if auth_error:
        return auth_error

    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format.")

    # Validate time format
    try:
        datetime.strptime(time, "%H:%M")
    except ValueError:
        raise ValueError("Invalid time format.")

    # Validate duration
    if duration <= 0:
        raise ValueError("Duration must be positive.")

    # Validate provider_id
    providers = _get_health_providers(store)
    provider_ids = {p["provider_id"] for p in providers}

    if provider_id not in provider_ids:
        raise ValueError("Provider not found!")

    profile = store.get("profile", {})
    appointments = profile.get("appointments", [])

    # Generate appointment ID
    appointment_id = generate_appointment_id(_id_seed, _tool_call_name, _tool_call_index)
    transaction_id = generate_transaction_id(_id_seed, _tool_call_name, _tool_call_index)

    # Create new appointment in exact format from configuration
    # Format: appointment_id, provider_id, date, time, duration, reason, status, transaction_id, fee
    fee = 5.0
    new_appointment = {
        "appointment_id": appointment_id,
        "provider_id": provider_id,
        "date": date,
        "time": time,
        "duration": duration,
        "reason": reason,
        "status": "pending",  # New appointments start as pending
        "transaction_id": transaction_id,
        "fee": fee,
    }

    appointments.append(new_appointment)

    return {
        "appointment": new_appointment
    }


def med_cancel_appointment(
        store: Dict[str, Any],
        appointment_id: str,
) -> Dict[str, Any]:
    """Cancel an existing appointment."""
    auth_error = _check_med_authorization(store)
    if auth_error:
        return auth_error

    profile = store.get("profile", {})
    appointments = profile.get("appointments", [])

    # Find and update the appointment
    found = False
    for apt in appointments:
        if apt.get("appointment_id") == appointment_id:
            apt["status"] = "cancelled"
            found = True
            break

    if not found:
        raise ValueError("Appointment not found!")

    return {
        "appointment_id": appointment_id,
        "status": "cancelled",
        "message": f"Appointment '{appointment_id}' has been cancelled."
    }


def med_update_user_profile(
        store: Dict[str, Any],
        basic_info: Optional[Dict[str, Any]] = None,
        health_risks: Optional[List[str]] = None,
        dietary_restrictions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Update the user's medical profile.
    Only the fields provided in the request will be modified.
    """
    auth_error = _check_med_authorization(store)
    if auth_error:
        return auth_error

    profile = store.get("profile", {})
    med_user_profile = profile.get("med_user_profile", {})

    # Update basic_info if provided
    if basic_info is not None:
        current_basic_info = med_user_profile.get("basic_info", {})
        # Only update fields that are provided
        if "name" in basic_info:
            current_basic_info["name"] = basic_info["name"]
        if "age" in basic_info:
            current_basic_info["age"] = basic_info["age"]
        if "gender" in basic_info:
            current_basic_info["gender"] = basic_info["gender"]
        med_user_profile["basic_info"] = current_basic_info

    # Update health_risks if provided (fully replaces existing)
    if health_risks is not None:
        med_user_profile["health_risks"] = health_risks

    # Update dietary_restrictions if provided (fully replaces existing)
    if dietary_restrictions is not None:
        med_user_profile["dietary_restrictions"] = dietary_restrictions

    return {
        "user_profile": med_user_profile
    }


# ============================================================
# Tool registry for dispatcher
# ============================================================

registered_med_tools = {
    "med-get_user_profile": med_get_user_profile,
    "med-get_provider_list": med_get_provider_list,
    "med-get_resources": med_get_resources,
    "med-create_appointment": med_create_appointment,
    "med-cancel_appointment": med_cancel_appointment,
    "med-update_user_profile": med_update_user_profile,
}

