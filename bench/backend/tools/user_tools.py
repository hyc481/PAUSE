# bench/backend/tools/user_tools.py

from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from bench.backend.tools.shopping_tools import _recompute_order_for_current_state

# ============================================================
# Helper functions
# ============================================================

def _find_transaction(store: Dict[str, Any], transaction_id: str) -> Dict[str, Any] | None:
    profile = store.get("profile", {})

    # ---- shopping transactions ----
    shopping = profile.get("shopping", {})
    for tx in shopping.get("transactions", []):
        if tx.get("transaction_id") == transaction_id:
            return tx

    # ---- medical appointments ----
    for apt in profile.get("appointments", []):
        if apt.get("transaction_id") == transaction_id:
            return apt

    return None



# ============================================================
# User tool implementations
# ============================================================

def update_source(
    store: Dict[str, Any],
    source_name: str,
    allow: bool,
) -> Dict[str, Any]:
    """
    Update or create a data source connection.

    Semantics:
    - If source exists:
        - allow=True  -> set connected=True
        - allow=False -> set connected=False
    - If source does NOT exist:
        - allow=True  -> create source + create device + connected=True
        - allow=False -> error
    """

    profile = store.get("profile", {})
    system_settings = profile.setdefault("system_settings", {})
    marketplaces = system_settings.setdefault("marketplaces", [])
    devices = system_settings.setdefault("devices", [])

    # canonical source → device mapping
    SOURCE_DEVICE_TEMPLATES = {
        "fitbit": {
            "device_id": "fitbit_versa_3",
            "type": "watch",
        },
        "google_fit": {
            "device_id": "pixel_watch_2",
            "type": "watch",
        },
        "apple_health": {
            "device_id": "apple_watch_series_8",
            "type": "watch",
        },
        "huawei_health": {
            "device_id": "huawei_band_8",
            "type": "band",
        },
        "samsung_tracking": {
            "device_id": "galaxy_watch_5",
            "type": "watch",
        },
        "garmin_connect": {
            "device_id": "garmin_forerunner_265",
            "type": "watch",
        },
        "oura": {
            "device_id": "oura_ring_gen3",
            "type": "ring",
        },
        "xiaomi_mi_fitness": {
            "device_id": "mi_band_8",
            "type": "band",
        },
        "polar_flow": {
            "device_id": "polar_vantage_v2",
            "type": "watch",
        },
        "withings": {
            "device_id": "withings_scanwatch",
            "type": "watch",
        },
    }

    # -------- 1) 查 marketplace 是否已存在 --------
    marketplace = None
    for m in marketplaces:
        if m.get("source") == source_name:
            marketplace = m
            break

    # -------- 2) source 不存在 --------
    if marketplace is None:
        if not allow:
            return {
                "error": {
                    "code": "SOURCE_NOT_FOUND",
                    "message": f"Source '{source_name}' does not exist and cannot be disabled."
                }
            }

        if source_name not in SOURCE_DEVICE_TEMPLATES:
            return {
                "error": {
                    "code": "UNKNOWN_SOURCE",
                    "message": f"Source '{source_name}' is not supported."
                }
            }

        # create marketplace entry
        marketplaces.append({
            "source": source_name,
            "connected": True,
        })

        # create canonical device
        tpl = SOURCE_DEVICE_TEMPLATES[source_name]
        devices.append({
            "device_id": tpl["device_id"],
            "source": source_name,
            "type": tpl["type"],
        })

        return {
            "source_name": source_name,
            "connected": True,
            "created": True,
            "message": f"Source '{source_name}' connected and device created."
        }

    # -------- 3) source 已存在：只更新 connected --------
    marketplace["connected"] = allow

    return {
        "source_name": source_name,
        "connected": allow,
        "created": False,
        "message": f"Source '{source_name}' connection status updated to {allow}."
    }



def set_raw_data_permission(
    store: Dict[str, Any],
    allow: bool,
) -> Dict[str, Any]:
    """Grant or revoke permission to access raw data."""
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    
    permissions["allow_raw_data_access"] = allow
    
    return {
        "permission": "allow_raw_data_access",
        "allowed": allow,
        "message": f"Raw data access permission {'granted' if allow else 'revoked'}."
    }


def set_user_notes_permission(
    store: Dict[str, Any],
    allow: bool,
) -> Dict[str, Any]:
    """Grant or revoke permission to access user notes."""
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    
    permissions["allow_user_notes_access"] = allow
    
    return {
        "permission": "allow_user_notes_access",
        "allowed": allow,
        "message": f"User notes access permission {'granted' if allow else 'revoked'}."
    }


def set_purchase_permission(
    store: Dict[str, Any],
    allow: bool,
) -> Dict[str, Any]:
    """Grant or revoke permission to access purchase-related data."""
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    
    permissions["allow_purchase"] = allow
    
    return {
        "permission": "allow_purchase",
        "allowed": allow,
        "message": f"Purchase permission {'granted' if allow else 'revoked'}."
    }


def set_med_assistant_permission(
    store: Dict[str, Any],
    allow: bool,
) -> Dict[str, Any]:
    """Grant or revoke permission for medical-assistant features."""
    profile = store.get("profile", {})
    system_settings = profile.get("system_settings", {})
    permissions = system_settings.get("permissions", {})
    
    permissions["allow_med_assistant"] = allow
    
    return {
        "permission": "allow_med_assistant",
        "allowed": allow,
        "message": f"Medical assistant permission {'granted' if allow else 'revoked'}."
    }


def top_up_wallet(
    store: Dict[str, Any],
    amount: float,
) -> Dict[str, Any]:
    """
    Add money to the user's wallet balance.
    Copied from tools/user_tools.py and adapted to work with store.
    """
    if amount is None or amount <= 0:
        return {
            "error": {
                "code": "INVALID_AMOUNT",
                "message": "Amount must be positive."
            }
        }
    
    profile = store.get("profile", {})
    shopping = profile.get("shopping", {})
    wallet = shopping.get("wallet", {})
    
    current_balance = wallet.get("balance", 0.0)
    wallet["balance"] = round(current_balance + float(amount), 2)
    
    return {
        "wallet": {
            "balance": wallet["balance"],
            "vip": wallet.get("vip", False),
            "vip_expiry": wallet.get("vip_expiry"),
            "vouchers": wallet.get("vouchers", []),
        }
    }


def authorize_checkout(
    store: Dict[str, Any],
    transaction_id: str,
) -> Dict[str, Any]:
    """
    Finalize a pending transaction (order or vip_membership).
    Deduct balance and update wallet state.
    Copied from tools/user_tools.py and adapted to work with store.

    as it depends on product_details and product_log which are not in the backend.
    For VIP membership purchases, pending orders will need to be recomputed separately
    if that functionality is needed.
    """
    tx = _find_transaction(store, transaction_id)
    if not tx:
        return {
            "error": {
                "code": "TRANSACTION_NOT_FOUND",
                "message": f"Transaction {transaction_id} does not exist."
            }
        }
    
    if tx.get("status") != "pending":
        return {
            "error": {
                "code": "TRANSACTION_NOT_PENDING",
                "message": "Only pending transactions can be authorized."
            }
        }
    
    profile = store.get("profile", {})
    shopping = profile.get("shopping", {})
    wallet = shopping.get("wallet", {})
    balance = wallet.get("balance", 0.0)
    
    # -----------------------------
    # Order transaction
    # -----------------------------
    if tx.get("type") == "order":
        amount = float(tx.get("final_total", 0.0))
        if balance < amount:
            return {
                "error": {
                    "code": "INSUFFICIENT_BALANCE",
                    "message": "Not enough balance to complete the purchase."
                }
            }
        
        wallet["balance"] = round(balance - amount, 2)
        
        tx["status"] = "completed"
        
        # Remove voucher if applied
        voucher_internal = tx.get("voucher_internal", {})
        if voucher_internal.get("applied"):
            voucher_id = tx.get("voucher_id")
            if voucher_id:
                vouchers = wallet.get("vouchers", [])
                wallet["vouchers"] = [
                    v for v in vouchers if v.get("voucher_id") != voucher_id
                ]
        
        return {
            "transaction": {
                "transaction_id": tx["transaction_id"],
                "type": tx["type"],
                "status": tx["status"],
                "final_total": tx["final_total"],
            },
            "wallet_balance": wallet["balance"]
        }
    
    # -----------------------------
    # VIP membership purchase
    # -----------------------------
    if tx.get("type") == "vip_membership":
        amount = float(tx.get("amount", 0.0))
        if balance < amount:
            return {
                "error": {
                    "code": "INSUFFICIENT_BALANCE",
                    "message": "Not enough balance to upgrade membership."
                }
            }
        
        wallet["balance"] = round(balance - amount, 2)
        
        # Mark VIP active
        wallet["vip"] = True
        
        # Recompute all pending orders under the new VIP state
        # Note: This is a simplified version - full recompute would need product_details
        transactions = shopping.get("transactions", [])
        for pending_tx in transactions:
            if pending_tx.get("status") == "pending" and pending_tx.get("type") == "order":
                _recompute_order_for_current_state(pending_tx, store)
        
        # Set VIP expiry based on duration
        duration = tx.get("vip_duration")
        now_str = profile.get("now") or ""
        try:
            now = datetime.fromisoformat(now_str)
        except Exception:
            try:
                now = datetime.strptime(now_str, "%Y-%m-%d %H:%M")
            except Exception:
                now = datetime.utcnow()
        
        if duration == "1_month":
            expiry = now + timedelta(days=30)
        elif duration == "3_months":
            expiry = now + timedelta(days=90)
        elif duration == "1_year":
            expiry = now + timedelta(days=365)
        else:
            expiry = now
        
        wallet["vip_expiry"] = expiry.date().isoformat()
        
        tx["status"] = "completed"
        
        return {
            "transaction": {
                "transaction_id": tx["transaction_id"],
                "type": tx["type"],
                "status": tx["status"],
                "vip_duration": tx["vip_duration"],
            },
            "wallet": {
                "balance": wallet["balance"],
                "vip": wallet["vip"],
                "vip_expiry": wallet["vip_expiry"],
            }
        }

    # -----------------------------
    # Medical appointment checkout
    # -----------------------------
    if tx.get("appointment_id"):
        amount = float(tx.get("fee", 0.0))  # 可选，没有就 0
        if amount <= 0:
            return {
                "error": {
                    "code": "INVALID_APPOINTMENT_FEE",
                    "message": "Appointment fee is missing or invalid."
                }
            }

        if balance < amount:
            return {
                "error": {
                    "code": "INSUFFICIENT_BALANCE",
                    "message": "Not enough balance to pay for appointment."
                }
            }

        wallet["balance"] = round(balance - amount, 2)

        tx["status"] = "confirmed"

        return {
            "appointment": {
                "appointment_id": tx.get("appointment_id"),
                "provider_id": tx.get("provider_id"),
                "status": tx["status"],
                "transaction_id": tx["transaction_id"],
            },
            "wallet_balance": wallet["balance"]
        }

    # Fallback for unknown transaction types
    return {
        "error": {
            "code": "UNKNOWN_TRANSACTION_TYPE",
            "message": f"Unsupported transaction type {tx.get('type')}."
        }
    }


# ============================================================
# Tool registry for dispatcher
# ============================================================

registered_user_tools = {
    "update_source": update_source,
    "set_raw_data_permission": set_raw_data_permission,
    "set_user_notes_permission": set_user_notes_permission,
    "set_purchase_permission": set_purchase_permission,
    "set_med_assistant_permission": set_med_assistant_permission,
    "top_up_wallet": top_up_wallet,
    "authorize_checkout": authorize_checkout,
}

