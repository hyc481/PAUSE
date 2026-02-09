# bench/backend/configuration/configuration.py

from __future__ import annotations
from typing import Dict, Any

from bench.backend.configuration.shopping_state import WALLET_STATE
from bench.backend.configuration.shopping_state import CART_STATE
from bench.backend.configuration.shopping_state import TRANSACTIONS

# ============================================================
# Debug Initialization Profile（不包含 wearable 配置）
# ============================================================

debug_profile: Dict[str, Any] = {
    "now": "2025-01-30T10:20",
    # -------- user_profile / personalization --------
    "user_profile": {
        "basic_info": {
            "name": "Alex",
            "age": 35,
            "gender": "F"
        },
        "health_risks": ["T2DM", "hypertension"],  # 慢病/风险标签
        "dietary_restrictions": ["lactose_intolerant"],
        "preferences": {
            "sport": ["cardio", "strength"],
            "food": ["high_protein", "low_sugar"]
        },
        "daily_goal": {
            "steps": 8000,
            "mets": 1000,
            "calories_burn": 2300,
            "AZM": 25,
            "calories_intake": 2200
          },
    },

    "med_user_profile": {
        "basic_info": {
            "name": "Alex",
            "age": 35,
            "gender": "F"
        },
        "health_risks": ["T2DM", "hypertension"],  # 慢病/风险标签
        "dietary_restrictions": ["lactose_intolerant"],
    },

    "source_profile": [
        {
            "source":"fitbit",
            "basic_info": {
            "name": "Alex",
                    "age": 35,
                    "gender": "F"
            },
            "daily_goal": {
                    "steps": 8000,
                    "mets": 1000,
                    "calories_burn": 2300,
                    "AZM": 60
            },
            "weekly_goal": {
                    "steps": 50000,
                    "mets": 6000,
                    "calories_burn": 16000,
                    "AZM": 400
            },
        }],

    # -------- system_settings --------
    "system_settings": {
        "marketplaces": [
            {"source": "fitbit",          "connected": True},
            {"source": "google_fit",      "connected": False},
            {"source": "apple_health",    "connected": False},
            {"source": "huawei_health",   "connected": False},
            {"source": "samsung_tracking","connected": False},
        ],
        "devices": [
            {"device_id": "fitbit_versa_3",        "source": "fitbit",           "type": "watch"},
            {"device_id": "google_pixel_watch",    "source": "google_fit",       "type": "watch"},
            {"device_id": "huawei_band_8",         "source": "huawei_health",    "type": "band"},
            {"device_id": "apple_watch_series_8",  "source": "apple_health",     "type": "watch"},
            {"device_id": "samsung_galaxy_watch5", "source": "samsung_tracking", "type": "watch"},
        ],
        
        "permissions": {
            "allow_raw_data_access": True,
            "allow_user_notes_access": True,
            "allow_purchase": False,
            "allow_med_assistant": False,
        }
    },
    # -------- Sports（运动事件/计划，和 session / record 相关的高层信息） --------
    # 这些通常会“挂”在某些 wearable session 上（通过 session_id或日期+时间）
    "sports": [
        {
            "sport_name": "run",
            "start_time": "2024-09-17T15:00:00",
            "end_time": "2024-09-17T16:00:00",
            "statistics": {
                "calories": 420,
                "azm": 28
            },
            "user_note": "",
            "intensity": "light"
        }
    ],

    # -------- sessions / meals（和 timeseries 绑定的记录类东西） --------
    # session 可以是运动 session、睡眠 session 等
    "sessions": [
        {
            "record_id": "session_0001",
            "session_type": "study",
            "start_time": "2016-04-10T07:30:00",
            "end_time":   "2016-04-10T08:10:00",
            "user_note": "prepare for exams",
        }
    ],

    "meals": [
        {
            "meal_id": "ld_tofu_veg_rice",
            "meal_type": "lunch_dinner",
            "items": [
                {"name": "Tofu", "amount_g": 180},
                {"name": "Stir-fried Vegetables", "amount_g": 220},
                {"name": "Brown Rice", "amount_g": 200},
            ],
            "nutrition": {"calories": 620, "sugar_g": 7.0, "fiber_g": 10.0, "fat_g": 18.0},
        },
    ],

    # -------- Notes & notifications --------
    "notes": [
        {
            "note_id": "note_0001",
            "time": "2016-04-09T18:00:00",
            "content": "Doctor suggested increasing weekly running time.",
        }
    ],

    "reminders":[
        {
            "reminder_id": "r_9913",
            "title": "晨跑",
            "time_of_day": "07:00",
            "repeat_days": ["Mon", "Wed", "Fri"]
        }
    ],

    # -------- Care plans (diet, exercise, medication, etc.) --------
    "health_providers": [],

    "care_plans": [
        {
            "plan_id": "cp_00000001",
            "provider_id": "provider_21341342",
            "created_at": "2016-04-10T20:00:00",
            "topics": ["exercise", "diet"],
            "note_text": "近期的血糖波动提示您需要更加规律的运动与饮食控制。建议每周至少进行 120 分钟的中等强度有氧运动，例如快走或轻度慢跑，并尽量将运动分散到至少 4 天完成。如出现头晕、心悸或明显疲劳，请适当降低强度。饮食方面，请减少含糖饮料、精制碳水与高热量零食的摄入，优先选择高纤维蔬菜、优质蛋白与适量全谷物。晚餐建议清淡，并避免睡前两小时内进食甜点或夜宵。"
        },
    ],

    # -------- Appointments --------
    "appointments": [
        {
            "appointment_id": "appt_00000002",
            "provider_id": "family_clinic",
            "date": "2016-07-05",
            "time": "12:00",
            "duration": 60,
            "reason": "最近几次家庭血压监测偏高，想咨询是否需要调整降压药或生活方式。",
            "status": "confirmed",
            "transaction_id": "txn_00000001"
        }
    ],
    
    # -------- source_assignment（按天 mask / 来源，用来制造复杂情况） --------
    # 这里是你打补丁的核心：决定每一天的数据来自哪个 source 或为 missing
    "source_assignment": {
        "2016-04-01": "fitbit",
        "2016-04-02": "fitbit",
        "2016-04-03": "missing",      # 故意缺失
        "2016-04-04": "fitbit",
        "2016-04-05": "google_fit",   # 可以模拟“某天换表”
        "2016-04-06": "fitbit",
        "2016-04-07": "missing"
    },

    "shopping": {
        "wallet": WALLET_STATE,
        "cart": CART_STATE,
        "transactions": TRANSACTIONS
    }
}

# ============================================================
# Profile Loader（你未来可扩展成 registry）
# ============================================================

from copy import deepcopy

_MINIMAL_PROFILE: Dict[str, Any] = {
    "now": None,
    "user_profile": {
        "basic_info": {},
        "health_risks": [],
        "dietary_restrictions": [],
        "preferences": {},
        "daily_goal": {},
    },
    "med_user_profile": {
        "basic_info": {},
        "health_risks": [],
        "dietary_restrictions": [],
    },
    "source_profile": [],
    "system_settings": {
        "marketplaces": [],
        "devices": [],
        "permissions": {
            "allow_raw_data_access": False,
            "allow_user_notes_access": False,
            "allow_purchase": False,
            "allow_med_assistant": False,
        },
    },
    "sports": [],
    "sessions": [],
    "meals": [],
    "notes": [],
    "reminders": [],
    "care_plans": [],
    "appointments": [],
    "source_assignment": {},
    "shopping": {
        "wallet": {
            "balance": 0,
            "vip": False,
            "vip_expiry": None,
            "vouchers":[]
        },
        "cart": [],
        "transactions": [],
    },
}



def load_initialization_profile(now: str, profile_id:str = None) -> Dict[str, Any]:
    if profile_id == "debug_profile":
        profile = deepcopy(debug_profile)
        profile["now"] = now
        return profile

    profile = deepcopy(_MINIMAL_PROFILE)
    profile["now"] = now
    return profile



