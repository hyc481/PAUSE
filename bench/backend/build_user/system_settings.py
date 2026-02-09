from __future__ import annotations

from typing import Dict, Any, List
import random


def generate_system_settings(
    source_assignment: Dict[str, str],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Generate system_settings based on *true* source_assignment.

    Semantics:
    - source_assignment is 100% real (backend truth).
    - system_settings reflects platform knowledge / connection state.
    - Three cases per source:
        A) created + connected
        B) created + disconnected
        C) not created
    - One canonical device per created source.
    - All permissions set to False.
    """

    # -------- source → realistic device templates --------
    SOURCE_DEVICE_TEMPLATES: Dict[str, Dict[str, str]] = {
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

    # -------- 1) 哪些 source 在真实世界里出现过 --------
    real_sources = {
        s for s in source_assignment.values()
        if s != "missing"
    }

    marketplaces: List[Dict[str, Any]] = []
    devices: List[Dict[str, Any]] = []

    # -------- 2) 为每个真实 source 采样平台状态 --------
    for src in sorted(real_sources):
        if src not in SOURCE_DEVICE_TEMPLATES:
            # safety fallback（理论上不会发生）
            continue

        # 三态采样（可调，但现在已经很合理）
        r = rng.random()
        if r < 0.5:
            state = "connected"          # A
        else:
            state = "disconnected"       # B
        """
        else:
            state = "not_created"        # C
        
        if state == "not_created":
            # source 未创建：marketplace / device 都不存在
            continue
        """
        # created
        connected = (state == "connected")
        marketplaces.append({
            "source": src,
            "connected": connected,
        })

        # one canonical device per source
        tpl = SOURCE_DEVICE_TEMPLATES[src]
        devices.append({
            "device_id": tpl["device_id"],
            "source": src,
            "type": tpl["type"],
        })

    # -------- 3) permissions：全部 False --------
    permissions = {
        "allow_raw_data_access": False,
        "allow_user_notes_access": False,
        "allow_purchase": False,
        "allow_med_assistant": False,
    }

    return {
        "marketplaces": marketplaces,
        "devices": devices,
        "permissions": permissions,
    }
