# bench/backend/store/build_store.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Literal, Optional, Tuple
import json
import random
from datetime import datetime, timedelta

import pandas as pd

from bench.utils.paths import DATA_DIR
from bench.backend.configuration.configuration import load_initialization_profile
from bench.pipeline.build_user.inject_sports import inject_sports
from bench.pipeline.build_user.source_assignment import generate_source_assignment
from bench.pipeline.build_user.system_settings import generate_system_settings
from bench.pipeline.build_user.inject_meals import generate_meals
from bench.pipeline.build_user.generate_profile import generate_profile


# ============================================================
# wearable tables 列表
# ============================================================

TABLE_FILES = {
    "daily_heartrate":      "heartrate_summary",
    "minute_calories":      "minuteCaloriesNarrow",
    "minute_intensities":   "minuteIntensitiesNarrow",
    "minute_mets":          "minuteMETsNarrow",
    "daily_sleep":          "sleep_summary",
    "minute_steps":         "minuteStepsNarrow",
}

FileFormat = Literal["csv", "parquet"]


# ============================================================
# 日期标准化 + mask（满足你的两种日期格式）
# ============================================================

def filter_df_by_user_and_date(
    df: pd.DataFrame,
    base_name: str,
    user_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    df = df.copy()

    # ---------- 1) 识别用户列（可能不存在） ----------
    if "Id" in df.columns:
        user_col = "Id"
    elif "id" in df.columns:
        user_col = "id"
    else:
        user_col = None

    # ---------- 2) 识别日期列 ----------
    if "Date" in df.columns:
        date_col = "Date"
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="raise")
    elif "ActivityMinute" in df.columns:
        date_col = "ActivityMinute"
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    else:
        raise ValueError(f"No recognized date column in {base_name}")

    # ---------- 3) 构造时间区间（半开区间） ----------
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    # ---------- 4) mask ----------
    if user_col is not None:
        mask = (
            (df[user_col].astype(str) == str(user_id)) &
            (df[date_col] >= start_ts) &
            (df[date_col] < end_ts)
        )
    else:
        mask = (
            (df[date_col] >= start_ts) &
            (df[date_col] < end_ts)
        )

    filtered = df.loc[mask].sort_values(by=date_col).reset_index(drop=True)
    return filtered


# ============================================================
# time shifting helpers
# ============================================================

def _shift_iso_dt_str(s: str, shift_days: int) -> str:
    """
    Shift an ISO datetime string like '2016-03-11T22:42:30' by integer days.
    (No timezone; fromisoformat is ok.)
    """
    dt = datetime.fromisoformat(s)
    return (dt + timedelta(days=shift_days)).isoformat(timespec="seconds")


def _shift_sleep_segments_json(seg_str: Any, shift_days: int) -> Any:
    """
    segments_json 是字符串，里面 list[dict] 包含 start_time/end_time。
    如果为空/NaN/非字符串，原样返回。
    """
    if seg_str is None:
        return seg_str
    if isinstance(seg_str, float) and pd.isna(seg_str):
        return seg_str

    s = str(seg_str).strip()
    if not s:
        return seg_str

    try:
        segs = json.loads(s)
        if not isinstance(segs, list):
            return seg_str
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            if "start_time" in seg and isinstance(seg["start_time"], str):
                seg["start_time"] = _shift_iso_dt_str(seg["start_time"], shift_days)
            if "end_time" in seg and isinstance(seg["end_time"], str):
                seg["end_time"] = _shift_iso_dt_str(seg["end_time"], shift_days)
        return json.dumps(segs, ensure_ascii=False)
    except Exception:
        # 不硬崩，保持原始
        return seg_str


def shift_df_time_columns(
    df: pd.DataFrame,
    base_name: str,
    shift_days: int,
) -> pd.DataFrame:
    """
    对单表做“整天平移”：
    - Date / ActivityMinute shift
    - 仅对 sleep_summary 额外 shift segments_json
    """
    out = df.copy()

    # 主时间列：Date / ActivityMinute
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="raise") + pd.Timedelta(days=shift_days)
        # 如果你希望继续用字符串（更贴近你截图里的形态），就转回去：
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    if "ActivityMinute" in out.columns:
        out["ActivityMinute"] = pd.to_datetime(out["ActivityMinute"], errors="raise") + pd.Timedelta(days=shift_days)
        # ActivityMinute 通常不需要转字符串（保留 datetime 更好）；如果你需要字符串也可自行 dt.strftime

    # 特例：sleep_summary 的 segments_json
    if base_name == "sleep_summary" and "segments_json" in out.columns:
        out["segments_json"] = out["segments_json"].apply(lambda x: _shift_sleep_segments_json(x, shift_days))

    return out


def choose_shifted_date_range(
    original_range: Tuple[str, str],
    target_year_start: int = 2024,
    target_year_end: int = 2026,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, int, str]:
    """
    从 [target_year_start, target_year_end] 随机采样 new_start（按天），
    保证 new_end = new_start + (orig_len_days-1) 仍落在范围内。
    返回：
      (new_start_str, new_end_str, shift_days, new_start_str)
    """
    if rng is None:
        rng = random.Random()

    orig_start = pd.to_datetime(original_range[0])
    orig_end = pd.to_datetime(original_range[1])

    if orig_end < orig_start:
        raise ValueError(f"Invalid original_range: {original_range}")

    # inclusive length in days
    length_days = int((orig_end - orig_start).days) + 1

    target_min = pd.Timestamp(f"{target_year_start}-01-01")
    target_max_end = pd.Timestamp(f"{target_year_end}-12-31")
    # new_start 的最大允许值（保证 new_end 不越界）
    latest_start = target_max_end - pd.Timedelta(days=length_days - 1)
    if latest_start < target_min:
        raise ValueError("Target year window too small for the original segment length.")

    # 随机采样 new_start（按天）
    span_days = int((latest_start - target_min).days)
    offset = rng.randint(0, span_days)
    new_start = target_min + pd.Timedelta(days=offset)
    new_end = new_start + pd.Timedelta(days=length_days - 1)

    shift_days = int((new_start - orig_start).days)

    return (
        new_start.strftime("%Y-%m-%d"),
        new_end.strftime("%Y-%m-%d"),
        shift_days,
        new_start.strftime("%Y-%m-%d"),
    )


def sample_now_on_end_date(
    end_date: str,
    hour_min: int = 18,
    hour_max: int = 22,
    rng: Optional[random.Random] = None,
) -> str:
    """
    在 end_date 当天的 [18,22] 随机采样整数点作为 now。
    """
    if rng is None:
        rng = random.Random()
    h = rng.randint(hour_min, hour_max)
    dt = datetime.fromisoformat(end_date).replace(hour=h, minute=0, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ============================================================
# 加载所有 wearable 表，统一过滤
# ============================================================
def normalize_wearable_precision(
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    统一 wearable minute 表的数值精度：
    - Calories / METs: round(1)
    - Steps: round to int
    - Intensity: int
    只处理存在的列，不假设全量 schema
    """
    out: Dict[str, pd.DataFrame] = {}

    for name, df in tables.items():
        if df is None or df.empty:
            out[name] = df
            continue

        df = df.copy()

        if "Calories" in df.columns:
            df["Calories"] = df["Calories"].round(2)

        if "METs" in df.columns:
            df["METs"] = df["METs"].round(2).astype("Int64")

        if "Steps" in df.columns:
            df["Steps"] = df["Steps"].round().astype("Int64")

        if "Intensity" in df.columns:
            # 防御性：避免 NaN -> float
            df["Intensity"] = df["Intensity"].round().astype("Int64")

        out[name] = df

    return out


def load_filtered_tables(
    user_id: str,
    date_range: Tuple[str, str],
    fmt: FileFormat = "csv",
) -> Dict[str, pd.DataFrame]:
    start_date, end_date = date_range
    results: Dict[str, pd.DataFrame] = {}

    for key, base_name in TABLE_FILES.items():
        # 读取文件
        if fmt == "csv":
            path = DATA_DIR / f"{base_name}.csv"
            df = pd.read_csv(path)
        else:
            path = DATA_DIR / f"{base_name}.parquet"
            df = pd.read_parquet(path)

        # 过滤用户和时间范围（原始静态时间段）
        results[key] = filter_df_by_user_and_date(
            df, base_name, user_id, start_date, end_date
        )

    return results


def shift_all_tables(
    tables: Dict[str, pd.DataFrame],
    shift_days: int,
) -> Dict[str, pd.DataFrame]:
    """
    对所有表做同一个 shift_days（整天平移）
    """
    shifted: Dict[str, pd.DataFrame] = {}
    for key, base_name in TABLE_FILES.items():
        df = tables.get(key)
        if df is None:
            continue
        shifted[key] = shift_df_time_columns(df, base_name=base_name, shift_days=shift_days)
    return shifted


# ============================================================
# store
# ============================================================

def build_store(
    user_id: str,
    date_range: Tuple[str, str],
    data_format: FileFormat = "csv",
    enable_time_shift: bool = True,
    target_year_start: int = 2024,
    target_year_end: int = 2026,
    rng_seed: Optional[int] = None,
    snapshot_dir: str = "/home/chy/state_aware_bench/bench/snapshots/debug_snapshot",
    profile_id: str = None
) -> Dict[str, Any]:
    """
    profile 初始化 + wearable 数据过滤 → store
    wearable data 的 userId / date_range（原始静态片段）由调用方控制

    enable_time_shift=True 时：
    - 在 2024~2026 采样 new_start
    - 计算 shift_days
    - 对所有表整体平移
    - 在 end_date 当天 18~22 点采样整数点作为 now
    """
    rng = random.Random(rng_seed)

    # 1) 先加载原始静态片段
    tables_orig = load_filtered_tables(
        user_id=user_id,
        date_range=date_range,
        fmt=data_format,
    )

    meta: Dict[str, Any] = {
        "user_id": user_id,
        "original_date_range": date_range,
        "data_format": data_format,
        "enable_time_shift": enable_time_shift,
        "rng_seed": rng_seed,
    }

    # 2) 时间平移（整天）
    if enable_time_shift:
        new_start, new_end, shift_days, sampled_new_start = choose_shifted_date_range(
            original_range=date_range,
            target_year_start=target_year_start,
            target_year_end=target_year_end,
            rng=rng,
        )
        tables = shift_all_tables(tables_orig, shift_days=shift_days)

        tables = normalize_wearable_precision(tables)

        # 3) 采样 now（end_date 当晚 18~22）
        now_str = sample_now_on_end_date(new_end, hour_min=18, hour_max=22, rng=rng)

        meta.update({
            "date_range": (new_start, new_end),
            "shift_days": shift_days,
            "sampled_new_start": sampled_new_start,
            "now": now_str,
            "target_year_window": (target_year_start, target_year_end),
        })
    else:
        raise ValueError("Time shift is now considered as necessary")

    profile = load_initialization_profile(now_str, profile_id)

    # Inject synthetic profile
    sports, tables, template = inject_sports(profile, tables, meta["date_range"], rng)
    profile["sports"] = sports
    profile["source_assignment"] = generate_source_assignment(
        date_range=meta["date_range"],
        rng=rng,
    )
    profile["system_settings"] = generate_system_settings(profile["source_assignment"], rng)
    profile["meals"] = generate_meals(meta["date_range"], rng)
    generation1 = generate_profile(meta["date_range"], profile, rng, tables, template)


    tables = normalize_wearable_precision(tables)
    save_store_snapshot(Path(snapshot_dir), meta, profile)

    return {
        "meta": meta,
        "profile": profile,              # 原样存放
        "wearable_tables": tables,
    }

def load_store_from_meta_profile(
    profile: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    - 加载 meta / profile
    - 重新从 CSV 读取 wearable 数据
    - 根据 meta 中的 shift_days 做时间平移
    - 使用相同 rng_seed 重新注入 sports，恢复对分钟级表的修改
    """

    # ---------- 1) load meta / profile ----------
    pass

    # ---------- 2) 反推 wearable loading 参数 ----------
    user_id = meta["user_id"]
    original_date_range = tuple(meta["original_date_range"])
    data_format = meta.get("data_format", "csv")
    shift_days = meta.get("shift_days", 0)

    # ---------- 3) 加载原始静态片段 ----------
    tables_orig = load_filtered_tables(
        user_id=user_id,
        date_range=original_date_range,
        fmt=data_format,
    )

    # ---------- 4) 时间平移（必须） ----------
    if shift_days is None:
        raise ValueError("Invalid snapshot: missing shift_days")

    tables = shift_all_tables(tables_orig, shift_days=shift_days)

    # ---------- 5) 重新注入 sports（确保表与 profile 一致） ----------
    rng = random.Random(meta.get("rng_seed"))
    sports, tables, template = inject_sports(profile, tables, meta.get("date_range"), rng)
    profile["sports"] = sports
    tables = normalize_wearable_precision(tables)

    return {
        "meta": meta,
        "profile": profile,
        "wearable_tables": tables,
    }

def save_store_snapshot(
    snapshot_dir: Path,
    meta: Dict[str, Any],
    profile: Dict[str, Any],
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    with open(snapshot_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(snapshot_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


# ============================================================
# Debug 运行测试
# ============================================================

if __name__ == "__main__":
    store = build_store(
        user_id="2026352035",
        date_range=("2016-04-01", "2016-04-30"),
        data_format="csv",
        enable_time_shift=True,
        target_year_start=2024,
        target_year_end=2026,
        rng_seed=46,
    )

    print("[OK] store loaded.")
    print("meta:", store["meta"])
    for name, df in store["wearable_tables"].items():
        print(name, df.shape)
