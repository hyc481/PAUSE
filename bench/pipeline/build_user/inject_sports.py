from __future__ import annotations

"""
Inject synthetic sports events into wearable tables with seed-driven randomness.
Focus is on semantic match (habit + intensity), not physiological accuracy.

This version:
- Natural-week aligned sampling (Mon-Sun)
- Stable weekly frequency: sample exact #training days per week (within range)
- DAILY PATTERNS: choose a day pattern per training day; each pattern defines multiple event windows
  (not "pick one slot from many"; pattern explicitly lists the events)
- No jitter
- Avoid overlaps within a day by interval bookkeeping
- Habit (high-level) != Sport (event-level)
- IMPORTANT: minute-level Intensity is modified FIRST with per-level perturbation.
  AZM is then derived as count(Intensity >= 2) over the event interval.
  Steps/METs/Calories are generated per-minute conditioned on the minute-level Intensity.
"""

import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

import pandas as pd


# --------------------------------------------------------------------------- #
# Day patterns (day-level structure)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EventWindow:
    """One event must be placed inside this window (half-open minutes-of-day)."""
    window: Tuple[int, int]  # (start_minute_of_day, end_minute_of_day)


@dataclass(frozen=True)
class DayPattern:
    """A day pattern is an explicit list of events (each with its own window)."""
    name: str
    events: List[EventWindow]


# --------------------------------------------------------------------------- #
# Habit templates (high-level)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HabitTemplate:
    name: str
    days_per_week: Tuple[int, int]
    duration_minutes: Tuple[int, int]
    intensity_mix: List[Tuple[str, float]]
    sports_mix: List[Tuple[str, float]]
    day_pattern_mix: List[Tuple["DayPattern", float]]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the habit template into a human- and LLM-readable dict.
        This is a semantic summary, not an execution spec.
        """

        def _ranked_labels(mix: List[Tuple[str, float]], top_k: int = 2):
            """
            Return labels ranked by weight, without exposing raw probabilities.
            """
            return [
                label for label, _ in
                sorted(mix, key=lambda x: x[1], reverse=True)[:top_k]
            ]

        def _bucketed_mix(mix: List[Tuple[str, float]]):
            """
            Convert weighted mix into coarse qualitative buckets.
            """
            total = sum(w for _, w in mix) or 1.0
            out = {}
            for label, weight in mix:
                ratio = weight / total
                if ratio >= 0.6:
                    out[label] = "primary"
                elif ratio >= 0.3:
                    out[label] = "secondary"
                else:
                    out[label] = "occasional"
            return out

        return {
            "name": self.name,

            # how often the user typically exercises
            "frequency": {
                "days_per_week": {
                    "min": self.days_per_week[0],
                    "max": self.days_per_week[1],
                }
            },

            # typical session length range
            "session_duration_minutes": {
                "min": self.duration_minutes[0],
                "max": self.duration_minutes[1],
            },

            # qualitative intensity profile
            "intensity_profile": _bucketed_mix(self.intensity_mix),

            # preferred sports (ranked, not probabilistic)
            "sport_preferences": _ranked_labels(self.sports_mix),

            # preferred time-of-day patterns (names only)
            "day_pattern_preferences": _bucketed_mix(
                [(pattern.name, weight) for pattern, weight in self.day_pattern_mix]
            ),
        }


# Example patterns for runner_focus
PAT_EVENING_SINGLE = DayPattern(
    name="evening_single",
    events=[EventWindow((18 * 60, 21 * 60))],
)

PAT_MORNING_SINGLE = DayPattern(
    name="morning_single",
    events=[EventWindow((6 * 60, 9 * 60))],
)

PAT_MORNING_EVENING = DayPattern(
    name="morning_evening",
    events=[
        EventWindow((6 * 60, 9 * 60)),
        EventWindow((18 * 60, 21 * 60)),
    ],
)

# Two sessions both in evening, but explicitly separated windows to reduce overlap pressure
PAT_EVENING_DOUBLE_SPLIT = DayPattern(
    name="evening_double_split",
    events=[
        EventWindow((18 * 60, 19 * 60 + 30)),
        EventWindow((19 * 60 + 30, 21 * 60)),
    ],
)

HABIT_TEMPLATES: List[HabitTemplate] = [
    # ------------------------------------------------------------------
    # 1) Runner focus（你原来的，略微保留）
    # ------------------------------------------------------------------
    HabitTemplate(
        name="runner_focus",
        days_per_week=(4, 6),
        duration_minutes=(35, 65),
        intensity_mix=[("moderate", 0.65), ("vigorous", 0.35)],
        sports_mix=[("running", 0.75), ("interval_run", 0.25)],
        day_pattern_mix=[
            (PAT_EVENING_SINGLE, 0.55),
            (PAT_MORNING_SINGLE, 0.10),
            (PAT_MORNING_EVENING, 0.25),
            (PAT_EVENING_DOUBLE_SPLIT, 0.10),
        ],
    ),

    # ------------------------------------------------------------------
    # 2) Casual jogger：频率中等，基本单次、强度偏低
    # ------------------------------------------------------------------
    HabitTemplate(
        name="casual_jogger",
        days_per_week=(2, 4),
        duration_minutes=(25, 45),
        intensity_mix=[("light", 0.55), ("moderate", 0.45)],
        sports_mix=[("jogging", 0.7), ("easy_run", 0.3)],
        day_pattern_mix=[
            (PAT_EVENING_SINGLE, 0.6),
            (PAT_MORNING_SINGLE, 0.4),
        ],
    ),

    # ------------------------------------------------------------------
    # 3) Morning discipline：早晨规律型，几乎不晚上练
    # ------------------------------------------------------------------
    HabitTemplate(
        name="morning_discipline",
        days_per_week=(3, 5),
        duration_minutes=(30, 55),
        intensity_mix=[("moderate", 0.7), ("vigorous", 0.3)],
        sports_mix=[("running", 0.6), ("tempo_run", 0.4)],
        day_pattern_mix=[
            (PAT_MORNING_SINGLE, 0.75),
            (PAT_MORNING_EVENING, 0.25),  # 偶尔 double
        ],
    ),

    # ------------------------------------------------------------------
    # 4) Busy professional：频率低但稳定，几乎全在晚上
    # ------------------------------------------------------------------
    HabitTemplate(
        name="busy_professional",
        days_per_week=(2, 3),
        duration_minutes=(30, 50),
        intensity_mix=[("moderate", 0.6), ("vigorous", 0.4)],
        sports_mix=[("running", 0.5), ("interval_run", 0.5)],
        day_pattern_mix=[
            (PAT_EVENING_SINGLE, 0.85),
            (PAT_EVENING_DOUBLE_SPLIT, 0.15),
        ],
    ),

    # ------------------------------------------------------------------
    # 5) Fitness enthusiast：频率高，双 session 常见
    # ------------------------------------------------------------------
    HabitTemplate(
        name="fitness_enthusiast",
        days_per_week=(5, 6),
        duration_minutes=(30, 60),
        intensity_mix=[("moderate", 0.5), ("vigorous", 0.5)],
        sports_mix=[
            ("running", 0.4),
            ("interval_run", 0.3),
            ("cardio_mix", 0.3),
        ],
        day_pattern_mix=[
            (PAT_MORNING_EVENING, 0.5),
            (PAT_EVENING_SINGLE, 0.3),
            (PAT_EVENING_DOUBLE_SPLIT, 0.2),
        ],
    ),

    # ------------------------------------------------------------------
    # 6) Recovery-oriented：低强度、多但轻（语义很不一样）
    # ------------------------------------------------------------------
    HabitTemplate(
        name="recovery_oriented",
        days_per_week=(3, 5),
        duration_minutes=(20, 40),
        intensity_mix=[("light", 0.7), ("moderate", 0.3)],
        sports_mix=[
            ("easy_run", 0.4),
            ("walk_jog", 0.4),
            ("recovery_cardio", 0.2),
        ],
        day_pattern_mix=[
            (PAT_MORNING_SINGLE, 0.5),
            (PAT_EVENING_SINGLE, 0.5),
        ],
    ),
]



# --------------------------------------------------------------------------- #
# Intensity mapping
# --------------------------------------------------------------------------- #

INTENSITY_LEVELS: Dict[str, Dict[str, Any]] = {
    "light": {
        "int_value": 1,
        "steps_per_min": (40, 60),
        "mets": (20, 40),
        "cal_per_min": (3.0, 5.0),
    },
    "moderate": {
        "int_value": 2,
        "steps_per_min": (60, 80),
        "mets": (40, 80),
        "cal_per_min": (6.0, 10.0),
    },
    "vigorous": {
        "int_value": 3,
        "steps_per_min": (80, 100),
        "mets": (80, 120),
        "cal_per_min": (9.0, 14.0),
    },
}

SEDENTARY_LEVEL = {
    "int_value": 0,
    "steps_per_min": (0, 25),
    "mets": (1.0, 1.5),
    "cal_per_min": (0.5, 1.5),
}

INT_VALUE_TO_LEVEL = {
    0: SEDENTARY_LEVEL,
    1: INTENSITY_LEVELS["light"],
    2: INTENSITY_LEVELS["moderate"],
    3: INTENSITY_LEVELS["vigorous"],
}

LABEL_TO_INT = {
    "light": 1,
    "moderate": 2,
    "vigorous": 3,
}

MINUTE_INT_PERTURB: Dict[str, List[Tuple[int, float]]] = {
    "light": [(1, 0.85), (0, 0.15)],
    "moderate": [(2, 0.70), (1, 0.20), (3, 0.10)],
    "vigorous": [(3, 0.85), (2, 0.15)],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _choose_template(rng: random.Random) -> HabitTemplate:
    return rng.choice(HABIT_TEMPLATES)


def _weighted_choice(pairs: List[Tuple[str, float]], rng: random.Random) -> str:
    labels, weights = zip(*pairs)
    return rng.choices(list(labels), weights=list(weights), k=1)[0]


def _weighted_choice_int(pairs: List[Tuple[int, float]], rng: random.Random) -> int:
    values, weights = zip(*pairs)
    return rng.choices(list(values), weights=list(weights), k=1)[0]


def _weighted_choice_obj(pairs: List[Tuple[Any, float]], rng: random.Random) -> Any:
    objs, weights = zip(*pairs)
    return rng.choices(list(objs), weights=list(weights), k=1)[0]


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _split_into_natural_weeks(start: pd.Timestamp, end: pd.Timestamp) -> List[List[pd.Timestamp]]:
    """Natural week: Monday..Sunday."""
    all_days = pd.date_range(start=start, end=end, freq="D")
    buckets: Dict[pd.Timestamp, List[pd.Timestamp]] = {}
    for d in all_days:
        week_start = (d - pd.Timedelta(days=int(d.weekday()))).normalize()  # Monday
        buckets.setdefault(week_start, []).append(d)
    weeks = [buckets[k] for k in sorted(buckets.keys())]
    for w in weeks:
        w.sort()
    return weeks


def _sample_training_days_by_week(
    date_range: Tuple[str, str],
    template: HabitTemplate,
    rng: random.Random,
) -> List[pd.Timestamp]:
    """Each natural week: sample exact #training days k and choose k days without replacement."""
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    weeks = _split_into_natural_weeks(start, end)

    training_days: List[pd.Timestamp] = []
    for week_days in weeks:
        k = rng.randint(template.days_per_week[0], template.days_per_week[1])
        k = min(k, len(week_days))
        chosen = rng.sample(week_days, k)
        chosen.sort()
        training_days.extend(chosen)

    training_days.sort()
    return training_days


def _interval_overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])  # half-open [start, end)


def _find_start_in_window_non_overlapping(
    window: Tuple[int, int],
    duration_min: int,
    occupied: List[Tuple[int, int]],
    rng: random.Random,
    max_tries: int = 60,
) -> Optional[int]:
    """
    Sample a start time inside a given window (no jitter), ensure no overlap with occupied.
    window is half-open [w_s, w_e). Event interval is [start, start+duration).
    """
    w_s, w_e = window
    w_s = _clamp(int(w_s), 0, 24 * 60)
    w_e = _clamp(int(w_e), 0, 24 * 60)

    if w_e <= w_s:
        return None

    latest_start = w_e - duration_min
    if latest_start < w_s:
        return None

    for _ in range(max_tries):
        start_min = rng.randint(w_s, latest_start)
        end_min = start_min + duration_min
        cand = (start_min, end_min)
        if any(_interval_overlaps(cand, itv) for itv in occupied):
            continue
        return start_min

    return None


def _sample_minute_intensity_vector(
    target_label: str,
    n_minutes: int,
    rng: random.Random,
) -> List[int]:
    dist = MINUTE_INT_PERTURB.get(target_label)
    if not dist:
        v = LABEL_TO_INT.get(target_label, 1)
        return [v] * n_minutes
    return [_weighted_choice_int(dist, rng) for _ in range(n_minutes)]


def _build_event_base(
    day: pd.Timestamp,
    start_minute: int,
    duration_min: int,
    target_intensity_label: str,
    sport_name: str,
) -> Dict[str, Any]:
    start_dt = datetime.combine(day.date(), datetime.min.time()) + timedelta(minutes=start_minute)
    end_dt = start_dt + timedelta(minutes=duration_min)
    return {
        "sport_type": "sport",
        "sport_name": sport_name,
        "start_time": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_time": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "intensity": target_intensity_label,
        "statistics": {
            "calories": 0.0,
            "azm": 0,
        },
        "user_note": "",
    }


def _ensure_datetime_col(df: pd.DataFrame, col: str) -> None:
    if df is None or df.empty:
        return
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")


def _apply_event_to_tables_and_derive_stats(
    tables: Dict[str, pd.DataFrame],
    event: Dict[str, Any],
    rng: random.Random,
) -> None:
    start = pd.to_datetime(event["start_time"])
    end = pd.to_datetime(event["end_time"])

    steps_df = tables.get("minute_steps")
    cal_df = tables.get("minute_calories")
    mets_df = tables.get("minute_mets")
    int_df = tables.get("minute_intensities")

    for df in (steps_df, cal_df, mets_df, int_df):
        if df is not None and not df.empty and "ActivityMinute" in df.columns:
            _ensure_datetime_col(df, "ActivityMinute")

    base_df = None
    for df in (int_df, steps_df, cal_df, mets_df):
        if df is not None and not df.empty:
            base_df = df
            break
    if base_df is None or base_df.empty or "ActivityMinute" not in base_df.columns:
        return

    base_mask = (base_df["ActivityMinute"] >= start) & (base_df["ActivityMinute"] < end)
    if not base_mask.any():
        return

    idx = base_df.index[base_mask]
    n = len(idx)
    if n <= 0:
        return

    target_label = event.get("intensity", "light")
    minute_ints = _sample_minute_intensity_vector(target_label, n, rng)

    # 1) write intensity first
    if int_df is not None and not int_df.empty and "Intensity" in int_df.columns and "ActivityMinute" in int_df.columns:
        int_mask = (int_df["ActivityMinute"] >= start) & (int_df["ActivityMinute"] < end)
        int_idx = int_df.index[int_mask]
        if len(int_idx) == n:
            int_df.loc[int_idx, "Intensity"] = minute_ints
        else:
            minute_ints_intdf = _sample_minute_intensity_vector(target_label, len(int_idx), rng)
            int_df.loc[int_idx, "Intensity"] = minute_ints_intdf

    # 2) generate per-minute values conditioned on intensity
    def _fill_by_int_vector(df: Optional[pd.DataFrame], col: str, noise_low: float, noise_high: float) -> float:
        if df is None or df.empty or col not in df.columns or "ActivityMinute" not in df.columns:
            return 0.0
        mask = (df["ActivityMinute"] >= start) & (df["ActivityMinute"] < end)
        if not mask.any():
            return 0.0
        sub_idx = df.index[mask]
        m = len(sub_idx)
        if m <= 0:
            return 0.0

        ints_vec = minute_ints if m == n else _sample_minute_intensity_vector(target_label, m, rng)

        vals: List[float] = []
        for iv in ints_vec:
            level = INT_VALUE_TO_LEVEL.get(int(iv), SEDENTARY_LEVEL)
            if col == "Steps":
                lo, hi = level["steps_per_min"]
            elif col == "METs":
                lo, hi = level["mets"]
            elif col == "Calories":
                lo, hi = level["cal_per_min"]
            else:
                return 0.0

            # 在 _fill_by_int_vector 里
            base = rng.uniform(lo, hi)
            noise = rng.uniform(noise_low, noise_high)
            val = base + noise

            if col in ("Steps", "METs"):
                vals.append(int(round(max(0.0, val))))
            elif col == "Calories":
                vals.append(round(max(0.0, val), 2))

        df.loc[sub_idx, col] = pd.Series(vals, index=sub_idx)
        return float(pd.Series(vals).sum())

    _fill_by_int_vector(steps_df, "Steps", noise_low=-8.0, noise_high=8.0)
    total_cal = _fill_by_int_vector(cal_df, "Calories", noise_low=-0.5, noise_high=0.5)
    _fill_by_int_vector(mets_df, "METs", noise_low=-0.3, noise_high=0.3)

    # 3) derive AZM
    if int_df is not None and not int_df.empty and "Intensity" in int_df.columns and "ActivityMinute" in int_df.columns:
        mask = (int_df["ActivityMinute"] >= start) & (int_df["ActivityMinute"] < end)
        azm = int((int_df.loc[mask, "Intensity"] >= 2).sum()) if mask.any() else int(sum(v >= 2 for v in minute_ints))
    else:
        azm = int(sum(v >= 2 for v in minute_ints))

    # derive calories
    if cal_df is not None and not cal_df.empty and "Calories" in cal_df.columns and "ActivityMinute" in cal_df.columns:
        mask = (cal_df["ActivityMinute"] >= start) & (cal_df["ActivityMinute"] < end)
        calories = float(cal_df.loc[mask, "Calories"].sum()) if mask.any() else float(total_cal)
    else:
        tmp = 0.0
        for iv in minute_ints:
            lvl = INT_VALUE_TO_LEVEL.get(int(iv), SEDENTARY_LEVEL)
            lo, hi = lvl["cal_per_min"]
            tmp += rng.uniform(lo, hi)
        calories = float(tmp)

    event["statistics"]["azm"] = azm
    event["statistics"]["calories"] = round(calories, 2)


# --------------------------------------------------------------------------- #
# Public API (interface unchanged)
# --------------------------------------------------------------------------- #

def inject_sports(
    profile: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    date_range: Tuple[str, str],
    rng: random.Random,
):
    """
    Generate sports events and modify wearable tables in-place.
    Returns (sports_events, tables).

    Sampling:
    1) Choose ONE habit template for the whole range
    2) Natural-week aligned: sample exact training days per week (within days_per_week)
    3) For each training day: choose ONE day pattern (explicit list of event windows)
    4) For each event window: sample duration, sample sport_name/intensity, sample start inside that window (no jitter)
    5) Avoid overlaps via occupied intervals
    6) Apply minute-level intensity perturbation first, then generate steps/METs/calories per-minute;
       AZM derived from minute_intensity >= 2
    """
    if not tables:
        return [], tables

    template = _choose_template(rng)
    training_days = _sample_training_days_by_week(date_range, template, rng)

    events: List[Dict[str, Any]] = []

    for day in training_days:
        pattern: DayPattern = _weighted_choice_obj(template.day_pattern_mix, rng)

        occupied: List[Tuple[int, int]] = []
        day_events: List[Dict[str, Any]] = []

        for ew in pattern.events:
            duration = rng.randint(template.duration_minutes[0], template.duration_minutes[1])

            start_minute = _find_start_in_window_non_overlapping(
                ew.window,
                duration,
                occupied,
                rng,
            )
            if start_minute is None:
                continue

            target_intensity = _weighted_choice(template.intensity_mix, rng)
            sport_name = _weighted_choice(template.sports_mix, rng)

            ev = _build_event_base(day, start_minute, duration, target_intensity, sport_name)
            day_events.append(ev)

            occupied.append((start_minute, start_minute + duration))
            occupied.sort(key=lambda x: x[0])

        day_events.sort(key=lambda e: e["start_time"])

        for ev in day_events:
            _apply_event_to_tables_and_derive_stats(tables, ev, rng)
            events.append(ev)

    events.sort(key=lambda e: e["start_time"])
    return events, tables, template.to_dict()
