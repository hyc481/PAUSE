from __future__ import annotations

from typing import Dict, Tuple, List
from datetime import datetime, timedelta
from collections import Counter
import math
import random


def generate_source_assignment(
    date_range: Tuple[str, str],
    rng: random.Random,
) -> Dict[str, str]:
    """
    Generate day-level source_assignment for the given date_range (inclusive).

    Semantics (per your latest definition):
    - source_assignment is NOT visible to assistant agent (backend truth).
    - "missing" means the user wore NO device that day (no data/events).
    - Non-missing values are source names (brands/platform-like strings).

    Constraints:
    - Internally sample 1/2/3 sources (no external 'sources' arg).
    - 1 source: constant + missing blocks.
    - 2 sources: either mid-switch OR middle-block; each source shouldn't be too short.
    - 3 sources: split into 3 contiguous segments; each segment shouldn't be too short.
    - Insert missing blocks: 1~2 blocks, each 1~3 days, contiguous.
    """

    # -------- source keyword pool (looks like common brands) --------
    SOURCE_POOL: List[str] = [
        "fitbit",
        "google_fit",
        "apple_health",
        "huawei_health",
        "samsung_tracking",
        "garmin_connect",
        "oura",
        "xiaomi_mi_fitness",
        "polar_flow",
        "withings",
    ]

    start_s, end_s = date_range
    start_dt = datetime.fromisoformat(start_s)
    end_dt = datetime.fromisoformat(end_s)
    if end_dt < start_dt:
        raise ValueError(f"Invalid date_range: {date_range}")

    n_days = (end_dt.date() - start_dt.date()).days + 1
    if n_days <= 0:
        raise ValueError(f"Invalid n_days from date_range: {date_range}")

    dates = [(start_dt.date() + timedelta(days=i)).isoformat() for i in range(n_days)]

    # -------- helper: choose k distinct sources from pool --------
    def _sample_sources(k: int) -> List[str]:
        if k > len(SOURCE_POOL):
            raise ValueError("SOURCE_POOL too small")
        # stable sampling under rng
        pool = SOURCE_POOL[:]
        rng.shuffle(pool)
        return pool[:k]

    # -------- helper: min length constraint (not obsessed with week) --------
    def _min_len(num_sources: int) -> int:
        """
        "不要太短" 的简单规则（可调）：
        - 2 sources: each >= ceil(0.20 * n_days)
        - 3 sources: each segment >= ceil(0.18 * n_days)
        同时给一个下限，避免小范围过严。
        """
        if num_sources == 1:
            return 1
        if num_sources == 2:
            return max(4, int(math.ceil(0.20 * n_days)))
        if num_sources == 3:
            return max(3, int(math.ceil(0.18 * n_days)))
        raise ValueError("num_sources must be 1/2/3")

    # -------- 0) sample how many sources (prefer 1/2, sometimes 3) --------
    if n_days < 9:
        num_sources = 1 if rng.random() < 0.7 else 2
    else:
        p = rng.random()
        if p < 0.3:
            num_sources = 1
        elif p < 0.7:
            num_sources = 2
        else:
            num_sources = 3

    minlen = _min_len(num_sources)

    # -------- 1) build base assignment without missing --------
    arr: List[str] = [""] * n_days  # day -> source/missing later

    if num_sources == 1:
        s1 = _sample_sources(1)[0]
        for i in range(n_days):
            arr[i] = s1

    elif num_sources == 2:
        s1, s2 = _sample_sources(2)
        mode = rng.choice(["mid_switch", "middle_block"])

        if mode == "mid_switch":
            # choose switch index so both sides >= minlen
            lo = minlen
            hi = n_days - minlen
            if lo >= hi:
                # fallback: relax
                lo = max(1, n_days // 3)
                hi = max(lo + 1, n_days - max(1, n_days // 3))
            switch = rng.randint(lo, hi)

            for i in range(0, switch):
                arr[i] = s1
            for i in range(switch, n_days):
                arr[i] = s2

        else:  # middle_block
            # pick dominant and other
            dominant, other = (s1, s2) if rng.random() < 0.5 else (s2, s1)
            for i in range(n_days):
                arr[i] = dominant

            # choose block length so both sources not too short
            max_block = n_days - minlen
            block_len_lo = minlen
            block_len_hi = max(block_len_lo, max_block)
            block_len = rng.randint(block_len_lo, block_len_hi)

            start_max = n_days - block_len
            start_idx = rng.randint(0, start_max)
            for i in range(start_idx, start_idx + block_len):
                arr[i] = other

    else:  # num_sources == 3
        s1, s2, s3 = _sample_sources(3)

        # split into 3 segments: len1, len2, len3; each >= minlen
        # choose len1, len2 then derive len3
        # ensure feasibility
        if 3 * minlen > n_days:
            # relax if needed
            minlen_relaxed = max(1, n_days // 4)
            minlen = minlen_relaxed

        len1 = rng.randint(minlen, n_days - 2 * minlen)
        len2 = rng.randint(minlen, n_days - len1 - minlen)
        len3 = n_days - len1 - len2
        # (len3 guaranteed >= minlen by construction)

        for i in range(0, len1):
            arr[i] = s1
        for i in range(len1, len1 + len2):
            arr[i] = s2
        for i in range(len1 + len2, n_days):
            arr[i] = s3

    # -------- 2) insert missing blocks (1~2 blocks, each 1~3 days) --------
    # Goal: always try to insert 1~2 blocks, but avoid making any real source too short.
    real_sources = [x for x in set(arr) if x != "missing"]
    counts = Counter(arr)

    # for "too short" protection: for 2/3 sources enforce minlen; for 1 source no strict need
    protect_min = minlen if num_sources in (2, 3) else 0

    def _try_place_missing_block(block_len: int) -> bool:
        nonlocal counts, arr
        # attempt many times to find a valid contiguous block
        for _ in range(300):
            start_idx = rng.randint(0, n_days - block_len)
            idxs = list(range(start_idx, start_idx + block_len))

            # skip if already missing (we keep blocks disjoint-ish)
            if any(arr[i] == "missing" for i in idxs):
                continue

            # simulate decrement counts
            dec = Counter()
            ok = True
            for i in idxs:
                src = arr[i]
                if src == "missing":
                    continue
                dec[src] += 1

            if protect_min > 0:
                for src, d in dec.items():
                    if (counts[src] - d) < protect_min:
                        ok = False
                        break

            if not ok:
                continue

            # apply
            for i in idxs:
                src = arr[i]
                if src != "missing":
                    counts[src] -= 1
                arr[i] = "missing"
            counts["missing"] += block_len
            return True

        return False

    # decide number of missing blocks
    if n_days <= 6:
        n_blocks = 1
    else:
        n_blocks = 1 if rng.random() < 0.6 else 2

    # place each block (len 1~3)
    for _b in range(n_blocks):
        blen = rng.randint(1, 3)
        placed = _try_place_missing_block(blen)
        if not placed:
            # relax protection a bit and try again (still prefer not breaking too much)
            saved = protect_min
            protect_min = max(0, protect_min - 1)
            placed2 = _try_place_missing_block(blen)
            protect_min = saved

            # still fail: force place onto currently most frequent source (but keep contiguity)
            if not placed2:
                # pick a window that hits the most frequent source
                # (this may violate protect_min in edge cases, but guarantees missing exists)
                major = None
                major_cnt = -1
                for s in real_sources:
                    if counts[s] > major_cnt:
                        major_cnt = counts[s]
                        major = s

                # find a block of length blen in major
                forced = False
                for i in range(0, n_days - blen + 1):
                    if all(arr[j] == major for j in range(i, i + blen)):
                        for j in range(i, i + blen):
                            counts[arr[j]] -= 1
                            arr[j] = "missing"
                        counts["missing"] += blen
                        forced = True
                        break

                if not forced and n_days - blen >= 0:
                    # last resort: just overwrite a random block
                    i = rng.randint(0, n_days - blen)
                    for j in range(i, i + blen):
                        if arr[j] != "missing":
                            counts[arr[j]] -= 1
                        arr[j] = "missing"
                    counts["missing"] += blen

    # -------- 3) pack to dict --------
    return {dates[i]: arr[i] for i in range(n_days)}


# -------------------------
# Example integration inside build_store():
# (place after meta["date_range"] is ready, and after profile is loaded)
# profile["source_assignment"] = generate_source_assignment(
#     date_range=meta["date_range"],
#     rng=rng,
# )
# -------------------------
