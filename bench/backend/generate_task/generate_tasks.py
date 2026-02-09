"""
Entry point for generating multiple tasks for a single user.

Workflow:
1) Build the user store (profile + wearable tables) via build_store.
2) Run TaskOrchestrator across configured branches.
3) Persist generated tasks (and lightweight context) to JSON.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Union, Dict

from bench.backend.store.build_store import build_store
from bench.backend.generate_task.orchestrator import TaskOrchestrator
# from bench.backend.generate_task.branches.wearable_data_casual import WearableDataCasual
# from bench.backend.generate_task.branches.wearable_data_advanced import WearableDataAdvanced
# from bench.backend.generate_task.branches.lifestyle_record_casual import LifeStyleCasual
# from bench.backend.generate_task.branches.lifestyle_record_advanced import LifeStyleAdvanced
from bench.backend.generate_task.branches.shopping import ShoppingBranch
from bench.backend.utils.clients import get_gen_route


def build_branches():
    """
    Assemble all task branches to run for a given model.
    Extend this list as more branches are added.
    """
    client_obj, model = get_gen_route()
    return [
        ShoppingBranch(client=client_obj, model=model),
    ]


def resolve_output_path(profile_id: Optional[str], output_dir: Optional[str | Path]) -> Path:
    """
    Decide where to save task JSON.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_tag = profile_id or "default"
    base = Path(output_dir) if output_dir else Path(__file__).resolve().parents[3] / "bench" / "runs"/ "debugging" / "tasks"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"tasks_{profile_tag}_{ts}.json"


def generate_tasks(
    profile_id: Optional[str],
    user_id: str,
    date_range: Tuple[str, str],
    data_format: str,
    rng_seed: Optional[int],
    target_year_start: int,
    target_year_end: int,
    snapshot_dir: str,
    output_path: Optional[str | Path] = None,
    runs_per_branch: Union[int, Dict[str, int]] = 5,
):
    """
    Build store, run branches, and save tasks.
    """
    store = build_store(
        user_id=user_id,
        date_range=date_range,
        data_format=data_format,
        enable_time_shift=True,
        target_year_start=target_year_start,
        target_year_end=target_year_end,
        rng_seed=rng_seed,
        snapshot_dir=snapshot_dir,
        profile_id=profile_id,
    )

    branches = build_branches()
    resolved_output = resolve_output_path(profile_id, output_path)

    orchestrator = TaskOrchestrator(
        store=store,
        branches=branches,
        save_path=resolved_output,
        runs_per_branch=runs_per_branch,
    )
    result = orchestrator.run()

    print(f"[OK] Generated {len(orchestrator.tasks)} task(s). Saved to: {resolved_output}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Generate tasks for a single user.")
    parser.add_argument(
        "--profile_id",
        default="developing",
        help="Decide the name of generated files. Set to debug_profile for debug initialization, otherwise empty initialization is used.",
    )
    parser.add_argument(
        "--user_num",
        type=int,
        default=30,
        help="Number of users to generate.",
    )
    parser.add_argument(
        "--runs_per_branch",
        type=int,
        default=1,
        help="How many times to run each branch (uniform setting).",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/chy/state_aware_bench/bench/runs/testrun5_inference2/tasks",
        help="Output directory for generated files. Defaults to bench/runs/debugging/tasks relative to repo.",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument("--user_id", default=None, help="User ID for wearable data.")
    parser.add_argument(
        "--start_date",
        default="2016-04-01",
        help="Start date (YYYY-MM-DD) for source data window.",
    )
    parser.add_argument(
        "--end_date",
        default="2016-04-30",
        help="End date (YYYY-MM-DD) for source data window.",
    )
    parser.add_argument(
        "--data_format",
        default="csv",
        choices=["csv", "parquet"],
        help="File format for wearable data.",
    )
    parser.add_argument(
        "--target_year_start",
        type=int,
        default=2024,
        help="Earliest year for time-shifted data.",
    )
    parser.add_argument(
        "--target_year_end",
        type=int,
        default=2026,
        help="Latest year for time-shifted data.",
    )
    parser.add_argument(
        "--snapshot_dir",
        default="/home/chy/state_aware_bench/bench/snapshots/debug_snapshot",
        help="Where to save store snapshots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    user_id_ls = ["2026352035", "2347167796", "4558609924", "5553957443", "5577150313", "6117666160", "6775888955",
                  "6962181067", "7007744171", "8792009665"]
    args = parse_args()
    for turn in range(args.user_num):
        if args.user_id is None:
            user_id = random.choice(user_id_ls)
        else:
            user_id = args.user_id

        if args.rng_seed is None:
            rng_seed = random.randint(1, 100000)
        else:
            rng_seed = args.rng_seed

        generate_tasks(
            profile_id=args.profile_id,
            user_id=user_id,
            date_range=(args.start_date, args.end_date),
            data_format=args.data_format,
            rng_seed=rng_seed,
            target_year_start=args.target_year_start,
            target_year_end=args.target_year_end,
            snapshot_dir=args.snapshot_dir,
            output_path=args.output_dir,
            runs_per_branch=args.runs_per_branch,
        )

