import json
import copy
from pathlib import Path
from typing import Dict, Any

from bench.backend.store.build_store import load_store_from_meta_profile
from bench.backend.utils.agent_runner import AgentRunner
from bench.backend.utils.clients import get_traj_model_routes
from bench.backend.utils.misc import load_branch_tool_allowlist, load_branch_assistant_guidance, group_routes_by_key
from concurrent.futures import ThreadPoolExecutor, as_completed


def _load_tasks_from_file(path: str, branch_name: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tasks_by_branch = payload.get("tasks_by_branch") or {}

    if branch_name != "all":
        if branch_name not in list(tasks_by_branch.keys()):
            raise ValueError(f"Branch '{branch_name}' not found.")
        tasks_by_branch = {branch_name: tasks_by_branch[branch_name]}

    return {
        "meta": payload.get("store_meta", {}),
        "profile": payload.get("profile", {}),
        "tasks_by_branch": tasks_by_branch,
    }


def _rebuild_store(meta: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    store = load_store_from_meta_profile(copy.deepcopy(profile), copy.deepcopy(meta))
    return store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_file", default="/home/chy/state_aware_bench/bench/runs/developing/tasks/tasks_developing_20260113_090755.json")
    parser.add_argument("--tasks_files", default="/home/chy/state_aware_bench/bench/runs/testrun5/tasks", help="If provided, run all task files under this directory (recursive), ignoring --tasks_file.")
    parser.add_argument("--output_dir", default="/home/chy/state_aware_bench/bench/runs/testrun5/inference_logs", help="If provided, output all generated trajs under this directory (recursive)")
    parser.add_argument("--branch", default="all")
    parser.add_argument("--inference", action="store_true", help="If set, assistant can access full platform_tools for inference without assistant_policy; otherwise restricted by branch.")
    # model/client are routed centrally; no CLI overrides needed
    args = parser.parse_args()

    runner = AgentRunner()
    traj_routes = get_traj_model_routes(args.inference)  # list of route dicts (see clients.py)
    if not traj_routes:
        raise ValueError("No traj routes configured.")

    if args.output_dir:
        logs_dir = Path(args.output_dir)
    else:
        logs_dir = Path(__file__).resolve().parent / "generations" / "logs_failed_models"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks_files:
        task_files = sorted(Path(args.tasks_files).rglob("*.json"))
        if not task_files:
            raise ValueError(f"No task files found under {args.tasks_files}")
    else:
        task_files = [Path(args.tasks_file)]

    for task_path in task_files:
        payload = _load_tasks_from_file(str(task_path), args.branch)
        if not payload:
            continue
        tasks_file_stem = task_path.stem
        for idx, task in enumerate(payload["tasks_by_branch"].items()):
            branch_name, branch_tasks = task
            for t_idx, t in enumerate(branch_tasks):
                trajs = []
                allowlist = None if args.inference else load_branch_tool_allowlist(branch_name)
                if (not args.inference) and (not allowlist):
                    raise ValueError(f"Branch {branch_name} does not expose tool allowlist.")
                assistant_policy = None if args.inference else load_branch_assistant_guidance(branch_name)

                # group routes by client key to serialize per client
                groups = group_routes_by_key(traj_routes, "key_assistant")

                def _run_group(key, routes):
                    group_trajs = []
                    for r in routes:
                        local_store = _rebuild_store(payload["meta"], payload["profile"])
                        try:
                            tr = runner.run(
                                store=local_store,
                                task_instruction_text=t["task_instruction"],
                                label=t.get("label", ""),
                                targets=t.get("targets", ""),
                                assistant_tool_allowlist=allowlist,
                                assistant_policy=assistant_policy,
                                traj_route=r,
                            )
                        except Exception as e:
                            tr = runner.build_error_traj(
                                store=local_store,
                                task_instruction_text=t["task_instruction"],
                                label=t.get("label", ""),
                                targets=t.get("targets", ""),
                                traj_route=r,
                                error=e,
                            )
                            print(
                                f"[traj error] branch={branch_name} task_idx={t_idx} "
                                f"assistant_model={tr.get('assistant_model','')} error={e}"
                            )
                        # pretty print per traj (thread-local)
                        print(f"[traj] branch={branch_name} task_idx={t_idx} assistant_model={tr.get('assistant_model','')}")
                        print(f"[traj full_messages]\n{json.dumps(tr.get('full_messages', []), ensure_ascii=False, indent=2)}")
                        group_trajs.append(tr)
                    return group_trajs

                with ThreadPoolExecutor(max_workers=len(groups) or 1) as ex:
                    futures = [ex.submit(_run_group, key, routes) for key, routes in groups.items()]
                    for fut in as_completed(futures):
                        trajs.extend(fut.result())

                save_path = logs_dir / f"{tasks_file_stem}_{branch_name}_{t_idx}.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "meta": payload["meta"],
                        "profile": payload["profile"],
                        "branch": branch_name,
                        "task_instruction": t["task_instruction"],
                        "label": t.get("label", ""),
                        "targets": t.get("targets", []),
                        "trajs": trajs,
                    }, f, indent=2, ensure_ascii=False)

                print(f"\nTrajectory saved to: {save_path}")
