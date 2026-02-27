#!/usr/bin/env python3
"""
Analyze task files with reference answers and compute statistics for the
selected trajectory: tool call counts and dialogue rounds.

Usage:
    python analyze_selected_traj_stats.py --tasks_dir <tasks_directory>
"""
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

def analyze_traj_stats(
    agent_traj: Optional[List[Dict[str, Any]]] = None,
    user_traj: Optional[List[Dict[str, Any]]] = None,
    full_messages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract statistics from trajectory data: tool call counts and dialogue
    rounds.

    Args:
        agent_traj: list of assistant trajectory messages
        user_traj: list of user trajectory messages
        full_messages: list of serialized full_messages strings

    Returns:
        A dict with the following fields:
        - total_tool_calls: total number of tool calls
        - assistant_tool_calls: number of assistant tool calls
        - user_tool_calls: number of user tool calls
        - rounds: dialogue rounds (outer-loop iterations; number of assistant
          replies without tool calls, excluding the first greeting)
        - assistant_tool_calls_by_name: per-tool counts for assistant
        - user_tool_calls_by_name: per-tool counts for user
    """
    stats = {
        "total_tool_calls": 0,
        "assistant_tool_calls": 0,
        "user_tool_calls": 0,
        "rounds": 0,
        "assistant_tool_calls_by_name": {},
        "user_tool_calls_by_name": {},
    }
    
    # Method 1: count from agent_traj and user_traj (more accurate)
    if agent_traj:
        assistant_rounds = 0
        first_assistant_without_tools = True  # mark first assistant message without tools (usually greeting)
        
        for msg in agent_traj:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            tool_calls = msg.get("tool_calls", [])
            
            # Count tool calls
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_name = func.get("name", "") if isinstance(func, dict) else ""
                    else:
                        func = getattr(tc, "function", None)
                        tool_name = getattr(func, "name", "") if func else ""
                    
                    if tool_name:
                        stats["assistant_tool_calls"] += 1
                        stats["total_tool_calls"] += 1
                        stats["assistant_tool_calls_by_name"][tool_name] = \
                            stats["assistant_tool_calls_by_name"].get(tool_name, 0) + 1
            
            # Count rounds: assistant final replies (content present and no tool_calls).
            # At the end of each round the assistant has a reply without tool_calls.
            # The first such reply is usually a greeting and should be excluded.
            if role == "assistant":
                content = msg.get("content", "") or ""
                if content and not tool_calls:
                    if first_assistant_without_tools:
                        # 跳过第一个（greeting）
                        first_assistant_without_tools = False
                    else:
                        assistant_rounds += 1
        
        stats["rounds"] = assistant_rounds
    
    if user_traj:
        for msg in user_traj:
            if not isinstance(msg, dict):
                continue
            tool_calls = msg.get("tool_calls", [])
            
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_name = func.get("name", "") if isinstance(func, dict) else ""
                    else:
                        func = getattr(tc, "function", None)
                        tool_name = getattr(func, "name", "") if func else ""
                    
                    if tool_name:
                        stats["user_tool_calls"] += 1
                        stats["total_tool_calls"] += 1
                        stats["user_tool_calls_by_name"][tool_name] = \
                            stats["user_tool_calls_by_name"].get(tool_name, 0) + 1
    
    # Method 2: if Method 1 did not produce rounds, fall back to full_messages.
    if stats["rounds"] == 0 and full_messages:
        # Count messages starting with [ASSISTANT] (excluding first greeting and tool calls).
        # At the end of each round the assistant has a message starting with [ASSISTANT]
        # that is not an [ASSISTANT TOOL CALL].
        assistant_count = 0
        for msg in full_messages:
            if isinstance(msg, str):
                msg_stripped = msg.strip()
                if msg_stripped.startswith("[ASSISTANT]") and not msg_stripped.startswith("[ASSISTANT TOOL CALL]"):
                    assistant_count += 1
        
        # Exclude the first greeting ([ASSISTANT] message).
        # rounds = outer-loop iterations = number of assistant replies without
        # tools, excluding the greeting.
        if assistant_count > 0:
            stats["rounds"] = assistant_count - 1 if assistant_count > 1 else 0
    
    # Method 3: if Method 1 did not see tool calls, fall back to full_messages.
    if stats["total_tool_calls"] == 0 and full_messages:
        for msg in full_messages:
            if isinstance(msg, str):
                msg_stripped = msg.strip()
                if msg_stripped.startswith("[ASSISTANT TOOL CALL]"):
                    stats["assistant_tool_calls"] += 1
                    stats["total_tool_calls"] += 1
                    # 尝试提取工具名
                    match = re.search(r'\[ASSISTANT TOOL CALL\]\s+(\w+)', msg)
                    if match:
                        tool_name = match.group(1)
                        stats["assistant_tool_calls_by_name"][tool_name] = \
                            stats["assistant_tool_calls_by_name"].get(tool_name, 0) + 1
                elif msg_stripped.startswith("[USER TOOL CALL]"):
                    stats["user_tool_calls"] += 1
                    stats["total_tool_calls"] += 1
                    # 尝试提取工具名
                    match = re.search(r'\[USER TOOL CALL\]\s+(\w+)', msg)
                    if match:
                        tool_name = match.group(1)
                        stats["user_tool_calls_by_name"][tool_name] = \
                            stats["user_tool_calls_by_name"].get(tool_name, 0) + 1
    
    return stats


def find_selected_traj(annotated_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find the final selected trajectory from an annotated log payload.

    Args:
        annotated_payload: JSON content loaded from an annotated log file

    Returns:
        The selected trajectory dict, or None if not found.
    """
    final_sel = annotated_payload.get("final_selection", {})
    if not isinstance(final_sel, dict):
        return None
    
    selected_model = str(final_sel.get("select_assistant_model", "")).strip()
    if not selected_model:
        return None
    
    # 先检查 rerun_trajs，再检查 trajs
    pools = [
        annotated_payload.get("rerun_trajs", []),
        annotated_payload.get("trajs", []),
    ]
    
    for pool in pools:
        if not isinstance(pool, list):
            continue
        for traj in pool:
            if not isinstance(traj, dict):
                continue
            traj_model = str(traj.get("assistant_model", "")).strip()
            if traj_model == selected_model:
                return traj
    
    return None


def process_task(
    task: Dict[str, Any],
    task_idx: int = 0,
    tasks_file: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process a single task: locate the selected trajectory and compute stats.

    Args:
        task: task dict
        task_idx: task index
        tasks_file: path to the task file (for error reporting)

    Returns:
        Statistics dict, or None if processing fails.
    """
    annotated_log_path = task.get("annotated_log_path", "")
    if not annotated_log_path:
        print(f"[WARN] Task {task_idx}: No annotated_log_path")
        return None
    
    annotated_path = Path(annotated_log_path).expanduser()
    if not annotated_path.exists():
        print(f"[WARN] Task {task_idx}: Annotated log not found: {annotated_path}")
        return None
    
    try:
        with annotated_path.open("r", encoding="utf-8") as f:
            annotated_payload = json.load(f)
    except Exception as e:
        print(f"[ERROR] Task {task_idx}: Failed to read annotated log {annotated_path}: {e}")
        return None
    
    # Find the final selected trajectory
    selected_traj = find_selected_traj(annotated_payload)
    if selected_traj is None:
        print(f"[WARN] Task {task_idx}: No selected traj found in {annotated_path}")
        return None
    
    # 提取traj数据
    agent_traj = selected_traj.get("assistant_traj", [])
    user_traj = selected_traj.get("user_traj", [])
    full_messages = selected_traj.get("full_messages", [])
    
    # 统计
    stats = analyze_traj_stats(
        agent_traj=agent_traj,
        user_traj=user_traj,
        full_messages=full_messages,
    )
    
    # 添加额外信息
    stats["task_idx"] = task_idx
    stats["task_label"] = task.get("label", "")
    stats["annotated_log_path"] = str(annotated_path)
    stats["assistant_model"] = selected_traj.get("assistant_model", "unknown")
    stats["user_model"] = selected_traj.get("user_model", "unknown")
    stats["terminated"] = selected_traj.get("terminated", False)
    stats["termination_reason"] = selected_traj.get("termination_reason", "")
    
    # 添加final_selection信息
    final_sel = annotated_payload.get("final_selection", {})
    if isinstance(final_sel, dict):
        stats["final_selection_status"] = final_sel.get("status", "")
        stats["final_selection_reason"] = final_sel.get("select_reason", "")
    
    if tasks_file:
        stats["tasks_file"] = str(tasks_file)
    
    return stats


def process_tasks_file(tasks_file: Path) -> List[Dict[str, Any]]:
    """
    Process a single tasks file and return statistics for all tasks.

    Args:
        tasks_file: path to the tasks JSON file

    Returns:
        List of statistics dicts.
    """
    try:
        with tasks_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read tasks file {tasks_file}: {e}")
        return []
    
    # 获取tasks列表
    tasks = payload.get("tasks", [])
    if not tasks:
        print(f"[WARN] {tasks_file}: No tasks found")
        return []
    
    results = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        
        stats = process_task(task, task_idx=idx, tasks_file=tasks_file)
        if stats:
            results.append(stats)
    
    return results


def print_stats_summary(all_stats: List[Dict[str, Any]]):
    """Print summary statistics over all finally selected trajectories."""
    if not all_stats:
        print("No statistics to display.")
        return
    
    print("\n" + "="*80)
    print("Summary (all finally selected trajectories)")
    print("="*80)
    print(f"Total tasks: {len(all_stats)}")
    
    # Compute overall averages
    total_tool_calls = [s["total_tool_calls"] for s in all_stats]
    assistant_tool_calls = [s["assistant_tool_calls"] for s in all_stats]
    user_tool_calls = [s["user_tool_calls"] for s in all_stats]
    rounds = [s["rounds"] for s in all_stats]
    
    print(f"\nTotal tool calls: avg={sum(total_tool_calls)/len(total_tool_calls):.2f}, "
          f"min={min(total_tool_calls)}, max={max(total_tool_calls)}, "
          f"total={sum(total_tool_calls)}")
    print(f"Assistant tool calls: avg={sum(assistant_tool_calls)/len(assistant_tool_calls):.2f}, "
          f"min={min(assistant_tool_calls)}, max={max(assistant_tool_calls)}, "
          f"total={sum(assistant_tool_calls)}")
    print(f"User tool calls: avg={sum(user_tool_calls)/len(user_tool_calls):.2f}, "
          f"min={min(user_tool_calls)}, max={max(user_tool_calls)}, "
          f"total={sum(user_tool_calls)}")
    print(f"Dialogue rounds: avg={sum(rounds)/len(rounds):.2f}, "
          f"min={min(rounds)}, max={max(rounds)}, total={sum(rounds)}")
    
    # Tool call distribution (aggregated over all trajectories)
    all_assistant_tools = defaultdict(int)
    all_user_tools = defaultdict(int)
    for s in all_stats:
        for tool, count in s["assistant_tool_calls_by_name"].items():
            all_assistant_tools[tool] += count
        for tool, count in s["user_tool_calls_by_name"].items():
            all_user_tools[tool] += count
    
    if all_assistant_tools:
        print(f"\nAssistant tool call distribution:")
        for tool, count in sorted(all_assistant_tools.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")
    
    if all_user_tools:
        print(f"\nUser tool call distribution:")
        for tool, count in sorted(all_user_tools.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")
    
    # Termination statistics
    terminated_count = sum(1 for s in all_stats if s.get("terminated", False))
    print(f"\nTerminated trajectories: {terminated_count}/{len(all_stats)}")
    
    # Model distribution (informational only, not used in stats)
    model_dist = defaultdict(int)
    for s in all_stats:
        model = s.get("assistant_model", "unknown")
        model_dist[model] += 1
    
    if model_dist:
        print(f"\nModel distribution (for reference only):")
        for model, count in sorted(model_dist.items()):
            print(f"  {model}: {count}")


def print_detailed_stats(all_stats: List[Dict[str, Any]]):
    """Print detailed statistics for each task."""
    print("\n" + "="*80)
    print("Detailed statistics (per task)")
    print("="*80)
    
    for stats in all_stats:
        print(f"\nTask #{stats['task_idx']} - {stats.get('task_label', 'unknown')}")
        print(f"  Tasks file: {Path(stats.get('tasks_file', '')).name}")
        print(f"  Annotated log: {Path(stats.get('annotated_log_path', '')).name}")
        print(f"  Assistant model: {stats['assistant_model']}")
        print(f"  User model: {stats['user_model']}")
        print(f"  Total tool calls: {stats['total_tool_calls']}")
        print(f"    - Assistant: {stats['assistant_tool_calls']}")
        print(f"    - User: {stats['user_tool_calls']}")
        print(f"  Dialogue rounds: {stats['rounds']}")
        print(f"  Terminated: {stats['terminated']} ({stats['termination_reason']})")
        print(f"  Final selection status: {stats.get('final_selection_status', 'unknown')}")
        print(f"  Selection reason: {stats.get('final_selection_reason', '')}")
        
        if stats['assistant_tool_calls_by_name']:
            print(f"  Assistant tool calls:")
            for tool, count in sorted(stats['assistant_tool_calls_by_name'].items(), key=lambda x: -x[1]):
                print(f"    {tool}: {count}")
        
        if stats['user_tool_calls_by_name']:
            print(f"  User tool calls:")
            for tool, count in sorted(stats['user_tool_calls_by_name'].items(), key=lambda x: -x[1]):
                print(f"    {tool}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tasks with reference answers and summarize stats of the selected trajectories."
    )
    parser.add_argument(
        "--tasks_dir",
        type=str,
        required=True,
        help="Directory containing task JSON files (searched recursively)."
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statistics per task."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save statistics to a JSON file."
    )
    
    args = parser.parse_args()
    
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"[ERROR] Directory does not exist: {tasks_dir}")
        return
    
    # Find all task files
    json_files = sorted(tasks_dir.rglob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found under {tasks_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    all_stats = []
    for json_file in json_files:
        try:
            stats_list = process_tasks_file(json_file)
            all_stats.extend(stats_list)
            print(f"[OK] {json_file.name}: processed {len(stats_list)} tasks")
        except Exception as e:
            print(f"[ERROR] Failed to process file {json_file}: {e}")
    
    if not all_stats:
        print("No valid task statistics found.")
        return
    
    # Print summary
    print_stats_summary(all_stats)
    
    # Print detailed statistics if requested
    if args.detailed:
        print_detailed_stats(all_stats)
    
    # Save to file if requested
    if args.output:
        output_file = Path(args.output)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {output_file}")


if __name__ == "__main__":
    main()

