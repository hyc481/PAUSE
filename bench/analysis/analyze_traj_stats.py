#!/usr/bin/env python3
"""
Analyze saved trajectory logs and extract statistics on tool call counts
and dialogue rounds.

Usage:
    python analyze_traj_stats.py --log_file <log_file.json>
    or
    python analyze_traj_stats.py --logs_dir <logs_directory>
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


def process_single_traj(traj: Dict[str, Any], traj_idx: int = 0) -> Dict[str, Any]:
    """处理单个轨迹并返回统计信息"""
    agent_traj = traj.get("assistant_traj", [])
    user_traj = traj.get("user_traj", [])
    full_messages = traj.get("full_messages", [])
    
    stats = analyze_traj_stats(
        agent_traj=agent_traj,
        user_traj=user_traj,
        full_messages=full_messages,
    )
    
    # 添加额外信息
    stats["traj_idx"] = traj_idx
    stats["assistant_model"] = traj.get("assistant_model", "unknown")
    stats["user_model"] = traj.get("user_model", "unknown")
    stats["terminated"] = traj.get("terminated", False)
    stats["termination_reason"] = traj.get("termination_reason", "")
    
    return stats


def process_log_file(log_file: Path) -> List[Dict[str, Any]]:
    """处理单个日志文件，返回所有轨迹的统计信息"""
    with log_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    
    trajs = payload.get("trajs", [])
    if not trajs:
        print(f"[WARN] {log_file}: No trajs found")
        return []
    
    results = []
    for idx, traj in enumerate(trajs):
        try:
            stats = process_single_traj(traj, traj_idx=idx)
            stats["log_file"] = str(log_file)
            results.append(stats)
        except Exception as e:
            print(f"[ERROR] {log_file} traj[{idx}]: {e}")
    
    return results


def print_stats_summary(all_stats: List[Dict[str, Any]], group_by_model: bool = True):
    """打印统计摘要"""
    if not all_stats:
        print("No statistics to display.")
        return
    
    if group_by_model:
        # 按模型分组统计
        by_model = defaultdict(list)
        for stats in all_stats:
            model = stats.get("assistant_model", "unknown")
            by_model[model].append(stats)
        
        print("\n" + "="*80)
        print("统计摘要 (按Assistant模型分组)")
        print("="*80)
        
        for model, model_stats in sorted(by_model.items()):
            print(f"\n[{model}] (共 {len(model_stats)} 个轨迹)")
            
            # 计算平均值
            total_tool_calls = [s["total_tool_calls"] for s in model_stats]
            assistant_tool_calls = [s["assistant_tool_calls"] for s in model_stats]
            user_tool_calls = [s["user_tool_calls"] for s in model_stats]
            rounds = [s["rounds"] for s in model_stats]
            
            print(f"  总工具调用次数: avg={sum(total_tool_calls)/len(total_tool_calls):.2f}, "
                  f"min={min(total_tool_calls)}, max={max(total_tool_calls)}, "
                  f"total={sum(total_tool_calls)}")
            print(f"  Assistant工具调用: avg={sum(assistant_tool_calls)/len(assistant_tool_calls):.2f}, "
                  f"min={min(assistant_tool_calls)}, max={max(assistant_tool_calls)}, "
                  f"total={sum(assistant_tool_calls)}")
            print(f"  User工具调用: avg={sum(user_tool_calls)/len(user_tool_calls):.2f}, "
                  f"min={min(user_tool_calls)}, max={max(user_tool_calls)}, "
                  f"total={sum(user_tool_calls)}")
            print(f"  对话轮数(rounds): avg={sum(rounds)/len(rounds):.2f}, "
                  f"min={min(rounds)}, max={max(rounds)}, total={sum(rounds)}")
            
            # 工具调用分布
            all_assistant_tools = defaultdict(int)
            all_user_tools = defaultdict(int)
            for s in model_stats:
                for tool, count in s["assistant_tool_calls_by_name"].items():
                    all_assistant_tools[tool] += count
                for tool, count in s["user_tool_calls_by_name"].items():
                    all_user_tools[tool] += count
            
            if all_assistant_tools:
                print(f"  Assistant工具调用分布:")
                for tool, count in sorted(all_assistant_tools.items(), key=lambda x: -x[1]):
                    print(f"    {tool}: {count}")
            
            if all_user_tools:
                print(f"  User工具调用分布:")
                for tool, count in sorted(all_user_tools.items(), key=lambda x: -x[1]):
                    print(f"    {tool}: {count}")
            
            # 终止统计
            terminated_count = sum(1 for s in model_stats if s.get("terminated", False))
            print(f"  终止轨迹数: {terminated_count}/{len(model_stats)}")
    
    # 总体统计
    print("\n" + "="*80)
    print("总体统计")
    print("="*80)
    print(f"总轨迹数: {len(all_stats)}")
    total_tool_calls = [s["total_tool_calls"] for s in all_stats]
    assistant_tool_calls = [s["assistant_tool_calls"] for s in all_stats]
    user_tool_calls = [s["user_tool_calls"] for s in all_stats]
    rounds = [s["rounds"] for s in all_stats]
    
    print(f"总工具调用次数: avg={sum(total_tool_calls)/len(total_tool_calls):.2f}, "
          f"min={min(total_tool_calls)}, max={max(total_tool_calls)}, "
          f"total={sum(total_tool_calls)}")
    print(f"Assistant工具调用: avg={sum(assistant_tool_calls)/len(assistant_tool_calls):.2f}, "
          f"min={min(assistant_tool_calls)}, max={max(assistant_tool_calls)}, "
          f"total={sum(assistant_tool_calls)}")
    print(f"User工具调用: avg={sum(user_tool_calls)/len(user_tool_calls):.2f}, "
          f"min={min(user_tool_calls)}, max={max(user_tool_calls)}, "
          f"total={sum(user_tool_calls)}")
    print(f"对话轮数(rounds): avg={sum(rounds)/len(rounds):.2f}, "
          f"min={min(rounds)}, max={max(rounds)}, total={sum(rounds)}")


def print_detailed_stats(all_stats: List[Dict[str, Any]]):
    """打印每个轨迹的详细统计"""
    print("\n" + "="*80)
    print("详细统计 (每个轨迹)")
    print("="*80)
    
    for stats in all_stats:
        print(f"\n轨迹 #{stats['traj_idx']} (来自 {Path(stats['log_file']).name})")
        print(f"  Assistant模型: {stats['assistant_model']}")
        print(f"  User模型: {stats['user_model']}")
        print(f"  总工具调用: {stats['total_tool_calls']}")
        print(f"    - Assistant: {stats['assistant_tool_calls']}")
        print(f"    - User: {stats['user_tool_calls']}")
        print(f"  对话轮数(rounds): {stats['rounds']}")
        print(f"  终止: {stats['terminated']} ({stats['termination_reason']})")
        
        if stats['assistant_tool_calls_by_name']:
            print(f"  Assistant工具调用:")
            for tool, count in sorted(stats['assistant_tool_calls_by_name'].items(), key=lambda x: -x[1]):
                print(f"    {tool}: {count}")
        
        if stats['user_tool_calls_by_name']:
            print(f"  User工具调用:")
            for tool, count in sorted(stats['user_tool_calls_by_name'].items(), key=lambda x: -x[1]):
                print(f"    {tool}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="分析已保存的轨迹日志，提取工具调用次数和对话轮数统计信息"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="单个日志文件路径"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        help="日志目录路径（会递归查找所有.json文件）"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="显示每个轨迹的详细统计"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="将统计结果保存到JSON文件"
    )
    
    args = parser.parse_args()
    
    if not args.log_file and not args.logs_dir:
        parser.error("Must specify either --log_file or --logs_dir.")
    
    all_stats = []
    
    if args.log_file:
        log_file = Path(args.log_file)
        if not log_file.exists():
            print(f"[ERROR] File does not exist: {log_file}")
            return
        stats_list = process_log_file(log_file)
        all_stats.extend(stats_list)
    
    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
        if not logs_dir.exists():
            print(f"[ERROR] Directory does not exist: {logs_dir}")
            return
        
        json_files = sorted(logs_dir.rglob("*.json"))
        if not json_files:
            print(f"[WARN] No JSON files found under {logs_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                stats_list = process_log_file(json_file)
                all_stats.extend(stats_list)
            except Exception as e:
                print(f"[ERROR] Failed to process file {json_file}: {e}")
    
    if not all_stats:
        print("No trajectory data found.")
        return
    
    # Print summary
    print_stats_summary(all_stats, group_by_model=True)
    
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

