#!/usr/bin/env python3
"""
分析已保存的轨迹日志，提取工具调用次数和对话轮数(rounds)统计信息。

用法:
    python analyze_traj_stats.py --log_file <log_file.json>
    或
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
    从保存的轨迹数据中提取统计信息：工具调用次数和对话轮数(rounds)。
    
    Args:
        agent_traj: Assistant轨迹消息列表
        user_traj: User轨迹消息列表
        full_messages: 完整的消息日志列表
    
    Returns:
        包含以下字段的字典:
        - total_tool_calls: 总工具调用次数
        - assistant_tool_calls: Assistant工具调用次数
        - user_tool_calls: User工具调用次数
        - rounds: 对话轮数（外层循环执行次数，即assistant最终没有工具调用的回复次数，排除第一个greeting）
        - assistant_tool_calls_by_name: Assistant各工具调用次数统计
        - user_tool_calls_by_name: User各工具调用次数统计
    """
    stats = {
        "total_tool_calls": 0,
        "assistant_tool_calls": 0,
        "user_tool_calls": 0,
        "rounds": 0,
        "assistant_tool_calls_by_name": {},
        "user_tool_calls_by_name": {},
    }
    
    # 方法1: 从agent_traj和user_traj统计（更准确）
    if agent_traj:
        assistant_rounds = 0
        first_assistant_without_tools = True  # 标记第一个没有工具调用的assistant消息（通常是greeting）
        
        for msg in agent_traj:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            tool_calls = msg.get("tool_calls", [])
            
            # 统计工具调用
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
            
            # 统计rounds: assistant的最终回复（有content且没有tool_calls）
            # 每一轮（round）结束时，assistant会有一个没有tool_calls的回复
            # 第一个这样的消息通常是greeting，应该排除
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
    
    # 方法2: 如果方法1没有统计到rounds，从full_messages统计（备用方法）
    if stats["rounds"] == 0 and full_messages:
        # 统计[ASSISTANT]开头的消息（排除第一个greeting和工具调用）
        # 每一轮结束时，assistant会有一个[ASSISTANT]开头的消息（不是[ASSISTANT TOOL CALL]）
        assistant_count = 0
        for msg in full_messages:
            if isinstance(msg, str):
                msg_stripped = msg.strip()
                if msg_stripped.startswith("[ASSISTANT]") and not msg_stripped.startswith("[ASSISTANT TOOL CALL]"):
                    assistant_count += 1
        
        # 排除第一个greeting（第一个[ASSISTANT]消息）
        # rounds = 外层循环执行次数 = assistant最终没有工具调用的回复次数（排除greeting）
        if assistant_count > 0:
            stats["rounds"] = assistant_count - 1 if assistant_count > 1 else 0
    
    # 方法3: 如果方法1没有统计到工具调用，从full_messages统计（备用方法）
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
        parser.error("必须指定 --log_file 或 --logs_dir")
    
    all_stats = []
    
    if args.log_file:
        log_file = Path(args.log_file)
        if not log_file.exists():
            print(f"[ERROR] 文件不存在: {log_file}")
            return
        stats_list = process_log_file(log_file)
        all_stats.extend(stats_list)
    
    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
        if not logs_dir.exists():
            print(f"[ERROR] 目录不存在: {logs_dir}")
            return
        
        json_files = sorted(logs_dir.rglob("*.json"))
        if not json_files:
            print(f"[WARN] 在 {logs_dir} 中未找到JSON文件")
            return
        
        print(f"找到 {len(json_files)} 个JSON文件")
        
        for json_file in json_files:
            try:
                stats_list = process_log_file(json_file)
                all_stats.extend(stats_list)
            except Exception as e:
                print(f"[ERROR] 处理文件 {json_file} 时出错: {e}")
    
    if not all_stats:
        print("未找到任何轨迹数据")
        return
    
    # 打印统计摘要
    print_stats_summary(all_stats, group_by_model=True)
    
    # 打印详细统计（如果请求）
    if args.detailed:
        print_detailed_stats(all_stats)
    
    # 保存到文件（如果请求）
    if args.output:
        output_file = Path(args.output)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

