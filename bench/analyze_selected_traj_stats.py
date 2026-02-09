#!/usr/bin/env python3
"""
分析有标准答案的任务文件，统计最终选择的traj的工具调用次数和对话轮数。

用法:
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


def find_selected_traj(annotated_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从annotated log中找到最终选择的traj。
    
    Args:
        annotated_payload: annotated log文件的JSON内容
    
    Returns:
        最终选择的traj字典，如果找不到则返回None
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
    处理单个任务，找到最终选择的traj并统计。
    
    Args:
        task: 任务字典
        task_idx: 任务索引
        tasks_file: 任务文件路径（用于错误报告）
    
    Returns:
        统计结果字典，如果处理失败则返回None
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
    
    # 找到最终选择的traj
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
    处理单个任务文件，返回所有任务的统计结果。
    
    Args:
        tasks_file: 任务文件路径
    
    Returns:
        统计结果列表
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
    """打印统计摘要 - 直接统计所有最终选择的traj的平均值"""
    if not all_stats:
        print("No statistics to display.")
        return
    
    print("\n" + "="*80)
    print("统计摘要 (所有最终选择的traj)")
    print("="*80)
    print(f"总任务数: {len(all_stats)}")
    
    # 计算总体平均值
    total_tool_calls = [s["total_tool_calls"] for s in all_stats]
    assistant_tool_calls = [s["assistant_tool_calls"] for s in all_stats]
    user_tool_calls = [s["user_tool_calls"] for s in all_stats]
    rounds = [s["rounds"] for s in all_stats]
    
    print(f"\n总工具调用次数: avg={sum(total_tool_calls)/len(total_tool_calls):.2f}, "
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
    
    # 工具调用分布（所有traj的汇总）
    all_assistant_tools = defaultdict(int)
    all_user_tools = defaultdict(int)
    for s in all_stats:
        for tool, count in s["assistant_tool_calls_by_name"].items():
            all_assistant_tools[tool] += count
        for tool, count in s["user_tool_calls_by_name"].items():
            all_user_tools[tool] += count
    
    if all_assistant_tools:
        print(f"\nAssistant工具调用分布:")
        for tool, count in sorted(all_assistant_tools.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")
    
    if all_user_tools:
        print(f"\nUser工具调用分布:")
        for tool, count in sorted(all_user_tools.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")
    
    # 终止统计
    terminated_count = sum(1 for s in all_stats if s.get("terminated", False))
    print(f"\n终止轨迹数: {terminated_count}/{len(all_stats)}")
    
    # 模型分布（仅作为信息展示，不用于统计）
    model_dist = defaultdict(int)
    for s in all_stats:
        model = s.get("assistant_model", "unknown")
        model_dist[model] += 1
    
    if model_dist:
        print(f"\n模型分布（仅供参考）:")
        for model, count in sorted(model_dist.items()):
            print(f"  {model}: {count}")


def print_detailed_stats(all_stats: List[Dict[str, Any]]):
    """打印每个任务的详细统计"""
    print("\n" + "="*80)
    print("详细统计 (每个任务)")
    print("="*80)
    
    for stats in all_stats:
        print(f"\n任务 #{stats['task_idx']} - {stats.get('task_label', 'unknown')}")
        print(f"  任务文件: {Path(stats.get('tasks_file', '')).name}")
        print(f"  Annotated log: {Path(stats.get('annotated_log_path', '')).name}")
        print(f"  Assistant模型: {stats['assistant_model']}")
        print(f"  User模型: {stats['user_model']}")
        print(f"  总工具调用: {stats['total_tool_calls']}")
        print(f"    - Assistant: {stats['assistant_tool_calls']}")
        print(f"    - User: {stats['user_tool_calls']}")
        print(f"  对话轮数(rounds): {stats['rounds']}")
        print(f"  终止: {stats['terminated']} ({stats['termination_reason']})")
        print(f"  最终选择状态: {stats.get('final_selection_status', 'unknown')}")
        print(f"  选择原因: {stats.get('final_selection_reason', '')}")
        
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
        description="分析有标准答案的任务文件，统计最终选择的traj的工具调用次数和对话轮数"
    )
    parser.add_argument(
        "--tasks_dir",
        type=str,
        required=True,
        help="任务文件目录路径（会递归查找所有.json文件）"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="显示每个任务的详细统计"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="将统计结果保存到JSON文件"
    )
    
    args = parser.parse_args()
    
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"[ERROR] 目录不存在: {tasks_dir}")
        return
    
    # 查找所有任务文件
    json_files = sorted(tasks_dir.rglob("*.json"))
    if not json_files:
        print(f"[WARN] 在 {tasks_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    all_stats = []
    for json_file in json_files:
        try:
            stats_list = process_tasks_file(json_file)
            all_stats.extend(stats_list)
            print(f"[OK] {json_file.name}: 处理了 {len(stats_list)} 个任务")
        except Exception as e:
            print(f"[ERROR] 处理文件 {json_file} 时出错: {e}")
    
    if not all_stats:
        print("未找到任何有效的任务数据")
        return
    
    # 打印统计摘要
    print_stats_summary(all_stats)
    
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

