#!/usr/bin/env python3
"""
分析工具调用次数和任务完成率的关系。

输入是saved_tasks目录，配套有inference_logs和evaluated_tasks目录。
统计不同模型的工具调用次数和任务完成率的关系。

用法:
    python analyze_tool_calls_vs_completion.py --saved_tasks_dir <saved_tasks_directory>
"""
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# 复用analyze_traj_stats中的统计函数
def analyze_traj_stats(
    agent_traj: Optional[List[Dict[str, Any]]] = None,
    user_traj: Optional[List[Dict[str, Any]]] = None,
    full_messages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    从保存的轨迹数据中提取统计信息：工具调用次数和对话轮数(rounds)。
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
        first_assistant_without_tools = True
        
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
            
            # 统计rounds
            if role == "assistant":
                content = msg.get("content", "") or ""
                if content and not tool_calls:
                    if first_assistant_without_tools:
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
        assistant_count = 0
        for msg in full_messages:
            if isinstance(msg, str):
                msg_stripped = msg.strip()
                if msg_stripped.startswith("[ASSISTANT]") and not msg_stripped.startswith("[ASSISTANT TOOL CALL]"):
                    assistant_count += 1
        
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
                    match = re.search(r'\[ASSISTANT TOOL CALL\]\s+(\w+)', msg)
                    if match:
                        tool_name = match.group(1)
                        stats["assistant_tool_calls_by_name"][tool_name] = \
                            stats["assistant_tool_calls_by_name"].get(tool_name, 0) + 1
                elif msg_stripped.startswith("[USER TOOL CALL]"):
                    stats["user_tool_calls"] += 1
                    stats["total_tool_calls"] += 1
                    match = re.search(r'\[USER TOOL CALL\]\s+(\w+)', msg)
                    if match:
                        tool_name = match.group(1)
                        stats["user_tool_calls_by_name"][tool_name] = \
                            stats["user_tool_calls_by_name"].get(tool_name, 0) + 1
    
    return stats


def find_inference_log(
    tasks_file: Path,
    branch: str,
    task_idx: int,
    inference_logs_dir: Path,
) -> Optional[Path]:
    """
    根据tasks_file、branch和task_idx找到对应的inference_log文件。
    
    Args:
        tasks_file: tasks文件路径
        branch: branch名称
        task_idx: 任务索引
        inference_logs_dir: inference_logs目录
    
    Returns:
        inference_log文件路径，如果找不到则返回None
    """
    tasks_stem = tasks_file.stem
    log_filename = f"{tasks_stem}_{branch}_{task_idx}.json"
    log_path = inference_logs_dir / log_filename
    
    return log_path if log_path.exists() else None


def process_task_evaluation(
    task: Dict[str, Any],
    task_idx: int,
    tasks_file: Path,
    inference_logs_dir: Path,
    evaluated_file: Path,
) -> List[Dict[str, Any]]:
    """
    处理单个task的evaluation，关联工具调用次数和完成率。
    
    Args:
        task: task字典
        task_idx: 任务索引
        tasks_file: tasks文件路径
        inference_logs_dir: inference_logs目录
        evaluated_file: evaluated_tasks文件路径
    
    Returns:
        统计结果列表
    """
    # 读取evaluated_tasks文件
    try:
        with evaluated_file.open("r", encoding="utf-8") as f:
            evaluated_payload = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read evaluated file {evaluated_file}: {e}")
        return []
    
    # 找到对应的evaluations
    evaluations_by_branch = evaluated_payload.get("evaluations_by_branch", {})
    if not isinstance(evaluations_by_branch, dict):
        return []
    
    results = []
    
    # 遍历所有branch
    for branch, eval_list in evaluations_by_branch.items():
        if not isinstance(eval_list, list):
            continue
        
        # 找到对应的task evaluation
        for eval_item in eval_list:
            if not isinstance(eval_item, dict):
                continue
            
            eval_task = eval_item.get("task", {})
            if not isinstance(eval_task, dict):
                continue
            
            # 检查是否是同一个task（通过label或annotated_log_path匹配）
            task_label = task.get("label", "")
            eval_label = eval_task.get("label", "")
            
            if task_label != eval_label:
                continue
            
            # 找到对应的inference_log
            log_path = find_inference_log(tasks_file, branch, task_idx, inference_logs_dir)
            if not log_path:
                print(f"[WARN] Inference log not found: {tasks_file.stem}_{branch}_{task_idx}.json")
                continue
            
            # 读取inference_log
            try:
                with log_path.open("r", encoding="utf-8") as f:
                    log_payload = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to read inference log {log_path}: {e}")
                continue
            
            trajs = log_payload.get("trajs", [])
            if not isinstance(trajs, list):
                continue
            
            # 处理每个evaluation
            evaluations = eval_item.get("evaluations", [])
            if not isinstance(evaluations, list):
                continue
            
            for evaluation in evaluations:
                if not isinstance(evaluation, dict):
                    continue
                
                traj_idx = evaluation.get("traj_idx", -1)
                assistant_model = evaluation.get("assistant_model", "unknown")
                eval_data = evaluation.get("evaluation", {})
                
                if not isinstance(eval_data, dict):
                    continue
                
                targets_achieved = eval_data.get("targets_achieved", [])
                if not isinstance(targets_achieved, list):
                    continue
                
                # 检查是否全部完成
                all_achieved = all(targets_achieved) if targets_achieved else False
                
                # 找到对应的traj并统计工具调用
                if 0 <= traj_idx < len(trajs):
                    traj = trajs[traj_idx]
                    if not isinstance(traj, dict):
                        continue
                    
                    agent_traj = traj.get("assistant_traj", [])
                    user_traj = traj.get("user_traj", [])
                    full_messages = traj.get("full_messages", [])
                    
                    tool_stats = analyze_traj_stats(
                        agent_traj=agent_traj,
                        user_traj=user_traj,
                        full_messages=full_messages,
                    )
                    
                    results.append({
                        "assistant_model": assistant_model,
                        "task_idx": task_idx,
                        "task_label": task_label,
                        "branch": branch,
                        "total_tool_calls": tool_stats["total_tool_calls"],
                        "assistant_tool_calls": tool_stats["assistant_tool_calls"],
                        "user_tool_calls": tool_stats["user_tool_calls"],
                        "rounds": tool_stats["rounds"],
                        "all_targets_achieved": all_achieved,
                        "num_targets": len(targets_achieved),
                        "targets_achieved": targets_achieved,
                    })
    
    return results


def process_tasks_file(
    tasks_file: Path,
    inference_logs_dir: Path,
    evaluated_tasks_dir: Path,
) -> List[Dict[str, Any]]:
    """
    处理单个tasks文件，返回所有任务的统计结果。
    
    Args:
        tasks_file: tasks文件路径
        inference_logs_dir: inference_logs目录
        evaluated_tasks_dir: evaluated_tasks目录
    
    Returns:
        统计结果列表
    """
    try:
        with tasks_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read tasks file {tasks_file}: {e}")
        return []
    
    tasks = payload.get("tasks", [])
    if not tasks:
        return []
    
    # 找到对应的evaluated_tasks文件（同名）
    evaluated_file = evaluated_tasks_dir / tasks_file.name
    if not evaluated_file.exists():
        print(f"[WARN] Evaluated file not found: {evaluated_file}")
        return []
    
    results = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        
        task_results = process_task_evaluation(
            task=task,
            task_idx=idx,
            tasks_file=tasks_file,
            inference_logs_dir=inference_logs_dir,
            evaluated_file=evaluated_file,
        )
        results.extend(task_results)
    
    return results


def get_tool_call_bucket(total_tool_calls: int) -> str:
    """
    根据工具调用次数返回区间标签。
    
    Args:
        total_tool_calls: 总工具调用次数
    
    Returns:
        区间标签: "<10", "10-15", "15-20", "20-25", "25-30", "30+"
    """
    if total_tool_calls < 10:
        return "<10"
    elif total_tool_calls < 15:
        return "10-15"
    elif total_tool_calls < 20:
        return "15-20"
    elif total_tool_calls < 25:
        return "20-25"
    elif total_tool_calls < 30:
        return "25-30"
    else:
        return "30+"


def print_stats_summary(all_stats: List[Dict[str, Any]]):
    """打印统计摘要：按工具调用次数区间统计任务完成率"""
    if not all_stats:
        print("No statistics to display.")
        return
    
    print("\n" + "="*80)
    print("工具调用次数区间 vs 任务完成率统计")
    print("="*80)
    
    # 按模型和区间分组
    by_model_and_bucket = defaultdict(lambda: defaultdict(lambda: {"total": 0, "completed": 0}))
    
    for stats in all_stats:
        model = stats.get("assistant_model", "unknown")
        total_tool_calls = stats.get("total_tool_calls", 0)
        all_achieved = stats.get("all_targets_achieved", False)
        
        bucket = get_tool_call_bucket(total_tool_calls)
        by_model_and_bucket[model][bucket]["total"] += 1
        if all_achieved:
            by_model_and_bucket[model][bucket]["completed"] += 1
    
    # 定义区间顺序
    bucket_order = ["<10", "10-15", "15-20", "20-25", "25-30", "30+"]
    
    # 按模型打印
    for model in sorted(by_model_and_bucket.keys()):
        model_buckets = by_model_and_bucket[model]
        print(f"\n[{model}]")
        
        for bucket in bucket_order:
            if bucket not in model_buckets:
                continue
            
            bucket_stats = model_buckets[bucket]
            total = bucket_stats["total"]
            completed = bucket_stats["completed"]
            completion_rate = (completed / total * 100) if total > 0 else 0
            
            print(f"  {bucket:>6}: {completed}/{total} ({completion_rate:.1f}%)")
    
    # 总体统计（跨所有模型）
    print("\n" + "="*80)
    print("总体统计 (所有模型)")
    print("="*80)
    
    by_bucket = defaultdict(lambda: {"total": 0, "completed": 0})
    for stats in all_stats:
        total_tool_calls = stats.get("total_tool_calls", 0)
        all_achieved = stats.get("all_targets_achieved", False)
        
        bucket = get_tool_call_bucket(total_tool_calls)
        by_bucket[bucket]["total"] += 1
        if all_achieved:
            by_bucket[bucket]["completed"] += 1
    
    for bucket in bucket_order:
        if bucket not in by_bucket:
            continue
        
        bucket_stats = by_bucket[bucket]
        total = bucket_stats["total"]
        completed = bucket_stats["completed"]
        completion_rate = (completed / total * 100) if total > 0 else 0
        
        print(f"{bucket:>6}: {completed}/{total} ({completion_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="分析工具调用次数和任务完成率的关系"
    )
    parser.add_argument(
        "--saved_tasks_dir",
        type=str,
        required=True,
        help="saved_tasks目录路径（同级应有inference_logs和evaluated_tasks目录）"
    )
    
    args = parser.parse_args()
    
    saved_tasks_dir = Path(args.saved_tasks_dir)
    if not saved_tasks_dir.exists():
        print(f"[ERROR] 目录不存在: {saved_tasks_dir}")
        return
    
    # 找到同级的inference_logs和evaluated_tasks目录
    base_dir = saved_tasks_dir.parent
    inference_logs_dir = base_dir / "inference_logs"
    evaluated_tasks_dir = base_dir / "evaluated_tasks"
    
    if not inference_logs_dir.exists():
        print(f"[ERROR] inference_logs目录不存在: {inference_logs_dir}")
        return
    
    if not evaluated_tasks_dir.exists():
        print(f"[ERROR] evaluated_tasks目录不存在: {evaluated_tasks_dir}")
        return
    
    # 查找所有tasks文件
    json_files = sorted(saved_tasks_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] 在 {saved_tasks_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个tasks文件")
    print(f"Inference logs目录: {inference_logs_dir}")
    print(f"Evaluated tasks目录: {evaluated_tasks_dir}")
    
    all_stats = []
    for json_file in json_files:
        try:
            stats_list = process_tasks_file(
                tasks_file=json_file,
                inference_logs_dir=inference_logs_dir,
                evaluated_tasks_dir=evaluated_tasks_dir,
            )
            all_stats.extend(stats_list)
            print(f"[OK] {json_file.name}: 处理了 {len(stats_list)} 个evaluations")
        except Exception as e:
            print(f"[ERROR] 处理文件 {json_file} 时出错: {e}")
    
    if not all_stats:
        print("未找到任何有效的统计数据")
        return
    
    # 打印统计摘要
    print_stats_summary(all_stats)


if __name__ == "__main__":
    main()

