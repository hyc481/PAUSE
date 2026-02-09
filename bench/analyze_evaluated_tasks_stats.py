#!/usr/bin/env python3
"""
分析evaluated_tasks文件，统计每个模型在不同targets数量下的任务完成率。

用法:
    python analyze_evaluated_tasks_stats.py --evaluated_tasks_dir <evaluated_tasks_directory>
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def process_evaluation(
    evaluation: Dict[str, Any],
    task: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    处理单个evaluation，提取统计信息。
    
    Args:
        evaluation: evaluation字典
        task: 对应的task字典
    
    Returns:
        统计信息字典，如果处理失败则返回None
    """
    eval_data = evaluation.get("evaluation", {})
    if not isinstance(eval_data, dict):
        return None
    
    targets_achieved = eval_data.get("targets_achieved", [])
    if not isinstance(targets_achieved, list):
        return None
    
    assistant_model = evaluation.get("assistant_model", "unknown")
    num_targets = len(targets_achieved)
    
    # 检查是否全部完成
    all_achieved = all(targets_achieved) if targets_achieved else False
    
    return {
        "assistant_model": assistant_model,
        "num_targets": num_targets,
        "all_achieved": all_achieved,
        "targets_achieved": targets_achieved,
        "task_label": task.get("label", ""),
    }


def process_evaluated_tasks_file(evaluated_file: Path) -> List[Dict[str, Any]]:
    """
    处理单个evaluated_tasks文件，返回所有evaluations的统计结果。
    
    Args:
        evaluated_file: evaluated_tasks文件路径
    
    Returns:
        统计结果列表
    """
    try:
        with evaluated_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read evaluated file {evaluated_file}: {e}")
        return []
    
    results = []
    
    # 处理evaluations_by_branch
    evaluations_by_branch = payload.get("evaluations_by_branch", {})
    if not isinstance(evaluations_by_branch, dict):
        return []
    
    for branch, evaluations in evaluations_by_branch.items():
        if not isinstance(evaluations, list):
            continue
        
        for eval_item in evaluations:
            if not isinstance(eval_item, dict):
                continue
            
            task = eval_item.get("task", {})
            if not isinstance(task, dict):
                continue
            
            eval_list = eval_item.get("evaluations", [])
            if not isinstance(eval_list, list):
                continue
            
            for evaluation in eval_list:
                if not isinstance(evaluation, dict):
                    continue
                
                stats = process_evaluation(evaluation, task)
                if stats:
                    stats["branch"] = branch
                    stats["evaluated_file"] = str(evaluated_file)
                    results.append(stats)
    
    return results


def print_stats_summary(all_stats: List[Dict[str, Any]]):
    """打印统计摘要"""
    if not all_stats:
        print("No statistics to display.")
        return
    
    # 按模型和targets数量分组
    by_model_and_targets = defaultdict(lambda: defaultdict(lambda: {"total": 0, "all_achieved": 0}))
    
    for stats in all_stats:
        model = stats.get("assistant_model", "unknown")
        num_targets = stats.get("num_targets", 0)
        all_achieved = stats.get("all_achieved", False)
        
        by_model_and_targets[model][num_targets]["total"] += 1
        if all_achieved:
            by_model_and_targets[model][num_targets]["all_achieved"] += 1
    
    print("\n" + "="*80)
    print("任务完成率统计 (按模型和targets数量分组)")
    print("="*80)
    
    # 按模型排序
    for model in sorted(by_model_and_targets.keys()):
        print(f"\n[{model}]")
        
        model_stats = by_model_and_targets[model]
        # 按targets数量排序（3, 4, 5, 6）
        for num_targets in sorted(model_stats.keys()):
            stats = model_stats[num_targets]
            total = stats["total"]
            all_achieved = stats["all_achieved"]
            completion_rate = (all_achieved / total * 100) if total > 0 else 0
            
            print(f"  Targets={num_targets}: {all_achieved}/{total} ({completion_rate:.1f}%)")
        
        # 总体统计（所有targets数量）
        total_all = sum(s["total"] for s in model_stats.values())
        all_achieved_all = sum(s["all_achieved"] for s in model_stats.values())
        overall_rate = (all_achieved_all / total_all * 100) if total_all > 0 else 0
        print(f"  总计: {all_achieved_all}/{total_all} ({overall_rate:.1f}%)")
    
    # 按targets数量汇总（跨所有模型）
    print("\n" + "="*80)
    print("按targets数量汇总 (所有模型)")
    print("="*80)
    
    by_targets = defaultdict(lambda: {"total": 0, "all_achieved": 0})
    for stats in all_stats:
        num_targets = stats.get("num_targets", 0)
        all_achieved = stats.get("all_achieved", False)
        
        by_targets[num_targets]["total"] += 1
        if all_achieved:
            by_targets[num_targets]["all_achieved"] += 1
    
    for num_targets in sorted(by_targets.keys()):
        stats = by_targets[num_targets]
        total = stats["total"]
        all_achieved = stats["all_achieved"]
        completion_rate = (all_achieved / total * 100) if total > 0 else 0
        
        print(f"Targets={num_targets}: {all_achieved}/{total} ({completion_rate:.1f}%)")
    
    # 总体统计
    total_all = sum(s["total"] for s in by_targets.values())
    all_achieved_all = sum(s["all_achieved"] for s in by_targets.values())
    overall_rate = (all_achieved_all / total_all * 100) if total_all > 0 else 0
    print(f"\n总计: {all_achieved_all}/{total_all} ({overall_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="分析evaluated_tasks文件，统计每个模型在不同targets数量下的任务完成率"
    )
    parser.add_argument(
        "--evaluated_tasks_dir",
        type=str,
        required=True,
        help="evaluated_tasks文件目录路径（会递归查找所有.json文件）"
    )
    
    args = parser.parse_args()
    
    evaluated_dir = Path(args.evaluated_tasks_dir)
    if not evaluated_dir.exists():
        print(f"[ERROR] 目录不存在: {evaluated_dir}")
        return
    
    # 查找所有evaluated_tasks文件
    json_files = sorted(evaluated_dir.rglob("*.json"))
    if not json_files:
        print(f"[WARN] 在 {evaluated_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    all_stats = []
    for json_file in json_files:
        try:
            stats_list = process_evaluated_tasks_file(json_file)
            all_stats.extend(stats_list)
            print(f"[OK] {json_file.name}: 处理了 {len(stats_list)} 个evaluations")
        except Exception as e:
            print(f"[ERROR] 处理文件 {json_file} 时出错: {e}")
    
    if not all_stats:
        print("未找到任何有效的evaluation数据")
        return
    
    # 打印统计摘要
    print_stats_summary(all_stats)


if __name__ == "__main__":
    main()

