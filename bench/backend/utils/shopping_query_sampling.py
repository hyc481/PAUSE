import random
import json
from typing import List, Tuple, Dict, Any
from dataclasses import asdict
from bench.backend.utils.resolver import (
    ShoppingQuery,
    solve_query,
    load_product_details,
    build_item_catalog,
)

BASE_TAGS = [
    "bar",
    "biscuit",
    "chip",
    "dairy",
    "drink",
    "fruit",
    "meal_replacement",
    "nut",
]

VARIANT_TAGS = [
    "diabetic_safe",
    "high_fiber",
    "high_protein",
    "low_calorie",
    "low_fat",
    "low_sodium",
    "low_sugar",
    "organic",
    "sugar_free",
]

DELIVERY_OPTIONS = ["today", "tomorrow", "3_days"]


def sample_shopping_query(
    *,
    allow_delivery: bool = True,
    user_is_vip: bool = False,
) -> List[Tuple[ShoppingQuery, Dict[str, Any]]]:
    """
    采样流程：
    1. 从 VARIANT_TAGS 中抽取 2-3 个元素
    2. 部分采样中加入 base_tags 和 delivery 采样
    3. 求解后根据结果处理
    
    Args:
        allow_delivery: 是否允许配送选项
        user_is_vip: 用户是否为 VIP，用于价格计算和商品过滤
    """
    # 加载商品目录
    product_details = load_product_details()
    catalog = build_item_catalog(product_details)
    
    # 1. 从 VARIANT_TAGS 中抽取 2-3 个元素
    num_variant_tags = random.choice([2, 3])
    selected_variant_tags = random.sample(VARIANT_TAGS, num_variant_tags)
    
    # 2. 部分采样中加入 base_tags 和 delivery
    # 随机决定是否添加 base_tags 和 delivery（仅考虑 today）
    use_base_tag = random.choice([True, False])
    use_delivery = True if allow_delivery else False
    
    base_tag = None
    if use_base_tag and BASE_TAGS:
        base_tag = random.choice(BASE_TAGS)
    
    delivery_option = None
    if use_delivery:
        delivery_option = "today"
    
    # 构建初始查询
    query = ShoppingQuery(
        include_tags_all=selected_variant_tags,
        top_k=10,
        user_is_vip=user_is_vip,
    )
    
    if base_tag:
        query.include_base_tags_all = [base_tag]
    
    if delivery_option:
        query.delivery_in = [delivery_option]
    
    # 求解
    result = solve_query(catalog, query)
    num_items = len(result.get("items", []))
    
    # 3. 根据结果处理
    if num_items == 1:
        # 情况1: 返回1个product，直接保留为list
        return [(query, result)]
    
    elif num_items == 0:
        # 情况2: 返回0个product，不保留该采样
        return []
    
    else:
        # 情况3: 返回多个product
        # 加入排序，按照price便宜或者calorie由低到高，选择第一个
        # 最终list加入排序这个条件，list只保留一个query
        
        # 随机选择排序方式：price 或 calorie
        sort_by = random.choice(["price", "calorie"])
        
        if sort_by == "price":
            objective = "effective_price"
            direction = "min"
        else:  # calorie
            objective = "calories"
            direction = "min"
        
        # 构建带排序的查询
        sorted_query = ShoppingQuery(
            include_tags_all=selected_variant_tags,
            objective=objective,
            direction=direction,
            user_is_vip=user_is_vip,
        )
        
        if base_tag:
            sorted_query.include_base_tags_all = [base_tag]
        
        if delivery_option:
            sorted_query.delivery_in = [delivery_option]
        
        # 求解
        sorted_result = solve_query(catalog, sorted_query)
        # 只保留一个最优 item（避免生成多个 targets）
        items = sorted_result.get("items", [])
        if items:
            sorted_result["items"] = [items[0]]
        
        # list只保留一个query（带排序的）
        return [(sorted_query, sorted_result)]


def clean_query_dict(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """清理query字典中的空项（空列表、None值等）"""
    cleaned = {}
    for key, value in query_dict.items():
        # 跳过空列表
        if isinstance(value, list) and len(value) == 0:
            continue
        # 跳过None值
        if value is None:
            continue
        # 跳过默认值（根据ShoppingQuery的默认值）
        if key == "user_is_vip" and value is False:
            continue
        if key == "allow_vip_only_products" and value is True:
            continue
        if key == "stock_min" and value == 0:
            continue
        if key == "top_k" and value == 10:
            continue
        cleaned[key] = value
    return cleaned


def clean_result_dict(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """清理result字典中的空项"""
    cleaned = {}
    for key, value in result_dict.items():
        # 跳过空列表
        if isinstance(value, list) and len(value) == 0:
            continue
        # 跳过None值
        if value is None:
            continue
        # result中的query字段已经在query中包含了，可以移除
        if key == "query":
            continue
        cleaned[key] = value
    return cleaned


def serialize_query_result(query: ShoppingQuery, result: Dict[str, Any]) -> Dict[str, Any]:
    """将 query 和 result 序列化为可 JSON 化的字典，并清理空项"""
    query_dict = asdict(query)
    cleaned_query = clean_query_dict(query_dict)
    cleaned_result = clean_result_dict(result)
    
    return {
        "query": cleaned_query,
        "result": cleaned_result
    }


if __name__ == "__main__":
    # 外层循环：多次采样
    num_iterations = 100  # 可以调整采样次数
    all_results = []
    
    print(f"开始采样，共 {num_iterations} 次迭代...")
    
    for i in range(num_iterations):
        result_list = sample_shopping_query()
        
        # 只保留非空 list
        if result_list:
            # 序列化每个 query-result 对（会自动清理空项）
            serialized_list = [
                serialize_query_result(query, result)
                for query, result in result_list
            ]
            all_results.append(serialized_list)
        
        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{num_iterations} 次迭代，当前有效采样数: {len(all_results)}")
    
    # 保存到 JSON 文件
    output_file = "shopping_bench/sampled_queries.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    lengths = [len(item) for item in all_results]
    status_counts = {}
    for item_list in all_results:
        for query_result in item_list:
            status = query_result["result"].get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n采样完成！")
    print(f"总迭代次数: {num_iterations}")
    print(f"有效采样数: {len(all_results)}")
    print(f"  - 长度为1的采样: {lengths.count(1)}")
    print(f"结果状态统计: {status_counts}")
    print(f"结果已保存到: {output_file}")
