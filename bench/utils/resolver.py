#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple shopping resolver for the health-agent shopping bench.

- 从 product_details.json 构建一个扁平的 item catalog（按 purchase_option 展开）
- ShoppingQuery：formalize 所有可用约束 + 单目标优化设置
- solve_query：多约束过滤 + 单目标优化（不做组合 / knapsack）

使用方式：
1. 确保 product_details.json 在当前目录
2. 在 __main__ 里看示例 query 如何写
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path

from bench.utils.paths import DATA_DIR

PRODUCT_DETAILS_PATH = DATA_DIR / "product_details.json"

# 你目前的品牌集合（按 name/tag 里出现的）
KNOWN_BRANDS = ["EverydayBite", "FamilyPack", "PureBalance", "EliteVital"]

# ----------------------------
# 1. Query / 约束定义
# ----------------------------

@dataclass
class ShoppingQuery:
    """
    Formalized query constraints.

    多约束过滤 + 单目标优化共享这一套字段。
    注意：全部约束都是“AND”逻辑；empty 列表代表不限定。

    ---- 基本标签约束 ----
    include_tags_all: 商品 tags 必须至少包含这些（交集 ⊇）
    include_tags_any: 商品 tags 与该集合至少有一个交集（交集 ≠ ∅）
    exclude_tags_any: 商品 tags 与该集合无交集（交集 = ∅）

    include_base_tags_all / any / exclude_base_tags_any:
        针对 base_tags，例如 ["biscuit", "dairy", "nuts", "drinks", ...] 之类

    ---- 品牌 / 配送 / 规格 ----
    brand_in / brand_not_in: 品牌过滤（从 name/tags 解析）
    delivery_in: 允许的配送方式，比如 ["today", "tomorrow"]
    size_in: 限定 purchase_options.size，例如 ["single_pack"] 只要单包

    ---- 价格 / VIP / 库存 ----
    price_min / price_max: 针对“有效价格”（考虑 discount_factor & VIP 权限之后）
    user_is_vip: 当前用户是否 VIP
    allow_vip_only_products:
        - False 时：非 VIP 用户会直接过滤掉 vip_required=True 的商品
        - True 时：非 VIP 用户仍可以看到，但不会享受 VIP 折扣
    stock_min: 最小库存要求（按 product 的 stock_count）

    ---- 营养约束 (per serving) ----
    calories_min / max
    sugar_g_min / max
    protein_g_min / max
    fat_g_min / max
    fiber_g_min / max
    sodium_mg_min / max

    ---- 优化设置 ----
    objective:
        None: 只做过滤，不做优化
        其它：单目标优化，例如：
          - "effective_price"
          - "protein_g"
          - "sugar_g"
          - "fiber_g"
          - "calories"
          - "sodium_mg"
          - "protein_per_price"
          - "fiber_per_price"
    direction: "max" or "min"（与 objective 搭配）
    top_k: 返回前多少个解
    """

    # 标签约束
    include_tags_all: List[str] = field(default_factory=list)
    include_tags_any: List[str] = field(default_factory=list)
    exclude_tags_any: List[str] = field(default_factory=list)

    include_base_tags_all: List[str] = field(default_factory=list)
    include_base_tags_any: List[str] = field(default_factory=list)
    exclude_base_tags_any: List[str] = field(default_factory=list)

    # 品牌 / 配送 / 规格
    brand_in: List[str] = field(default_factory=list)
    brand_not_in: List[str] = field(default_factory=list)
    delivery_in: List[str] = field(default_factory=list)
    size_in: List[str] = field(default_factory=list)  # e.g. ["single_pack", "pack_3"]

    # 价格 / VIP / 库存
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    user_is_vip: bool = False
    allow_vip_only_products: bool = True
    stock_min: int = 0

    # 营养（每份）
    calories_min: Optional[float] = None
    calories_max: Optional[float] = None
    sugar_g_min: Optional[float] = None
    sugar_g_max: Optional[float] = None
    protein_g_min: Optional[float] = None
    protein_g_max: Optional[float] = None
    fat_g_min: Optional[float] = None
    fat_g_max: Optional[float] = None
    fiber_g_min: Optional[float] = None
    fiber_g_max: Optional[float] = None
    sodium_mg_min: Optional[float] = None
    sodium_mg_max: Optional[float] = None

    # 单目标优化
    objective: Optional[str] = None  # 参考上面 docstring
    direction: Optional[Literal["max", "min"]] = None
    top_k: int = 10


# ----------------------------
# 2. Catalog 构建
# ----------------------------

def load_product_details(path: str=PRODUCT_DETAILS_PATH) -> Dict[str, Any]:
    """读取 product_details.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_brand(product: Dict[str, Any]) -> Optional[str]:
    """
    从 tags 或 name 中解析品牌。
    品牌可能出现在 tags（EverydayBite / FamilyPack / PureBalance / EliteVital），
    也可能只在 name 里出现。
    """
    name = product.get("name", "")
    tags = set(product.get("tags", []))
    for b in KNOWN_BRANDS:
        if b in tags or b in name:
            return b
    return None


def build_item_catalog(
    product_details: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    将每个 product 的 purchase_options 展开成 item 级别。

    返回的每个 item 例如：
    {
        "product_id": ...,
        "name": ...,
        "brand": ...,
        "tags": set([...]),
        "base_tags": set([...]),
        "stock_count": int,
        "discount_factor": float,
        "vip_required": bool,
        "unit": "g"/"ml",
        "weight": int/float,
        "nutrition": {...},
        "delivery": "today"/"tomorrow"/"standard_3_days",
        "size": "single_pack"/"pack_3"/"pack_5",
        "base_price": float   # 未加 discount_factor 的 pack 价格
    }
    """
    items: List[Dict[str, Any]] = []

    for pid, p in product_details.items():
        brand = infer_brand(p)
        tags = set(p.get("tags", []))
        base_tags = set(p.get("base_tags", []))
        stock_count = p.get("stock_count", 0)
        discount_info = p.get("discount_info", {})
        discount_factor = discount_info.get("discount_factor", 1.0)
        vip_required = discount_info.get("vip_required", False)
        packaging = p.get("packaging", {})
        unit = packaging.get("unit")
        weight = packaging.get("weight")
        nutrition = packaging.get("nutrition_per_serving", {})
        delivery = p.get("delivery")

        for opt in p.get("purchase_options", []):
            item = {
                "product_id": pid,
                "name": p.get("name"),
                "brand": brand,
                "tags": tags,
                "base_tags": base_tags,
                "stock_count": stock_count,
                "discount_factor": float(discount_factor),
                "vip_required": bool(vip_required),
                "unit": unit,
                "weight": weight,
                "nutrition": nutrition,
                "delivery": delivery,
                "size": opt.get("size"),
                "base_price": float(opt.get("price")),
            }
            items.append(item)

    return items


# ----------------------------
# 3. 过滤 & 优化
# ----------------------------

def effective_price(item: Dict[str, Any], user_is_vip: bool) -> float:
    """
    计算该 item 的有效价格：
    - pack 基础价格为 base_price
    - discount_factor 为额外折扣系数
    - 如果 vip_required 且用户不是 VIP，则不享受 discount_factor
    - 否则按 discount_factor 折扣
    """
    base_price = item["base_price"]
    df = item["discount_factor"]
    if item["vip_required"] and not user_is_vip:
        return base_price
    return base_price * df


def passes_numeric_range(
    value: Optional[float],
    min_val: Optional[float],
    max_val: Optional[float]
) -> bool:
    """通用的数值区间检查。"""
    if value is None:
        return False  # 对于营养字段，缺失就当作不满足
    if (min_val is not None) and (value < min_val):
        return False
    if (max_val is not None) and (value > max_val):
        return False
    return True


def filter_items(
    items: List[Dict[str, Any]],
    query: ShoppingQuery
) -> List[Dict[str, Any]]:
    """按 ShoppingQuery 做多约束过滤，并补上 effective_price 字段。"""

    include_tags_all = set(query.include_tags_all)
    include_tags_any = set(query.include_tags_any)
    exclude_tags_any = set(query.exclude_tags_any)

    include_base_tags_all = set(query.include_base_tags_all)
    include_base_tags_any = set(query.include_base_tags_any)
    exclude_base_tags_any = set(query.exclude_base_tags_any)

    brand_in = set(query.brand_in)
    brand_not_in = set(query.brand_not_in)
    delivery_in = set(query.delivery_in)
    size_in = set(query.size_in)

    filtered: List[Dict[str, Any]] = []

    for item in items:
        tags = item["tags"]
        base_tags = item["base_tags"]

        # 库存
        if item["stock_count"] < query.stock_min:
            continue

        # 品牌过滤
        brand = item["brand"]
        if brand_in and (brand not in brand_in):
            continue
        if brand_not_in and (brand in brand_not_in):
            continue

        # 配送
        if delivery_in and (item["delivery"] not in delivery_in):
            continue

        # 规格大小
        if size_in and (item["size"] not in size_in):
            continue

        # tags 约束
        if include_tags_all and not tags.issuperset(include_tags_all):
            continue
        if include_tags_any and not (tags & include_tags_any):
            continue
        if exclude_tags_any and (tags & exclude_tags_any):
            continue

        # base_tags 约束（biscuit/dairy/nuts/...）
        if include_base_tags_all and not base_tags.issuperset(include_base_tags_all):
            continue
        if include_base_tags_any and not (base_tags & include_base_tags_any):
            continue
        if exclude_base_tags_any and (base_tags & exclude_base_tags_any):
            continue

        # VIP 权限约束
        if (not query.allow_vip_only_products) and item["vip_required"] and (not query.user_is_vip):
            # 不允许非 VIP 用户看到只对 VIP 开放的折扣商品
            continue

        # 有效价格（要在价格/优化时用）
        eff_price = effective_price(item, query.user_is_vip)

        # 价格范围
        if not passes_numeric_range(eff_price, query.price_min, query.price_max):
            continue

        # 营养约束（per serving）
        nutr = item["nutrition"] or {}
        if not passes_numeric_range(nutr.get("calories"), query.calories_min, query.calories_max):
            continue
        if not passes_numeric_range(nutr.get("sugar_g"), query.sugar_g_min, query.sugar_g_max):
            continue
        if not passes_numeric_range(nutr.get("protein_g"), query.protein_g_min, query.protein_g_max):
            continue
        if not passes_numeric_range(nutr.get("fat_g"), query.fat_g_min, query.fat_g_max):
            continue
        if not passes_numeric_range(nutr.get("fiber_g"), query.fiber_g_min, query.fiber_g_max):
            continue
        if not passes_numeric_range(nutr.get("sodium_mg"), query.sodium_mg_min, query.sodium_mg_max):
            continue

        # 通过所有过滤，复制一份并挂上 effective_price
        item_copy = dict(item)
        item_copy["tags"] = list(item["tags"])
        item_copy["base_tags"] = list(item["base_tags"])
        item_copy["effective_price"] = eff_price

        filtered.append(item_copy)

    return filtered


def compute_score(item: Dict[str, Any], objective: str) -> float:
    """
    计算单目标优化的 score。
    支持：
      - "effective_price"
      - "protein_g" / "sugar_g" / "fiber_g" / "calories" / "sodium_mg"
      - "protein_per_price" / "fiber_per_price"
    """
    nutr = item.get("nutrition", {}) or {}
    price = item.get("effective_price")

    if objective == "effective_price":
        return price

    if objective in {"protein_g", "sugar_g", "fiber_g", "calories", "sodium_mg"}:
        return float(nutr.get(objective.replace("_g", "_g"), nutr.get(objective, 0.0)))

    if objective == "protein_per_price":
        if price is None or price <= 0:
            return float("-inf")
        return float(nutr.get("protein_g", 0.0)) / price

    if objective == "fiber_per_price":
        if price is None or price <= 0:
            return float("-inf")
        return float(nutr.get("fiber_g", 0.0)) / price

    raise ValueError(f"Unknown objective: {objective}")


def solve_query(
    items: List[Dict[str, Any]],
    query: ShoppingQuery
) -> Dict[str, Any]:
    """
    综合入口：
    - 如果 query.objective is None：只做过滤，默认按 effective_price 升序返回前 top_k
    - 否则：在过滤结果上做单目标优化（max/min），返回前 top_k
    
    注意：返回结果按 product_id 去重，每个 product 只返回一次（选择最优的 option），
    并且移除 size 字段（因为货物几乎总是充足的）。
    """

    feasible = filter_items(items, query)

    if not feasible:
        return {
            "status": "no_solution",
            "query": asdict(query),
            "items": []
        }

    # 只过滤，不优化
    if query.objective is None:
        feasible.sort(key=lambda x: x["effective_price"])
        # 按 product_id 去重，每个 product 只保留最便宜的 option
        seen_products = {}
        for item in feasible:
            pid = item["product_id"]
            if pid not in seen_products:
                seen_products[pid] = item
        unique_items = list(seen_products.values())[: query.top_k]
        
        # 移除 size 字段
        for item in unique_items:
            item.pop("size", None)
        
        return {
            "status": "feasible_only",
            "query": asdict(query),
            "items": unique_items
        }

    # 单目标优化
    direction = query.direction or "max"
    for it in feasible:
        it["score"] = compute_score(it, query.objective)

    reverse = True if direction == "max" else False
    feasible.sort(key=lambda x: x["score"], reverse=reverse)
    
    # 按 product_id 去重，每个 product 只保留最优的 option
    seen_products = {}
    for item in feasible:
        pid = item["product_id"]
        if pid not in seen_products:
            seen_products[pid] = item
    unique_items = list(seen_products.values())[: query.top_k]
    
    # 移除 size 字段
    for item in unique_items:
        item.pop("size", None)

    return {
        "status": "optimal",
        "query": asdict(query),
        "objective": query.objective,
        "direction": direction,
        "items": unique_items
    }


def solve_bundle_queries(
    items: List[Dict[str, Any]],
    queries: List[ShoppingQuery],
) -> Dict[str, Any]:
    """
    Solve multiple queries and aggregate their first feasible items.

    如果 objective=None，则：
      - chosen_item = 最便宜的
      - additional_field: all_candidate_product_ids 列出所有可行商品 product_id
    """
    bundle_items: List[Dict[str, Any]] = []
    total_price = 0.0

    for idx, query in enumerate(queries):
        res = solve_query(items, query)

        if res.get("status") == "no_solution" or not res.get("items"):
            return {
                "status": "no_solution",
                "failed_query_index": idx,
                "query": asdict(query),
                "items": [],
            }

        feasible_items = res["items"]

        # -------- 新增逻辑：objective=None 时返回所有 product_id --------
        all_candidate_pids = None
        if query.objective is None:
            all_candidate_pids = [it["product_id"] for it in feasible_items]

        # 选最优（或最便宜）商品
        chosen = feasible_items[0]
        total_price += float(chosen.get("effective_price", 0.0))

        entry = {
            "query": asdict(query),
            "chosen_item": chosen,
        }

        if all_candidate_pids is not None:
            entry["all_candidate_product_ids"] = all_candidate_pids

        bundle_items.append(entry)

    return {
        "status": "bundle_feasible",
        "items": bundle_items,
        "total_price": round(total_price, 2),
    }


if __name__ == "__main__":
    sq = ShoppingQuery(
        objective="calories",
        direction="min",
        include_tags_all=["low_calorie", "low_sugar"],
    )
    product_details = load_product_details()
    catalog = build_item_catalog(product_details)
    result = solve_query(catalog, sq)
    print(result)

