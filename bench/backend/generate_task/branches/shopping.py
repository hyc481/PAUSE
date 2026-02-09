from typing import Dict, List, Any, Tuple
import random
import copy
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import asdict

from bench.backend.generate_task.task_branch_base import TaskBranch
from bench.backend.utils.generate_ids import (
    generate_note_id,
    generate_transaction_id,
    generate_cart_item_id,
    generate_care_plan_id,
    generate_health_provider_id,
)
from bench.prompts.agent_interplay_prompt import shopping_user_roleplay_prompt

from bench.backend.utils.shopping_query_sampling import sample_shopping_query
from bench.backend.utils.resolver import ShoppingQuery


PRODUCT_DETAILS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "product_details.json"
)

root = Path(__file__).resolve().parents[2]
tool_schemas_path = root / "tool_schemas" / "platform_tools.json"
with open(tool_schemas_path, "r") as f:
    tool_schemas = json.load(f)

tool_schemas_path = root / "tool_schemas" / "source_tools.json"
with open(tool_schemas_path, "r") as f:
    source_tool_schemas = json.load(f)

tool_schemas_path = root / "tool_schemas" / "med_tools.json"
with open(tool_schemas_path, "r") as f:
    med_tool_schemas = json.load(f)

involved_tool_names = [
    "browse_items",
    "inspect_item",
    "add_to_cart",
    "remove_from_cart",
    "get_cart",
    "get_wallet",
    "prepare_order",
    "upgrade_membership_request",
    "get_transactions",
]

involved_source_tool_names_bool = True

involved_med_tool_names_bool = True

involved_tool_schemas = [
    {"name": tool_schema["function"]["name"], "description": tool_schema["function"]["description"]} \
    for tool_schema in tool_schemas if tool_schema["function"]["name"] in involved_tool_names
]

if involved_source_tool_names_bool:
    involved_tool_schemas.extend([
        {"name": source_tool_schema["function"]["name"], "description": source_tool_schema["function"]["description"]} \
        for source_tool_schema in source_tool_schemas
    ])

if involved_med_tool_names_bool:
    involved_tool_schemas.extend([
        {"name": med_tool_schema["function"]["name"], "description": med_tool_schema["function"]["description"]} \
        for med_tool_schema in med_tool_schemas
        ])


def _unique_timestamp(
    base_dt: datetime,
    used: set[str],
    rng: random.Random,
    day_window: int = 6,
) -> str:
    """
    Generate a unique ISO timestamp within the last `day_window` days.
    """
    for _ in range(50):
        candidate = (base_dt - timedelta(days=rng.randint(0, day_window))).replace(microsecond=0)
        ts = candidate.isoformat(timespec="seconds")
        if ts not in used:
            used.add(ts)
            return ts
    # Fallback: nudge by minutes until unique
    candidate = base_dt.replace(microsecond=0)
    while True:
        ts = candidate.isoformat(timespec="seconds")
        if ts not in used:
            used.add(ts)
            return ts
        candidate -= timedelta(minutes=1)


def _query_to_natural_language(query: ShoppingQuery) -> str:
    """
    Convert shopping query constraints to natural language description.
    """
    parts = []
    
    # Variant tags (health/nutritional attributes)
    if query.include_tags_all:
        tag_descriptions = []
        for tag in query.include_tags_all:
            if tag == "low_sugar":
                tag_descriptions.append("low sugar")
            elif tag == "high_protein":
                tag_descriptions.append("high protein")
            elif tag == "high_fiber":
                tag_descriptions.append("high fiber")
            elif tag == "low_calorie":
                tag_descriptions.append("low calorie")
            elif tag == "low_sodium":
                tag_descriptions.append("low sodium")
            elif tag == "low_fat":
                tag_descriptions.append("low fat")
            elif tag == "sugar_free":
                tag_descriptions.append("sugar-free")
            elif tag == "lactose_free":
                tag_descriptions.append("lactose-free")
            elif tag == "gluten_free":
                tag_descriptions.append("gluten-free")
            elif tag == "diabetic_safe":
                tag_descriptions.append("diabetic-safe")
            elif tag == "organic":
                tag_descriptions.append("organic")
            elif tag == "vegan":
                tag_descriptions.append("vegan")
            else:
                tag_descriptions.append(tag.replace("_", " "))
        
        if tag_descriptions:
            parts.append(f"products that are {', '.join(tag_descriptions)}")
    
    # Base tags (product categories)
    if query.include_base_tags_all:
        base_tag_descriptions = []
        for tag in query.include_base_tags_all:
            if tag == "biscuit":
                base_tag_descriptions.append("biscuits")
            elif tag == "dairy":
                base_tag_descriptions.append("dairy products")
            elif tag == "drink":
                base_tag_descriptions.append("drinks")
            elif tag == "nut":
                base_tag_descriptions.append("nuts")
            elif tag == "bar":
                base_tag_descriptions.append("bars")
            elif tag == "chip":
                base_tag_descriptions.append("chips")
            elif tag == "fruit":
                base_tag_descriptions.append("fruit products")
            elif tag == "meal_replacement":
                base_tag_descriptions.append("meal replacements")
            else:
                base_tag_descriptions.append(tag.replace("_", " "))
        
        if base_tag_descriptions:
            parts.append(f"in categories like {', '.join(base_tag_descriptions)}")
    
    # Delivery options
    if query.delivery_in:
        delivery_descriptions = []
        for delivery in query.delivery_in:
            if delivery == "today":
                delivery_descriptions.append("same-day delivery")
            elif delivery == "tomorrow":
                delivery_descriptions.append("next-day delivery")
            elif delivery == "3_days":
                delivery_descriptions.append("standard delivery")
            else:
                delivery_descriptions.append(delivery)
        
        if delivery_descriptions:
            parts.append(f"with {', '.join(delivery_descriptions)}")
    
    # Objective (sorting)
    if query.objective:
        if query.objective == "effective_price":
            if query.direction == "min":
                parts.append("preferring the cheapest option")
            else:
                parts.append("preferring higher-priced options")
        elif query.objective == "calories":
            if query.direction == "min":
                parts.append("preferring the lowest calorie option")
            else:
                parts.append("preferring higher calorie options")
        elif query.objective == "protein_per_price":
            parts.append("preferring the best protein value for money")
        elif query.objective == "fiber_per_price":
            parts.append("preferring the best fiber value for money")
    
    if not parts:
        return "products matching your preferences"
    
    return " ".join(parts)


def _clean_query_dict(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty/None/default fields from query dict."""
    cleaned: Dict[str, Any] = {}
    for key, value in query_dict.items():
        if isinstance(value, list) and not value:
            continue
        if value is None:
            continue
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


def _load_product_details() -> Dict[str, Any]:
    with open(PRODUCT_DETAILS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _map_name_to_product_ids(product_details: Dict[str, Any]) -> Dict[str, List[str]]:
    name_map: Dict[str, List[str]] = {}
    for pid, detail in product_details.items():
        name = detail.get("name")
        if not name:
            continue
        name_map.setdefault(name, []).append(pid)
    return name_map


def _get_option_price(detail: Dict[str, Any], size: str) -> float:
    options = detail.get("purchase_options") or []
    for opt in options:
        if opt.get("size") == size:
            try:
                return float(opt.get("price", 0.0))
            except (TypeError, ValueError):
                return 0.0
    if options:
        try:
            return float(options[0].get("price", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _split_quantity_to_sizes(total_qty: int) -> List[Dict[str, Any]]:
    if total_qty <= 0:
        return []
    pack_qty = total_qty // 5
    remainder = total_qty % 5
    items: List[Dict[str, Any]] = []
    if pack_qty > 0:
        items.append({"size": "pack_5", "quantity": pack_qty})
    if remainder > 0:
        items.append({"size": "single_pack", "quantity": remainder})
    return items


def _compute_required_vouchers(
    wallet: Dict[str, Any],
    expected_items: List[Dict[str, Any]],
    product_details: Dict[str, Any],
    now_date: str,
) -> List[str]:
    required: List[str] = []
    vouchers = wallet.get("vouchers", []) or []
    is_vip = bool(wallet.get("vip"))

    # Precompute per-item price and tags (match prepare_order pricing logic)
    item_info = []
    for item in expected_items:
        pid = item.get("product_id")
        size = item.get("size", "")
        qty = int(item.get("quantity", 0))
        detail = product_details.get(pid, {}) if pid else {}
        tags = list(detail.get("tags", []) or [])  # Create a copy of tags list to avoid mutation
        base_price = _get_option_price(detail, size)
        discount_info = detail.get("discount_info", {}) or {}
        discount_factor = float(discount_info.get("discount_factor", 1.0))
        vip_required = bool(discount_info.get("vip_required", False))
        if vip_required and not is_vip:
            effective_discount = 1.0
        else:
            effective_discount = discount_factor
        discounted_price = round(base_price * effective_discount, 2)
        item_info.append({
            "pid": pid,
            "size": size,
            "qty": qty,
            "tags": tags,
            "base_price": base_price,
            "discounted_price": discounted_price,
        })

    best_voucher_id = None
    best_savings = 0.0
    for v in vouchers:
        voucher_id = v.get("voucher_id")
        if not voucher_id:
            continue
        if v.get("vip_required") and not is_vip:
            continue
        expiry = v.get("expiry_date")
        if expiry and now_date > expiry:
            continue
        brand = v.get("brand")
        min_amount = float(v.get("min_amount", 0.0))

        eligible_subtotal_original = 0.0
        eligible_subtotal_discounted = 0.0
        for info in item_info:
            if not info["pid"]:
                continue
            if brand is None or brand in info["tags"]:
                eligible_subtotal_original += info["base_price"] * info["qty"]
                eligible_subtotal_discounted += info["discounted_price"] * info["qty"]

        if eligible_subtotal_original < min_amount:
            continue

        discount_type = v.get("discount_type", "percent")
        if discount_type != "percent":
            continue
        factor = float(v.get("discount_factor", 1.0))
        savings = eligible_subtotal_discounted * (1.0 - factor)
        if savings > best_savings:
            best_savings = savings
            best_voucher_id = voucher_id

    if best_voucher_id:
        required.append(best_voucher_id)
    return required


def _build_targets(
    input_query_template: Dict[str, Any],
    wallet: Dict[str, Any],
    product_details: Dict[str, Any],
    now_date: str,
) -> List[Dict[str, Any]]:
    expected_items: List[Dict[str, Any]] = []
    product_qty: Dict[str, int] = {}

    for q in input_query_template.get("queries", []):
        if not isinstance(q, dict):
            continue
        result = q.get("result", {}) if isinstance(q.get("result"), dict) else {}
        status = result.get("status")
        if status == "no_solution":
            continue
        items = result.get("items", [])
        for it in items or []:
            pid = it.get("product_id")
            qty = int(it.get("num", 0) or it.get("quantity", 0) or 0)
            if pid and qty > 0:
                product_qty[pid] = product_qty.get(pid, 0) + qty

    for pid, total_qty in product_qty.items():
        for entry in _split_quantity_to_sizes(total_qty):
            expected_items.append({
                "product_id": pid,
                "size": entry["size"],
                "quantity": entry["quantity"],
            })

    required_vouchers = _compute_required_vouchers(
        wallet=wallet,
        expected_items=expected_items,
        product_details=product_details,
        now_date=now_date,
    )

    return [{
        "transaction": {
            "type": "order",
            "status": "completed",
            "items": expected_items,
            "voucher_ids": required_vouchers,
            "wallet_balance_max": 5.0,
            "vip": bool(wallet.get("vip")),
        }
    }]


def _build_input_query_template(
    query_lists: List[List[Tuple[ShoppingQuery, Dict[str, Any]]]],
    retrieval_infos: List[Dict[str, Any]],
    accept_vip: bool,
    delivery_in_today: bool,
    rng: random.Random,
    product_details: Dict[str, Any],
) -> Dict[str, Any]:
    name_to_ids = _map_name_to_product_ids(product_details)
    queries = []
    for idx, query_list in enumerate(query_lists):
        if not query_list:
            continue
        query, result = query_list[0]
        query_dict = _clean_query_dict(asdict(query))
        if not query_dict:
            continue

        status = result.get("status", "no_solution")
        retrieval_info = retrieval_infos[idx]
        retrieval_type = retrieval_info.get("type", "none")
        retrieval_payload = {"type": retrieval_type}
        if retrieval_type in {"note", "care_plan"}:
            created_at = retrieval_info.get("created_at")
            if not created_at:
                continue
            retrieval_payload["created_at"] = created_at

        query_dict.pop("delivery_in", None)
        entry = {
            "index": idx,
            "retrieval": retrieval_payload,
            "query": query_dict,
            "result": {"status": status},
        }

        items = result.get("items", [])
        if status != "no_solution" and items:
            entry["result"]["items"] = []
            for item in items:
                name = item.get("name")
                if not name:
                    pid = item.get("product_id")
                    if pid:
                        name = product_details.get(pid, {}).get("name")
                if not name:
                    continue
                pid = item.get("product_id")
                if not pid:
                    pids = name_to_ids.get(name, [])
                    if len(pids) == 1:
                        pid = pids[0]
                entry["result"]["items"].append({
                    "name": name,
                    "num": rng.randint(3, 8),
                    "product_id": pid,
                })

        queries.append(entry)

    return {
        "accept_vip": accept_vip,
        "delivery_in_today": delivery_in_today,
        "queries": queries,
    }


def _build_shopping_prompt(input_query_template: Dict[str, Any]) -> str:
    return (
        shopping_user_roleplay_prompt
        + "\n\n### Hidden Input JSON\n"
        + json.dumps(input_query_template, ensure_ascii=False, indent=2)
    )


def _generate_voucher_id(rng: random.Random) -> str:
    """Generate a voucher ID."""
    year = datetime.now().year
    month = rng.randint(1, 12)
    return f"v{year}_{month:02d}"


def _inject_context_for_tasks(
    store: Dict[str, Any]
) -> Tuple[
    List[List[Tuple[ShoppingQuery, Dict[str, Any]]]],
    List[Dict[str, Any]],
    bool,
]:
    """
    Inject shopping-related context into store:
    1. Sample 1-2 shopping query lists
    2. Inject queries as notes or care_plans (converted to natural language)
    3. Setup initial shopping state (wallet, cart, transactions)
    
    Returns:
        Tuple of (sampled_query_lists, retrieval_infos, delivery_in_today)
        or ([], [], False) if sampling fails
    """
    rng = random.Random(None)
    profile = store.get("profile", {})
    used_times: set[str] = set()
    now_str = profile.get("now")
    now_dt = datetime.fromisoformat(now_str)
    
    # Initialize shopping state if not exists
    shopping = profile.setdefault("shopping", {})
    wallet = shopping.setdefault("wallet", {
        "balance": 0.0,
        "vip": False,
        "vip_expiry": None,
        "vouchers": []
    })
    shopping.setdefault("cart", [])
    shopping.setdefault("transactions", [])
    
    # Setup initial shopping state - VIP status needs to be set before query sampling
    # Wallet: set balance, VIP status, and vouchers
    wallet["balance"] = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # Random balance between 0-5
    wallet["vip"] = rng.choice([True, False])
    if wallet["vip"]:
        wallet["vip_expiry"] = datetime.strptime(profile["now"], "%Y-%m-%dT%H:%M:%S") + timedelta(days=20)
        wallet["vip_expiry"] = wallet["vip_expiry"].strftime("%Y-%m-%dT%H:%M:%S")
    
    # Sample 1-2 shopping query lists (non-empty, non-duplicate)
    num_query_lists = rng.choice([1, 2])
    sampled_query_lists: List[List[Tuple[ShoppingQuery, Dict[str, Any]]]] = []
    retrieval_infos: List[Dict[str, Any]] = []
    seen_query_keys: set[str] = set()
    max_attempts = 100
    note_used = 0
    care_plan_used = 0
    delivery_in_today = rng.choice([True, False])

    while len(sampled_query_lists) < num_query_lists and max_attempts > 0:
        candidates = ["none"]
        if note_used < 1:
            candidates.append("note")
        if care_plan_used < 1:
            candidates.append("care_plan")
        retrieval_type = rng.choice(candidates)
        query_list = sample_shopping_query(
            allow_delivery=delivery_in_today,
            user_is_vip=wallet["vip"],
        )
        max_attempts -= 1
        if not query_list:
            continue
        query, _result = query_list[0]
        query_key = json.dumps(_clean_query_dict(asdict(query)), sort_keys=True)
        if not query_key or query_key in seen_query_keys:
            continue
        seen_query_keys.add(query_key)
        sampled_query_lists.append(query_list)
        retrieval_infos.append({"type": retrieval_type, "created_at": None})
        if retrieval_type == "note":
            note_used += 1
        if retrieval_type == "care_plan":
            care_plan_used += 1

    if len(sampled_query_lists) < 1:
        return ([], [], False)

    # Inject queries as notes or care_plans when needed
    notes_list = profile.get("notes", [])
    care_plans_list = profile.get("care_plans", [])
    
    for idx, query_list in enumerate(sampled_query_lists):
        # Convert first query in each list to natural language
        if query_list:
            first_query, first_result = query_list[0]
            query_description = _query_to_natural_language(first_query)
            
            mode = retrieval_infos[idx]["type"]
            if mode == "note":
                # Inject as note
                note_time = _unique_timestamp(now_dt, used_times, rng, day_window=6)
                note_content = f"I need to find {query_description} for my dietary needs."
                notes_list.append({
                    "note_id": generate_note_id(),
                    "time": note_time,
                    "content": note_content,
                })
                retrieval_infos[idx]["created_at"] = note_time.split("T")[0]
            elif mode == "care_plan":
                # Inject as care_plan (health-related shopping needs)
                created_at = _unique_timestamp(now_dt, used_times, rng, day_window=6)
                # Generate a provider if needed
                providers = profile.get("health_providers", [])
                if not providers:
                    # Create a provider if none exists
                    provider_id = generate_health_provider_id()
                    providers.append({
                        "provider_id": provider_id,
                        "doctor": rng.choice(["Dr. Lee", "Dr. Chen", "Dr. Patel"]),
                        "clinic": rng.choice(["Family Clinic", "Wellness Center", "Community Health"]),
                        "address": rng.choice(["Main St 12", "River Rd 8", "Oak Ave 23"]),
                    })
                    profile["health_providers"] = providers
                else:
                    provider_id = rng.choice(providers).get("provider_id")
                
                care_plan_text = f"Consider purchasing {query_description} to support your health goals."
                care_plans_list.append({
                    "plan_id": generate_care_plan_id(),
                    "provider_id": provider_id,
                    "created_at": created_at.split("T")[0],
                    "topics": ["diet", "nutrition"],
                    "note_text": care_plan_text,
                })
                retrieval_infos[idx]["created_at"] = created_at.split("T")[0]
    
    profile["notes"] = notes_list
    profile["care_plans"] = care_plans_list
    
    # Setup vouchers (VIP status already set above)
    num_vouchers = 4
    brands = ["EverydayBite", "FamilyPack", "PureBalance", "EliteVital"]
    for _ in range(num_vouchers):
        voucher_id = _generate_voucher_id(rng)
        brand = rng.choice(brands)
        min_amount = rng.choice([20, 25, 30, 35])
        discount_factor = rng.choice([0.85, 0.90, 0.95])  # 15%, 10%, or 5% off
        vip_required = rng.choice([True, False])
        
        # Set expiry date (within next 6 months)
        expiry_days = rng.randint(30, 180)
        expiry_date = (now_dt + timedelta(days=expiry_days)).date().isoformat()
        
        discount_percent = int((1 - discount_factor) * 100)
        description = f"{discount_percent}% off {brand} orders over {min_amount:.0f}"
        if vip_required:
            description += " (VIP required)"
        
        wallet["vouchers"].append({
            "voucher_id": voucher_id,
            "description": description,
            "brand": brand,
            "vip_required": vip_required,
            "expiry_date": expiry_date,
            "min_amount": round(min_amount, 2),
            "discount_type": "percent",
            "discount_factor": discount_factor,
            "discount_rule": "The minimum spend requirement for vouchers is calculated based on the original prices of eligible items. Discounts can be applied cumulatively."
        })
    
    # Cart: optionally pre-populate with some items (randomly)
    if rng.random() < 0.3 and not shopping.get("cart"):  # 30% chance to have items in cart
        product_details = _load_product_details()
        product_ids = list(product_details.keys())
        if product_ids:
            cart_items = shopping.get("cart", [])
            num_cart_items = rng.randint(1, 2)
            for _ in range(num_cart_items):
                pid = rng.choice(product_ids)
                detail = product_details.get(pid, {}) or {}
                options = detail.get("purchase_options") or []
                if not options:
                    continue
                size = rng.choice(options).get("size")
                if not size:
                    continue
                cart_items.append({
                    "cart_item_id": generate_cart_item_id(),
                    "product_id": pid,
                    "size": size,
                    "quantity": rng.randint(1, 3),
                })
            shopping["cart"] = cart_items
    
    return (sampled_query_lists, retrieval_infos, delivery_in_today)


class ShoppingBranch(TaskBranch):
    branch_name = "shopping"
    
    def is_applicable(self, store: Dict[str, Any]) -> bool:
        """Shopping branch is always applicable."""
        return True
    
    def build_prompt(self, store: Dict[str, Any]) -> str:
        """Not used in shopping branch - we generate tasks directly from queries."""
        pass
    
    def postprocess(
        self,
        llm_output: str,
        store: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Not used in shopping branch."""
        pass
    
    def run(self, store: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run shopping task generation:
        1. Inject shopping context (queries, notes, care_plans, shopping state)
        2. Generate tasks from sampled queries
        3. Compute final state annotations
        """
        # Step 1: Inject context and sample queries
        sampled_query_lists, retrieval_infos, delivery_in_today = _inject_context_for_tasks(store)
        
        if not sampled_query_lists:
            return []
        if len(sampled_query_lists) < 1:
            return []
        if len(sampled_query_lists) != len(retrieval_infos):
            return []
        
        # Use all sampled queries (no random subset selection)
        selected_query_lists = sampled_query_lists
        selected_retrieval_infos = retrieval_infos
        
        # Step 3: Build prompt input and task
        tasks = []
        profile = store.get("profile", {})
        shopping = profile.get("shopping", {})
        # Deep copy wallet to avoid mutating the original (vouchers is a nested list)
        initial_wallet = copy.deepcopy(shopping.get("wallet", {}))
        initial_cart = shopping.get("cart", []).copy()

        product_details = _load_product_details()
        accept_vip = any(info.get("type") == "care_plan" for info in selected_retrieval_infos)
        # Use a fresh random instance for query template generation
        template_rng = random.Random(None)
        input_query_template = _build_input_query_template(
            query_lists=selected_query_lists,
            retrieval_infos=selected_retrieval_infos,
            accept_vip=accept_vip,
            delivery_in_today=delivery_in_today,
            rng=template_rng,
            product_details=product_details,
        )
        if not input_query_template.get("queries"):
            return []

        task_instruction = _build_shopping_prompt(input_query_template)
        now_date = profile.get("now", "").split("T")[0]
        targets = _build_targets(
            input_query_template=input_query_template,
            wallet=initial_wallet,
            product_details=product_details,
            now_date=now_date,
        )
        task = {
            "task_instruction": task_instruction,
            "input_query_template": input_query_template,
            "label": "shopping",
            "targets": targets,
        }
        tasks.append(task)
        
        return tasks

