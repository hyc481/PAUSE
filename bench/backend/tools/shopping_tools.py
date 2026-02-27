# shopping_tools.py
"""
Tool implementations for the shopping demo.

- browse_items(page)   -> list products with pagination
- inspect_item(product_id) -> detailed product info
- update_cart(product_id, size, quantity)
- get_cart()
- prepare_order(voucher_id)
- get_wallet()
- upgrade_membership_request(vip_duration)
- get_transactions()
"""
import json
import copy
import os
from typing import Dict, Any, List, Optional
from datetime import date

from bench.utils.generate_ids import (
    generate_cart_item_id,
    generate_transaction_id,
)

# Import global timestamp helper


# -----------------------------
# Load product data
# -----------------------------

BASE_DIR = os.path.dirname(__file__) or "."
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

with open(os.path.join(DATA_DIR, "product_log.json"), "r", encoding="utf-8") as f:
    product_log: List[Dict[str, Any]] = json.load(f)

with open(os.path.join(DATA_DIR, "product_details.json"), "r", encoding="utf-8") as f:
    product_details: Dict[str, Dict[str, Any]] = json.load(f)

# Map for quick lookup
product_log_index: Dict[str, Dict[str, Any]] = {
    item["product_id"]: item for item in product_log
}

def _shopping_state(store: Dict[str, Any]) -> Dict[str, Any]:
    """Return the shopping substate from store.profile, initializing if missing."""
    profile = store.setdefault("profile", {})
    shopping = profile.setdefault("shopping", {})
    shopping.setdefault("wallet", {})
    shopping.setdefault("cart", [])
    shopping.setdefault("transactions", [])
    return shopping


def _recompute_order_for_current_state(tx: Dict[str, Any], store: Dict[str, Any]) -> None:
    """根据当前 WALLET_STATE / voucher 规则，重新计算一个 pending 订单的价格。"""
    if tx.get("type") != "order" or tx.get("status") != "pending":
        return

    shopping = _shopping_state(store)
    wallet = shopping.get("wallet", {})
    is_vip = bool(wallet.get("vip", False))
    voucher_id = tx.get("voucher_id")
    voucher = _find_voucher(store, voucher_id)

    new_items = []
    original_total = 0.0

    # 1) 先按当前 VIP 重新算商品自身折扣
    for item in tx.get("items", []):
        pid = item["product_id"]
        size = item["size"]
        qty = int(item.get("quantity", 0))

        base_price = _get_purchase_option_price(pid, size)
        if base_price is None or qty <= 0:
            continue

        detail = product_details.get(pid, {}) or {}
        discount_info = detail.get("discount_info", {}) or {}
        discount_factor = float(discount_info.get("discount_factor", 1.0))
        vip_required = bool(discount_info.get("vip_required", False))

        if vip_required and not is_vip:
            effective_discount = 1.0
        else:
            effective_discount = discount_factor

        unit_price = round(base_price * effective_discount, 2)
        subtotal = round(unit_price * qty, 2)

        new_items.append({
            "product_id": pid,
            "name": item.get("name", ""),
            "size": size,
            "quantity": qty,
            "unit_price": unit_price,
            "discount_factor_applied": effective_discount,
            "subtotal": subtotal,
        })

        original_total += subtotal

    final_total = original_total
    voucher_applied = False
    fail_reasons: List[str] = []
    eligible_subtotal_discounted = 0.0  # Initialize for voucher_internal

    # 2) 再按当前 VIP / voucher 规则重新算 voucher
    profile = store.get("profile", {})
    now = profile.get("now", "")
    if voucher:
        if voucher.get("vip_required") and not is_vip:
            fail_reasons.append("vip_required")

        expiry_str = voucher.get("expiry_date")
        if expiry_str:
            try:
                expiry = date.fromisoformat(expiry_str)
                if now.date() > expiry:
                    fail_reasons.append("expired")
            except Exception:
                pass

        brand = voucher.get("brand")
        min_amount = float(voucher.get("min_amount", 0.0))

        # For min_amount check: use original price (before product discounts)
        # For voucher discount: use discounted price (after product discounts)
        eligible_subtotal_original = 0.0  # For min_amount check
        eligible_subtotal_discounted = 0.0  # For voucher discount calculation
        
        for item in new_items:
            pid = item["product_id"]
            detail = product_details.get(pid, {}) or {}
            tags = detail.get("tags", []) or []
            if brand is None or brand in tags:
                # Calculate original price (before product discount)
                size = item.get("size")
                qty = int(item.get("quantity", 0))
                base_price = _get_purchase_option_price(pid, size)
                if base_price is not None and qty > 0:
                    original_subtotal = base_price * qty
                    eligible_subtotal_original += original_subtotal
                # Use discounted subtotal for voucher discount
                eligible_subtotal_discounted += float(item["subtotal"])

        # Check min_amount against original price (before product discounts)
        if eligible_subtotal_original < min_amount:
            fail_reasons.append("min_amount_not_met")

        if not fail_reasons:
            discount_type = voucher.get("discount_type", "percent")
            if discount_type == "percent":
                factor = float(voucher.get("discount_factor", 1.0))
                # Apply voucher discount to discounted eligible subtotal
                discount_amount = eligible_subtotal_discounted * (1.0 - factor)
                final_total = round(original_total - discount_amount, 2)
                voucher_applied = True

    # 覆盖 tx
    tx["items"] = new_items
    tx["final_total"] = round(final_total, 2)
    tx["voucher_internal"] = {
        "applied": voucher_applied,
        "fail_reasons": fail_reasons,
        "eligible_subtotal": round(eligible_subtotal_discounted, 2),
    }


def _get_purchase_option_price(product_id: str, size: str) -> Optional[float]:
    """
    Look up base unit price from product_details.purchase_options by size.
    Falls back to product_log.price if missing.
    """
    detail = product_details.get(product_id, {})
    options = detail.get("purchase_options") or []
    for opt in options:
        if opt.get("size") == size:
            try:
                return float(opt.get("price", 0.0))
            except (TypeError, ValueError):
                return 0.0

    # Fallback: use product_log price (treated as default size price)
    log_info = product_log_index.get(product_id)
    if log_info is not None:
        try:
            return float(log_info.get("price", 0.0))
        except (TypeError, ValueError):
            return 0.0

    return None


def _iter_cart_items(store: Dict[str, Any]):
    """
    Iterate over cart items.
    Unified CART_STATE structure:

    CART_STATE = [
        {
            "cart_item_id": "...",
            "product_id": "...",
            "size": "...",
            "quantity": int,
        },
        ...
    ]
    """
    cart_state = _shopping_state(store).get("cart", [])

    for item in cart_state:
        pid = item.get("product_id")
        size = item.get("size")
        qty = item.get("quantity", 0)
        cid = item.get("cart_item_id")

        if pid and size and qty > 0:
            yield cid, pid, size, qty


def _compute_cart_summary(store: Dict[str, Any]) -> Dict[str, Any]:
    """Compute current cart summary based on CART_STATE and wallet vip."""
    items: List[Dict[str, Any]] = []
    total_price = 0.0
    total_items = 0
    shopping = _shopping_state(store)
    wallet = shopping.get("wallet", {})
    is_vip = bool(wallet.get("vip", False))

    for cid, pid, size, qty in _iter_cart_items(store):
        if qty is None or qty <= 0:
            continue

        log_info = product_log_index.get(pid)
        detail = product_details.get(pid)

        # Skip unknown products silently (should not happen in normal flows)
        if not log_info or not detail:
            # if you want to debug: raise ValueError(f"Unknown product in cart: {pid}")
            continue

        # Base price now comes from purchase_options[size]
        base_price = _get_purchase_option_price(pid, size)
        if base_price is None:
            # If still cannot find a price, skip this line defensively
            continue

        discount_info = detail.get("discount_info", {}) or {}
        discount_factor = float(discount_info.get("discount_factor", 1.0))
        vip_required = bool(discount_info.get("vip_required", False))

        if vip_required and not is_vip:
            effective_discount = 1.0
        else:
            effective_discount = discount_factor

        unit_price = round(base_price * effective_discount, 2)
        subtotal = round(unit_price * qty, 2)

        items.append(
            {
                "cart_item_id": cid,
                "product_id": pid,
                "name": log_info.get("name", ""),
                "size": size,
                "quantity": int(qty),
                "unit_price": unit_price,
                "discount_factor_applied": effective_discount,
                "subtotal": subtotal,
            }
        )

        total_price += subtotal
        total_items += int(qty)

    return {
        "items": items,
        "total_items": total_items,
        "total_price": round(total_price, 2)
    }


def _find_voucher(store: Dict[str, Any], voucher_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not voucher_id:
        return None
    wallet = _shopping_state(store).get("wallet", {})
    for v in wallet.get("vouchers", []):
        if v.get("voucher_id") == voucher_id:
            return v
    return None


def _public_transaction_view(tx: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of tx without internal-only fields."""
    public = dict(tx)
    public.pop("voucher_internal", None)
    return public


# -----------------------------
# Tools: browsing
# -----------------------------
def browse_items(store: Dict[str, Any], page: int) -> Dict[str, Any]:
    """
    Browse items with pagination.
    Page size fixed to 15.
    """
    log_len = len(product_log)
    page_size = 15
    max_page = log_len // page_size + (1 if log_len % page_size else 0)

    if page < 1 or page > max_page:
        return {"page": page, "max_page": max_page, "content": []}

    start = page_size * (page - 1)
    end = min(log_len, page_size * page)

    page2return = copy.deepcopy(product_log[start:end])
    for item in page2return:
        # hide internal tags
        item.pop("base_tags", None)

    return {"page": page, "max_page": max_page, "content": page2return}

def inspect_item(store: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    """
    Get detailed information of an item.
    """
    product2return = copy.deepcopy(product_details.get(product_id, {}))
    if product2return:
        product2return.pop("base_tags", None)
        product2return.pop("index", None)
    return product2return


# -----------------------------
# Tools: cart
# -----------------------------
def add_to_cart(
    store: Dict[str, Any],
    product_id: str,
    size: str,
    quantity: int,
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
):
    """
    Append-only cart operation.
    Always creates a new cart item.
    """
    # 校验 product 存在
    if product_id not in product_details:
        return {
            "error": {
                "code": "INVALID_PRODUCT_ID",
                "message": f"Product {product_id} does not exist."
            }
        }

    detail = product_details[product_id]
    purchase_options = detail.get("purchase_options") or []
    valid_sizes = {opt["size"] for opt in purchase_options if opt.get("size")}

    # size 校验（如果该商品定义了 purchase_options）
    if valid_sizes and size not in valid_sizes:
        return {
            "error": {
                "code": "INVALID_SIZE",
                "message": (
                    f"Size '{size}' is not valid for product {product_id}. "
                    f"Valid sizes: {sorted(valid_sizes)}"
                )
            }
        }

    if quantity <= 0:
        return {
            "error": {
                "code": "INVALID_QUANTITY",
                "message": "Quantity must be >= 1."
            }
        }

    cart_item_id = generate_cart_item_id(_id_seed, _tool_call_name, _tool_call_index)

    shopping = _shopping_state(store)
    cart_state = shopping.get("cart", [])

    cart_state.append({
        "cart_item_id": cart_item_id,
        "product_id": product_id,
        "size": size,
        "quantity": quantity
    })

    return {
        "success": True,
        "cart_item_id": cart_item_id,
        "cart_size": len(cart_state)
    }


def remove_from_cart(store: Dict[str, Any], cart_item_id: str):
    """
    Remove a cart item by its cart_item_id.
    """

    shopping = _shopping_state(store)
    cart_state = shopping.get("cart", [])

    before = len(cart_state)
    cart_state[:] = [item for item in cart_state if item.get("cart_item_id") != cart_item_id]
    after = len(cart_state)

    if before == after:
        # No such id
        return {
            "error": {
                "code": "CART_ITEM_NOT_FOUND",
                "message": f"Cart item {cart_item_id} does not exist."
            }
        }

    return {
        "success": True,
    }



def get_cart(store: Dict[str, Any]) -> Dict[str, Any]:
    """Get current cart summary."""
    return _compute_cart_summary(store)


# -----------------------------
# Tools: wallet
# -----------------------------
def get_wallet(store: Dict[str, Any]) -> Dict[str, Any]:
    """Return current wallet state (read-only)."""
    shopping = _shopping_state(store)
    wallet = shopping.get("wallet", {})
    return copy.deepcopy(wallet)


# -----------------------------
# Tools: orders & membership
# -----------------------------
def prepare_order(
    store: Dict[str, Any],
    voucher_id: Optional[str] = None,
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare an order based on current cart and optional voucher.
    final_total is calculated based on original_total, which has already considered base discounts.
    Creates a pending transaction but does not finalize payment.
    Clear the cart.
    voucher 存在但失败 = 订单照常创建，fail_reasons 记录原因
    voucher 成功 = 按 percent 折扣计算 final_total
    永远不会因为 voucher 失败而拒绝订单（除非你希望做修改）
    """
    profile = store.get("profile", {})
    now = profile.get("now", "")

    summary = _compute_cart_summary(store)
    if summary["total_items"] <= 0:
        return {
            "error": {
                "code": "CART_EMPTY",
                "message": "Your cart is empty. Please add items before preparing an order.",
            }
        }

    original_total = float(summary["total_price"])
    final_total = original_total

    voucher = _find_voucher(store, voucher_id)
    voucher_applied = False
    fail_reasons: List[str] = []
    eligible_subtotal_discounted = 0.0  # Initialize for voucher_internal

    if voucher:
        # VIP requirement
        shopping = _shopping_state(store)
        wallet = shopping.get("wallet", {})
        if voucher.get("vip_required") and not wallet.get("vip", False):
            fail_reasons.append("vip_required")

        # Expiry check
        expiry_str = voucher.get("expiry_date")
        if expiry_str:
            try:
                expiry = date.fromisoformat(expiry_str)
                today = now.date()
                if today > expiry:
                    fail_reasons.append("expired")
            except Exception:
                # bad format -> ignore in demo
                pass

        # Brand & min_amount check
        brand = voucher.get("brand")
        min_amount = float(voucher.get("min_amount", 0.0))

        # Compute eligible subtotal (per brand or whole cart)
        # For min_amount check: use original price (before product discounts)
        # For voucher discount: use discounted price (after product discounts)
        eligible_subtotal_original = 0.0  # For min_amount check
        eligible_subtotal_discounted = 0.0  # For voucher discount calculation
        
        for item in summary["items"]:
            pid = item["product_id"]
            detail = product_details.get(pid, {})
            tags = detail.get("tags", []) or []
            if brand is None or brand in tags:
                # Calculate original price (before product discount)
                size = item.get("size")
                qty = int(item.get("quantity", 0))
                base_price = _get_purchase_option_price(pid, size)
                if base_price is not None and qty > 0:
                    original_subtotal = base_price * qty
                    eligible_subtotal_original += original_subtotal
                # Use discounted subtotal for voucher discount
                eligible_subtotal_discounted += float(item["subtotal"])

        # Check min_amount against original price (before product discounts)
        if eligible_subtotal_original < min_amount:
            fail_reasons.append("min_amount_not_met")

        # Apply discount if all conditions pass
        if not fail_reasons:
            discount_type = voucher.get("discount_type", "percent")
            if discount_type == "percent":
                factor = float(voucher.get("discount_factor", 1.0))
                # Apply voucher discount to discounted eligible subtotal
                discount_amount = eligible_subtotal_discounted * (1.0 - factor)
                final_total = round(original_total - discount_amount, 2)
                voucher_applied = True
            else:
                # Unknown type: do not apply in minimal demo
                pass

    # Create pending transaction
    tx = {
        "transaction_id": generate_transaction_id(_id_seed, _tool_call_name, _tool_call_index),
        "type": "order",
        "status": "pending",
        "items": summary["items"],
        "final_total": round(final_total, 2),
        "voucher_id": voucher_id,
        "voucher_internal": {
            "applied": voucher_applied,
            "fail_reasons": fail_reasons,
            "eligible_subtotal": round(eligible_subtotal_discounted, 2),
        },
    }

    shopping = _shopping_state(store)
    transactions = shopping.get("transactions", [])
    transactions.append(tx)

    cart_state = shopping.get("cart", [])
    cart_state.clear()

    return {"pending_transaction": _public_transaction_view(tx)}


VIP_PRICING = {
    "1_month": 20.0,
    "3_months": 50.0,
    "1_year": 150.0,
}

def upgrade_membership_request(
    store: Dict[str, Any],
    vip_duration: str,
    _tool_call_name: Optional[str] = None,
    _tool_call_index: Optional[int] = None,
    _id_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Request upgrade of membership (VIP).
    Does not finalize payment — creates a pending 'vip_membership' transaction.
    """
    if vip_duration not in VIP_PRICING:
        return {
            "error": {
                "code": "INVALID_VIP_DURATION",
                "message": f"vip_duration must be one of {list(VIP_PRICING.keys())}.",
            }
        }

    amount = VIP_PRICING[vip_duration]

    tx = {
        "transaction_id": generate_transaction_id(_id_seed, _tool_call_name, _tool_call_index),
        "type": "vip_membership",
        "status": "pending",
        "vip_duration": vip_duration,
        "amount": round(amount, 2),
    }

    transactions = _shopping_state(store).get("transactions", [])
    transactions.append(tx)

    return {"pending_transaction": _public_transaction_view(tx)}


def get_transactions(store: Dict[str, Any]) -> Dict[str, Any]:
    """Return history of transactions (orders and membership)."""
    transactions = _shopping_state(store).get("transactions", [])
    return {"transactions": [_public_transaction_view(tx) for tx in transactions]}


# -----------------------------
# Simple manual test
# -----------------------------

if __name__ == "__main__":
    # Example quick checks (will only work if product data files exist)
    _store = {"profile": {"shopping": {"wallet": {}, "cart": [], "transactions": []}}}

    print("Browse page 1:")
    print(browse_items(_store, 1))

    example_pid = product_log[0]["product_id"] if product_log else ""
    if example_pid:
        print("\nInspect first product:")
        print(inspect_item(_store, example_pid))


# ============================================================
# Tool registry for dispatcher
# ============================================================

registered_shopping_tools = {
    "browse_items": browse_items,
    "inspect_item": inspect_item,
    "add_to_cart": add_to_cart,
    "remove_from_cart": remove_from_cart,
    "get_cart": get_cart,
    "get_wallet": get_wallet,
    "prepare_order": prepare_order,
    "upgrade_membership_request": upgrade_membership_request,
    "get_transactions": get_transactions,
}
