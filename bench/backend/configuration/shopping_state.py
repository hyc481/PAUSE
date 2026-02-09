# shopping_state.py
"""
Global state for shopping demo: wallet, cart, transactions.
You can edit these values manually to create different scenarios.
"""

from typing import List, Dict, Any
from datetime import datetime



"""
# shopping state 0
# Wallet: balance, VIP status, and vouchers
WALLET_STATE: Dict[str, Any] = {
    "balance": 0.5,
    "vip": False,
    "vip_expiry": None,
    "vouchers": [
        {
            "voucher_id": "v2025_01",
            "description": "10% off EverydayBite over 30 (VIP required)",
            "brand": "EverydayBite",        # 对应商品 tags 里的品牌
            "vip_required": True,
            "expiry_date": "2025-6-9",    # YYYY-MM-DD
            "min_amount": 10.0,
            "discount_type": "percent",
            "discount_factor": 0.9,         # 再打 9 折
        }
    ],
}

# Cart: product_id -> quantity
CART_STATE: Dict[str, int] = []
"""



# shopping state 1
# Wallet: balance, VIP status, and vouchers
WALLET_STATE: Dict[str, Any] = {
    "balance": 0.5,
    "vip": False,
    "vip_expiry": None,
    "vouchers": [
        {
            "voucher_id": "v2025_01",
            "description": "10% off EverydayBite over 30 (VIP required)",
            "brand": "EverydayBite",        # 对应商品 tags 里的品牌
            "vip_required": True,
            "expiry_date": "2025-6-9",    # YYYY-MM-DD
            "min_amount": 10.0,
            "discount_type": "percent",
            "discount_factor": 0.9,         # 再打 9 折
        },
        {
            "voucher_id": "v2025_02",
            "description": "15% off PureBalance bundle orders over 60",
            "brand": "PureBalance",
            "vip_required": False,
            "expiry_date": "2025-12-31",
            "min_amount": 60.0,
            "discount_type": "percent",
            "discount_factor": 0.85,
        },
    ],
}

# Cart: product_id -> quantity
CART_STATE: Dict[str, int] = [
    {
        "cart_item_id": "cart_03d73bec",
        "product_id": "p_fbfdd7_002",
        "size": "pack_3",
        "quantity": 1,
    }
]

# Transactions: list of order / vip membership transactions
TRANSACTIONS: List[Dict[str, Any]] = []


def reset_state() -> None:
    """Reset global state to initial values."""
    global WALLET_STATE, CART_STATE, TRANSACTIONS

    WALLET_STATE = {
        "balance": 120.0,
        "vip": False,
        "vip_expiry": None,
        "vouchers": [
            {
                "voucher_id": "v2025_01",
                "description": "10% off EverydayBite over 30 (VIP required)",
                "brand": "EverydayBite",
                "vip_required": True,
                "expiry_date": "2025-12-31",
                "min_amount": 30.0,
                "discount_type": "percent",
                "discount_factor": 0.9,
            }
        ],
    }
    CART_STATE = []
    TRANSACTIONS = []

