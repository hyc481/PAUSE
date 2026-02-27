from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Tuple
from bench.utils.generate_ids import generate_meal_id


# ============================================================
# Atom meals (meal-level nutrition, items are display-only)
# ============================================================

ATOM_MEALS = [
    # ===================== breakfast =====================
    {
        "meal_id": "bf_eggs_toast",
        "meal_type": "breakfast",
        "items": [
            {"name": "Scrambled Eggs", "amount_g": 120},
            {"name": "Whole Wheat Toast", "amount_g": 60},
        ],
        "nutrition": {"calories": 480, "sugar_g": 5.0, "fiber_g": 5.5, "fat_g": 20.0},
    },
    {
        "meal_id": "bf_oatmeal_fruit",
        "meal_type": "breakfast",
        "items": [
            {"name": "Oatmeal", "amount_g": 250},
            {"name": "Blueberries", "amount_g": 80},
        ],
        "nutrition": {"calories": 520, "sugar_g": 10.0, "fiber_g": 10.0, "fat_g": 12.0},
    },
    {
        "meal_id": "bf_yogurt_granola",
        "meal_type": "breakfast",
        "items": [
            {"name": "Greek Yogurt", "amount_g": 220},
            {"name": "Granola", "amount_g": 60},
        ],
        "nutrition": {"calories": 560, "sugar_g": 14.0, "fiber_g": 6.0, "fat_g": 16.0},
    },
    {
        "meal_id": "bf_avocado_toast",
        "meal_type": "breakfast",
        "items": [
            {"name": "Avocado Toast", "amount_g": 160},
        ],
        "nutrition": {"calories": 500, "sugar_g": 3.5, "fiber_g": 8.0, "fat_g": 22.0},
    },
    {
        "meal_id": "bf_pancake_syrup",
        "meal_type": "breakfast",
        "items": [
            {"name": "Pancakes", "amount_g": 180},
            {"name": "Maple Syrup", "amount_g": 30},
        ],
        "nutrition": {"calories": 620, "sugar_g": 26.0, "fiber_g": 3.0, "fat_g": 18.0},
    },
    {
        "meal_id": "bf_smoothie_banana",
        "meal_type": "breakfast",
        "items": [
            {"name": "Banana Smoothie", "amount_g": 350},
        ],
        "nutrition": {"calories": 420, "sugar_g": 22.0, "fiber_g": 6.0, "fat_g": 8.0},
    },
    {
        "meal_id": "bf_peanut_butter_toast",
        "meal_type": "breakfast",
        "items": [
            {"name": "Toast", "amount_g": 80},
            {"name": "Peanut Butter", "amount_g": 30},
        ],
        "nutrition": {"calories": 540, "sugar_g": 7.0, "fiber_g": 5.0, "fat_g": 26.0},
    },
    {
        "meal_id": "bf_cereal_milk",
        "meal_type": "breakfast",
        "items": [
            {"name": "Breakfast Cereal", "amount_g": 90},
            {"name": "Milk", "amount_g": 200},
        ],
        "nutrition": {"calories": 480, "sugar_g": 18.0, "fiber_g": 4.0, "fat_g": 14.0},
    },

    # ===================== lunch / dinner =====================
    {
        "meal_id": "ld_chicken_rice_broccoli",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Grilled Chicken Breast", "amount_g": 150},
            {"name": "Steamed Rice", "amount_g": 200},
            {"name": "Broccoli", "amount_g": 120},
        ],
        "nutrition": {"calories": 540, "sugar_g": 4.0, "fiber_g": 7.0, "fat_g": 9.0},
    },
    {
        "meal_id": "ld_salmon_quinoa",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Baked Salmon", "amount_g": 160},
            {"name": "Quinoa", "amount_g": 180},
        ],
        "nutrition": {"calories": 640, "sugar_g": 5.0, "fiber_g": 8.0, "fat_g": 24.0},
    },
    {
        "meal_id": "ld_beef_noodles",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Beef Stir Fry", "amount_g": 180},
            {"name": "Noodles", "amount_g": 240},
        ],
        "nutrition": {"calories": 760, "sugar_g": 7.0, "fiber_g": 5.0, "fat_g": 24.0},
    },
    {
        "meal_id": "ld_pasta_meat_sauce",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Pasta", "amount_g": 260},
            {"name": "Meat Sauce", "amount_g": 180},
        ],
        "nutrition": {"calories": 820, "sugar_g": 11.0, "fiber_g": 6.0, "fat_g": 28.0},
    },
    {
        "meal_id": "ld_tofu_veg_rice",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Tofu", "amount_g": 180},
            {"name": "Stir-fried Vegetables", "amount_g": 220},
            {"name": "Brown Rice", "amount_g": 200},
        ],
        "nutrition": {"calories": 620, "sugar_g": 7.0, "fiber_g": 10.0, "fat_g": 18.0},
    },
    {
        "meal_id": "ld_fried_chicken",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Fried Chicken", "amount_g": 220},
        ],
        "nutrition": {"calories": 780, "sugar_g": 3.0, "fiber_g": 2.0, "fat_g": 42.0},
    },
    {
        "meal_id": "ld_pizza",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Pizza Slices", "amount_g": 300},
        ],
        "nutrition": {"calories": 880, "sugar_g": 12.0, "fiber_g": 4.0, "fat_g": 36.0},
    },
    {
        "meal_id": "ld_burrito_bowl",
        "meal_type": "lunch_dinner",
        "items": [
            {"name": "Burrito Bowl", "amount_g": 420},
        ],
        "nutrition": {"calories": 700, "sugar_g": 6.0, "fiber_g": 9.0, "fat_g": 22.0},
    },

    # ===================== snack =====================
    {
        "meal_id": "sn_fruit_nuts",
        "meal_type": "snack",
        "items": [
            {"name": "Apple", "amount_g": 160},
            {"name": "Mixed Nuts", "amount_g": 30},
        ],
        "nutrition": {"calories": 320, "sugar_g": 18.0, "fiber_g": 6.0, "fat_g": 16.0},
    },
    {
        "meal_id": "sn_yogurt",
        "meal_type": "snack",
        "items": [
            {"name": "Low Sugar Yogurt", "amount_g": 220},
        ],
        "nutrition": {"calories": 240, "sugar_g": 9.0, "fiber_g": 0.0, "fat_g": 6.0},
    },
    {
        "meal_id": "sn_cookie",
        "meal_type": "snack",
        "items": [
            {"name": "Cookies", "amount_g": 70},
        ],
        "nutrition": {"calories": 360, "sugar_g": 22.0, "fiber_g": 1.5, "fat_g": 18.0},
    },
    {
        "meal_id": "sn_chips",
        "meal_type": "snack",
        "items": [
            {"name": "Potato Chips", "amount_g": 60},
        ],
        "nutrition": {"calories": 340, "sugar_g": 2.0, "fiber_g": 2.0, "fat_g": 22.0},
    },
    {
        "meal_id": "sn_protein_bar",
        "meal_type": "snack",
        "items": [
            {"name": "Protein Bar", "amount_g": 60},
        ],
        "nutrition": {"calories": 280, "sugar_g": 7.0, "fiber_g": 5.0, "fat_g": 10.0},
    },
    {
        "meal_id": "sn_ice_cream",
        "meal_type": "snack",
        "items": [
            {"name": "Ice Cream", "amount_g": 120},
        ],
        "nutrition": {"calories": 300, "sugar_g": 24.0, "fiber_g": 1.0, "fat_g": 16.0},
    },
]



# ============================================================
# Meal synthesis
# ============================================================
def _jitter_time_str(
    day_str: str,
    base_hhmm: str,
    rng: random.Random,
    jitter_min: int,
) -> str:
    """
    day_str: "YYYY-MM-DD"
    base_hhmm: "HH:MM"
    jitter_min: e.g., 30 means +/- 30 minutes
    """
    base_dt = datetime.strptime(f"{day_str}T{base_hhmm}", "%Y-%m-%dT%H:%M")
    dt = base_dt + timedelta(minutes=rng.randint(-jitter_min, jitter_min))
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def _scale_items(items, scale):
    out = []
    for it in items:
        out.append({
            "name": it["name"],
            "amount": {
                "value": round(it["amount_g"] * scale),
                "unit": "g",
            }
        })
    return out


def generate_meals(
    date_range: Tuple[str, str],
    rng: random.Random,
    snack_prob: float = 0.3,
    scale_low: float = 0.9,
    scale_high: float = 1.1,
    time_jitter: Dict[str, int] | None = None,  # minutes
) -> List[Dict[str, Any]]:

    # 默认每个 slot 的扰动范围（分钟）
    if time_jitter is None:
        time_jitter = {
            "breakfast": 20,  # +/- 20
            "lunch": 30,
            "dinner": 45,
            "snack": 60,
        }

    start = datetime.strptime(date_range[0], "%Y-%m-%d").date()
    end = datetime.strptime(date_range[1], "%Y-%m-%d").date()

    meals: List[Dict[str, Any]] = []

    def pick(meal_type: str) -> Dict[str, Any]:
        return rng.choice([m for m in ATOM_MEALS if m["meal_type"] == meal_type])

    cur = start
    while cur <= end:
        day = cur.strftime("%Y-%m-%d")

        plan = [
            ("breakfast", "07:30", "breakfast"),
            ("lunch_dinner", "12:30", "lunch"),
            ("lunch_dinner", "18:30", "dinner"),
        ]
        if rng.random() < snack_prob:
            plan.append(("snack", "15:30", "snack"))

        for meal_type, base_hhmm, slot in plan:
            atom = pick(meal_type)
            scale = rng.uniform(scale_low, scale_high)
            ts = _jitter_time_str(day, base_hhmm, rng, time_jitter.get(slot, 0))

            meals.append({
                "record_id": generate_meal_id(),
                "time": ts,
                "meal_type": slot,
                # ---- scaled items (truth) ----
                "items": [
                    {
                        "name": it["name"],
                        "amount_g": int(round(it["amount_g"] * scale/10)*10),
                    }
                    for it in atom["items"]
                ],
                # ---- scaled nutrition (truth) ----
                "nutrition": {
                    k: round(v * scale, 1)
                    for k, v in atom["nutrition"].items()
                },
            })

        cur += timedelta(days=1)

    meals.sort(key=lambda x: x["time"])
    return meals


