from typing import Dict, List, Any
import random
from datetime import datetime, timedelta
from bench.pipeline.generate_task.task_branch_base import TaskBranch
from bench.prompts.generation_prompt import (
    base_task_generation_prompt,
    base_task_valid_prompt,
    platform_overview
)
from bench.utils.misc import strip_code_fences
from bench.utils.generate_ids import (
    generate_appointment_id,
    generate_transaction_id,
    generate_care_plan_id,
    generate_health_provider_id,
    generate_note_id,
    generate_reminder_id,
)
from pathlib import Path
import json


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
    "get_daily_summary",
    "get_range_summary",
    "get_user_profile",
    "update_profile",
    "get_sport_records",
    "get_session_records",
    "get_meal_records",
    "create_session_record",
    "create_meal_record",
    "delete_record",
    "plot_time_series",
    # additional tools
    "get_source_features",
    "get_system_settings",
    "list_daily_reminders",
    "create_daily_reminder",
    "delete_daily_reminder",
    "get_med_features",
    "list_notes",
    "add_note",
    "delete_note",
    "recommend_sports",
    "get_wallet",
    "get_transactions",
    "upgrade_membership_request",
]

involved_source_tool_names_bool = True

involved_med_tool_names_bool = True

notes = [
    "I think I should avoid exercising too late at night, since it tends to affect my sleep.",
    "I should try not to schedule high-intensity workouts on back-to-back days.",
    "I think limiting high-intensity workouts to about three times per week would help me recover better.",
    "I should aim to keep my daily calorie intake beneath 2,500 kJ.",
    "I think I should make time for at least one workout per week that lasts longer than 30 minutes.",
    "I think I should avoid having very sugary breakfasts, since they tend to make my energy drop later in the morning.",
    "I think I should keep my total daily calorie intake within a reasonable range, rather than letting one heavy meal push it too high.",
    "I should try not to let my total daily sugar intake go beyond roughly 50 g, especially from drinks or snacks.",
    "I think keeping my daily fat intake below around 70 g makes my meals feel lighter and easier to recover from.",
    "I think I should try to have more high-fiber meals and generally aim for around 30 g of fiber per day."
]

care_plans = [
    "To protect sleep quality and recovery, avoid scheduling workouts too late in the evening whenever possible.",
    "To reduce injury risk and allow sufficient recovery, avoid performing high-intensity workouts on consecutive days.",
    "To support overall energy expenditure and metabolic health, aim to keep total daily calorie intake beneath 2,500 kJ.",
    "To build endurance capacity, complete at least one workout per week that lasts 30 minutes or longer at a sustained effort.",
    "To prevent overuse and excessive fatigue, avoid extending any single workout beyond 90 minutes.",
    "To support stable morning energy levels, avoid consuming very sugary breakfasts whenever possible.",
    "To maintain balanced energy intake across the day, aim to keep total daily calorie intake within a reasonable range rather than concentrating it in a single heavy meal.",
    "To reduce excessive sugar intake, try to keep total daily sugar consumption below roughly 50 g, with particular attention to drinks and snacks.",
    "To support easier digestion and recovery, aim to keep total daily fat intake below around 70 g.",
    "To promote satiety and digestive health, prioritize higher-fiber meals and aim for approximately 30 g of fiber per day."
]

reminders = [
    "This week, try to avoid very sugary breakfasts to keep your morning energy steady.",
    "This week, try to keep your total daily calorie intake within a reasonable range.",
    "This week, try to keep your daily sugar intake below about 50 g, especially from drinks or snacks.",
    "This week, try to keep your daily fat intake below around 70 g.",
    "This week, try to include more high-fiber meals and aim for roughly 30 g of fiber per day."
]

care_plan_topics = [
    ["exercise", "diet"],
]


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


def get_involved_platform_tool_names() -> List[str]:
    """
    Branch tool allowlist for non-inference trajectory runs.
    Other branches should expose the same helper or define `involved_tool_names`.
    """
    return list(involved_tool_names)

lifestyle_record_advanced_generator_prompt = """
### Branch-Specific Instructions
If this is a generation task, produce exactly one executable lifestyle-monitoring task that strictly follows all instructions below.
If this is a validation or refinement task, verify whether the provided task satisfies all instructions.

#### Time Range Coverage
You should actively use **multiple, varied time ranges** when composing a single task to encourage diversity and realism. Prefer mixing different human-like time windows within the same task, such as:
- relative ranges (e.g., “the past three days”, “the last two weeks”),
- calendar-based references (e.g., “since last Monday”, “over the past week”),
- or a combination of both (e.g., “last week vs. the past five days”).
You should try to involve more than 2 different time ranges when composing the task.

### User Configuration
User configuration provides existing meal records, sport records, and profile information (preference, health risks, dietary restrictions...).
All questions, comparisons, or references to existing records must be grounded in the provided **user configuration**.

When referring to a specific meal or sport session, use **coarse, human-like time descriptions** (e.g., “last Friday’s evening run”, “breakfast 2 days ago”) rather than record IDs or exact timestamps.

#### Create Tasks
The composed second-person task_instruction (You’re… You want the assistant to help you… You expect the assistant to…) should combine 2~3 following action types.
A should be your main focus. However, always try to introduce B, C, D as side quests. You can ask the assistant to:
A. Record Review (Meal / Sport)
- Review recent **meal and/or sport records** and identify subsets or extremes (e.g., snack days, highest-calorie breakfasts, late meals).
- Perform **coarse comparisons** (e.g., calorie intake on days with vs. without snacks; calories burned with vs. without workouts).
- Use recent meal records to request **food recommendations**.
- Visualize a recorded sport session**, showing effort distribution or minute-level intensity.
- Sync AZM or calorie-burn goals** across all connected source apps.
B. Visualization for Insight
- Request **simple visualizations** to illustrate patterns or comparisons.
- Visualized data may include daily calorie intake, calories burned, workout AZM, or other high-level metrics.
C. Notes & Reminders (Read / Write)
- Retrieve, create, update, or cancel notes and reminders.
- Examples include reviewing recent activity and then adding a note or setting a reminder, or retrieving an existing note/reminder to check goal adherence.
D. Care Plans & Appointments (Read / Write)
- Retrieve, create, update, or cancel care plans and healthcare appointments.
- Examples include retrieving a doctor-provided care plan and checking goal completion, or reviewing recent activity and rescheduling an existing appointment.

"""


few_shot_examples = [
    {
      "label": "appointment_management_with_activity_context_note_and_reminder",
      "task_instruction": (
        "You're Samantha. You want to schedule a healthcare appointment and make sure it doesn’t conflict with your usual workout routine in the past 2 weeks.\n"
        "First, you want the assistant to review your recent activity pattern to recommend a time that won’t disrupt your workouts. Then you expect the assistant to find an appropriate provider "
        "and create an appointment. After scheduling, save a note with the appointment details and set a reminder so you don’t forget."
      ),
      "targets": [
        "Retrieve sport records over the past 2 weeks through today to infer typical workout time-of-day (e.g., mornings vs evenings).",
        "Retrieve provider resources to find available appointment slots.",
        "Create an appointment for a slot that avoids the inferred typical workout window.",
        "Add a note with: provider, appointment time, and the reasoning (chosen to avoid usual workout window).",
        "Create a daily reminder leading up to the appointment date (e.g., reminder the day before and the morning-of, expressed as daily reminders if that’s the available primitive)."
      ]
    },
    {
        "label": "appointment_recreation_after_azm_review_visualization_and_goal_reset",
        "task_instruction": "You're Alex. You want to evaluate how well you've been meeting your Active Zone Minutes (AZM) goals over the past two weeks. You expect the assistant to review your AZM completion day by day for the past 14 days and visualize the total AZM for each day so you can clearly see your consistency and peaks. Based on this review, you want the assistant to adjust and set a new AZM goal across all your connected activity app sources that better reflects your current training load. After updating the goal, you want the assistant to cancel your existing healthcare appointment, check which healthcare provider previously issued your care plan, and then create a new appointment with that same provider.",
        "targets": [
            "Retrieve daily AZM summaries for the past 14 days through today",
            "Visualize total daily AZM over the two-week period",
            "Assess daily AZM completion patterns to inform a realistic goal adjustment and sync the updated AZM goal across all connected activity app sources",
            "Retrieve the existing upcoming healthcare appointment and cancel it",
            "Retrieve prior care plans and identify the healthcare provider who issued them",
            "Create a new healthcare appointment with the identified provider"
        ],
    },
    {
        "label": "snacking_days_intake_burn_goal_check_with_azm_adjustment",
        "targets": [
            "Identify days with snack-related meal records in the past week",
            "Compare calories intake vs calories burned on those days",
            "Check daily calorie intake and burn goal completion for those days",
            "Summarize whether snacking days tend to exceed burn or not",
            "Review Active Zone Minutes (AZM) goal completion over the past two weeks",
            "Propose a more realistic AZM goal based on recent performance and sync it to all connected source apps"
        ],
        "task_instruction": (
            "You’re Maya. You want the assistant to help you understand whether snacking is disrupting your weekly balance. "
            "You expect the assistant to review your meal records from the past 7 days and identify which days include snacks. "
            "For those snack days, you want to compare calories intake versus calories burned and see which is higher. "
            "You also want to check whether you met your daily intake and burn goals on those days. "
            "In addition, you want to step back and review how consistently you’ve met your Active Zone Minutes goal over the past two weeks. "
            "Based on that pattern, you want the assistant to suggest a more realistic AZM goal and sync it across all your connected source apps. "
            "Your purpose is to decide whether you need stricter snack control or a small adjustment to your overall activity targets."
        )
    },
    {
        "label": "breakfast_intensity_activity_comparison_with_notes_and_care_plan_check",
        "targets": [
            "Rank days in the previous week by breakfast calories and select the top three",
            "Review workout presence on those high-breakfast days",
            "Compare activity patterns between high-breakfast days and remaining days",
            "Check daily goal completion differences between the two groups",
            "Retrieve the most recent personal note and doctor-provided healthcare advice",
            "Check whether the goals or guidance mentioned there were met over the past five days"
        ],
        "task_instruction": (
            "You’re Daniel Reyes. You want the assistant to help you test whether heavier breakfasts are linked to being more active. "
            "You expect the assistant to review your meal records from last week and find the three days with the most calories at breakfast. "
            "You then want to see how those days compare to the rest of the week in terms of workouts and overall activity. "
            "You also want to compare daily goal completion between these two groups. "
            "In addition, you want to retrieve your most recent personal note and the healthcare advice provided by your doctor, "
            "and check whether you’ve followed that guidance over the past five days. "
            "Your purpose is to decide whether you should intentionally eat more at breakfast while staying aligned with your health guidance."
        )
    },
    {
        "label": "evening_meal_workout_pattern_with_visualization_and_appointment",
        "targets": [
            "Identify days with notable evening food intake since last Wednesday",
            "Check whether those days include evening workouts",
            "Compare calorie intake and burn patterns on those days",
            "Visualize evening-related calorie intake and consumption trends",
            "Update profile preferences based on observed patterns",
            "Create a healthcare appointment related to evening fatigue or recovery concerns"
        ],
        "task_instruction": (
            "You’re Alex. You want the assistant to help you understand whether late-evening eating has been clashing with your evening workouts "
            "and leaving you overly tired lately. You expect the assistant to review your records starting from last Wednesday and identify days with "
            "noticeable evening food intake, then see how those days line up with evening workouts. "
            "You also want a simple visualization comparing calorie intake and calories burned across those days to make the pattern clear. "
            "Based on what’s observed, you want the assistant to update your profile with one practical evening preference to reduce this fatigue cycle. "
            "Because the fatigue has been lingering and you want professional input, you also want the assistant to help you schedule a short healthcare "
            "appointment to discuss evening recovery and energy management. "
            "Your purpose is to build a more sustainable evening routine and make sure nothing important is being overlooked."
        )
    },
    {
      "label": "healthy_meal_selection_note_and_longest_workout_minute_visualization",
      "targets": [
        "Review meal records from the past three days",
        "Select the three healthiest meal records based on nutritional balance",
        "Recommend one suitable food choice based on those meals",
        "Add a note summarizing the selected meals and the recommended food",
        "Retrieve sport records from the past week",
        "Identify the single workout session with the longest duration",
        "Visualize minute-level activity intensity for that longest workout session"
      ],
      "task_instruction": (
        "You’re Alex. You want the assistant to help you reflect on how well you’ve been eating recently and how that fits with your training. "
        "You expect the assistant to review your meal records from the past three days and pick out the three meals that look the healthiest overall. "
        "Based on those meals, you want the system to recommend one food choice that would fit well with your current eating pattern. "
        "You also want the assistant to save a short note summarizing the selected meals and the recommended food so you can refer back to it later. "
        "After that, you want to look at your workouts from the past week, identify the single session that lasted the longest, "
        "and visualize its minute-by-minute activity intensity to better understand how effort was distributed across that workout. "
        "Your purpose is to reinforce good eating habits while staying aware of how your longer workouts are actually paced."
      )
    },
    {
      "label": "two_week_activity_frequency_meal_review_and_goal_sync",
      "targets": [
        "Compare sport records between this week and last week to determine which period had more workout sessions",
        "Summarize differences in workout frequency and overall activity level between the two weeks",
        "Review meal records from the current week to assess overall dietary patterns",
        "Select meal records from the most recent three days and use them to recommend a suitable food choice",
        "Sync the user’s calorie burn goal across all connected source apps",
        "Sync the user’s Active Zone Minutes (AZM) goal across all connected source apps"
      ],
      "task_instruction": (
        "You’re Alex. You want the assistant to help you reflect on how your activity and eating habits have been trending recently. "
        "You expect the assistant to review your sport records from this week and last week, "
        "and compare the two periods to see which week you worked out more frequently and how your overall activity level differed. "
        "At the same time, you want to take a closer look at your eating habits this week by reviewing your meal records "
        "and identifying any clear patterns. "
        "From there, you want the assistant to focus on your most recent three days of meals "
        "and recommend one food choice that would fit well with how you’ve been eating lately. "
        "Finally, based on this broader review of your activity and diet, "
        "you want to make sure your calorie burn goal and your Active Zone Minutes goal are properly synced "
        "across all of your connected source apps. "
        "Your purpose is to keep your goals aligned with how you’re actually living week to week."
      )
    }
]

"""
{
        "label": "create_recent_meal_records_note_and_peak_workout_visualization",
        "targets": [
            "Create missing meal records with explicit food items and timestamps",
            "Review the updated meal records over the past three days and add a note summarizing recent dietary patterns",
            "Retrieve all sport records from the current week",
            "Identify the single workout session with the highest overall intensity",
            "Visualize detailed minute-level activity intensity patterns for that highest-intensity workout"
        ],
        "task_instruction": (
            "You’re Alex. You want the assistant to help you fill in missing meals so you can better understand recent patterns. "
            "You expect the assistant to create meal records for the past three days: "
            "on Tuesday at 08:00, breakfast with oatmeal (60 g) and almond milk (200 ml); "
            "on Wednesday at 13:00, lunch with grilled chicken (180 g) and mixed vegetables (150 g); "
            "and on Thursday at 19:30, dinner with rice (200 g) and tofu (120 g). "
            "After recording these meals, you want to review the updated meal records from the past three days "
            "and save a short note summarizing what your recent eating pattern looks like. "
            "You also want to step back and review all of your workouts from this week, "
            "identify the single session that felt the most intense, "
            "and visualize its detailed minute-level activity pattern to better understand where that effort came from. "
            "Your purpose is to make your recent data complete enough to guide small routine adjustments."
        )
    }
    {
        "label": "create_recent_meal_records_note_and_peak_workout_visualization",
        "targets": [
            "Create missing meal records with explicit food items and timestamps",
            "Review the updated meal records over the past three days and add a note summarizing recent dietary patterns",
            "Retrieve all sport records from the current week",
            "Identify the single workout session with the highest overall intensity",
            "Visualize detailed minute-level activity intensity patterns for that highest-intensity workout"
        ],
        "task_instruction": (
            "You’re Alex. You want the assistant to help you fill in missing meals so you can better understand recent patterns. "
            "You expect the assistant to create meal records for the past three days: "
            "on Tuesday at 08:00, breakfast with oatmeal (60 g) and almond milk (200 ml); "
            "on Wednesday at 13:00, lunch with grilled chicken (180 g) and mixed vegetables (150 g); "
            "and on Thursday at 19:30, dinner with rice (200 g) and tofu (120 g). "
            "After recording these meals, you want to review the updated meal records from the past three days "
            "and save a short note summarizing what your recent eating pattern looks like. "
            "You also want to step back and review all of your workouts from this week, "
            "identify the single session that felt the most intense, "
            "and visualize its detailed minute-level activity pattern to better understand where that effort came from. "
            "Your purpose is to make your recent data complete enough to guide small routine adjustments."
        )
    }
    {
      "label": "meal_record_cleanup_session_creation_profile_and_weekly_azm_review",
      "targets": [
        "Update or delete inaccurate meal records",
        "Create lifestyle session records with clear activity descriptions",
        "Check alignment between records and profile preferences",
        "Summarize daily Active Zone Minutes (AZM) goal completion over the past week"
      ],
      "task_instruction": (
        "You’re Samantha. You want the assistant to help you clean up your records so they reflect what actually happened. "
        "You expect the assistant to remove a dinner record from last Saturday that was logged by mistake, "
        "and update a lunch record from Sunday to reflect that you ate a smaller portion than originally recorded. "
        "You also want to create session records for recent non-sport activities: "
        "a personal errands session on Tuesday from 16:00 to 17:30, "
        "and a focused home organization session on Thursday from 20:00 to 21:00. "
        "After that, you want to review whether these patterns still align with your profile food and activity preferences, "
        "and make a small update if needed. "
        "You also want to step back and summarize how consistently you’ve met your daily Active Zone Minutes goal over the past week. "
        "Your purpose is to keep your records, activity patterns, and preferences consistent."
      )
    }
    {
      "label": "meal_pattern_activity_intensity_comparison",
      "targets": [
        "Identify a subset of days based on a meal-related pattern",
        "Compare steps and METs between selected days and remaining days",
        "Assess differences in daily activity intensity",
        "Check goal completion differences across the two groups"
      ],
      "task_instruction": (
        "You’re Samantha. You want the assistant to help you understand how your eating patterns relate to how active you are during the day. "
        "You expect the assistant to review your recent meal records and identify a meaningful subset of days, such as days with heavier dinners "
        "or higher overall intake. You then want to compare those days with the remaining days using steps and METs to see whether activity "
        "intensity differs. You also want to check how daily goal completion compares between the two groups. "
        "Your purpose is to adjust your eating habits to better support consistent daily activity."
      )
    },
    {
        "label": "create_work_study_sessions_and_compare_with_sport",
        "targets": [
            "Create multiple lifestyle session records with explicit timestamps",
            "Compare sport activity levels on days with heavy work or study sessions",
            "Review daily goal completion differences on those days"
        ],
        "task_instruction": (
            "You’re Daniel Reyes. Your purpose is to see whether work intensity is crowding out movement. You want the assistant to help you understand how your busy work and study days affect your activity. "
            "You expect the assistant to create session records for the following activities you remember from the past week: "
            "a focused work session on Monday from 09:00 to 12:00, a study session on Wednesday from 19:30 to 21:00, "
            "and a long work block on Friday from 10:00 to 14:00. "
            "After recording these sessions, you want to compare those days with your sport records to see whether physical activity "
            "tends to drop or stay stable on heavy work or study days. You also want to check how your daily goals look on those days. "
        )
    },
     {
      "label": "healthy_meal_selection_note_and_longest_workout_minute_visualization",
      "targets": [
        "Review meal records from the past three days",
        "Select the three healthiest meal records based on nutritional balance",
        "Recommend one suitable food choice based on those meals",
        "Add a note summarizing the selected meals and the recommended food",
        "Retrieve sport records from the past week",
        "Identify the single workout session with the longest duration",
        "Visualize minute-level activity intensity for that longest workout session"
      ],
      "task_instruction": (
        "You’re Alex. You want the assistant to help you reflect on how well you’ve been eating recently and how that fits with your training. "
        "You expect the assistant to review your meal records from the past three days and pick out the three meals that look the healthiest overall. "
        "Based on those meals, you want the system to recommend one food choice that would fit well with your current eating pattern. "
        "You also want the assistant to save a short note summarizing the selected meals and the recommended food so you can refer back to it later. "
        "After that, you want to look at your workouts from the past week, identify the single session that lasted the longest, "
        "and visualize its minute-by-minute activity intensity to better understand how effort was distributed across that workout. "
        "Your purpose is to reinforce good eating habits while staying aware of how your longer workouts are actually paced."
      )
    },
    """


lifestyle_record_advanced_tool_selection_prompt = """
### Tool Selection Instructions (wearable_data_advanced)
The assistant must select tools carefully based on data availability, permissions,
and task intent. Follow the rules below strictly.

#### 1. Permission Handling
- If required data or operations are restricted by permissions, and a corresponding
  user tool exists, the assistant must first clearly explain what permission is needed
  and then prompt the user to perform the appropriate user action before proceeding.
Available user actions include:
- update_source(source_name, allow)
- set_raw_data_permission(allow)
- set_user_notes_permission(allow)
- set_med_assistant_permission(allow)

#### 2. Minute-Level Analysis
To determine which minute-level data tools are available, first use `get_source_features`
to inspect supported intraday / minute-level capabilities.

#### 3. Healthcare / Medical Information
For tasks involving healthcare providers, doctor guidance, medical records, care plans, 
or appointments, always start by calling `get_med_features` to identify available medical
data and supported operations.

#### 4. Missing or Unavailable Data Reasoning
When encountering missing or unavailable data, you can choose to:
- Call get_system_settings to inspect marketplace connections and overall system configuration.
If at least one data source is connected, prompt the user to connect one additional disconnected source using update_source(source_name, allow=True).
If data remains unavailable after this step, the requested data may originate from another still-disconnected source.
If data remains unavailable after all the sources are connected, conclude that the data was not recorded for the requested period.
- Use `get_daily_summary` (or range summaries) to inspect day-level visibility
    and infer whether data is missing due to disconnection or true absence.

#### 5. Membership and Payment Flow
- If an operation requires VIP membership:
  - First check membership status using `get_wallet`.
  - If VIP is not enabled, first ask the user whether they want to upgrade. Only after the user explicitly approves the
  membership upgrade and specifies the VIP duration should you initiate upgrade_membership_request. You must wait for an
  explicit user decision before proceeding.
- If wallet balance is insufficient:
  - Prompt the user to top up using `top_up_wallet(amount)`.
  - Retrieve the resulting transaction identifier using 'get_transactions'.
  - Ask the user to authorize the transaction with authorize_checkout(transaction_id)`.
- Do NOT proceed with restricted operations until payment and authorization are fully completed.

#### 6. General Tool Selection Principles
- You are NOT allowed to call user tools! You MUST prompt the user to call intended tools.
- Prefer higher-level, aggregated tools whenever they can satisfy the task.
- Avoid redundant or unnecessary tool calls.
- Select tools that align with the temporal granularity required by the task (range → daily → hourly → minute).
- All tool calls must be logically justified by the task context.
"""


def filter_sport_records(sports, now_str):
    now = datetime.fromisoformat(now_str)

    # 若数量 < 15，保留全部
    if len(sports) < 15:
        return [
            {
                "sport_name": s["sport_name"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
            }
            for s in sports
        ]

    # 否则仅保留最近 10 天
    cutoff = now - timedelta(days=10)
    filtered = []
    for s in sports:
        start = datetime.fromisoformat(s["start_time"])
        if start >= cutoff:
            filtered.append({
                "sport_name": s["sport_name"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
            })
    return filtered


def filter_meal_records(meals, now_str):
    now = datetime.fromisoformat(now_str)
    cutoff = now - timedelta(days=3)
    filtered = []
    for m in meals:
        meal_time = datetime.fromisoformat(m["time"])
        if meal_time >= cutoff:
            nutrition = m.get("nutrition", {}) if isinstance(m, dict) else {}
            filtered.append({
                "record_id": m.get("record_id"),
                "time": m.get("time"),
                "meal_type": m.get("meal_type"),
                "items": m.get("items", []),
                "total_calories": nutrition.get("calories"),
            })
    return filtered

def _pick_provider_id(profile: Dict[str, Any], rng: random.Random) -> str:
    providers = profile.get("health_providers") or []
    if providers:
        return rng.choice(providers).get("provider_id")
    return generate_health_provider_id()


def _ensure_health_provider_template(profile: Dict[str, Any], rng: random.Random) -> str:
    providers = profile.get("health_providers") or []
    if providers:
        return rng.choice(providers).get("provider_id")

    provider_id = generate_health_provider_id()
    providers.append({
        "provider_id": provider_id,
        "doctor": rng.choice(["Dr. Lee", "Dr. Chen", "Dr. Patel"]),
        "clinic": rng.choice(["Family Clinic", "Wellness Center", "Community Health"]),
        "address": rng.choice(["Main St 12", "River Rd 8", "Oak Ave 23"]),
    })
    profile["health_providers"] = providers
    return provider_id


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


def _inject_context_for_tasks(store: Dict[str, Any]) -> None:
    meta = store.get("meta", {})
    injected = meta.setdefault("injected_context", {})
    if injected.get("wearable_data_advanced"):
        return

    rng = random.Random(meta.get("rng_seed"))
    profile = store.get("profile", {})
    used_times: set[str] = set()
    now_str = profile.get("now")
    now_dt = datetime.fromisoformat(now_str)

    # Notes: sample 2-3 and inject
    if notes:
        note_count = min(len(notes), rng.randint(2, 3))
        notes_list = profile.get("notes") or []
        for note in rng.sample(notes, k=note_count):
            note_time = _unique_timestamp(now_dt, used_times, rng, day_window=6)
            notes_list.append({
                "note_id": generate_note_id(),
                "time": note_time,
                "content": note,
            })
        profile["notes"] = notes_list

    # Care plans: sample 2-3 and inject
    care_plans_list = profile.get("care_plans") or []
    if care_plans:
        plan_count = min(len(care_plans), rng.randint(2, 3))
        provider_id = _ensure_health_provider_template(profile, rng)
        for _text in rng.sample(care_plans, k=plan_count):
            created_at = _unique_timestamp(now_dt, used_times, rng, day_window=6)
            care_plans_list.append({
                "plan_id": generate_care_plan_id(),
                "provider_id": provider_id,
                "created_at": created_at,
                "topics": ["exercise"],
                "note_text": _text,
            })
        profile["care_plans"] = care_plans_list

    # Reminders: sample 1 and inject
    if reminders:
        repeat_days = rng.sample(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            k=3,
        )
        reminders_list = profile.get("reminders") or []
        reminders_list.append({
            "reminder_id": generate_reminder_id(),
            "title": rng.choice(reminders),
            "time_of_day": rng.choice(["07:00", "12:00", "18:00"]),
            "repeat_days": repeat_days,
        })
        profile["reminders"] = reminders_list

    # Appointment: inject one record at now + 3 days, 12:00
    appointments = profile.get("appointments") or []
    appointment_date = (now_dt + timedelta(days=3)).date().strftime("%Y-%m-%d")
    appointment_provider_id = _ensure_health_provider_template(profile, rng)
    appointments.append({
        "appointment_id": generate_appointment_id(),
        "provider_id": appointment_provider_id,
        "date": appointment_date,
        "time": "12:00",
        "duration": 60,
        "reason": "health consultation",
        "status": "confirmed",
        "transaction_id": generate_transaction_id(),
    })
    profile["appointments"] = appointments

    injected["wearable_data_advanced"] = True

class LifeStyleAdvanced(TaskBranch):
    branch_name = "lifestyle_record_advanced"

    def is_applicable(self, store: Dict[str, Any]) -> bool:
        """
        Only run when there is at least one accessible date.
        """
        return True

    def build_prompt(self, store: Dict[str, Any]) -> str:
        pass

    def _build_generation_prompt(
        self,
        user_configuration: Dict[str, Any],
        sampled_few_shots: List[Dict[str, Any]],
    ) -> str:
        sections = [
            base_task_generation_prompt,
            platform_overview,
            lifestyle_record_advanced_generator_prompt,
            "### Tools\n" + json.dumps(involved_tool_schemas, indent=2),
            "### Few-shot Task Examples\n"
            + json.dumps(sampled_few_shots, indent=2),
            "### Input User Configuration\n" + json.dumps(user_configuration, indent=2),
        ]
        return "\n\n---\n\n".join(sections)

    def _build_valid_prompt(
        self,
        user_configuration: Dict[str, Any],
        sampled_few_shots: List[Dict[str, Any]],
        initial_task: str,
    ) -> str:
        sections = [
            base_task_valid_prompt,
            platform_overview,
            lifestyle_record_advanced_generator_prompt,
            "### Tools\n" + json.dumps(involved_tool_schemas, indent=2),
            "### Input User Configuration\n" + json.dumps(user_configuration, indent=2),
            "### Candidate Task (JSON)\n" + initial_task,
        ]
        return "\n\n---\n\n".join(sections)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        return json.loads(strip_code_fences(text))

    def run(self, store: Dict[str, Any]) -> List[Dict[str, Any]]:
        for source_dict in store["profile"]["system_settings"]["marketplaces"]:
            source_dict["connected"] = False

        _inject_context_for_tasks(store)

        user_configuration = {
            "user_profile": store["profile"]["user_profile"],
            "user_persona": store["profile"]["narrative_summary"],
            "now": store["profile"]["now"],
            "sport_records": filter_sport_records(store["profile"]["sports"], store["profile"]["now"]),
            "meal_records": filter_meal_records(store["profile"]["meals"], store["profile"]["now"]),
            "notes": store["profile"].get("notes", []),
            "care_plans": store["profile"].get("care_plans", []),
            "reminders": store["profile"].get("reminders", []),
            "appointments": store["profile"].get("appointments", []),
        }


        # Try at most 3 times. Any failure (gen/valid/parse or invalid decision) => continue.
        for _ in range(3):
            # Step 1: generate
            try:
                sampled_few_shots = random.sample(
                    few_shot_examples, k=min(3, len(few_shot_examples))
                )
                gen_prompt = self._build_generation_prompt(user_configuration, sampled_few_shots)
                current_task_str = self.call_llm(gen_prompt)
                current_task = self._parse_json(current_task_str)
                required_keys = ["task_instruction", "label", "targets"]
                missing = [k for k in required_keys if k not in current_task]
                if missing:
                    continue
            except Exception:
                continue

            # Step 2: validate + (optional) rewrite-on-invalid
            # Per generation_prompt.py: when task_valid/alignment is False, the validator may rewrite
            # task_instruction and targets with minimal necessary changes. We apply and re-validate once.
            validated_ok = False
            for _rewrite_round in range(2):
                try:
                    valid_prompt = self._build_valid_prompt(
                        user_configuration=user_configuration,
                        sampled_few_shots=sampled_few_shots,
                        initial_task=current_task_str,
                    )
                    valid_output = self.call_llm(valid_prompt)
                    valid_decision = self._parse_json(valid_output)
                except Exception:
                    break

                task_valid = valid_decision.get("task_valid")
                alignment = valid_decision.get("alignment")

                if (task_valid is True) and (alignment is True):
                    validated_ok = True
                    break

                # try rewrite if provided
                rewritten_instruction = valid_decision.get("task_instruction")
                rewritten_targets = valid_decision.get("targets")
                if rewritten_instruction or rewritten_targets:
                    if rewritten_instruction:
                        current_task["task_instruction"] = rewritten_instruction
                    if rewritten_targets:
                        current_task["targets"] = rewritten_targets
                    current_task_str = json.dumps(current_task, ensure_ascii=False, indent=2)
                    continue

                # no rewrite info -> regenerate
                break

            if not validated_ok:
                continue

            return [current_task]

        # all attempts failed
        return []

    def postprocess(
            self,
            llm_output: str,
            store: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
       pass


