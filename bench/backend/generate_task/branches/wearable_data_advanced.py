from typing import Dict, List, Any
import random
from datetime import datetime, timedelta
from bench.backend.generate_task.task_branch_base import TaskBranch
from bench.prompts.generation_prompt import (
    base_task_generation_prompt,
    base_task_valid_prompt,
    platform_overview
)
from bench.backend.utils.misc import strip_code_fences
from bench.backend.utils.generate_ids import (
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
    # Compared to casual, sleep summary removed,
    "get_daily_summary",
    "get_range_summary",
    "get_hourly_mets",
    "get_hourly_steps",
    "get_hourly_calories",
    "get_hourly_activity",
    "get_user_profile",
    "update_profile",
    "get_sport_records",
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
    "I think I'd better make sure I reach at least 20 Active Zone Minutes on most weekdays to stay consistent.",
    "I should keep my daily steps above 7,000, especially on days when I don't have a formal workout.",
    "I think I should avoid exercising too late at night, since it tends to affect my sleep.",
    "I should try not to schedule high-intensity workouts on back-to-back days.",
    "I think limiting high-intensity workouts to about three times per week would help me recover better.",
    "I should aim to keep my daily calorie burn above 2,500 kJ.",
    "I think I need to make sure my daily METs consumption stays above 20,000.",
    "I should try to include at least two moderate-or-higher intensity workouts each week.",
    "I think I should make time for at least one workout per week that lasts longer than 30 minutes.",
    "I should be careful not to let any single workout run longer than 90 minutes."
]

care_plans = [
    "To support baseline cardiovascular health and weekday activity consistency, aim to reach at least 20 Active Zone Minutes on most weekdays rather than relying on weekend activity alone.",
    "To maintain general mobility and prevent inactivity on rest days, keep daily step counts above 7,000, with particular attention to non-training days.",
    "To protect sleep quality and recovery, avoid scheduling workouts too late in the evening whenever possible.",
    "To reduce injury risk and allow sufficient recovery, avoid performing high-intensity workouts on consecutive days.",
    "To balance training stimulus and recovery, limit high-intensity workouts to no more than three sessions per week.",
    "To support overall energy expenditure and metabolic health, aim to keep total daily calorie expenditure above 2,500 kJ.",
    "To ensure sufficient daily activity load, target a total daily METs consumption above 20,000.",
    "To maintain aerobic fitness and strength, include at least two workouts per week with moderate or higher intensity.",
    "To build endurance capacity, complete at least one workout per week that lasts 30 minutes or longer at a sustained effort.",
    "To prevent overuse and excessive fatigue, avoid extending any single workout beyond 90 minutes."
]

reminders = [
    "This week, try to reach at least 20 Active Zone Minutes on most weekdays.",
    "This week, keep my daily steps above 7,000, especially on non-training days.",
    "This week, avoid scheduling workouts too late in the evening.",
    "This week, limit high-intensity workouts and avoid doing them on back-to-back days.",
    "This week, make sure to include at least two workouts with moderate or higher intensity."
]

care_plan_topics = [
    ["exercise", "diet"],
    ["exercise"],
    ["diet"],
    ["sleep"],
    ["exercise", "sleep"],
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

wearable_data_advanced_generator_prompt = """
### Branch-Specific Instructions
If this is a generation task, produce exactly one executable wearable-monitoring task that strictly follows all instructions below.
If this is a validation or refinement task, verify whether the provided task satisfies all instructions.
#### Source Assignment and Available Data
Wearable data is only visible on days where a data source is both assigned and connected.
- `source_assignment` indicates which wearable source (e.g., Fitbit, Google Fit) generated data on a given day,
  or is missing if no data exists.
- A source being assigned does not guarantee visibility. Data from an assigned source is visible only if
  that source is connected to the platform (i.e., appears in marketplaces / connected devices).
- If a day is marked as missing, data for that day is unavailable regardless of connection status.
- If a source is not connected, all collected data from this source is invisible.
#### Create Tasks
The composed second-person task_instruction (You're... You want to...) should involve at least 1 of the following action types:
A. Minute-Level Workout Analysis
- Select ONE recorded workout session.
- Use minute-level data within the session window to explain effort distribution and further ask for sport recommendation based on recent sport records.
B. Partial-Availability Reasoning
- Perform comparisons or summaries over a date range that may contain missing or disconnected days.
- Do NOT mention missing data explicitly in the task instruction.
C. Notes & Reminders (Read / Write): Retrieve, create, update, or cancel notes and reminders. Examples include:
- Reviewing recent activity and then adding a note or setting a reminder.
- Asking the assistant to retrieve an existing note or reminder to check whether goals are being followed.
D. Care Plans & Appointments (Read / Write): Retrieve, create, update, or cancel care plans and appointments from healthcare providers. Examples include:
- Asking the assistant to retrieve a doctor-provided care plan and check whether its goals are met.
- Reviewing recent activity and rescheduling an existing appointment.
  

#### Task Composition Requirements
When composing tasks, always ensure that:
- All referenced resources (sport records, notes, appointments...) exist in the user configuration. When requesting record retrieval, phrase the request in natural, everyday language that a human user would realistically use。
- Do not leak any information from "source_assignment" into the generated task_instruction.
- Read/write operations are semantically justified by the task context.
- Avoid vague temporal phrases such as “the past few days” and exact timestamp expressions that rarely appear in everyday language.
  Use natural, daily-life, human-preferred time descriptions that provide just enough information for the assistant to infer the correct records and tool arguments.
- The composed task should be **grounded in the `user_persona`**, reflecting the user’s personal situation, habits, or lifestyle.
- Do not try to cover all scenarios in a single task, as this makes the task difficult to execute and unnatural in context.
"""

few_shot_examples = [
    {
      "label": "top3_azm_workouts_minute_azm_calories_plus_daily_context",
      "task_instruction": (
        "You're Samantha. Over the past week, you want to understand what your three most effortful workouts looked like, "
        "and whether those high-effort sessions were driven by long sustained intensity or short bursts.\n"
        "Review your sport sessions from the past 5 days up through today and find the three workouts that correspond to the highest overall effort "
        "(use Active Zone Minutes as the primary signal). For each of those three workouts, zoom in on the session window and visualize your "
        "minute-by-minute AZM and calories burned to pinpoint where effort peaked.\n"
        "Then step back and pull the daily summary for each of those dates, and explain how the workout shows up in that day’s overall AZM pattern "
        "and goal completion — for example, whether AZM is concentrated around the workout or spread across the day."
      ),
      "targets": [
        "Retrieve sport records for the past 5 days through today.",
        "Identify the three workout sessions/dates with the highest overall effort (AZM-first; use duration/intensity as tie-breakers if needed).",
        "For each of the three workouts, retrieve minute-level AZM and minute-level calories over the session window (optionally include a small padding window, e.g., ±15 minutes).",
        "Visualize minute-level AZM and calories for each workout and identify the peak minute(s) for AZM and for calories.",
        "Retrieve the daily summary for each of the three dates.",
        "Explain, for each date, how the workout is reflected in the day’s AZM distribution and goal completion (e.g., concentrated around the session vs spread out)."
      ]
    },
    {
      "label": "best_day_daily_to_hourly_to_minute_peak_concentration",
      "task_instruction": (
        "You're Samantha. You want to identify which day in the past week was your strongest overall day and understand *why* it was strong.\n"
        "First, look back over the last week and pick the day where your overall activity effort was highest (use daily AZM and calories together). "
        "On that day, look at your evening activity by hour to see which hour contributed most to your effort.\n"
        "Then, find the main workout session on that day and zoom into the session window to see the minute-by-minute pattern of effort. "
        "Visualize the hourly trend for the evening and the minute-level trend for the workout window, and explain whether your daily performance was "
        "driven by one concentrated workout or by sustained activity across multiple hours."
      ),
      "targets": [
        "Retrieve range summary (or daily summaries) in the last week through today to compare daily AZM and calories.",
        "Select the single strongest day based on daily AZM + calories (clear selection rationale).",
        "Retrieve hourly activity (or hourly METs/calories) for that day over an evening window (e.g., 17:00–22:00) and identify the top-contributing hour.",
        "Retrieve sport records for that selected day and identify the main workout session with a concrete start/end window.",
        "Retrieve minute-level AZM and minute-level calories for the workout session window (optionally ±15 minutes).",
        "Visualize the hourly evening series and the minute-level workout series; explain whether effort is concentrated in the session vs distributed across hours.",
        "Retrieve the daily summary for that day and tie the conclusion back to daily goal completion."
      ]
    },
    {
      "label": "weekly_goal_day_minute_azm_with_note_and_reminder",
      "task_instruction": (
        "You're Samantha. You want to better understand what helped you meet your Active Zone Minutes (AZM) goal recently, "
        "so you can build a repeatable habit.\n"
        "Review the past week up through today and identify the days where you met your AZM goal. Pick one successful day "
        "and look at the workout session(s) recorded on that date.\n"
        "Zoom into the main workout session and visualize your minute-by-minute AZM to see how effort accumulated. "
        "Summarize what you notice and save it as a short note for future reference.\n"
        "Based on that pattern, choose a suitable workout style to repeat and create a daily reminder for the coming week "
        "to encourage consistency."
      ),
      "targets": [
        "Retrieve daily summaries for the past week and identify which days met the AZM goal; select one successful day.",
        "Retrieve sport records for the selected day and choose a primary workout session with a clear start/end window.",
        "Retrieve and visualize minute-level AZM for the session window (optional small padding allowed), and describe how effort accumulated.",
        "Add a note summarizing the observed minute-level effort pattern and why the day was successful.",
        "Recommend a suitable workout style that matches the observed effort pattern and the user’s AZM goal.",
        "Create a daily reminder for the coming week that nudges the user to repeat a similar workout routine."
      ]
    },
    {
      "label": "weekly_goal_shortfall_with_recommendation_and_reminder",
      "task_instruction": (
        "You're Samantha. You feel your activity level has been inconsistent lately and want a simple plan to improve it.\n"
        "Review the past 10 days up through today and check how often you met your steps and Active Zone Minutes (AZM) goals. "
        "Look at your activity pattern by hour to understand when your movement tends to drop.\n"
        "Using that context, pick a practical workout approach that fits naturally into your routine and set up a reminder "
        "to help you follow through. Save a short note explaining the plan so you can refer back to it."
      ),
      "targets": [
        "Retrieve daily summaries for the past 10 days and summarize how many days met the steps goal and AZM goal.",
        "Retrieve and visualize hourly activity (steps and/or METs) over the same period to identify a common low-activity time window.",
        "Recommend a workout style that suits the user’s recent activity pattern and goal needs.",
        "Create a daily reminder aligned with the identified low-activity window to encourage consistent activity.",
        "Add a note summarizing the weekly goal performance, the chosen activity window, and the recommended plan."
      ]
    },
    {
      "label": "appointment_management_with_activity_context_note_and_reminder",
      "task_instruction": (
        "You're Samantha. You want to schedule a healthcare appointment and make sure it doesn’t conflict with your usual workout routine in the past 2 weeks.\n"
        "First, you want the assistant to review your recent activity pattern to recommend a time that won’t disrupt your workouts. Then you expect the assistant to find an appropriate provider "
        "and create an appointment. After scheduling, save a note with the appointment details and set a reminder so you don’t forget."
      ),
      "targets": [
        "Retrieve sport records over the past 2 weeks through today to infer typical workout time-of-day (e.g., mornings vs evenings).",
        "Retrieve provider resources (med.get_resources) to find available appointment slots.",
        "Create an appointment (med.create_appointment) for a slot that avoids the inferred typical workout window.",
        "Add a note with: provider, appointment time, and the reasoning (chosen to avoid usual workout window).",
        "Create a daily reminder leading up to the appointment date (e.g., reminder the day before and the morning-of, expressed as daily reminders if that’s the available primitive)."
      ]
    },
    {
      "label": "weekly_high_effort_workout_with_doctor_care_plan_and_note",
      "task_instruction": (
        "You're Samantha. You want to review one of your more effortful workouts from the past week and understand "
        "how that session fits into the guidance provided by your doctor.\n"
        "Look back over the past 5 days and pick a workout that required relatively high effort. "
        "Zoom into the session window and visualize your minute-by-minute Active Zone Minutes (AZM) and calories burned "
        "to see how effort accumulated during the workout.\n"
        "Then review the care plans provided by your doctor along with your own personal notes, and explain whether this workout "
        "aligns with that guidance. Summarize your findings and save a short note capturing what you learned."
      ),
      "targets": [
        "Retrieve sport records for the past 5 days and select one relatively high-effort workout session.",
        "Retrieve and visualize minute-level AZM and minute-level calories for the selected session window, and describe how effort accumulated over time.",
        "Retrieve the daily summary for the workout date and briefly explain how the session contributes to that day’s AZM distribution and goal completion.",
        "Retrieve doctor-provided care plans and personal notes, and assess whether the workout aligns with the stated medical guidance (e.g., intensity, duration, timing).",
        "Add a new note summarizing the workout pattern and whether it aligns with the doctor-provided care plans and existing notes."
      ]
    },
    {
      "label": "appointment_reschedule_with_notes_care_plan_activity_and_time_constraints",
      "task_instruction": (
        "You're Samantha. You want to reschedule an existing healthcare appointment after reviewing your recent health guidance and activity.\n"
        "Start by checking your most recent personal notes and the care plans provided by your doctor to remind yourself of any relevant guidance. "
        "Then review your activity over the past 7 days to get a sense of how things have been going.\n"
        "After that, you want the assistant to look at your upcoming healthcare appointment, cancel it, and confirm a new appointment with the same provider."
        "at a more suitable time within the next two weeks. Finally, save a note summarizing what you reviewed and set a reminder so you don’t forget "
        "the updated appointment."
      ),
      "targets": [
        "Retrieve the most recent personal notes and doctor-provided care plans, and summarize key guidance relevant to the user’s health or activity.",
        "Retrieve daily/range summaries and/or sport records for the past 7 days through today to review recent activity at a high level.",
        "Retrieve medical user profile and provider list, and identify the healthcare provider associated with the existing appointment.",
        "Cancel the existing appointment using its appointment identifier and confirm a new appointment with the same provider at an available time within the next two weeks..",
        "Add a note summarizing the reviewed notes, care plans, recent activity, and the new appointment date and time.",
        "Create a reminder tied to the new appointment date (e.g., the day before or the morning of the appointment)."
      ]
    },
    {
      "label": "range_extreme_comparison_with_source_goal_sync",
      "task_instruction": (
        "You're Samantha. You’re trying to manage weight and reduce diabetes risk by keeping your daily movement consistent, "
        "but you know you sometimes have low-movement days.\n"
        "Review roughly the last two weeks up through today and identify your highest-step day and lowest-step day. "
        "For each of those two dates, compare your Active Zone Minutes (AZM), activity intensity distribution, "
        "and whether you met your steps and AZM goals.\n"
        "After that, check how these results relate to the goals set in your connected wearable apps, "
        "and make sure your system activity data is aligned with those sources."
      ),
      "targets": [
        "Retrieve daily summaries (or range summary) for roughly the past two weeks and identify the days with the highest and lowest step counts.",
        "For each of the identified days, retrieve Active Zone Minutes, activity intensity distribution, and steps/AZM goal completion status.",
        "Retrieve source features for the user’s connected wearable apps to understand which sources are active and what goals are configured.",
        "Compare the system-recorded steps and AZM results for the two extreme days against the goals defined in the connected sources.",
        "Confirm that the system activity data for those days is synchronized and consistent with the connected wearable apps, "
        "and summarize any notable differences at a high level."
      ]
    },
]

wearable_data_advanced_tool_selection_prompt = """
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


class WearableDataAdvanced(TaskBranch):
    branch_name = "wearable_data_advanced"

    def get_accessible_dates(self, store: Dict[str, Any]) -> List[str]:
        """
        A date is accessible if:
        - source_assignment[date] != "missing"
        - assigned source is connected in marketplaces
        """
        source_assignment = store["profile"].get("source_assignment", {})
        marketplaces = {
            m["source"]: m["connected"]
            for m in store["profile"]["system_settings"]["marketplaces"]
        }

        accessible_dates = []
        for date, source in source_assignment.items():
            if source == "missing":
                continue
            if marketplaces.get(source, False):
                accessible_dates.append(date)
        return sorted(accessible_dates)

    def is_applicable(self, store: Dict[str, Any]) -> bool:
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
            wearable_data_advanced_generator_prompt,
            "### Tools\n" + json.dumps(involved_tool_schemas, indent=2),
            "### Few-shot Task Examples\n"
            + json.dumps(sampled_few_shots, indent=2),
            "### Input User Configuration\n" + json.dumps(user_configuration, indent=2),
        ]
        return "\n\n---\n\n".join(sections)

    def _build_valid_prompt(
        self,
        user_configuration: Dict[str, Any],
        initial_task: str,
    ) -> str:
        sections = [
            base_task_valid_prompt,
            platform_overview,
            wearable_data_advanced_generator_prompt,
            "### Tools\n" + json.dumps(involved_tool_schemas, indent=2),
            "### Input User Configuration\n" + json.dumps(user_configuration, indent=2),
            "### Candidate Task (JSON)\n" + initial_task,
        ]
        return "\n\n---\n\n".join(sections)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        return json.loads(strip_code_fences(text))

    def run(self, store: Dict[str, Any]) -> List[Dict[str, Any]]:
        _inject_context_for_tasks(store)
        user_configuration = {
            "user_profile": store["profile"]["user_profile"],
            "user_persona": store["profile"]["narrative_summary"],
            "now": store["profile"]["now"],
            "sport_records": filter_sport_records(store["profile"]["sports"], store["profile"]["now"]),
            "notes": store["profile"].get("notes", []),
            "care_plans": store["profile"].get("care_plans", []),
            "reminders": store["profile"].get("reminders", []),
            "appointments": store["profile"].get("appointments", []),
            "source_assignment": store["profile"].get("source_assignment", {})
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
        try:
            data = self._parse_json(llm_output)
        except Exception:
            # If parsing fails, return empty list
            return []

        additional_instruction = """
At the start of conversation, play as the user and express your entire task and information need in one initial message using plain, natural language.
After that, do not introduce new requests or additional information unless the assistant explicitly asks you to confirm or restate task-relevant details.
If the assistant asks you for VIP upgrade duration, you always prefer 1 month option.
End the conversation only after the assistant explicitly confirms that all requested tasks have been fully completed.
If the assistant claims completion when the task is in fact unfinished, do not correct or guide it—simply end the conversation immediately."""
        
        # Ensure task_instruction exists and is a string before appending
        if "task_instruction" not in data:
            data["task_instruction"] = ""
        elif not isinstance(data["task_instruction"], str):
            # Convert to string if it's not already
            data["task_instruction"] = str(data["task_instruction"])
        
        data["task_instruction"] += additional_instruction
        return [data]


