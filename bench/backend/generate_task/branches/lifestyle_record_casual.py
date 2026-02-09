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
from pathlib import Path
import json


root = Path(__file__).resolve().parents[2]
tool_schemas_path = root / "tool_schemas" / "platform_tools.json"
with open(tool_schemas_path, "r") as f:
    tool_schemas = json.load(f)
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
    "plot_time_series"
]
involved_tool_schemas = [
    {"name": tool_schema["function"]["name"], "description": tool_schema["function"]["description"]} \
    for tool_schema in tool_schemas if tool_schema["function"]["name"] in involved_tool_names
]


def get_involved_platform_tool_names() -> List[str]:
    """
    Branch tool allowlist for non-inference trajectory runs.
    Other branches should expose the same helper or define `involved_tool_names`.
    """
    return list(involved_tool_names)

lifestyle_record_casual_generator_prompt = """
### Branch-Specific Instructions
If this is a generation task, produce exactly one executable lifestyle-monitoring task that strictly follows all instructions below.
If this is a validation or refinement task, verify whether the provided task satisfies all instructions.
#### Daily Event Records
- **Meal records** log daily food intake and nutrition-related events.
- **Sport records** log workouts synced from wearable sources and are read-only.
- **Session records** are user-created entries for non-sport events such as studying, working, or other daily activities.
User configuration provides existing meal records, sport records, and profile information (preference, health risks, dietary restrictions...).
- **Refer / Compare / Retrieve Existing Records**  
  All questions, comparisons, or references to existing records must be grounded in the provided **user configuration**.
  The task must not assume, invent, or imply records that are not present in the configuration.
- **Create New Records**  
  Tasks may create new records beyond the user configuration to, but the task instruction must explicitly provide all required information:
  - For session records: clear start time, end time, and session type.
  - For meal records: explicit timestamp, concrete food items, and quantities (value and unit).

#### Create Tasks
The composed second-person task_instruction (You’re… You want the assistant to help you… You expect the assistant to…) should combine 2~3 following action types:
A. Record Review & Comparison (Meal / Sport)
- Review recent **meal and/or sport records**.
- Identify subsets or extremes (e.g., snack days, highest-calorie breakfasts, late meals).
- Perform **coarse comparisons** (e.g., calories intake on days with vs without snacks; calories burned with vs without workouts).
B. Goal & Trend Reasoning
- Compare recent behavior against **daily goals** using daily or range summaries (e.g., calories intake/burn, AZM).
- Summarize trends or consistency across a date range.
C. Profile Inspection & Update
- Retrieve and inspect the **user profile** (preferences, dietary restrictions, health risks, goals).
- Check alignment between profile settings and recent records.
- Apply updates to preferences or goals when justified.
D. Record Creation & Cleanup (Non-sport)
- Create, update, or delete **meal records** or **session records**.
- When creating a meal record, you MUST specify a timestamp (e.g, yesterday 11 am...) and included food items (e.g., 200 g beef, 100 ml milk).
- When the task involves creating **session records**, the user instruction must explicitly specify the session start time and end time, as well as the session type.
E. Visualization for Insight
- Request **simple visualizations** to illustrate patterns or comparisons.
- Visualized data may come from daily calorie intake, calories burned, workout AZM, or similar high-level metrics.
"""

few_shot_examples = [
    {
      "label": "snacking_days_intake_burn_goal_check",
      "targets": [
        "Identify days with snack-related meal records in the past week",
        "Compare calories intake vs calories burned on those days",
        "Check daily calorie intake and burn goal completion for those days",
        "Summarize whether snacking days tend to exceed burn or not"
      ],
      "task_instruction": (
        "You’re Maya. You want the assistant to help you understand whether snacking is disrupting your weekly balance. "
        "You expect the assistant to review your meal records from the past 7 days and identify which days include snacks. "
        "For those snack days, you want to compare calories intake versus calories burned and see which is higher. "
        "You also want to check whether you met your daily intake and burn goals on those days. "
        "Your purpose is to decide whether you need stricter snack control or a small adjustment to your goals."
      )
    },
    {
      "label": "breakfast_intensity_activity_comparison",
      "targets": [
        "Rank recent days by breakfast calories and select the top three",
        "Review workout presence on those high-breakfast days",
        "Compare activity patterns between high-breakfast days and remaining days",
        "Check goal completion differences between the two groups"
      ],
      "task_instruction": (
        "You’re Daniel Reyes. You want the assistant to help you test whether heavier breakfasts are linked to being more active. "
        "You expect the assistant to review your meal records from the previous week and find the three days with the most calories at breakfast. "
        "You then want to see how those days compare to the rest of the week in terms of workouts and overall activity. "
        "You also want to compare daily goal completion between these two groups. "
        "Your purpose is to decide whether you should intentionally eat more at breakfast to support your activity level."
      )
    },
    {
      "label": "evening_meal_workout_pattern_with_visualization",
      "targets": [
        "Identify days with notable evening food intake",
        "Check whether those days include evening workouts",
        "Compare calorie intake and burn patterns on those days",
        "Visualize evening-related calorie intake and consumption trends",
        "Update profile preferences based on observed patterns"
      ],
      "task_instruction": (
        "You’re Alex. You want the assistant to help you understand whether late-evening eating is clashing with your evening workouts "
        "and leaving you overly tired. You expect the assistant to review the past week and identify days with noticeable evening food intake, "
        "then see how those days line up with evening workouts. You also want a simple visualization comparing calorie intake and calories burned "
        "across those days to make the pattern clear. Based on what’s observed, you want the assistant to update your profile with one practical "
        "evening preference to reduce this fatigue cycle. Your purpose is to build a more sustainable evening routine."
      )
    },
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
            "You’re Daniel Reyes. You want the assistant to help you understand how your busy work and study days affect your activity. "
            "You expect the assistant to create session records for the following activities you remember from the past week: "
            "a focused work session on Monday from 09:00 to 12:00, a study session on Wednesday from 19:30 to 21:00, "
            "and a long work block on Friday from 10:00 to 14:00. "
            "After recording these sessions, you want to compare those days with your sport records to see whether physical activity "
            "tends to drop or stay stable on heavy work or study days. You also want to check how your daily goals look on those days. "
            "Your purpose is to see whether work intensity is crowding out movement."
        )
    },
    {
      "label": "create_recent_meal_records_and_activity_visualization",
      "targets": [
        "Create missing meal records with explicit food items and timestamps",
        "Compare activity levels on days with newly created meals",
        "Visualize calorie intake versus activity intensity patterns"
      ],
      "task_instruction": (
        "You’re Alex. You want the assistant to help you fill in missing meals so you can better understand recent patterns. "
        "You expect the assistant to create meal records for the past three days: "
        "on Tuesday at 08:00, breakfast with oatmeal (60 g) and almond milk (200 ml); "
        "on Wednesday at 13:00, lunch with grilled chicken (180 g) and mixed vegetables (150 g); "
        "and on Thursday at 19:30, dinner with rice (200 g) and tofu (120 g). "
        "After recording these meals, you want to compare those days with your sport records to see how activity levels line up. "
        "You also want a simple visualization showing calories intake alongside activity intensity. "
        "Your purpose is to make your recent data complete enough to guide small routine adjustments."
      )
    },
    {
      "label": "meal_record_cleanup_session_creation_and_profile_check",
      "targets": [
        "Update or delete inaccurate meal records",
        "Create lifestyle session records with clear activity descriptions",
        "Check alignment between records and profile preferences"
      ],
      "task_instruction": (
        "You’re Samantha. You want the assistant to help you clean up your records so they reflect what actually happened. "
        "You expect the assistant to remove a dinner record from last Saturday that was logged by mistake, "
        "and update a lunch record from Sunday to reflect that you ate a smaller portion than originally recorded. "
        "You also want to create session records for recent non-sport activities: "
        "a personal errands session on Tuesday from 16:00 to 17:30, "
        "and a focused home organization session on Thursday from 20:00 to 21:00. "
        "After that, you want to review whether these patterns still align with your profile food and activity preferences, "
        "and make a small update if needed. Your purpose is to keep your records and preferences consistent."
      )
    }
]

lifestyle_record_casual_tool_selection_prompt = """### Branch-specific instructions
1. The assistant should correctly identify the time span implied by the task instruction,
such as specific dates, a short recent range, or a particular time of day.
2. The assistant should select tools with a time granularity appropriate to the question:
- Range-level summaries for multi-day trends or comparisons  
- Daily summaries for analyzing or comparing specific days  
- You should prioritize higher-level tools with coarser time granularity.
Only when such tools cannot provide the information required by the targets
should you resort to tools with finer time granularity for additional detail.
3. The assistant should use a sufficient and appropriate set of tools to answer the task,
avoiding unnecessary detours, redundant calls, or over-complicated analysis.
4. The assistant must carefully and thoroughly analyze the user’s possible task targets, 
including any explicit or implicit informational or statistical objectives.
5. The assistant must respect record mutability:
- Sport records are read-only and must never be created, modified, or deleted.
- Meal records and session records are platform-native and may be created, updated, or deleted only when explicitly required by the task.
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
    cutoff = now - timedelta(days=7)
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


class LifeStyleCasual(TaskBranch):
    branch_name = "lifestyle_record_casual"

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
            lifestyle_record_casual_generator_prompt,
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
            lifestyle_record_casual_generator_prompt,
            "### Tools\n" + json.dumps(involved_tool_schemas, indent=2),
            "### Input User Configuration\n" + json.dumps(user_configuration, indent=2),
            "### Candidate Task (JSON)\n" + initial_task,
        ]
        return "\n\n---\n\n".join(sections)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        return json.loads(strip_code_fences(text))

    def run(self, store: Dict[str, Any]) -> List[Dict[str, Any]]:
        # accessible_dates = self.get_accessible_dates(store)
        for source_dict in store["profile"]["system_settings"]["marketplaces"]:
            source_dict["connected"] = True
        user_configuration = {
            "user_profile": store["profile"]["user_profile"],
            "now": store["profile"]["now"],
            "sport_records": filter_sport_records(store["profile"]["sports"], store["profile"]["now"]),
            "meal_records": filter_meal_records(store["profile"]["meals"], store["profile"]["now"]),
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
        data = self._parse_json(llm_output)
        additional_instruction = """
Express your entire task and information need in one initial message using plain, natural language.
After that, do not introduce new requests or additional information unless the assistant explicitly asks you to confirm or restate task-relevant details.
This task does not involve any raw data or permission requirements, if the assistant asks for further clarification, refuse their requests.
End the conversation once the assistant indicates that all necessary information has been provided."""
        # Ensure task_instruction is either string or dict (allow richer payloads)
        data["task_instruction"] += additional_instruction
        return [data]


