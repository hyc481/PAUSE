from typing import Dict, List, Any
import random
from datetime import datetime, timedelta
from pathlib import Path
import json

from bench.pipeline.generate_task.task_branch_base import TaskBranch
from bench.prompts.generation_prompt import (
    base_task_generation_prompt,
    base_task_valid_prompt,
    platform_overview,
)
from bench.utils.misc import strip_code_fences
from bench.utils.paths import TOOL_SCHEMAS_DIR


tool_schemas_path = TOOL_SCHEMAS_DIR / "platform_tools.json"
with open(tool_schemas_path, "r") as f:
    tool_schemas = json.load(f)
involved_tool_names = [
    "get_daily_summary",
    "get_range_summary",
    "get_hourly_mets",
    "get_hourly_steps",
    "get_hourly_calories",
    "get_hourly_activity",
    "get_user_profile",
    "update_profile",
    "get_sport_records",
    "get_daily_sleep_summary",
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

wearable_data_casual_generator_prompt = """
### Branch-Specific Instructions
If this is a generation task, produce exactly one executable wearable-monitoring task that strictly follows all instructions below.
If this is a validation or refinement task, verify whether the provided task satisfies all instructions.
1. Only involves **accessible dates** listed in the user configuration and does not induce missing data analysis, disconnected source diagnosis and raw or minute-level sensor inspection.
2. Reflects a **natural, casual user need for everyday data tracking**,  such as checking daily activity, sleep quality, sport record checks, basic hourly breakdowns or short-term patterns that plausibly arise in the user’s daily life.
3. Avoid vague or relative phrases such as "the past few days" or similar ambiguous temporal references.
4. The task should be phrased such that a **clear, finite, and tool-consistent sequence of calls** can satisfy it,  
   without relying on hidden assumptions, unavailable data, or implicit user knowledge.
5. The composed task should be **grounded in the `user_persona`**, reflecting the user’s personal situation, habits, or lifestyle  
   (e.g., work schedule, exercise routines, sleep patterns), so that the request corresponds to a plausible real-world scenario and concrete user need.
6. If sport records are involved in the task_instruction, refer to sport records provided in user configuration. Do not invent any sport outside the user's history. 

**Do NOT**
- Invent new dates
- Reference unavailable days
- Explain system internals
- Mention data visibility explicitly.
"""

example_user_profile = {
  "user_profile": {
    "basic_info": {
      "name": "Samantha",
      "age": 35,
      "gender": "F"
    },
    "health_risks": [
      "T2DM",
      "hypertension"
    ],
    "dietary_restrictions": [
      "lactose_intolerant"
    ],
    "preferences": {
      "sport": [
        "cardio",
        "strength"
      ],
      "food": [
        "high_protein",
        "low_sugar"
      ]
    },
    "daily_goal": {
      "steps": 8000,
      "mets": 1000,
      "calories_burn": 2300,
      "AZM": 25,
      "calories_intake": 2200
    }
  }
}

example_user_configuration = {
    "user_profile": example_user_profile["user_profile"],
    "user_persona": "Samantha is a moderately active runner who trains 4~6 days a week, often mixing steady runs with occasional interval work. Her recent activity logs_failed_models show consistent moderate intensity, with several higher-effort days reaching over 12,000 steps. She maintains a balanced diet with an emphasis on high-fiber, high-protein, and low-sugar foods, often choosing organic, vegan options and lactose- or gluten-free meals. Meal replacement bars and nuts serve as convenient fuel around workouts. While she stays fairly active, there are periodic lower-movement days, so her current goals aim to boost daily steps and active minutes slightly above her recent averages to support weight management and reduce diabetes risk.",
    "now": "2025-06-05T19:00:00",
    "accessible_dates": [ "2025-05-24", "2025-05-25", "2025-05-26", "2025-05-27", "2025-05-28", "2025-05-29", "2025-05-30", "2025-05-31", "2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05" ] }

few_shot_examples = [
    {
        "label": "range_extreme_comparison",
        "task_instruction": (
            "You're Samantha. You’re trying to manage weight and reduce diabetes risk by keeping your daily movement consistent, "
            "but you know you sometimes have low-movement days. Review roughly the last two weeks up through today and identify "
            "your highest-step day and lowest-step day. For each of those two dates, compare AZM, the activity intensity distribution, "
            "and whether you met your steps/AZM goals.\n"
            ),
        "targets": [
            "Identify the day with the highest and lowest step counts over the past two weeks.",
            "For those two days, retrieve the Active Zone Minutes and activity intensity distribution.",
            "Check whether the step and AZM goals were met on each of those days."
        ]
    },
    {
        "label": "daily_peak_drilldown",
        "task_instruction": (
            "You're Samantha. Today you want to pinpoint the exact time of day when you were most active so you can plan future runs "
            "more consistently. Identify your peak hour for steps and your peak hour for METs for today, then inspect a tight 3-hour "
            "window around each peak to explain what kind of movement pattern produced those peaks.\n"
            ),
        "targets": [
            "Identify the peak hour for steps and the peak hour for METs for today.",
            "For each peak hour, examine a tight 3-hour window centered on that peak.",
        ]
    },
    {
        "label": "day_to_day_comparison",
        "task_instruction": (
            "You're Samantha. You’re training regularly (mixing steady runs and occasional intervals), and you want to compare two "
            "recent days—earlier this week vs. today—to see which one better supports your goal of slightly higher steps and active minutes. "
            "Compare those two days using daily totals (steps, METs, calories, AZM) and goal completion. Then compare 07:00–12:00 on both days "
            "to explain which morning set you up for a stronger day and how (more steps, higher intensity minutes, or both).\n"
            ),
        "targets": [
            "Compare an earlier day this week and today using daily totals for steps, METs, calories, and Active Zone Minutes, including goal completion.",
            "Compare the 07:00–12:00 time window on both days in terms of steps and intensity-related metrics.",
            "Determine which morning better supported a stronger overall day and whether it was driven by higher volume, higher intensity, or both."
        ]
    },
    {
        "label": "heartrate_activity_link",
        "task_instruction": (
            "You're Samantha. You suspect that your cardiovascular state may be influencing your daily consistency—especially on days when your activity "
            "falls below expectations. Review your daily heart summary from a day earlier this week, and then evaluate your activity on the following day "
            "(steps, AZM, intensity distribution, and goal completion). You also want to "
            "inspect the 08:00–12:00 window on that day to determine whether low activity was present from the morning or if it declined later.\n"
        ),
        "targets": [
            "Retrieve the daily heart summary for one day earlier this week.",
            "Evaluate activity on the following day, including steps, AZM, intensity distribution, and goal completion.",
            "Inspect the 08:00–12:00 window on that day to see whether low activity started in the morning or declined later."
        ]
    },
    {
        "label": "sleep_activity_link",
        "task_instruction": (
            "You're Samantha. You suspect your sleep quality is affecting your consistency—especially on days when you end up moving less "
            "than you want. Check your sleep summary for the night a few days ago (early this week) and then evaluate your activity on the following day "
            "(steps, AZM, intensity distribution, and goal completion). You also want to"
            "inspect 08:00–12:00 on that day to see whether you started the day low-energy or whether the drop happened later.\n"
            ),
        "targets": [
            "Retrieve the sleep summary for one night earlier this week.",
            "Evaluate activity on the following day, including steps, AZM, intensity distribution, and goal completion.",
            "Inspect the 08:00–12:00 window on that day to see whether low activity started in the morning or declined later."
        ]
    },
    {
        "label": "multi_metric_hourly_alignment",
        "task_instruction": (
            "You're Samantha. Looking back at yesterday, you want to understand whether your movement and effort were 'steady' or 'spiky'—"
            "because that affects how you schedule steady runs vs. interval days. Pull hourly steps, hourly calories, and hourly METs for 08:00–20:00, "
            "visualize them by hour, and describe whether the peaks happen in the same hours or whether steps, calories, and METs peak at different times.\n"
            ),
        "targets": [
          "Retrieve hourly steps, calories, and METs for yesterday between 08:00 and 20:00.",
          "Visualize the three metrics by hour and identify the peak hour for each metric.",
          "Determine whether the peaks occur in the same hour or at different times, indicating steady versus spiky activity."
        ]
    },
    {
        "label": "sport_to_daily_explanation",
        "task_instruction": (
            "You're Samantha. You want to confirm that your hardest workout from roughly the past week actually showed up in your daily metrics "
            "Look at sport records from about a week ago up through today and pick the most demanding workout day based on intensity/calories/AZM. "
            "For that date, explain how the workout is reflected in AZM, intensity distribution, and goal completion. You also want to inspect the AZM time window around the workout record to see when the intensity was concentrated.. "
            ),
        "targets": [
            "From the past week through today, identify the most demanding workout day based on intensity, calories, and AZM.",
            "For that day, examine how the workout is reflected in AZM, activity intensity distribution, and step/AZM goal completion.",
            "Inspect the AZM time window around the workout record to see when the intensity was concentrated."
        ]
    },
    {
        "label": "sport_volume_vs_overall_activity",
        "task_instruction": (
            "You're Samantha. You train 4–6 days per week, and you want a clear recap of how much structured exercise you actually did recently—and how it relates "
            "to your overall movement goals. Summarize your sport records from roughly the last 10 days up through today (workout count, total workout calories, total workout AZM if available). "
            "Then compare that with your overall activity in the same period (average daily steps and AZM) and explain whether your training volume seems consistent with your routine.\n"
            ),
        "targets": [
            "Summarize structured workout activity over the past 10 days, including workout count, total workout calories, and total workout AZM if available.",
            "Compute overall movement metrics for the same period, specifically average daily steps and average daily AZM.",
            "Compare structured training volume with overall activity levels to assess consistency with the stated training routine."
        ]
    },
    {
        "label": "workout_vs_nonworkout_comparison",
        "task_instruction": (
            "You're Samantha. You want to know whether your workout days reliably push you over your steps and AZM goals, or if you still need better non-workout-day habits "
            "for weight management and diabetes risk reduction. Over roughly the past week up to today, identify which dates had a recorded workout. Compare those workout dates "
            "against at least two non-workout dates in the same window using goal completion and intensity distribution, then conclude whether workouts consistently translate into better goal outcomes for you.\n"
            ),
        "targets": [
          "Identify which dates in the past week had a recorded workout.",
          "Compare those workout dates with at least two non-workout dates using step/AZM goal completion and activity intensity distribution.",
          "Determine whether workout days consistently result in better goal outcomes than non-workout days."
        ]
    },
    {
        "label": "trend_visualization",
        "task_instruction": (
            "You're Samantha. You want a quick visual check of whether you’ve been trending toward more consistent movement—since that supports weight management and lower diabetes risk. "
            "Across the last seven days up through today, collect each day’s total steps and plot a day-level time series starting at the first day in that window with unit 'steps'. "
            "Then summarize whether the trend looks upward, downward, or mixed over the week.\n"
            ),
        "targets": [
          "Collect total daily step counts for each day over the last seven days up through today.",
          "Construct a day-level time series of steps starting from the first day in that window.",
          "Assess whether the weekly step trend appears upward, downward, or mixed."
        ]
    },
    {
        "label": "goal_adjustment_decision",
        "task_instruction": (
            "You're Samantha. You’re generally moderately active but want to nudge your daily steps and active minutes slightly above your recent averages. "
            "Review roughly the last two weeks up through today and check how often you met your steps goal and AZM goal. If you met them on most days, increase your steps goal by 500 and your AZM goal by 5; "
            "otherwise keep them unchanged. Apply the update and confirm the final goal values.\n"
            ),
        "targets": [
            "Over the past two weeks up through today, count how many days the steps goal and AZM goal were met.",
            "Determine whether the goals were met on most days within that period.",
            "If so, increase the steps goal by 500 and the AZM goal by 5, then confirm the final goal values; otherwise keep the goals unchanged."
        ]
    },
    {
        "label": "multiday_very_active_minutes_sum",
        "task_instruction": (
            "You're Samantha. You want to understand how much truly intense activity you accumulated recently, since very active minutes matter most for improving fitness and reducing diabetes risk. "
            "Calculate your total very active minutes across the last week up through today, and report the final sum along with a brief explanation of how concentrated or spread out that effort was across the days.\n"
            ),
        "targets": [
            "Calculate the total number of very active minutes accumulated over the last week up through today.",
            "Report the final summed value of very active minutes for that period.",
            "Assess whether those very active minutes were concentrated on a few days or spread out across the week."
        ]
    },
    {
        "label": "goal_alignment_with_recent_performance",
        "task_instruction": (
            "You're Samantha. You want to make sure your current daily goals actually reflect how you’ve been performing recently, rather than being outdated. "
            "Review your activity from about the last week up through today and compare it with your current steps and AZM goals. If your recent averages are clearly above your goals, slightly raise them to stay challenged; "
            "if not, keep them as-is. Update your profile accordingly and confirm the final goal settings.\n"
            ),
        "targets": [
            "Compute your average daily steps and Active Zone Minutes over the past week up through today.",
            "Compare these recent averages with your current steps and AZM goals.",
            "If the averages are clearly above the goals, slightly increase the goals; otherwise keep them unchanged and confirm the final settings."
        ]
    }
]

wearable_data_casual_tool_selection_prompt = """### Branch-specific instructions
1. The assistant should correctly identify the time span implied by the task instruction,
such as specific dates, a short recent range, or a particular time of day.
2. The assistant should select tools with a time granularity appropriate to the question:
- Range-level summaries for multi-day trends or comparisons  
- Daily summaries for analyzing or comparing specific days  
- Hour-level tools for within-day or time-of-day analysis
- You should prioritize higher-level tools with coarser time granularity.
Only when such tools cannot provide the information required by the targets
should you resort to tools with finer time granularity for additional detail.
3. The assistant should use a sufficient and appropriate set of tools to answer the task,
avoiding unnecessary detours, redundant calls, or over-complicated analysis.
4. The assistant must carefully and thoroughly analyze the user’s possible task targets, 
including any explicit or implicit informational or statistical objectives.
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


class WearableDataCasual(TaskBranch):
    branch_name = "wearable_data_casual"

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
        """
        Only run when there is at least one accessible date.
        """
        accessible_dates = self.get_accessible_dates(store)
        return len(accessible_dates) > 0

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
            wearable_data_casual_generator_prompt,
            "### Tools\n" + json.dumps(involved_tool_schemas, indent=2),
            "### Few-shot User Configuration and Task Examples\n"
            + json.dumps(example_user_configuration, indent=2) +"\n"
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
            wearable_data_casual_generator_prompt,
            "### Tools\n" + json.dumps(involved_tool_schemas, indent=2),
            "### Input User Configuration\n" + json.dumps(user_configuration, indent=2),
            "### Candidate Task (JSON)\n" + initial_task,
        ]
        return "\n\n---\n\n".join(sections)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        return json.loads(strip_code_fences(text))

    def run(self, store: Dict[str, Any]) -> List[Dict[str, Any]]:
        accessible_dates = self.get_accessible_dates(store)
        user_configuration = {
            "user_profile": store["profile"]["user_profile"],
            "user_persona": store["profile"]["narrative_summary"],
            "now": store["profile"]["now"],
            "accessible_dates": accessible_dates,
            "sport_records": filter_sport_records(store["profile"]["sports"], store["profile"]["now"]),
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


