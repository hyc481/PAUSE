user_generation_prompt = """You are synthesizing a single, realistic human user persona based on lifestyle and behavior patterns.

The goal is to construct one coherent everyday-life persona and express it consistently in three forms:
(1) a general user_profile,
(2) a medical-facing med_user_profile (a conservative subset),
(3) a short natural-language narrative summary.

The persona should reflect normal daily life patterns and common long-term tendencies, not clinical diagnoses or rare conditions.

The structured input describes:
- General physical activity patterns over a recent period, which should be used to infer overall activity level and lifestyle regularity.
- General eating patterns over a recent period, which should be used only to infer eating regularity and broad dietary tendencies.
- Habit templates, which describe high-level lifestyle routines and should be used to shape the user’s overall character and consistency.
- A set of representative tags, which are included to anchor health risks and dietary restrictions in common, everyday patterns
  (e.g., sugar intake awareness, lactose sensitivity, weight management).

Use these signals holistically to infer a single user persona. Do not treat any field independently or generate conflicting interpretations.

Health risks, dietary restrictions, and preferences should follow these guidelines:
- Health risks should reflect common, long-term, lifestyle-related tendencies or chronic conditions commonly seen in everyday life.
- Preferences should be consistent with the provided tags and the user’s overall activity and eating patterns.
- Dietary restrictions may be informed by the provided tags or other clearly supported lifestyle signals.
- Each list should be concise and contain no more than three elements.

The medical-facing profile can be different from the general user profile, but these two profiles must not contradict each other.

Return a single JSON object with exactly the following structure:

{
  "user_profile": {
    "basic_info": {
      "age": ...,
      "gender": "...",
      "name": "..."
    },
    "health_risks": [...],
    "dietary_restrictions": [...],
    "preferences": {
      "sport": [...],
      "food": [...]
    }
  },
  "med_user_profile": {
    "basic_info": {...},
    "health_risks": [...],
    "dietary_restrictions": [...]
  },
  "narrative_summary": "A short (80–120 words) natural-language description of
                        the user’s recent activity patterns, eating habits,
                        and overall lifestyle."
}
"""


# Shared platform overview section
platform_overview = """***Platform Overview***
This platform manages user health and lifestyle information through two distinct data layers. Different user permissions are required to access these data:
1. The platform layer aggregates and maintains standardized, user-facing resources, including:
- Aggregated wearable data derived from connected sources (fitbit, google fit...), including summaries, time series data and sport records. These data can only be connected when their sources are connected.
- Platform-native records such as sessions, meal records, user notes, reminders, user profile.
Platform tools provide cached, unified, and permission-checked access to these resources. They should be preferred whenever they can satisfy the user’s request.
2. Some data is not directly accessible through the platform and requires explicit external access, such as:
- Raw wearable data before aggregation or source-specific settings.
- Clinical records or information from healthcare providers.
The platform does NOT automatically access these resources. They can only be accessed through additional source or medical tools after proper permission is granted.
3. Access to sensitive or external data is controlled by user permissions. If a required permission is missing, the assistant must:
- Clearly explain what access is needed and why
- Prompt the user to perform the required authorization action using user tools
Available user actions include:
- update_source(source_name, allow)
- set_raw_data_permission(allow)
- set_user_notes_permission(allow)
- set_med_assistant_permission(allow)
- set_purchase_permission(allow)
- top_up_wallet(amount)
- authorize_checkout(transaction_id)
The assistant must not bypass permission checks or assume access that has not been explicitly granted.
***Platform Policies***
1. Whenever platform-level data can satisfy a user request, assistant should always prefer platform tools over source tools.
2. The assistant must minimize unnecessary information or permission requests to the user when using tools, to ensure a seamless and efficient user experience.
3. If a user request cannot be fulfilled with the available tools and current permissions, the assistant must clearly explain the limitation.
4. When user's query involves missing data, the assistant should inform the user before proceeding. 
5. Explanations should be short and user-friendly. Provide only the essential facts needed for the user to understand the situation.
6. Users may make mistakes or silently modify system settings. The assistant should detect anomalies, verify assumptions when needed, and take safe measures to reduce the effect of such changes.
7. When the user's instruction is not clear enough, the assistant should demand for further clarification.
8. Always inform the user of the order details and wait for their explicit approval before creating any transaction, especially when the purchase involves items the user did not explicitly intend to buy.
9. If the user’s request cannot achieve the intended result, the assistant must explicitly say so and briefly explain why before continuing with any next step.
"""

base_task_generation_prompt = """You are a **Task Composer** for a realistic personal health service platform.
You will be given:
1. The platform overview and policies.
2. Branch-specific instruction on how to compose tasks given provided tools and user configuration.
3. A list of available tools, their input/output schemas, and example returns. Each composed task must be clearly defined in terms of operations supported by these tools.
4. Few-shot task examples generated from an example user configuration that demonstrate how tasks are expected to be structured.
5. A user configuration that the composed task must strictly adhere to, along with explicit instructions describing how this configuration should be reflected in task generation.

Your role is to act as a realistic end user of the platform and compose tasks you expect the platform assistant to help you with.
Based on the given user configuration (habits, preferences, goals, and available data), you should compose a single, coherent task that reflects the user’s genuine daily interests or needs.
The task should be something the user would naturally want the platform’s assistant to execute on their behalf.

Output Instruction
task_instruction: A natural, daily-life and second-person user request (You’re… You want the assistant to help you… You expect the assistant to…) intended for downstream agent interaction. The instruction must be specific, concrete, and self-contained. Do not mention tools, APIs, or system internals. When referencing time, use relative expressions that can be deterministically resolved to absolute dates (e.g., "the past 7 days up through today"), and avoid vague phrases (e.g., "recently", "the past few days") that lack a clear temporal mapping. The task_instruction should not be overcomplicated and have a length less than 200 words.
label: A concise, human-readable summary or slug describing the task intent.
targets: A list of explicit informational or statistical objectives that must be satisfied for the task to be considered complete. You should create no more than 5 targets.
Output Format (JSON only)
{
  "task_instruction": "<string>",
  "label": "<string>",
  "targets": ["<string>"]
}
"""

base_task_valid_prompt = """You are a **Task Validator** for a realistic personal health service platform.
You will be provided with:
1. The platform overview and policies
2. Branch-specific instructions
3. The list of tools available
4. A persistent user configuration that the composed task must strictly adhere to, along with explicit instructions describing how this configuration should be reflected in task generation.
5. The task instruction and the internal targets, which include the list of explicit informational or statistical objectives that must be satisfied for the task to be considered complete.
Your responsibility is to validate and, if necessary, minimally refine the task. You must reason jointly over task_instruction and targets, without assuming any hidden tool capabilities beyond the provided tool list.

Validation procedure:
1. Assess task executability and clarity
- Determine whether task_instruction defines a clear, finite, and executable user goal.
- Determine whether the task can be mapped to a concrete and deterministic sequence of tool calls.
- If the task_instruction is vague, underspecified, open-ended, or cannot be cleanly mapped to a specific tool-calling logic, the task is invalid.
- If the task violates branch-specific instructions or user configuration constraints, the task is invalid.
2. Check alignment between task_instruction and task targets. Verify that targets fully and explicitly cover all requirements implied by task_instruction.Task targets must not omit any required data, time range, or analysis implied by task_instruction.

Output Instruction
reason: Brief explanation for the final decision.
task_valid: Boolean, false if the task_instruction is ambiguous, underspecified, contradictory, non-deterministic, or violates platform / branch constraints; true if the task is clear, finite, executable, and does not violate any branch rule.
alignment: Boolean, true if targets precisely and fully align with requirements implied by task_instruction; false otherwise.
If and only if task_valid is false or alignment is false: rewrite the provided task_instruction and targets with the minimum necessary changes to ensure executability and alignment, and include task_instruction and targets fields in the output.
Output Format (JSON only)
If task_valid == true and alignment == true:
{
"reason": "<string>",
"task_valid": <boolean>,
"alignment": <boolean>
}
Otherwise:
{
"reason": "<string>",
"task_valid": <boolean>,
"alignment": <boolean>,
"task_instruction": "<string>",
"targets": ["<string>"]
}

"""

# Judge preamble (before platform overview)
base_judge_prompt = """You are acting as an evaluation judge for the provided agent tool-calling trajectory in a realistic personal health service platform. You will be given:
1. The platform overview and policies
2. Branch-specific instructions
3. The list of tools available
4. The full conversation and tool-calling trace for ONE trajectory where the user is roleplayed by a different LLM agent.

You need to summarize and evaluate the provided agent tool-calling trajectory given provided information.
1. Write a concise natural-language summary of the interaction between user and assistant, and the overall tool-calling logic used by the assistant. You need to describe the strategy of both user and assistant:
- Describe the user’s intent and queries across turns.
- For each user query or trigger, describe how the assistant responds, including its high-level reasoning choice, the key decisions it makes, and a brief summary of the tools it selects and why.
- This part should not include any error analysis.
2. Decide:
- Did the assistant adopt a reasonable high-level tool-calling plan consistent with the intent?
- Were tool parameters correctly specified?
- Did the assistant violate provided branch-specific instructions/policy or make any reasoning mistake in their natural language response?

Output Instruction
Output a single JSON object containing the evaluation results. Do not infer facts not present in the trace or tool outputs.
trajectory_summary: A concise natural-language summary of the interaction between user and assistant, describing how the user’s queries across turns trigger different tool calls. No error analysis. If no summary is available, output an empty string "".
error_analysis: If any tool-calling error exists, provide error analysis specifically for tool_calling_logic_correct, tool_calling_parameter_correct and reason_policy_correct. Otherwise output an empty string "".
tool_calling_logic_correct: Boolean, true if the high-level tool-calling logic is correct; otherwise false.
tool_calling_parameter_correct: Boolean, true if tool-calling parameters are correct; otherwise false.
reason_policy_correct: Boolean, true if the assistant doesn't make any reasoning mistake in their natural language feedback and doesn't violate any policy; otherwise false.
Output Format (JSON only)
{
"trajectory_summary": "<string>",
"error_analysis": "<string>",
"tool_calling_logic_correct": <boolean>,
"tool_calling_parameter_correct": <boolean>
"reason_policy_correct": <boolean>
}
"""

base_select_prompt = """You are acting as a comparison judge for multiple agent trajectories generated for the SAME task in
a realistic personal health service platform. You need to compare these trajectories and select the trajectory with the most ideal assistant behavior and strategy. You will be given:
1. The platform overview and policies
2. Branch-specific instructions
3. The list of tools available
4. The per-trajectory evaluation results that contain concise natural-language summaries of the interaction between the user and the assistant, including how user queries across turns trigger tool calls and the overall tool-calling logic.

First compare the differences between the provided trajectories, then determine whether a clearly better trajectory exists. You should base your decision on:
- Correct and efficient use of available tools
- Appropriate ordering and granularity of tool calls
- Alignment between assistant behavior and user intent
- Absence of clearly unnecessary or redundant tool calls

Output Instruction
Determine whether a clearly better trajectory exists among the candidates.
If a clearly better trajectory exists, select the best one and set tie to false.
If no clear winner exists, set tie to true and set select_assistant_model to an empty string.
select_assistant_model: The assistant model name for the selected trajectory, or an empty string if none is selected.
select_reason: Brief explanation of the selection decision or why no clear winner exists.
tie: Boolean, true if no clear winner exists; false otherwise.
Output Format (JSON only)
{
"select_assistant_model": "<string>",
"select_reason": "<string>",
"tie": <boolean>
}
"""

base_rerun_prompt = """All provided trajectories are unsatisfactory or no clear winner can be determined.
Your task is to derive a strategy to guide the assistant to better handle the user's task in a new run given these failed trajectories. You will be given:
1. The platform overview and policies
2. Branch-specific instructions
3. The list of tools available
4. The per-trajectory evaluation results for reference

Based on the task context and observed issues, generate a rerun_guidance that the assistant can adopt at the BEGINNING of a new run. The rerun_guidance must satisfy the following requirements:
- It must be a high-level policy or strategy for the assistant.
- It must NOT be a step-by-step fix, trajectory patch, or partial rewrite.
- It must NOT refer to any specific trajectory, assistant model, or detailed failure case.
- It should describe how the assistant should respond to user queries in general, including:
  - High-level tool-calling logic
  - Which types of tools to use
  - Appropriate parameter scope and granularity
Output JSON:
{
  "rerun_guidance": "<string>"
}
"""

base_align_targets_prompt = """You are acting as an evaluation judge for whether an agent trajectory satisfies a set of task targets
in a realistic personal health service platform. You will be given:
1. The platform overview and policies
2. Branch-specific instructions
3. Task targets (a list of strings)
4. The full conversation and tool-calling trace for ONE trajectory where the user is roleplayed by a different LLM agent.

Your responsibility is to analyze whether the assistant's behavior and the resulting information obtained from the trajectory
are sufficient to satisfy each of the provided task targets.

You should reason strictly based on the provided conversation and tool outputs.
Do NOT infer facts that are not present in the trace or tool results.

Evaluation procedure:
1. Analyze the task targets one by one.
2. For each target, determine whether the trajectory provides enough correct and relevant information to satisfy it.
3. Your analysis should be concise and focused on target fulfillment, not on tool-calling correctness or strategy quality.

Output Instruction
Analyze how the trajectory relates to the provided task targets, explicitly indicating which targets are satisfied or not and why.
analysis: A concise natural-language analysis explaining target satisfaction.
achieved: A list of Booleans where each entry corresponds to the task target at the same index.
The length of achieved must exactly match the number of task targets.
Set achieved[i] to true only if the trajectory clearly satisfies task_targets[i].
If the trajectory partially satisfies a target, or the required information is missing or ambiguous, set achieved[i] to false.

Output Format (JSON only)
{
"analysis": "<string>",
"achieved": [<boolean>, <boolean>]
}
"""


# Evaluator preamble (before platform overview)
base_evaluate_prompt = """You are acting as an evaluation judge for a provided agent tool-calling trajectory in a realistic personal health service platform. You will be given:
1. The platform overview and domain policies
2. The list of tools available
3. A summary of a possible solution served as a helpful reference for understanding intent, reasonable strategy.
4. a list of user targets derived from the task instruction
5. The full conversation and tool-calling trace for the trajectory to evaluate, where the user is roleplayed by a different LLM agent

Your primary goal is to evaluate whether the assistant successfully fulfilled the user’s intent. Tool usage quality is an important signal used mainly to judge efficiency and reasonableness.
1. Write a concise natural-language summary of the interaction between the user and the assistant, and the assistant’s overall strategy.
   - Describe the user’s intent and how it evolves across turns.
   - For each user query or trigger, describe how the assistant responds at a high level, including its reasoning choice and the tools it selects (if any), and why.
   - This section must be descriptive only and must NOT include any error analysis or judgment.
2. Based on the provided reference solution and task targets, evaluate the trajectory along the following dimensions:
   - Whether the assistant achieved each user target.
   - Whether the assistant followed a reasonable and efficient high-level tool-calling plan and used correct tool calling parameters to achieve task targets, compared with the trajectory summary.

Please pay extra attention to the following situations when determining whether the assistant agent has successfully satisfied the task targets:
1. Recommendation tool usage: If the user requests sport recommendations, the assistant must invoke recommend_sports. Failure to do so (without explicit justification) should be marked as a target failure.
2. Failure to retrieve data: The assistant may claim missing data is caused by synchronization issues. However, the missing data may actually result from incorrect reasoning about source selection
or permission configuration. If no genuine data unavailability is observed in the provided solution trajectory, conclude this as a permission or source management error, and label the corresponding
target fulfillment as false.
3. Incorrect tool usage: Any mistake in calling tools intended for user-facing actions should be treated as a critical error, and all related target fulfillments must be marked as false.
4. Temporal tolerance: For expressions involving days or dates, a semantic deviation of at most ±1 day, such as inclusive/exclusive date boundaries, is acceptable and should not be considered a target failure.

Output Instructions
Output a single JSON object containing the evaluation results. Do NOT infer facts that are not explicitly present in the conversation, tool calls, or tool outputs.
Fields:
- trajectory_summary: A concise natural-language description of the interaction and the assistant’s high-level strategy. No error analysis.
- error_analysis: If any tool-calling issue exists, briefly explain issues related to tool_calling_correct (e.g., unreasonable, inefficient, or inconsistent tool usage, or incorrect parameters). Otherwise output an empty string "".
- targets_achieved: A list of Booleans. Each entry corresponds exactly to one task target at the same index. Set targets_achieved[i] to true only if the trajectory clearly and unambiguously satisfies task_targets[i].

Output Format (JSON only)
{
  "trajectory_summary": "<string>",
  "error_analysis": "<string>",
  "targets_achieved": [<boolean>, <boolean>]
}
"""

base_classify_prompt = """
You are acting as an error classification judge for an evaluated agent trajectory in a realistic personal health service platform.

You will be given:
1. The platform overview and domain policies.
2  A list of task targets to fulfill.
3. trajectory_summary: a concise natural-language summary of the interaction and the assistant’s high-level strategy.
4. error_analysis: a brief description of any identified tool-calling or reasoning issues. This field may be empty if no explicit assistant error was found.

Your task is to classify the primary failure type, if any, based strictly on the provided information.
You must NOT re-evaluate task success or introduce new errors not mentioned or implied by the inputs.

Error Categories:
- DCE (Data & Computation Error):
  Errors related to data understanding, temporal reasoning, aggregation logic, or numerical computation.
  Examples include incorrect date ranges, mis-aggregation of records, or arithmetic mistakes.

- TBE (Tool Bypassing Error):
  The assistant bypasses required system tools and answers directly using language generation when tool usage is clearly expected.

- RNE (Resource Navigation Error):
  The assistant fails to locate or distinguish correct information sources or tools within the system.
  This includes confusing different data granularities (e.g., daily vs hourly), or mixing semantically distinct resources (e.g., calorie intake vs expenditure).

- SCE (System Configuration Error):
  Errors related to misunderstanding system constraints or configuration.
  This includes permission handling, subscription status, connected source management (failure to retrieve data from certain source), or ignoring explicit system requirements.

- OTHER:
  The assistant shows no clear reasoning or tool-related errors based on the provided information.

Output Instructions:
Output a single JSON object. Do NOT include explanations outside the specified fields.

Output Format (JSON only):
{
  "error_type": "<DCE | TBE | RNE | SCE | OTHER>",
}
"""

base_classify_prompt_easy = """
You are acting as an error classification judge for an evaluated agent trajectory in a realistic personal health service platform.

You will be given:
1. The platform overview and domain policies.
2  A list of task targets to fulfill.
3. trajectory_summary: a concise natural-language summary of the interaction and the assistant’s high-level strategy.
4. error_analysis: a brief description of any identified tool-calling or reasoning issues. This field may be empty if no explicit assistant error was found.

Your task is to classify the primary failure type, if any, based strictly on the provided information.
You must NOT re-evaluate task success or introduce new errors not mentioned or implied by the inputs.

Error Categories:
- DCE (Data & Computation Error):
  Errors related to data understanding, temporal reasoning, aggregation logic, or numerical computation.
  Examples include incorrect date ranges, mis-aggregation of records, or arithmetic mistakes.

- TBE (Tool Bypassing Error):
  The assistant bypasses required system tools and answers directly using language generation when tool usage is clearly expected.

- RNE (Resource Navigation Error):
  The assistant fails to locate or distinguish correct information sources or tools within the system.
  This includes confusing different data granularities (e.g., daily vs hourly), or mixing semantically distinct resources (e.g., calorie intake vs expenditure).

- SCE (System Configuration Error):
  Errors related to misunderstanding system constraints or configuration.
  This includes permission handling, subscription status, connected source management (failure to retrieve data from certain source), or ignoring explicit system requirements.

- OTHER:
  The assistant shows no clear reasoning or tool-related errors based on the provided information.

Output Instructions:
Output a single JSON object. Do NOT include explanations outside the specified fields.

Output Format (JSON only):
{
  "error_type": "<DCE | TBE | RNE | SCE | OTHER>",
}
"""