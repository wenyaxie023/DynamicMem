import json
from typing import Any, Dict, List, Optional

from tce_contracts import CURRENT_TASK_CONTRACT_VERSION


TASK_C_V2_USER_COMMUNICATION_TASK_INSTRUCTION = (
    "Draft a specific reminder message for the user in this scenario."
)
TASK_C_V2_INFORMATION_REQUEST_TASK_INSTRUCTION = (
    "Help the user set the search filters in this scenario."
)
TASK_C_V2_ACTION_CONFIGURATION_TASK_INSTRUCTION = (
    "Help the user complete the setup or form fields in this scenario."
)
SCHEDULE_DATE_ENCODING_TEXT = (
    "`day_of_week` and `days_of_week` use zero-based weekday indexes: "
    "0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday. "
    "`day_of_month` and `days_of_month` use calendar day numbers 1-31; they are not zero-based. "
    "`week_of_month` uses ordinal week numbers within the month: "
    "1=first, 2=second, 3=third, 4=fourth, 5=fifth."
)
SCHEDULE_DATE_ENCODING_BULLETS = """- schedule date encodings:
  - `day_of_week` and `days_of_week` use zero-based weekday indexes: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday.
  - `day_of_month` and `days_of_month` use calendar day numbers 1-31; they are not zero-based.
  - `week_of_month` uses ordinal week numbers within the month: 1=first, 2=second, 3=third, 4=fourth, 5=fifth."""
SCHEDULE_FORMAT_SUPPLEMENT = """- Habit schedule format vocabulary:
  - `daily`: every day. Format: `{"frequency_type": "daily"}`
  - `weekly`: every week on the listed weekday(s), such as every Tuesday and Thursday. Format: `{"frequency_type": "weekly", "days_of_week": [0-6 integers, 0=Monday ... 6=Sunday]}`
  - `biweekly`: every two weeks on one listed weekday; `start_date` is the first occurrence and anchors the alternating-week pattern. Format: `{"frequency_type": "biweekly", "days_of_week": [single integer 0-6], "start_date": "YYYY-MM-DD"}`
  - `monthly_by_date`: every month on specific calendar date(s), such as the 1st or 15th of each month. Format: `{"frequency_type": "monthly_by_date", "days_of_month": [1-28 integers]}`
  - `monthly_nth_weekday`: every month on an ordinal weekday, such as the first Monday, third Friday, or last Sunday of the month. Format: `{"frequency_type": "monthly_nth_weekday", "week_of_month": "1-4 or last", "day_of_week": "0-6 integer"}`
  Use these exact vocabulary values when the evidence supports them."""


def _coerce_inline_memory_blocks(
    *,
    inline_memory_blocks: Optional[List[str]],
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
) -> List[str]:
    if inline_memory_blocks is not None:
        return [str(block) for block in inline_memory_blocks if str(block).strip()]
    if context_logs is None:
        return []
    return [str(log_to_text(log)) for log in context_logs]


def _render_inline_memory_section(
    *,
    inline_memory_blocks: List[str],
) -> str:
    context = "\n<->\n".join(str(block) for block in inline_memory_blocks if str(block).strip())
    return f"""[Memory]
{context}
[/Memory]"""


def _render_agent_memory_section() -> str:
    return """[Memory]
Use the system's stored checkpoint-bounded memory about the user's trajectory to answer.
[/Memory]"""


def _is_habit_state_key(state_key: str) -> bool:
    return str(state_key or "").strip().startswith("habits_state:")


def _state_completion_schedule_instruction(target_keys: List[str]) -> str:
    if not any(_is_habit_state_key(key) for key in target_keys):
        return ""
    return f"- Schedule date encoding: {SCHEDULE_DATE_ENCODING_TEXT}\n{SCHEDULE_FORMAT_SUPPLEMENT}\n"


def build_task_c_task_body(
    *,
    scenario: str,
    task_instruction: str,
    output_template: Any = None,
) -> str:
    scenario = str(scenario or "").strip()
    task_instruction = str(task_instruction or "").strip()
    parts = [
        "[Scenario]\n{}".format(scenario),
        "[Task Instruction]\n{}".format(task_instruction),
    ]
    if output_template is not None:
        parts.append("[Required Output Object]\n{}".format(json.dumps(output_template, ensure_ascii=False, indent=2)))
    return "\n\n".join(parts)


def _build_state_completion_prompt(
    *,
    memory_section: str,
    task_query: str,
    target_keys: List[str],
    target_value_templates: Dict[str, Any],
) -> str:
    user_state_template = {k: target_value_templates.get(k, "<fill the blank>") for k in target_keys}
    evidence_template = {
        k: [{"app_log_id": "<app_log_id>", "evidence_content": "<supporting snippet>"}]
        for k in target_keys
    }
    fill_template = {
        "user_state": user_state_template,
        "evidence": evidence_template,
    }
    fill_template_block = json.dumps(fill_template, ensure_ascii=False, indent=2)
    schedule_instruction = _state_completion_schedule_instruction(target_keys)

    return f"""[Instructions]
- Answer the question below using only the system memory about the user's trajectory provided in [Memory]...[/Memory].
- **Make each answer value as detailed and accurate as the memory supports.** Preserve specific names, times, dates, places, labels, and constraints instead of giving vague summaries.
- Follow the template exactly, and fill every requested field with the most precise supported value.
{schedule_instruction}- `evidence` for each key must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only.
- No extra keys.

[Output format]
{{
  "user_state": {{
    "<key>": "<detailed, accurate value or nested object following the template>",
    "...": "<same structure as template>"
  }},
  "evidence": {{
    "<key>": [
      {{
        "app_log_id": "<app_log_id>",
        "evidence_content": "<supporting snippet from the same log>"
      }}
    ],
    "...": []
  }}
}}

[Question]
{task_query}
[/Question]

{memory_section}

Concrete JSON skeleton to fill:
{fill_template_block}
"""


def build_state_completion_prompt_with_inline_memory(
    *,
    context_logs: Optional[List[Dict[str, Any]]],
    task_query: str,
    target_keys: List[str],
    target_value_templates: Dict[str, Any],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    return _build_state_completion_prompt(
        memory_section=_render_inline_memory_section(
            inline_memory_blocks=_coerce_inline_memory_blocks(
                inline_memory_blocks=inline_memory_blocks,
                context_logs=context_logs,
                log_to_text=log_to_text,
            )
        ),
        task_query=task_query,
        target_keys=target_keys,
        target_value_templates=target_value_templates,
    )


def build_state_completion_prompt_with_agent_memory(
    *,
    context_logs: Optional[List[Dict[str, Any]]],
    task_query: str,
    target_keys: List[str],
    target_value_templates: Dict[str, Any],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    del context_logs, log_to_text, inline_memory_blocks
    return _build_state_completion_prompt(
        memory_section=_render_agent_memory_section(),
        task_query=task_query,
        target_keys=target_keys,
        target_value_templates=target_value_templates,
    )


def _build_change_reasoning_prompt(
    *,
    memory_section: str,
    task_query: str,
    changed_keys: List[str],
    changed_value_templates: Dict[str, Any],
) -> str:
    change_template = {k: changed_value_templates.get(k, {}) for k in changed_keys}
    fill_template = {"change_analysis": change_template}
    fill_template_block = json.dumps(fill_template, ensure_ascii=False, indent=2)

    return f"""{task_query}

{memory_section}

[Instructions]
- Answer this user question based on the memory.
- `evidence` for each key must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only.
- No extra keys.

[Output format]
{{
  "change_analysis": {{
    "<key>": {{
      "before": "<value or nested object following template.before>",
      "after": "<value or nested object following template.after>",
      "change_reason": "<one concise reason or attribution>",
      "evidence": [
        {{
          "app_log_id": "<app_log_id>",
          "evidence_content": "<supporting snippet from the same log>"
        }}
      ]
    }}
  }}
}}

Template:
{fill_template_block}
"""


def build_change_reasoning_prompt_with_inline_memory(
    *,
    context_logs: Optional[List[Dict[str, Any]]],
    task_query: str,
    changed_keys: List[str],
    changed_value_templates: Dict[str, Any],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    return _build_change_reasoning_prompt(
        memory_section=_render_inline_memory_section(
            inline_memory_blocks=_coerce_inline_memory_blocks(
                inline_memory_blocks=inline_memory_blocks,
                context_logs=context_logs,
                log_to_text=log_to_text,
            )
        ),
        task_query=task_query,
        changed_keys=changed_keys,
        changed_value_templates=changed_value_templates,
    )


def build_change_reasoning_prompt_with_agent_memory(
    *,
    context_logs: Optional[List[Dict[str, Any]]],
    task_query: str,
    changed_keys: List[str],
    changed_value_templates: Dict[str, Any],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    del context_logs, log_to_text, inline_memory_blocks
    return _build_change_reasoning_prompt(
        memory_section=_render_agent_memory_section(),
        task_query=task_query,
        changed_keys=changed_keys,
        changed_value_templates=changed_value_templates,
    )


def _build_personalized_service_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_type: str,
    state_key: str,
    state_value: Any,
    item_count: int = 2,
) -> str:
    item_count = max(1, int(item_count))
    item_template = [
        {
            "service_category": "<one service category name>",
            "question": "<one concise free-form question asking what service should be provided for the user>",
            "reference_answer": "<short canonical answer>",
        }
        for i in range(item_count)
    ]
    return """[Task Instruction]
Generate personalized service free-form QA items that test whether an LLM can remember and correctly use one user state item
to provide the right personalized service for the user.

The generated question will be asked directly to the responding model, which should naturally act as the assistant serving the user.
The question must require the given user state item to answer correctly.

Do not generate questions that mainly test state recall, paraphrase, trivial lookup, or generic common-sense advice.
Instead, generate questions where the best answer depends on using the provided user state item to decide what service should be provided for the user.

Each item must include:
- one service_category
- one question
- one reference_answer

[Definitions]
- user state item: one structured memory item about the user.
- state_type: the type of the user state item. It must be exactly one of:
  - attribute
  - habit
  - preference

- attribute: a relatively stable fact about the user, such as memberships, affiliations, roles, social context, access, or standing constraints.
- habit: a recurring behavior pattern, routine, or repeated timing pattern that affects when or how service should be delivered.
- preference: a favored option, favored format, or comparative inclination that affects which service action is best for the user.

- personalized service: a service decision or action tailored to the user, such as what should be recommended, scheduled, sent, chosen, routed, or prioritized for the user.

- service_category: the type of service decision the item is about. It must name the service context in a short and concrete way.
  Allowed examples include:
  - shopping recommendation
  - plan recommendation
  - scheduling
  - outreach
  - event recommendation
  - provider choice
  - notification strategy
  - community engagement
  - content recommendation
  - support routing
  The service_category is a label for the kind of service being personalized, not a full scenario description.

- question: one concise free-form question asking what service should be provided for the user.
  The question should be written so that the responding model naturally acts as the assistant.
  The question must not be multiple choice and must not list answer options.
  Good patterns include:
  - What policy should the assistant apply for the user ...?
  - How should the assistant prioritize or route ...?
  - What service configuration should the assistant adopt for the user ...?
  - How should the assistant respond when ...?

- reference_answer: a short canonical answer naming the best personalized service action for the user.
  It should be specific, action-oriented, and concise.

- strong item: an item where the given state item is necessary for the correct answer, and if the state changed to a reasonable alternative, the best answer would likely change.
- weak item: an item that can be answered correctly without using the state item, or by generic common sense alone, or by simply paraphrasing the state.

[Constraints]
1. Generate exactly {item_count} items in list order.
2. Each item must use exactly one given user state item as the key personalization signal.
3. Each item must be a free-form QA item.
4. Each item must be about a realistic personalized service decision.
5. The question must directly ask what service should be provided for the user.
6. The question must be concise and ask for one best service action.
7. Do not include explicit answer options in the question, and do not frame the question as a named A/B menu of candidate actions.
8. The correct answer must depend on the given state item; if the state changed to a reasonable alternative, the best answer would likely change.
9. Do not rely on any user state other than the provided state item.
10. Do not address the user as "you", "your", or "yours".
11. Keep the reference_answer short, specific, and action-oriented.
12. The program will assign internal item ids automatically. Do not generate `qa_id` or any other internal identifier field.
13. Prefer service decisions that fit the state type:
   - for attribute: route/prioritize policy, specialist-plan choice, eligibility/access handling, provider choice, or bounded support triage
   - for habit: interruption policy, batching policy, defer-vs-escalate policy, notification suppression/surfacing, or service timing strategy around a recurring routine
   - for preference: recommendation format, communication format, plan format, or recommend-vs-avoid policy where the best action depends on the user's comparative preference
14. Prefer bounded policy or service-decision questions such as:
   - defer vs escalate policy
   - route / prioritize policy
   - recommend / avoid policy
   - configure / suppress / surface policy
15. Avoid question types that mainly ask for:
   - exact capacity calculation
   - lot or tactic optimization
   - monitoring-rule design
   - threshold-setting
   - exact inventory / participant arithmetic
16. The question must require a real decision tension: at least two plausible service actions should exist, and the provided state item must be what makes one action better than the alternative.
17. For habit states, schedule-grounded service operations are valid when the question asks what the assistant should do around the routine, such as silence, batch, defer, protect, or flag conflicts; do not ask the model to merely report the routine time/date or compute a timestamp from it.
18. Do not turn the item into a pure lookup, direct state restatement, trivial extraction, or simple calendar arithmetic problem with no real assistant-action layer.
19. Reject questions whose scenario merely echoes the same wording, label, or semantic category already present in the state; if the question can be answered by copying the state language into the scenario, the item is invalid.
20. Reject questions whose main work is exact timestamp calculation, duration calculation, buffer calculation, generic troubleshooting, or simple fit/capacity matching.
21. Reject questions where the scenario itself injects the decisive rule and the user state becomes unnecessary.
22. Before finalizing an item, internally test a reasonable counterfactual version of the state; if the best action would not change, discard that candidate and generate a stronger item.
23. The reference_answer must not introduce concrete facts, parameters, product specifications, thresholds, or background knowledge that are not explicit in `state_value` or in the question itself.
24. The reference_answer must not introduce extra methods, tactics, metrics, escalation mechanisms, capacity calculations, or domain-specific tricks unless they are already explicit in `state_value` or in the question.
25. Make the reference_answer a complete action rather than a vague label or partial sub-step.
26. Output JSON only, strictly matching schema, with no extra fields.

[Example]
[Example Input 1]
state_type: "preference"
state_key: "learning_modality"
state_value: {{
  "statement": "Prefers in-depth, self-paced technical white papers and webinars over large conferences",
  "signals": [
    "Completed a self-paced technical certificate program instead of attending a live bootcamp",
    "Saved several specialist white papers for later review",
    "Declined a large conference in favor of webinar-based continuing education"
  ]
}}

[Example Output 1]
{{
  "items": [
    {{
      "service_category": "training plan recommendation",
      "question": "What training format should the assistant arrange for the user's upcoming technical upskilling plan?",
      "reference_answer": "Arrange a self-paced package of technical white papers and on-demand webinars instead of registering the user for a live conference."
    }}
  ]
}}

[Example Input 2]
state_type: "habit"
state_key: "sunday_family_dinner"
state_value: {{
  "schedule": {{
    "frequency_type": "weekly",
    "days_of_week": [0]
  }},
  "timing": {{
    "start_time": "15:00",
    "end_time": "17:00"
  }},
  "priority": "high"
}}

[Example Output 2]
{{
  "items": [
    {{
      "service_category": "notification strategy",
      "question": "How should the assistant handle non-urgent notifications during the user's recurring Sunday family dinner block?",
      "reference_answer": "Silence non-urgent notifications until the dinner block ends, and surface only urgent family coordination issues immediately."
    }}
  ]
}}

[Example Input 3]
state_type: "attribute"
state_key: "professional_certifications"
state_value: "Certified to advise clients on new environmental compliance standards for industrial coating operations."

[Example Output 3]
{{
  "items": [
    {{
      "service_category": "support routing",
      "question": "How should the assistant route this week's incoming client advisory work for the user?",
      "reference_answer": "Route environmental compliance advisory requests that match the user's certification to the user, and route generic relationship-management follow-ups elsewhere."
    }}
  ]
}}

[Bad Example]
- Bad question: "What exact time interval should the assistant reserve for the user's final October budget review if the session needs a 15-minute lead-in and lasts 60 minutes?"
- Why bad:
  - It mainly reduces to schedule lookup plus one-step arithmetic.
  - It does not test a meaningful personalized service decision.
  - It is likely to fail `service_decision_quality`.

[Bad Example 2]
- Bad question: "What specific lot-selection strategy should the assistant utilize when the user requests to liquidate a portion of the portfolio for an upcoming expense?"
- Bad answer: "Prioritize selling investment lots with the highest cost basis to minimize realized capital gains."
- Why bad:
  - The answer imports a specific financial tactic that is not explicit in the state or the question.
  - It is likely to fail `answer_groundedness`.

[Bad Example 3]
- Bad question: "What type of care recommendation should the assistant prioritize when the user reports a minor, non-acute health issue?"
- Why bad:
  - It mainly restates the preference into a matching scenario rather than creating a real service decision.
  - It can be answered by echoing the same state wording, so it is likely to fail `service_decision_quality`.

[Input/Output Format]
Input:
- checkpoint_timestamp: checkpoint timestamp string
- state_type: one of "attribute", "habit", "preference"
- state_key: one validated state item key
- state_value: structured value for that state item

Actual Input:
- checkpoint_timestamp: {checkpoint_timestamp}
- state_type: {state_type}
- state_key: {state_key}
- state_value: {state_value}

Output JSON ONLY:
{{
  "items": [
    {{
      "service_category": "...",
      "question": "...",
      "reference_answer": "..."
    }}
  ]
}}

[Schema Template]
{item_template}
""".format(
        item_count=item_count,
        checkpoint_timestamp=checkpoint_timestamp,
        state_type=json.dumps(state_type, ensure_ascii=False),
        state_key=json.dumps(state_key, ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False),
        item_template=json.dumps(item_template, ensure_ascii=False, indent=2),
    )


def _infer_apply_state_type(state_key: str) -> str:
    prefix = str(state_key or "").split(":", 1)[0].strip().lower()
    if prefix == "habits_state":
        return "habit"
    if prefix == "preferences_state":
        return "preference"
    return "attribute"


def _fill_placeholder_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _fill_placeholder_template(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_fill_placeholder_template(child) for child in value]
    return "<fill>"


def _task_c_v2_contract_preference_state_input(state_value: Any) -> Any:
    if not isinstance(state_value, dict):
        return state_value
    statement = state_value.get("statement")
    if statement is None:
        return state_value
    return {"statement": statement}


def build_rq3_apply_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_key: str,
    state_value: Any,
    item_count: int = 2,
) -> str:
    return _build_personalized_service_question_pack_prompt(
        checkpoint_timestamp=checkpoint_timestamp,
        state_type=_infer_apply_state_type(state_key),
        state_key=state_key,
        state_value=state_value,
        item_count=item_count,
    )

def _build_task_c_v2_user_communication_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_type: str,
    state_key: str,
    state_value: Any,
) -> str:
    del checkpoint_timestamp
    if str(state_type or "").strip() != "habit":
        raise ValueError("Task C v2 user_communication prompt requires state_type='habit'.")

    fixed_task_instruction = TASK_C_V2_USER_COMMUNICATION_TASK_INSTRUCTION
    return f"""[Task]
Generate exactly one habit-conditioned communication task for a user-facing assistant.

Each item contains:
- `scenario`: synthesize this field.
- `task_instruction`: copy the fixed string exactly.
- `reference_answer`: synthesize this field as the intended correct assistant message.

The task should require the assistant to use the user's habit state, not only the scenario.
The reference_answer must be exactly one proactive assistant-to-user message for this moment.
It must be free-form natural language.

[Design Principle]
Keep `scenario` leakage-safe:
- `task_instruction` is fixed and must be copied exactly.
- `scenario` may set up the current moment and local situation,
- but it must not restate, paraphrase, or strongly imply the user-state facts that should instead be recovered from `state_value`.

[Definitions]
- terminal field: one leaf field in state_value whose value is scalar or array-valued and not further decomposed.
- leaf path: the path to a terminal field using dot notation, for example: schedule.days_of_week or timing.start_time.
- world-background scenario: a short third-person description of what is true right now in the world. It is not spoken by the assistant, not spoken by the user, and not written from the user's point of view.
- current-world anchor: the weekday, calendar date, and current clock time in `scenario`. These anchors are allowed and are not leakage by themselves; they make the current-moment service task answerable.
{SCHEDULE_DATE_ENCODING_BULLETS}

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "reference_answer"
4. Copy task_instruction exactly as this fixed string:
   {fixed_task_instruction}
5. scenario must be short, concrete, and written as third-person world background.
6. scenario must not use first-person or second-person wording such as "I", "we", "you", "your", or "you've".
7. scenario must anchor the current moment clearly enough for the task to be answerable:
   - for weekly or weekday-specific routines: include the current weekday + current clock time;
   - for monthly or date-like routines: include the current calendar anchor + current clock time;
   - if `state_value` contains `schedule.days_of_week`, `schedule.day_of_week`, `schedule.days_of_month`, `schedule.day_of_month`, or `schedule.week_of_month`, a clock time alone is not enough.
   - interpret all schedule weekday/month fields using the schedule date encodings above; for example, days_of_week: [1] means Tuesday, not Monday.
8. scenario may include only:
   - the current moment,
   - whether something has or has not happened yet,
   - whether something has or has not been prepared,
   - at most one additional situational fact that plausibly matters right now.
9. scenario must not restate or paraphrase the routine action, frequency, stored start time, stored end time, location, or any other personalized habit fact already present in state_value. It may state the current weekday/date/time as world background, but it must not say the routine itself starts, occurs, meets, happens, or is located at a state-derived value.
10. reference_answer must be exactly one natural assistant-to-user message, not a meta description.
11. reference_answer must be complete enough that a fully correct answer would use every terminal field in state_value.
12. Before finalizing, silently confirm the item would later pass the same five semantic validation criteria:
   - answerability: scenario plus the fixed task_instruction define one clear current-moment communication task.
   - service_realism: the item describes a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state.
   - full_field_dependency: a fully correct message needs all non-derived field paths in state_value.
   - low_leakage: scenario does not restate or strongly imply the habit facts that should come from state_value.
   - output_groundedness: reference_answer is a short natural-language assistant message whose personalized content is supported by state_value without adding unsupported user-specific facts.

[Good Example A Input]
state_key: {json.dumps("habits_state:client_technical_briefing", ensure_ascii=False)}
state_value: {json.dumps({
    "schedule": {
        "frequency_type": "weekly",
        "days_of_week": [0],
    },
    "timing": {
        "start_time": "10:00",
    },
    "location": "regional corporate headquarters",
}, ensure_ascii=False, indent=2)}

[Good Example A Output]
{json.dumps({"item": {
    "scenario": "It is Monday at 09:20. Nothing has been started yet this morning.",
    "task_instruction": fixed_task_instruction,
    "reference_answer": (
        "Your weekly client technical briefing is at 10:00 today at the regional corporate headquarters. "
        "Since Monday is the scheduled day, it is almost time to get ready."
    )
}}, ensure_ascii=False, indent=2)}

[Good Example B Input]
state_key: {json.dumps("habits_state:monthly_hoa_meeting", ensure_ascii=False)}
state_value: {json.dumps({
    "schedule": {
        "frequency_type": "monthly_nth_weekday",
        "week_of_month": 3,
        "day_of_week": 0,
    },
    "timing": {
        "start_time": "12:00",
    },
    "location": "Wexford community center hall",
}, ensure_ascii=False, indent=2)}

[Good Example B Output]
{json.dumps({"item": {
    "scenario": "It is Monday, January 20th at 11:15 AM. No travel has been initiated yet.",
    "task_instruction": fixed_task_instruction,
    "reference_answer": (
        "It is the third Monday of the month, and your monthly HOA meeting starts at 12:00 at the "
        "Wexford community center hall. It is almost time to head over."
    )
}}, ensure_ascii=False, indent=2)}

[Good Example C Input]
state_key: {json.dumps("habits_state:outdoor_cycling", ensure_ascii=False)}
state_value: {json.dumps({
    "schedule": {
        "frequency_type": "weekly",
        "days_of_week": [4, 5],
    },
    "timing": {
        "start_time": "06:30",
        "end_time": "08:00",
    },
    "location": "North Hills trail system",
}, ensure_ascii=False, indent=2)}

[Good Example C Output]
{json.dumps({"item": {
    "scenario": "It is Friday at 06:15. The user has just woken up and is checking their phone.",
    "task_instruction": fixed_task_instruction,
    "reference_answer": (
        "Your weekly ride at the North Hills trail system starts at 06:30 and runs until 08:00. "
        "Since Friday is one of the scheduled days, it is almost time to head out."
    )
}}, ensure_ascii=False, indent=2)}

[Bad Example — Do Not Imitate]
{{
  "item": {{
    "scenario": "It is 07:50, and you've just sat down at your desk for the morning.",
    "task_instruction": {json.dumps(fixed_task_instruction, ensure_ascii=False)},
    "reference_answer": "Your weekly review starts soon."
  }}
}}

Why the bad example fails:
- The scenario is written as if the assistant were inhabiting the user's perspective.
- The current moment is weakly grounded.
- The reference answer is too generic.

[Bad Example 2 — Do Not Imitate]
{{
  "item": {{
    "scenario": "It is 09:25. The morning is quiet and nothing has been started yet.",
    "task_instruction": {json.dumps(fixed_task_instruction, ensure_ascii=False)},
    "reference_answer": "Your weekly client technical briefing is at 10:00 today at the regional corporate headquarters. Since Tuesday is the scheduled day, it is almost time to get ready."
  }}
}}

Why the bad example fails:
- For a weekly or weekday-specific routine, clock time alone does not establish that today is the scheduled day.
- A better scenario would include the current weekday and current clock time, without naming the routine, cadence, location, or stored start time as an appointment fact.

[Input]
- state_key: {state_key}
- state_value: {state_value}

[Output JSON ONLY]
{{
  "item": {{
    "scenario": "...",
    "task_instruction": {json.dumps(fixed_task_instruction, ensure_ascii=False)},
    "reference_answer": "..."
  }}
}}
"""
def _build_task_c_v2_information_request_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_type: str,
    state_key: str,
    state_value: Any,
) -> str:
    del checkpoint_timestamp
    if str(state_type or "").strip() != "preference":
        raise ValueError("Task C v2 information_request_construction prompt requires state_type='preference'.")

    fixed_task_instruction = TASK_C_V2_INFORMATION_REQUEST_TASK_INSTRUCTION
    fixed_task_instruction_json = json.dumps(fixed_task_instruction, ensure_ascii=False)

    return f"""[Task]
Generate exactly one preference-conditioned search-filter task for an assistant helping the user browse, search, compare, or plan options.

Each item contains:
- `scenario`: synthesize this field.
- `task_instruction`: copy the fixed string exactly.
- `output_template`: synthesize this field.
- `reference_output`: synthesize this field as the intended correct filled object.

The task should require the assistant to use the user's preference state to fill search/filter fields, not only the scenario.
The reference_output must fill a structured search/filter object for the assistant to apply before showing matching options.
It must not be a final recommendation, a ranked list, or a free-form explanation.

[Key design goal]
This is a search-filter task, not a copy-the-statement task.
The generated item should require the answering assistant to translate the user's preference statement into semantically meaningful filters.

[Design Principle]
Keep `scenario` leakage-safe:
- `task_instruction` is fixed and must be copied exactly.
- `scenario` may set up the current product or service moment and local completion task,
- but it must not restate, paraphrase, or strongly imply the user-state facts that should instead be recovered from `state_value`.

You may synthesize request-facing keys.

That means:
- output_template should use synthesized, request-facing keys;
- reference_output should be one coherent canonical fill of that template.

[Definitions]
- preference statement: the value in state_value.statement.
- world-background scenario: a short third-person or neutral description of what is happening right now in the product or assistant context. It is not spoken by the assistant, not spoken by the user, and not written from the user's point of view.
- canonical reference_output: one valid answer, but not necessarily the only valid answer.
- fill leaf: one `"<fill>"` slot in output_template and the corresponding filled value in reference_output.
- reference_anchors: audit notes tying each fill leaf to the state/reference basis used to create it. Anchors are for traceability, not for scoring.
- core fill: a fill leaf whose value captures the field-local core preference needed for this service object. Core is leaf/field-level, not state-level.
- detail fill: an optional second fill leaf that adds grounded precision, qualification, or exclusion for the same service object. It must be useful for the service task, not filler.
{SCHEDULE_DATE_ENCODING_BULLETS}

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "output_template"
   - "reference_output"
   - "reference_anchors"
4. Copy task_instruction exactly as this fixed string:
   {fixed_task_instruction}
5. scenario must be short, natural, and written as world background.
6. scenario must not use first-person or second-person wording such as "I", "we", "you", "your", or "you've".
7. scenario must make the filtering situation feel like a plausible user product moment, not like a backend log line.
8. Prefer natural situations such as:
   - the user is browsing options in an app,
   - the user is searching for options,
   - the user is comparing choices,
   - the assistant is setting filters before showing matches.
9. Avoid robotic phrasing such as:
   - "a filtering step is about to run"
   - "a screening request is about to be sent"
   - "a downstream module will execute now"
   - "a shortlist is being prepared before anything is shown"
10. scenario may include only:
   - the immediate user goal or option space,
   - the fact that the assistant is setting search/filter fields before showing matches,
   - at most one additional situational fact that plausibly matters right now.
11. scenario must not restate or paraphrase the user's actual preference content.
12. output_template and reference_output must both be top-level JSON objects.
13. output_template and reference_output must have exactly the same nested shape.
14. Every leaf in output_template must be the string "<fill>".
15. output_template must contain one or two fill leaves total.
16. At least one fill leaf must be a core fill for this item. A second fill leaf may be a detail fill when it is directly grounded and service-useful.
17. reference_anchors must contain exactly one object for each fill leaf and no extra objects.
18. Each reference_anchors object must include:
   - "target_path": dot path to the reference_output leaf
   - "role": either "core" or "detail"
   - "state_reference": the state_value field or statement phrase that grounds the filled value
   - "anchor_note": short explanation of why this fill is grounded
19. Do not use a fixed universal key like only "preference_statement". Instead, synthesize request-facing keys and grouping that fit the domain implied by the preference statement.
20. The synthesized schema should decompose the preference into meaningful filtering dimensions when appropriate, such as:
   - preferred types or formats,
   - desired attributes,
   - required features,
   - avoided or deprioritized options,
   - priorities or goals.
21. reference_output must be a coherent canonical fill of output_template.
22. reference_output is the intended structured gold answer; every filled value must be supported by state_value.
23. If state_value contains schedule-like weekday/month fields, interpret them using the schedule date encodings above.
24. Before finalizing, silently confirm the item would later pass the same five semantic validation criteria:
   - answerability: scenario plus the fixed task_instruction define one clear current-moment structured completion task.
   - service_realism: the item describes a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state.
   - full_field_dependency: every fill leaf must be necessary for the service task and must require the preference statement in state_value, not only the scenario. At least one fill leaf must capture the field-local core preference; any detail leaf must add grounded service-relevant precision.
   - low_leakage: scenario does not restate or strongly imply the preference facts that should come from state_value.
   - output_groundedness: output_template plus reference_output define a task-appropriate filtering object grounded in state_value rather than a raw state copy or unsupported content.

[Good Example A Input]
state_key: {json.dumps("preferences_state:learning_modality", ensure_ascii=False)}
state_value: {json.dumps({
    "statement": "Prefers in-depth, self-paced technical white papers and webinars over large live conferences"
}, ensure_ascii=False, indent=2)}

[Good Example A Output]
{json.dumps({"item": {
    "scenario": (
        "The user is browsing professional-development resources in a learning portal. "
        "The assistant is setting search filters before showing matching options."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "content_search_filters": {
            "resource_formats": "<fill>",
            "avoid_setting": "<fill>"
        }
    },
    "reference_output": {
        "content_search_filters": {
            "resource_formats": "in-depth, self-paced technical white papers or webinars",
            "avoid_setting": "large live conferences"
        }
    },
    "reference_anchors": [
        {
            "target_path": "content_search_filters.resource_formats",
            "role": "core",
            "state_reference": "statement: in-depth, self-paced technical white papers and webinars over large live conferences",
            "anchor_note": "This fill captures the field-local core learning-resource preference."
        },
        {
            "target_path": "content_search_filters.avoid_setting",
            "role": "detail",
            "state_reference": "statement: over large live conferences",
            "anchor_note": "This detail fill records the grounded exclusion needed for filtering."
        }
    ],
}}, ensure_ascii=False, indent=2)}

[Good Example B Input]
state_key: {json.dumps("preferences_state:capital_allocation", ensure_ascii=False)}
state_value: {json.dumps({
    "statement": "Prefers long-term capital preservation and tax-efficient growth over high-risk speculative trading"
}, ensure_ascii=False, indent=2)}

[Good Example B Output]
{json.dumps({"item": {
    "scenario": (
        "The user is reviewing investment options for an upcoming planning session. "
        "The assistant is setting strategy filters before showing matching options."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "investment_filters": {
            "strategy_goal": "<fill>",
            "avoid_strategy": "<fill>"
        }
    },
    "reference_output": {
        "investment_filters": {
            "strategy_goal": "long-term capital preservation and tax-efficient growth",
            "avoid_strategy": "high-risk speculative trading"
        }
    },
    "reference_anchors": [
        {
            "target_path": "investment_filters.strategy_goal",
            "role": "core",
            "state_reference": "statement: long-term capital preservation and tax-efficient growth over high-risk speculative trading",
            "anchor_note": "This fill captures the field-local core investment strategy preference."
        },
        {
            "target_path": "investment_filters.avoid_strategy",
            "role": "detail",
            "state_reference": "statement: over high-risk speculative trading",
            "anchor_note": "This detail fill adds the grounded strategy exclusion."
        }
    ],
}}, ensure_ascii=False, indent=2)}

[Good Example C Input]
state_key: {json.dumps("preferences_state:coffee_shop_style", ensure_ascii=False)}
state_value: {json.dumps({
    "statement": "Prefers quiet neighborhood coffee shops with table seating over loud chain cafes"
}, ensure_ascii=False, indent=2)}

[Good Example C Output]
{json.dumps({"item": {
    "scenario": (
        "The user is choosing a place for a casual conversation later today. "
        "The assistant is setting venue filters before showing nearby options."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "venue_filters": {
            "venue_match": "<fill>"
        }
    },
    "reference_output": {
        "venue_filters": {
            "venue_match": "quiet neighborhood coffee shop with table seating rather than loud chain cafes"
        }
    },
    "reference_anchors": [
        {
            "target_path": "venue_filters.venue_match",
            "role": "core",
            "state_reference": "statement: quiet neighborhood coffee shops with table seating over loud chain cafes",
            "anchor_note": "This fill captures the field-local core venue preference."
        }
    ],
}}, ensure_ascii=False, indent=2)}

[Bad Example — Do Not Imitate]
{{
  "item": {{
    "scenario": "The user prefers quiet neighborhood coffee shops over loud chain cafes, and a shortlist is being prepared.",
    "task_instruction": {fixed_task_instruction_json},
    "output_template": {{
      "filtering_params": {{
        "preference_statement": "<fill>"
      }}
    }},
    "reference_output": {{
      "filtering_params": {{
        "preference_statement": "Prefers quiet neighborhood coffee shops with table seating over loud chain cafes"
      }}
    }}
  }}
}}

Why the bad example fails:
- The scenario leaks the user's actual preference.
- The schema is not synthesized.
- The output collapses the entire preference into one copied statement instead of search/filter fields.

[Input]
- state_key: {state_key}
- state_value: {state_value}

[Output JSON ONLY]
{{
  "item": {{
    "scenario": "...",
    "task_instruction": {fixed_task_instruction_json},
    "output_template": {{
      "<synthesized_request_key>": {{
        "<synthesized_filter_key>": "<fill>"
      }}
    }},
    "reference_output": {{
      "<same synthesized shape as output_template>": "..."
    }},
    "reference_anchors": [
      {{
        "target_path": "<reference_output leaf path>",
        "role": "core|detail",
        "state_reference": "<state_value field or statement phrase>",
        "anchor_note": "<why this fill is grounded>"
      }}
    ]
  }}
}}
"""
def _build_task_c_v2_action_configuration_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_type: str,
    state_key: str,
    state_value: Any,
) -> str:
    del checkpoint_timestamp
    if str(state_type or "").strip() != "attribute":
        raise ValueError("Task C v2 action_configuration prompt requires state_type='attribute'.")

    fixed_task_instruction = TASK_C_V2_ACTION_CONFIGURATION_TASK_INSTRUCTION
    fixed_task_instruction_json = json.dumps(fixed_task_instruction, ensure_ascii=False)

    return f"""[Task]
Generate exactly one attribute-conditioned action-configuration task for an assistant helping the user set up, connect, complete, or submit something.

Each item contains:
- `scenario`: synthesize this field.
- `task_instruction`: copy the fixed string exactly.
- `output_template`: synthesize this field.
- `reference_output`: synthesize this field as the intended correct filled object.

The task should require the assistant to use the user's attribute state to fill execution fields, not only the scenario.
The reference_output must fill a structured action-configuration object for a user-facing tool, setup flow, form, or executable service.
It must not be a user-facing message, a retrieval request, or a free-form explanation.

[Key design goal]
This is a setup/form-configuration task, not a copy-the-attribute task.
The generated item should require the answering assistant to translate the user's known attributes into the specific fields needed to complete a user-facing action.

[Design Principle]
Keep `scenario` leakage-safe:
- `task_instruction` is fixed and must be copied exactly.
- `scenario` may set up the current product or service moment and local execution task,
- but it must not restate, paraphrase, or strongly imply the user-state facts that should instead be recovered from `state_value`.

You may synthesize configuration-facing keys.

That means:
- output_template should use synthesized, configuration-facing keys;
- reference_output should be one coherent canonical fill of that template.

[Definitions]
- attribute value: the information contained in state_value.
- world-background scenario: a short third-person or neutral description of what is happening right now in the product or assistant context. It is not spoken by the assistant, not spoken by the user, and not written from the user's point of view.
- canonical reference_output: one valid answer, but not necessarily the only valid answer.
- grounded decomposition: a raw attribute string may be split into multiple configuration leaves only when each resulting leaf is directly supported by the wording of state_value and serves a distinct execution role.
- deterministic auto-fill: every filled value should be determined by state_value and the neutral setup/form context, not by an extra user choice.
- fill leaf: one `"<fill>"` slot in output_template and the corresponding filled value in reference_output.
- reference_anchors: audit notes tying each fill leaf to the state/reference basis used to create it. Anchors are for traceability, not for scoring.
- core fill: a fill leaf whose value captures the field-local core attribute needed for this service object. Core is leaf/field-level, not state-level.
- detail fill: an optional second fill leaf that adds grounded precision, qualification, or execution detail for the same service object. It must be useful for the service task, not filler.
{SCHEDULE_DATE_ENCODING_BULLETS}

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "output_template"
   - "reference_output"
   - "reference_anchors"
4. Copy task_instruction exactly as this fixed string:
   {fixed_task_instruction}
5. scenario must be short, natural, and written as world background.
6. scenario must not use first-person or second-person wording such as "I", "we", "you", "your", or "you've".
7. scenario must make the execution moment feel like a plausible user product moment, not like a backend log line.
8. Prefer natural situations such as:
   - the user is completing checkout,
   - the user is finishing a setup flow,
   - the user is preparing a profile or form before submission,
   - the user is connecting a device or account,
   - the assistant is auto-filling setup/form/configuration fields.
9. Avoid robotic phrasing such as:
   - "an action configuration is about to be sent"
   - "a downstream workflow will execute now"
   - "a payload is being prepared for a module"
   - "a coordinator is processing a dispatch request"
10. scenario may include only:
   - the immediate user goal or action being completed,
   - the fact that the assistant is filling setup, form, or configuration fields,
   - at most one additional situational fact that plausibly matters right now.
11. scenario must not restate or paraphrase the user's actual attribute values.
12. output_template and reference_output must both be top-level JSON objects.
13. output_template and reference_output must have exactly the same nested shape.
14. Every leaf in output_template must be the string "<fill>".
15. output_template must contain one or two fill leaves total.
16. At least one fill leaf must be a core fill for this item. A second fill leaf may be a detail fill when it is directly grounded and service-useful.
17. reference_anchors must contain exactly one object for each fill leaf and no extra objects.
18. Each reference_anchors object must include:
   - "target_path": dot path to the reference_output leaf
   - "role": either "core" or "detail"
   - "state_reference": the state_value field or phrase that grounds the filled value
   - "anchor_note": short explanation of why this fill is grounded
19. Prefer configuration-facing schemas that decompose compound attribute strings into execution-relevant fields when the decomposition is directly supported by state_value.
20. Do not invent facts that are not directly stated in state_value.
21. Use scenarios where the assistant only auto-fills values determined by state_value. Avoid scenarios that require an extra user choice not in state_value, such as choosing a subset, quantity, recipient, priority, destination, or commitment.
22. reference_output must preserve the selected grounded attribute facts needed by the synthesized configuration schema.
23. reference_output is the intended structured gold answer; every filled value must be supported by state_value.
24. For list-valued state_value, preserve source order when the configuration represents per-item entries.
25. If state_value contains schedule-like weekday/month fields, interpret them using the schedule date encodings above.
26. Before finalizing, silently confirm the item would later pass the same five semantic validation criteria:
   - answerability: scenario plus the fixed task_instruction define one clear current-moment structured completion task.
   - service_realism: the item describes a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state.
   - full_field_dependency: every fill leaf must be necessary for the service task and must require the attribute value in state_value, not only the scenario. At least one fill leaf must capture the field-local core attribute; any detail leaf must add grounded service-relevant precision.
   - low_leakage: scenario does not restate or strongly imply the attribute facts that should come from state_value.
   - output_groundedness: output_template plus reference_output define a task-appropriate action-configuration object grounded in state_value rather than a raw state copy or unsupported content.

[Good Example A Input]
state_key: {json.dumps("user_attributes_state:primary_job_role", ensure_ascii=False)}
state_value: {json.dumps("Senior Coatings Consultant at PPG Industries (specializing in heavy-duty infrastructure and marine protection)", ensure_ascii=False, indent=2)}

[Good Example A Output]
{json.dumps({"item": {
    "scenario": (
        "The user is completing registration for a technical industry symposium. "
        "The assistant is filling the professional credential fields before submission."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "symposium_registration": {
            "professional_profile": {
                "role_title": "<fill>",
                "specialization": "<fill>"
            }
        }
    },
    "reference_output": {
        "symposium_registration": {
            "professional_profile": {
                "role_title": "Senior Coatings Consultant at PPG Industries",
                "specialization": "heavy-duty infrastructure and marine protection"
            }
        }
    },
    "reference_anchors": [
        {
            "target_path": "symposium_registration.professional_profile.role_title",
            "role": "core",
            "state_reference": "Senior Coatings Consultant at PPG Industries (specializing in heavy-duty infrastructure and marine protection)",
            "anchor_note": "This fill captures the field-local core professional identity."
        },
        {
            "target_path": "symposium_registration.professional_profile.specialization",
            "role": "detail",
            "state_reference": "specializing in heavy-duty infrastructure and marine protection",
            "anchor_note": "This detail fill adds the grounded specialization needed by the credential fields."
        }
    ],
}}, ensure_ascii=False, indent=2)}

[Good Example B Input]
state_key: {json.dumps("user_attributes_state:fitness_technology", ensure_ascii=False)}
state_value: {json.dumps([
    "Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)",
    "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)"
], ensure_ascii=False, indent=2)}

[Good Example B Output]
{json.dumps({"item": {
    "scenario": (
        "The user is setting up a wellness app. "
        "The assistant is filling the connected-device sync settings before health data syncing starts."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "wearable_sync_setup": {
            "connected_health_sources": "<fill>"
        }
    },
    "reference_output": {
        "wearable_sync_setup": {
            "connected_health_sources": "Apple Watch Series 9 for daily heart rate and step tracking; Oura Ring Gen3 for sleep staging and recovery metrics"
        }
    },
    "reference_anchors": [
        {
            "target_path": "wearable_sync_setup.connected_health_sources",
            "role": "core",
            "state_reference": "Apple Watch Series 9 ... daily heart rate and step tracking; Oura Ring Gen3 ... sleep staging and recovery metrics",
            "anchor_note": "This fill captures the field-local core wearable sync sources."
        }
    ],
}}, ensure_ascii=False, indent=2)}

[Good Example C Input]
state_key: {json.dumps("user_attributes_state:digital_subscriptions", ensure_ascii=False)}
state_value: {json.dumps([
    "Audible Premium Plus (used for listening to non-fiction during 45-minute commutes)",
    "Disney Bundle including Hulu and ESPN+ (family entertainment and sports coverage)",
    "MasterClass (annual subscription used for learning technical crafting and cooking skills)"
], ensure_ascii=False, indent=2)}

[Good Example C Output]
{json.dumps({"item": {
    "scenario": (
        "The user is connecting subscriptions in a content and services hub. "
        "The assistant is filling linked-service fields before the services are shown."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "subscription_entitlements": {
            "linked_services": "<fill>",
            "usage_context": "<fill>"
        }
    },
    "reference_output": {
        "subscription_entitlements": {
            "linked_services": "Audible Premium Plus; Disney Bundle including Hulu and ESPN+; MasterClass annual subscription",
            "usage_context": "non-fiction during 45-minute commutes, family entertainment and sports coverage, and learning technical crafting and cooking skills"
        }
    },
    "reference_anchors": [
        {
            "target_path": "subscription_entitlements.linked_services",
            "role": "core",
            "state_reference": "Audible Premium Plus; Disney Bundle including Hulu and ESPN+; MasterClass annual subscription",
            "anchor_note": "This fill captures the field-local core subscriptions to link."
        },
        {
            "target_path": "subscription_entitlements.usage_context",
            "role": "detail",
            "state_reference": "non-fiction during 45-minute commutes; family entertainment and sports coverage; learning technical crafting and cooking skills",
            "anchor_note": "This detail fill adds grounded usage context for the linked-service setup."
        }
    ],
}}, ensure_ascii=False, indent=2)}

[Bad Example — Do Not Imitate]
{{
  "item": {{
    "scenario": "The user has an Apple Watch Series 9 and an Oura Ring Gen3, so the sync setup is being prepared.",
    "task_instruction": "Write the best setup recommendation for the user right now.",
    "output_template": {{
      "devices": ["<fill>", "<fill>"]
    }},
    "reference_output": {{
      "devices": [
        "Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)",
        "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)"
      ]
    }}
  }}
}}

Why the bad example fails:
- The scenario leaks the attribute values.
- The task_instruction asks for a free-form recommendation instead of a structured action configuration.
- The schema fails to decompose execution-relevant parts of the attribute strings into meaningful configuration fields.

[Bad Example — Do Not Imitate]
{{
  "item": {{
    "scenario": "A logistics coordinator is processing a freight dispatch request for a regional sports event. The system needs standard ground priority.",
    "task_instruction": {fixed_task_instruction_json},
    "output_template": {{
      "dispatch_request": {{
        "item_name": "<fill>",
        "priority_level": "<fill>"
      }}
    }},
    "reference_output": {{
      "dispatch_request": {{
        "item_name": "YETI Trailhead Camp Chairs",
        "priority_level": "standard_ground"
      }}
    }}
  }}
}}

Why the bad example fails:
- The scenario invents a coordinator workflow instead of a natural user setup/form action.
- The scenario leaks or supplies an operational value that should not come from the user's attribute.
- The reference_output includes an unsupported dispatch priority.

[Bad Example — Do Not Imitate]
{{
  "item": {{
    "scenario": "The user is registering gear for a local youth league's equipment drive. The assistant is filling the donation details before the form is submitted.",
    "task_instruction": {fixed_task_instruction_json},
    "output_template": {{
      "equipment_donation": {{
        "quantity_to_donate": "<fill>",
        "items": ["<fill>", "<fill>"]
      }}
    }},
    "reference_output": {{
      "equipment_donation": {{
        "quantity_to_donate": 10,
        "items": ["practice soccer balls", "cones"]
      }}
    }}
  }}
}}

Why the bad example fails:
- state_value may say the user has soccer gear, but it does not determine what subset or quantity the user wants to donate.
- The correct donation fields depend on an extra user choice, so this is not a deterministic auto-fill task.

[Input]
- state_key: {state_key}
- state_value: {state_value}

[Output JSON ONLY]
{{
  "item": {{
    "scenario": "...",
    "task_instruction": {fixed_task_instruction_json},
    "output_template": {{
      "<synthesized_configuration_key>": {{
        "<synthesized_execution_field>": "<fill>"
      }}
    }},
    "reference_output": {{
      "<same synthesized shape as output_template>": "..."
    }},
    "reference_anchors": [
      {{
        "target_path": "<reference_output leaf path>",
        "role": "core|detail",
        "state_reference": "<state_value field or phrase>",
        "anchor_note": "<why this fill is grounded>"
      }}
    ]
  }}
}}
"""

def build_task_c_v2_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_key: str,
    state_value: Any,
    service_family: str,
) -> str:
    state_type = _infer_apply_state_type(state_key)
    normalized_family = str(service_family or "").strip()
    if normalized_family == "user_communication":
        return _build_task_c_v2_user_communication_question_pack_prompt(
            checkpoint_timestamp=checkpoint_timestamp,
            state_type=state_type,
            state_key=state_key,
            state_value=state_value,
        )
    if normalized_family == "information_request_construction":
        return _build_task_c_v2_information_request_question_pack_prompt(
            checkpoint_timestamp=checkpoint_timestamp,
            state_type=state_type,
            state_key=state_key,
            state_value=_task_c_v2_contract_preference_state_input(state_value),
        )
    if normalized_family == "action_configuration":
        return _build_task_c_v2_action_configuration_question_pack_prompt(
            checkpoint_timestamp=checkpoint_timestamp,
            state_type=state_type,
            state_key=state_key,
            state_value=state_value,
        )
    raise ValueError(f"Unsupported Task C v2 service_family: {normalized_family}")


def _build_task_c_runtime_prompt(
    *,
    memory_section: str,
    response_mode: str,
    task_body: str,
    output_template: Any = None,
) -> str:
    normalized_mode = str(response_mode or "").strip().lower()
    if normalized_mode not in {"text", "structured"}:
        raise ValueError("Unsupported Task C response_mode: {}".format(response_mode))
    task_body = str(task_body or "").strip()
    if normalized_mode == "structured":
        instructions = """[Instructions]
- Use [Assistant Task] and the system-maintained memory of the user's trajectory in [Memory]...[/Memory] to complete the assistant task.
- Put the completed task result in `answer`.
- Put the memory evidence supporting that result in `evidence`.
- `evidence` must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only; do not write any text outside the JSON object.

[Output format]
Output JSON ONLY:
{{
  "answer": {output_template},
  "evidence": [
    {{
      "app_log_id": "<app_log_id>",
      "evidence_content": "<supporting snippet from the same log>"
    }}
  ]
}}""".format(
            output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        )
    else:
        instructions = """[Instructions]
- Use [Assistant Task] and the system-maintained memory of the user's trajectory in [Memory]...[/Memory] to complete the assistant task.
- Put the completed task result in `answer`.
- Put the memory evidence supporting that result in `evidence`.
- `evidence` must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only; do not write any text outside the JSON object.

[Output format]
Output JSON ONLY:
{{
  "answer": "<specific and complete assistant message>",
  "evidence": [
    {{
      "app_log_id": "<app_log_id>",
      "evidence_content": "<supporting snippet from the same log>"
    }}
  ]
}}"""
    return "{instructions}\n\n[Assistant Task]\n{task_body}\n[/Assistant Task]\n\n{memory_section}\n".format(
        instructions=instructions,
        task_body=task_body,
        memory_section=memory_section,
    )


def _build_structured_service_completion_prompt(
    *,
    memory_section: str,
    scenario: str,
    task_instruction: str,
    output_template: Any,
) -> str:
    return _build_task_c_runtime_prompt(
        memory_section=memory_section,
        response_mode="structured",
        task_body=build_task_c_task_body(
            scenario=scenario,
            task_instruction=task_instruction,
            output_template=output_template,
        ),
        output_template=output_template,
    )


def _build_task_c_v2_user_communication_answer_prompt(
    *,
    memory_section: str,
    scenario: str,
    task_instruction: str,
) -> str:
    return _build_task_c_runtime_prompt(
        memory_section=memory_section,
        response_mode="text",
        task_body=build_task_c_task_body(
            scenario=scenario,
            task_instruction=task_instruction,
        ),
    )


def build_task_c_prompt_with_inline_memory(
    *,
    response_mode: str,
    task_body: str,
    output_template: Any = None,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    return _build_task_c_runtime_prompt(
        memory_section=_render_inline_memory_section(
            inline_memory_blocks=_coerce_inline_memory_blocks(
                inline_memory_blocks=inline_memory_blocks,
                context_logs=context_logs,
                log_to_text=log_to_text,
            )
        ),
        response_mode=response_mode,
        task_body=task_body,
        output_template=output_template,
    )


def build_task_c_prompt_with_agent_memory(
    *,
    response_mode: str,
    task_body: str,
    output_template: Any = None,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    del context_logs, log_to_text, inline_memory_blocks
    return _build_task_c_runtime_prompt(
        memory_section=_render_agent_memory_section(),
        response_mode=response_mode,
        task_body=task_body,
        output_template=output_template,
    )


def build_structured_service_completion_prompt_with_inline_memory(
    *,
    scenario: str,
    task_instruction: str,
    output_template: Any,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    return _build_structured_service_completion_prompt(
        memory_section=_render_inline_memory_section(
            inline_memory_blocks=_coerce_inline_memory_blocks(
                inline_memory_blocks=inline_memory_blocks,
                context_logs=context_logs,
                log_to_text=log_to_text,
            )
        ),
        scenario=scenario,
        task_instruction=task_instruction,
        output_template=output_template,
    )


def build_structured_service_completion_prompt_with_agent_memory(
    *,
    scenario: str,
    task_instruction: str,
    output_template: Any,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    del context_logs, log_to_text, inline_memory_blocks
    return _build_structured_service_completion_prompt(
        memory_section=_render_agent_memory_section(),
        scenario=scenario,
        task_instruction=task_instruction,
        output_template=output_template,
    )


def build_task_c_v2_user_communication_prompt_with_inline_memory(
    *,
    scenario: str,
    task_instruction: str,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    return _build_task_c_v2_user_communication_answer_prompt(
        memory_section=_render_inline_memory_section(
            inline_memory_blocks=_coerce_inline_memory_blocks(
                inline_memory_blocks=inline_memory_blocks,
                context_logs=context_logs,
                log_to_text=log_to_text,
            )
        ),
        scenario=scenario,
        task_instruction=task_instruction,
    )


def build_task_c_v2_user_communication_prompt_with_agent_memory(
    *,
    scenario: str,
    task_instruction: str,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    del context_logs, log_to_text, inline_memory_blocks
    return _build_task_c_v2_user_communication_answer_prompt(
        memory_section=_render_agent_memory_section(),
        scenario=scenario,
        task_instruction=task_instruction,
    )


def build_rq3_apply_validation_prompt(
    *,
    state_key: str,
    state_value: Any,
    service_category: str = "",
    question: str = "",
    reference_answer: str = "",
    apply_scenario: str = "",
    apply_question: str = "",
    apply_reference_answer: str = "",
) -> str:
    question = str(question or apply_question or "").strip()
    reference_answer = str(reference_answer or apply_reference_answer or "").strip()
    service_category = str(service_category or "").strip()
    criteria_template = [
        {
            "criterion": "personalization_necessity",
            "analysis": "<why the answer does or does not materially depend on the user state>",
            "pass": "<bool>",
        },
        {
            "criterion": "service_decision_quality",
            "analysis": "<whether the item asks for a real assistant/service decision rather than recall, lookup, or an explicit option menu>",
            "pass": "<bool>",
        },
        {
            "criterion": "answer_groundedness",
            "analysis": "<whether the reference answer stays within information explicit in the state_value or the question>",
            "pass": "<bool>",
        },
    ]
    return """[Task Instruction]
Validate whether this service-decision item is strong.
Judge it using the three required criteria and give one short analysis for each.
Do not output an overall verdict; only return the per-criterion judgments.

[Definitions]
- personalization_necessity: The correct answer must materially depend on the user's state, not merely on generic service logic or scenario-only constraints.
- service_decision_quality: The item must ask for a real assistant/service decision rather than a fact lookup, pure classification, raw state recall, or an explicit A/B menu. For habit states, schedule-grounded service execution, conflict handling, batching, silencing, defer/escalate, or routine-protection questions can still pass when they ask what the assistant should do around the routine rather than merely asking what time or date the routine occurs.
- answer_groundedness: The reference answer must stay within information explicit in the `state_value` or the `question`, and be specific enough that a point-based rubric can score predictions stably.
- grounded reference answer: a reference answer that uses only information explicit in the `state_value` or the `question`, without adding new concrete facts, thresholds, product specs, or background knowledge.
- candidate QA: the proposed service category, question, and reference answer to judge.

[Constraints]
1. Evaluate exactly the three required criteria in this fixed order: personalization_necessity, service_decision_quality, answer_groundedness.
2. Output exactly one object for each required criterion.
3. Use the criterion names exactly as given; do not rename, reorder, omit, or add criteria.
4. Set `pass` to true only when the criterion is clearly satisfied.
5. Keep each `analysis` concise but specific, and explain why the criterion passes or fails.
6. Mark `service_decision_quality` as failed if the item mainly collapses to raw state recall, event-name lookup, timestamp/date lookup, one-step arithmetic with no real assistant-action layer, or an explicit A/B or named-option menu instead of a natural assistant service request.
7. For habit states, allow schedule-grounded service operations such as silence, batch, defer, protect, or flag conflicts when the question is asking what the assistant should do.
8. Mark `personalization_necessity` as failed if the scenario itself injects the decisive rule and the user state becomes unnecessary.
9. Mark `answer_groundedness` as failed if the answer is too vague for stable scoring or introduces new concrete facts, thresholds, tactics, metrics, product specs, or background knowledge that are not explicit in `state_value` or `question`.
10. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "preferences_state:learning_modality"
state_value: {{"statement": "Prefers in-depth, self-paced technical white papers and webinars over large conferences"}}
candidate_qa: {{
  "service_category": "content recommendation",
  "question": "What professional development resource should be recommended for the user this quarter?",
  "reference_answer": "Recommend an in-depth self-paced bundle of technical white papers and on-demand webinars."
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "personalization_necessity",
      "analysis": "The best action depends on the user's preference for self-paced, in-depth learning rather than generic quality alone.",
      "pass": true
    }},
    {{
      "criterion": "service_decision_quality",
      "analysis": "The item asks the assistant to make one concrete learning-resource decision rather than perform recall or choose from an explicit menu.",
      "pass": true
    }},
    {{
      "criterion": "answer_groundedness",
      "analysis": "The reference answer names one bounded service action without importing any new details beyond the state and question.",
      "pass": true
    }}
  ]
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- candidate QA: object with `service_category`, `question`, `reference_answer`

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- candidate_qa: {{
    "service_category": {service_category},
    "question": {question},
    "reference_answer": {reference_answer}
  }}

Output JSON ONLY:
{{
  "criteria": [
    {{
      "criterion": "<one required criterion name>",
      "analysis": "<why it passes or fails>",
      "pass": "<bool>"
    }}
  ]
}}

[Schema Template]
{criteria_template}
""".format(
        state_key=state_key,
        state_value=json.dumps(state_value, ensure_ascii=False),
        service_category=json.dumps(service_category, ensure_ascii=False),
        question=json.dumps(question, ensure_ascii=False),
        reference_answer=json.dumps(reference_answer, ensure_ascii=False),
        criteria_template=json.dumps(criteria_template, ensure_ascii=False, indent=2),
    )


def build_task_c_v2_validation_prompt(
    *,
    state_key: str,
    state_value: Any,
    service_family: str,
    scenario: str,
    task_instruction: str,
    output_template: Any,
    reference_output: Any,
    reference_anchors: Any = None,
    reference_answer: str = "",
) -> str:
    normalized_family = str(service_family or "").strip()
    if normalized_family == "user_communication":
        criteria_template = [
            {
                "criterion": "answerability",
                "analysis": "<whether scenario plus task_instruction define one clear current-moment communication task that can be answered from the provided state>",
                "pass": "<bool>",
            },
            {
                "criterion": "service_realism",
                "analysis": "<whether the item describes a realistic assistant-mediated service action that a user could naturally be doing now>",
                "pass": "<bool>",
            },
            {
                "criterion": "full_field_dependency",
                "analysis": "<whether answering well requires all non-derived fields in the provided state>",
                "pass": "<bool>",
            },
            {
                "criterion": "low_leakage",
                "analysis": "<whether item-specific scenario wording avoids restating personalized habit facts while allowing current-world anchors>",
                "pass": "<bool>",
            },
            {
                "criterion": "output_groundedness",
                "analysis": "<whether each personalized part of reference_answer is grounded by the relevant fields in the provided state or by the state_key routine label>",
                "pass": "<bool>",
            },
        ]
        return """[Task Instruction]
Validate whether this item defines a realistic reminder-message task:
the assistant should draft a reminder message for the user in the given scenario,
and the correct message should depend on the provided routine information rather than on the scenario alone.
Judge it using the five required criteria and give one short analysis for each.
Do not output an overall verdict; only return the per-criterion judgments.

[Definitions]
- answerability: `scenario` plus `task_instruction` must define one clear communication task for the current moment. Judge this by checking whether the current moment is anchored well enough and whether the assistant can tell what kind of message should be sent now. For day/date-specific routines, a clock time alone is not enough; the scenario needs the current weekday or calendar anchor.
- service_realism: the item should describe a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state. For habits, judge whether the reminder/message would be useful and natural at this moment, not merely a recall of stored routine data.
- full_field_dependency: answering well should require all non-derived fields in `state_value`; dropping an important field path should make the communication materially weaker or incorrect. Judge this by checking which field paths in `state_value` are actually needed by the ideal message. For weekly habits, `frequency_type: "weekly"` is a meaningful cadence field when the ideal message says the routine is weekly; do not fail merely because `days_of_week` also identifies the current day.
- low_leakage: item-specific `scenario` wording should describe only the local situation and current communication task; it must not restate or strongly imply personalized habit facts that should come from `state_value`. Current weekday/date/time are allowed as world-background anchors and are not leakage by themselves. The fixed generic `task_instruction` may mention using the user's routine details; do not count that generic wording as item leakage.
- output_groundedness: `reference_answer` must be a short natural-language assistant response whose personalized content is grounded in `state_value` or the human-readable routine label in `state_key`. Judge this by checking which parts of `reference_answer` are supported by state fields or the state_key suffix, and fail if it adds unsupported user-specific facts.
{schedule_date_encoding_bullets}

[Constraints]
1. Evaluate exactly the five required criteria in this fixed order: answerability, service_realism, full_field_dependency, low_leakage, output_groundedness.
2. Output exactly one object for each required criterion.
3. Use the criterion names exactly as given; do not rename, reorder, omit, or add criteria.
4. Set `pass` to true only when the criterion is clearly satisfied.
5. Keep each `analysis` concise but specific.
6. Mark `answerability` as failed if the current-moment task is vague, underspecified, unclear about what the assistant should send now, or gives only a clock time when `state_value` requires a scheduled weekday/date/nth-weekday.
7. Mark `service_realism` as failed if the item feels like a backend placeholder, arbitrary workflow, contrived form, a task invented only to expose the state, raw state recall, or a generic check-in rather than a natural assistant-mediated service action.
8. Mark `full_field_dependency` as failed if one or more important field paths in `state_value` are unused, optionalized, or collapsible without materially changing the ideal message. Do not fail weekly `frequency_type` just because the current weekday is also present; a weekly cadence claim is still a state-dependent requirement.
9. Mark `low_leakage` as failed if item-specific `scenario` wording restates or paraphrases the habit action/identity, cadence, stored start/end time, location, modality, or priority. Do not fail low_leakage merely because scenario states the current weekday, calendar date, or current clock time, and do not fail merely because the fixed generic task_instruction says to use the user's routine details.
10. Mark `answerability` as failed if `scenario` uses a weekday/date anchor that conflicts with encoded schedule fields in `state_value`.
11. Mark `output_groundedness` as failed if `reference_answer` introduces unsupported tactics, thresholds, user-specific facts absent from `state_value` and `state_key`, or a weekday/date claim that conflicts with encoded schedule fields in `state_value`.
12. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "habits_state:client_technical_briefing"
state_value: {{"schedule": {{"frequency_type": "weekly", "days_of_week": [0]}}, "timing": {{"start_time": "10:00"}}, "location": "regional corporate headquarters"}}
candidate_item: {{
  "scenario": "It is Monday at 09:20. Nothing has been started yet this morning.",
  "task_instruction": "Draft a specific reminder message for the user in this scenario.",
  "reference_answer": "Your weekly client technical briefing is at 10:00 today at the regional corporate headquarters. Since Monday is the scheduled day, it is almost time to get ready."
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "answerability",
      "analysis": "The current moment is anchored clearly enough that one reminder message can be written now.",
      "pass": true
    }},
    {{
      "criterion": "service_realism",
      "analysis": "The item asks for one concrete assistant reminder message rather than raw recall.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "The weekly cadence, scheduled day, start time, and location all matter for a fully correct reminder.",
      "pass": true
    }},
    {{
      "criterion": "low_leakage",
      "analysis": "The scenario gives the local current weekday and time without restating the routine identity, cadence, stored start time, or location from the state.",
      "pass": true
    }},
    {{
      "criterion": "output_groundedness",
      "analysis": "The answer stays grounded in the schedule, timing, and location information without adding unsupported user facts.",
      "pass": true
    }}
  ]
}}

[Example 2]
[Example Input]
state_key: "habits_state:family_movie_night"
state_value: {{"schedule": {{"frequency_type": "weekly", "days_of_week": [5]}}, "timing": {{"start_time": "19:30"}}}}
candidate_item: {{
  "scenario": "It is Saturday at 19:15. The living room is currently empty and the television is off.",
  "task_instruction": "Draft a specific reminder message for the user in this scenario.",
  "reference_answer": "It's 19:15 on Saturday, and your weekly family movie night is starting at 19:30. Since the television is still off, would you like to get started?"
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "answerability",
      "analysis": "The current moment is anchored to Saturday at 19:15, matching the scheduled weekday and giving a clear reminder moment.",
      "pass": true
    }},
    {{
      "criterion": "service_realism",
      "analysis": "The item asks for one concrete user-facing message to send now, not a raw recall answer.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "The weekly cadence, Saturday schedule, and 19:30 start time all matter for the ideal reminder.",
      "pass": true
    }},
    {{
      "criterion": "low_leakage",
      "analysis": "The scenario gives only current-world context and does not name the routine, cadence, or stored start time; the fixed generic task instruction is not item-specific leakage.",
      "pass": true
    }},
    {{
      "criterion": "output_groundedness",
      "analysis": "The routine label is grounded by the state_key suffix, and the weekly cadence, Saturday schedule, and 19:30 time are grounded by state_value.",
      "pass": true
    }}
  ]
}}

[Example 3]
[Example Input]
state_key: "habits_state:evening_walk"
state_value: {{"timing": {{"start_time": "19:00"}}, "schedule": {{"days_of_week": [2, 4, 6]}}}}
candidate_item: {{
  "scenario": "It is 18:50. The evening is quiet and nothing has been started.",
  "task_instruction": "Draft a specific reminder message for the user in this scenario.",
  "reference_answer": "Your regular evening walk normally starts at 19:00, so this may be a good time to get ready."
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "answerability",
      "analysis": "The scenario gives only a clock time; because the state has scheduled weekdays, it does not establish whether today is one of the scheduled days.",
      "pass": false
    }},
    {{
      "criterion": "service_realism",
      "analysis": "The task is still framed as writing one concrete assistant reminder message rather than doing recall.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "A good reminder would still need both the time and recurring-day information from the state.",
      "pass": true
    }},
    {{
      "criterion": "low_leakage",
      "analysis": "The scenario stays local and does not restate the actual walk identity, scheduled weekdays, cadence, or stored start time from the state.",
      "pass": true
    }},
    {{
      "criterion": "output_groundedness",
      "analysis": "The reference answer is short and grounded in the provided state without adding unsupported user facts.",
      "pass": true
    }}
  ]
}}

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- candidate_item: {{
    "scenario": {scenario},
    "task_instruction": {task_instruction},
    "reference_answer": {reference_answer}
  }}

Output JSON ONLY:
{{
  "criteria": [
    {{
      "criterion": "<one required criterion name>",
      "analysis": "<why it passes or fails>",
      "pass": "<bool>"
    }}
  ]
}}
""".format(
            state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
            state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
            scenario=json.dumps(str(scenario or ""), ensure_ascii=False),
            task_instruction=json.dumps(str(task_instruction or ""), ensure_ascii=False),
            reference_answer=json.dumps(str(reference_answer or ""), ensure_ascii=False),
            schedule_date_encoding_bullets=SCHEDULE_DATE_ENCODING_BULLETS,
        )
    if normalized_family == "information_request_construction":
        validation_task_definition = "Validate whether this item defines a realistic search-filter task:\nthe assistant should help set search filters for the user in the given scenario,\nand the correct filled filters should depend on the provided user information rather than on the scenario alone."
        service_definition = "The item should describe a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state. For preferences, the browsing/search/filtering moment should feel natural for the user's option space."
        output_definition = "`output_template` plus `reference_output` must define one task-appropriate search/filter object with one or two filled leaves grounded in `state_value`. Fail if the item merely copies the raw preference statement, mirrors the raw state schema, produces a final recommendation, invents unsupported content, has zero or more than two filled leaves, or lacks one `reference_anchors` object per filled output leaf. `reference_anchors` should identify the state basis and role (`core` or `detail`) for each filled output leaf."
        answerability_fail = "Mark `answerability` as failed if the user browsing/search/comparison moment is vague, backend-like, underspecified, or unclear about which search/filter object should be completed now."
        service_fail = "Mark `service_realism` as failed if the item feels like a backend placeholder, arbitrary workflow, contrived form, task invented only to expose the state, free-form recommendation, raw preference recall, or unexplained shortlist operation."
        output_fail = "Mark `output_groundedness` as failed if the item simply copies the preference statement, mirrors the raw state schema, fails to produce a top-level search/filter object, writes a final recommendation, or invents unsupported preference content."
        positive_example = """[Example]
[Example Input]
state_key: "preferences_state:learning_modality"
state_value: {"statement": "Prefers in-depth, self-paced technical white papers and webinars over large live conferences"}
candidate_item: {
  "scenario": "The user is browsing professional-development resources in a learning portal. The assistant is setting search filters before showing matching options.",
  "task_instruction": "Help the user set the search filters in this scenario.",
  "output_template": {"content_search_filters": {"resource_formats": "<fill>", "avoid_setting": "<fill>"}},
  "reference_output": {"content_search_filters": {"resource_formats": "in-depth, self-paced technical white papers or webinars", "avoid_setting": "large live conferences"}},
  "reference_anchors": [
    {"target_path": "content_search_filters.resource_formats", "role": "core", "state_reference": "in-depth, self-paced technical white papers and webinars", "anchor_note": "field-local core preference used as the main format filter"},
    {"target_path": "content_search_filters.avoid_setting", "role": "detail", "state_reference": "over large live conferences", "anchor_note": "grounded exclusion used as a service-useful detail filter"}
  ]
}

[Example Output]
{
  "criteria": [
    {"criterion": "answerability", "analysis": "The user browsing moment and the assistant's search-filter action are clear enough that one bounded object can be completed now.", "pass": true},
    {"criterion": "service_realism", "analysis": "The item is framed as filling search filters rather than recommending content or recalling the preference.", "pass": true},
    {"criterion": "full_field_dependency", "analysis": "Both filled leaves are necessary for the search-filter task, the core leaf captures the main preference, and the detail leaf adds a grounded exclusion.", "pass": true},
    {"criterion": "low_leakage", "analysis": "The scenario describes the local product moment without restating the user's actual learning preferences.", "pass": true},
    {"criterion": "output_groundedness", "analysis": "The structured output is a synthesized search-filter object rather than a raw state copy, and the filled value is grounded in the preference statement.", "pass": true}
  ]
}"""
        negative_example = """[Bad Example]
[Example Input]
state_key: "preferences_state:coffee_shop_style"
state_value: {"statement": "Prefers quiet neighborhood coffee shops with table seating over loud chain cafes"}
candidate_item: {
  "scenario": "The user prefers quiet neighborhood coffee shops over loud chain cafes, and a shortlist is being prepared.",
  "task_instruction": "Help the user set the search filters in this scenario.",
  "output_template": {"filtering_params": {"preference_statement": "<fill>"}},
  "reference_output": {"filtering_params": {"preference_statement": "Prefers quiet neighborhood coffee shops with table seating over loud chain cafes"}},
  "reference_anchors": [{"target_path": "filtering_params.preference_statement", "role": "core", "state_reference": "full statement", "anchor_note": "copied raw statement"}]
}

[Example Output]
{
  "criteria": [
    {"criterion": "answerability", "analysis": "The scenario mentions a shortlist, but it is less natural than a clear user browsing/search moment.", "pass": false},
    {"criterion": "service_realism", "analysis": "The item is structured, but it collapses toward raw preference transfer rather than meaningful search-filter dimensions.", "pass": false},
    {"criterion": "full_field_dependency", "analysis": "The output has one state-dependent leaf, but it is a raw statement copy rather than a meaningful service fill.", "pass": false},
    {"criterion": "low_leakage", "analysis": "The scenario restates the user's actual coffee-shop preference.", "pass": false},
    {"criterion": "output_groundedness", "analysis": "The object mirrors the raw state statement instead of forming a task-appropriate search-filter object.", "pass": false}
  ]
}"""
    elif normalized_family == "action_configuration":
        validation_task_definition = "Validate whether this item defines a realistic setup-or-form completion task:\nthe assistant should help complete setup or form fields for the user in the given scenario,\nand the correct filled fields should depend on the provided user information rather than on the scenario alone."
        service_definition = "The item should describe a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state. For attributes, the setup/form/configuration moment should feel natural for the known attribute."
        output_definition = "`output_template` plus `reference_output` must define one task-appropriate action-configuration object with one or two filled leaves grounded in `state_value`. Fail if the item merely copies raw attribute strings, mirrors the raw state schema, behaves like filtering/retrieval, invents unsupported content, has zero or more than two filled leaves, lacks one `reference_anchors` object per filled output leaf, or fills values that require an extra user choice not determined by state_value. `reference_anchors` should identify the state basis and role (`core` or `detail`) for each filled output leaf."
        answerability_fail = "Mark `answerability` as failed if the user setup/form/configuration moment is vague, backend-like, underspecified, unclear about which configuration fields should be completed now, or depends on an extra user choice not determined by state_value."
        service_fail = "Mark `service_realism` as failed if the item feels like a backend placeholder, arbitrary workflow, contrived form, task invented only to expose the state, free-form recommendation, raw attribute recall, or coordinator workflow."
        output_fail = "Mark `output_groundedness` as failed if the item simply copies raw attribute strings, mirrors the raw state schema, fails to produce a top-level executable/configuration object, behaves like retrieval/filtering, invents unsupported attribute content, or makes the correct output depend on an extra user choice such as subset, quantity, recipient, priority, destination, or commitment."
        positive_example = """[Example]
[Example Input]
state_key: "user_attributes_state:fitness_technology"
state_value: ["Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)", "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)"]
candidate_item: {
  "scenario": "The user is setting up a wellness app. The assistant is filling the connected-device sync settings before health data syncing starts.",
  "task_instruction": "Help the user complete the setup or form fields in this scenario.",
  "output_template": {"wearable_sync_setup": {"connected_devices": "<fill>", "sync_metrics": "<fill>"}},
  "reference_output": {"wearable_sync_setup": {"connected_devices": "Apple Watch Series 9; Oura Ring Gen3", "sync_metrics": "daily heart rate and step tracking; sleep staging and recovery metrics"}},
  "reference_anchors": [
    {"target_path": "wearable_sync_setup.connected_devices", "role": "core", "state_reference": "Apple Watch Series 9; Oura Ring Gen3", "anchor_note": "field-local core devices used for sync setup"},
    {"target_path": "wearable_sync_setup.sync_metrics", "role": "detail", "state_reference": "daily heart rate and step tracking; sleep staging and recovery metrics", "anchor_note": "grounded metric details needed for the sync configuration"}
  ]
}

[Example Output]
{
  "criteria": [
    {"criterion": "answerability", "analysis": "The user setup moment and wearable-sync fields are clear enough that one bounded configuration can be completed now.", "pass": true},
    {"criterion": "service_realism", "analysis": "The item is framed as completing an executable sync configuration rather than making a recommendation or recalling devices.", "pass": true},
    {"criterion": "full_field_dependency", "analysis": "Both filled leaves are necessary for the sync setup, the core leaf captures the devices, and the detail leaf captures supported metrics.", "pass": true},
    {"criterion": "low_leakage", "analysis": "The scenario describes the setup moment without naming the actual devices or metrics.", "pass": true},
    {"criterion": "output_groundedness", "analysis": "The structured output converts the attribute value into an action-configuration field, and the filled value is grounded in state_value.", "pass": true}
  ]
}"""
        negative_example = """[Bad Example]
[Example Input]
state_key: "user_attributes_state:family_sports_gear"
state_value: "Set of 10 practice soccer balls and cones"
candidate_item: {
  "scenario": "The user is registering gear for a local youth league's equipment drive. The assistant is filling the donation details before the form is submitted.",
  "task_instruction": "Help the user complete the setup or form fields in this scenario.",
  "output_template": {"equipment_donation": {"quantity_to_donate": "<fill>", "items": ["<fill>", "<fill>"]}},
  "reference_output": {"equipment_donation": {"quantity_to_donate": 10, "items": ["practice soccer balls", "cones"]}},
  "reference_anchors": [{"target_path": "equipment_donation.quantity_to_donate", "role": "core", "state_reference": "Set of 10 practice soccer balls and cones", "anchor_note": "incorrectly treats available quantity as donation intent"}]
}

[Example Output]
{
  "criteria": [
    {"criterion": "answerability", "analysis": "The donation form requires choosing what quantity or subset to donate, which is not determined by the state.", "pass": false},
    {"criterion": "service_realism", "analysis": "The item asks the assistant to fill donation choices rather than only auto-fill known setup/profile fields.", "pass": false},
    {"criterion": "full_field_dependency", "analysis": "The output has three filled leaves and includes a donation quantity that requires an extra user choice.", "pass": false},
    {"criterion": "low_leakage", "analysis": "The scenario does not restate the exact gear details.", "pass": true},
    {"criterion": "output_groundedness", "analysis": "The state says what gear exists, but not that all 10 balls and cones should be donated.", "pass": false}
  ]
}"""
    else:
        raise ValueError(f"Unsupported Task C v2 service_family: {normalized_family}")

    return """[Task Instruction]
{validation_task_definition}
Judge it using the five required criteria and give one short analysis for each.
Do not output an overall verdict; only return the per-criterion judgments.

[Definitions]
- answerability: `scenario` plus `task_instruction` must define one clear structured completion task for the current moment. Judge this by checking whether the current user product/service moment is clear and whether the assistant can tell what object should be completed now.
- service_realism: {service_definition}
- full_field_dependency: the item should have one or two filled `reference_output` leaves. Every filled leaf should be necessary for the service task and dependent on `state_value`, not answerable from the scenario alone. At least one filled leaf should be a field-local `core` leaf; a `detail` leaf is valid only when it adds grounded service-relevant precision.
- low_leakage: `scenario` and `task_instruction` should describe only the local product/service situation and current completion task; they must not restate or strongly imply the key user-state facts that should come from `state_value`. Judge this by comparing `scenario` and `task_instruction` against the field paths in `state_value`.
- output_groundedness: {output_definition}
{schedule_date_encoding_bullets}

[Constraints]
1. Evaluate exactly the five required criteria in this fixed order: answerability, service_realism, full_field_dependency, low_leakage, output_groundedness.
2. Output exactly one object for each required criterion.
3. Use the criterion names exactly as given; do not rename, reorder, omit, or add criteria.
4. Set `pass` to true only when the criterion is clearly satisfied.
5. Keep each `analysis` concise but specific.
6. {answerability_fail}
7. {service_fail}
8. Mark `full_field_dependency` as failed if `reference_output` has zero or more than two filled leaves, lacks a field-local core leaf, has missing or extra anchors for its filled leaves, or includes any filled leaf that is optional, scenario-only, unsupported, or not tied to `state_value`.
9. Mark `low_leakage` as failed if `scenario` or `task_instruction` restates or paraphrases the key user-state facts that should come from `state_value`.
10. If state_value contains schedule-like weekday/month fields, apply the schedule date encodings when checking answerability and output groundedness.
11. {output_fail}
12. Output JSON ONLY with no markdown and no extra keys.

{positive_example}

{negative_example}

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- candidate_item: {{
    "scenario": {scenario},
    "task_instruction": {task_instruction},
    "output_template": {output_template},
    "reference_output": {reference_output},
    "reference_anchors": {reference_anchors}
  }}

Output JSON ONLY:
{{
  "criteria": [
    {{
      "criterion": "<one required criterion name>",
      "analysis": "<why it passes or fails>",
      "pass": "<bool>"
    }}
  ]
}}
""".format(
        validation_task_definition=validation_task_definition,
        service_definition=service_definition,
        output_definition=output_definition,
        answerability_fail=answerability_fail,
        service_fail=service_fail,
        output_fail=output_fail,
        positive_example=positive_example,
        negative_example=negative_example,
        schedule_date_encoding_bullets=SCHEDULE_DATE_ENCODING_BULLETS,
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
        scenario=json.dumps(str(scenario or ""), ensure_ascii=False),
        task_instruction=json.dumps(str(task_instruction or ""), ensure_ascii=False),
        output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        reference_output=json.dumps(reference_output, ensure_ascii=False, indent=2),
        reference_anchors=json.dumps(reference_anchors if reference_anchors is not None else [], ensure_ascii=False, indent=2),
    )


def build_state_questionability_validation_prompt(
    *,
    state_key: str,
    state_value: Any,
    askable_fields: List[str],
    evidence_logs: List[Dict[str, Any]],
) -> str:
    normalized_fields = [
        str(path).strip().lower()
        for path in list(askable_fields or [])
        if str(path).strip()
    ]
    output_template = {
        "field_verdicts": {
            path: {
                "reason_analysis": "<why this field's semantic value is or is not inferable from evidence>",
                "is_valid": "<bool>",
            }
            for path in normalized_fields
        }
    }
    return """[Task Instruction]
Validate whether this state is inferable from evidence.
Evaluate field-level inferability.
Success means each candidate field is marked valid when the semantic value represented by that field is supported by evidence.

[Definitions]
- state_key: the state item being validated.
- state_value: the structured target value for this state_key at checkpoint time.
- candidate_field_paths: field paths to evaluate for inferability. The target value for each path is found inside state_value.
- evidence_logs: related app logs up to checkpoint time; these are the primary evidence.
- field_verdicts: an object keyed exactly by candidate field path. Each value is one field-level decision with `{{reason_analysis, is_valid}}`.
- semantic alignment: the evidence supports the same user-specific meaning as the candidate field value. Exact wording, exact labels, or verbatim phrasing are not required.
- structured field: a schedule, date, time, numeric, enum-like, object, or list field. Validate the normalized meaning of the value, not the literal surface form. For example, evidence saying "every Sunday at 9 AM" semantically supports `days_of_week=[6]` and `start_time="09:00"`.
- statement/text field: a preference, attribute, or other natural-language field. Validate whether the core user-specific claims, comparisons, and important qualifiers are explicitly supported or implicitly supported by the evidence. Do not require the target statement to appear verbatim.
- implicit behavioral support: evidence can support a state through the user's behavior, choices, repeated use, complex operations, professional context, or demonstrated expertise even when the user never states the state directly. For example, updating a Microsoft Project timeline, adjusting resource leveling, and revising a critical path can implicitly support advanced Microsoft Project skill for timeline/resource management.
- unsupported central qualifier: a meaning-changing detail in the target value that the evidence does not explicitly or implicitly support, such as ownership, primary/main status, exact purpose/use, or a comparative preference.
{schedule_date_encoding_bullets}

[Constraints]
1. Evaluate each path in `candidate_field_paths` independently.
2. Output exactly one `field_verdicts` entry per candidate field path.
3. Use exactly the field keys shown in the output template; do not add, remove, or rename field keys.
4. For each candidate field, compare the target value in `state_value` against the evidence logs.
5. Set `is_valid=true` when evidence logs explicitly or implicitly support the semantic value of the candidate field.
6. Do not require exact wording, exact labels, repeated evidence, or verbatim target phrasing when one or more clear evidence logs support the field's core user-specific meaning.
7. For structured schedule/date/time/list fields, validate semantic equivalence after interpreting date encodings and normalized forms. Do not fail merely because evidence uses natural language, calendar words, or paraphrased item names.
8. For statement/text fields, allow reasonable summarization and general wording when the core personalized claims are supported explicitly or implicitly.
9. For skill, expertise, preference, or stable-attribute claims, treat demonstrated complex behavior or repeated choices as valid implicit support when the behavior would be unlikely without that state.
10. Fail only when an unsupported central qualifier, comparison, scope, or concrete detail materially changes the field meaning. Weak signals such as viewing a feed item, receiving a newsletter, or a single generic search do not by themselves establish ownership, membership, primary/main status, or strong preference.
11. If evidence is missing, ambiguous, contradicted, or supports only a materially weaker/different claim, set `is_valid=false` and explain the gap in `reason_analysis`.
12. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "habits_state:budget_review"
state_value: {{
  "schedule": {{
    "days_of_week": [6]
  }},
  "timing": {{
    "start_time": "09:00",
    "end_time": "10:00"
  }}
}}
candidate_field_paths: ["schedule.days_of_week", "timing.start_time", "timing.end_time"]
evidence_logs: [
  {{
    "app_log_id": "log_0101",
    "api_name": "scheduleService",
    "response": "Created a recurring Sunday 09:00 reminder for the user's budget review."
  }}
]

[Example Output]
{{
  "field_verdicts": {{
    "schedule.days_of_week": {{
      "reason_analysis": "The evidence says the budget review reminder is recurring on Sunday, which semantically matches days_of_week=[6] under the weekday encoding.",
      "is_valid": true
    }},
    "timing.start_time": {{
      "reason_analysis": "The evidence gives the reminder time as 09:00, matching the semantic time value of start_time.",
      "is_valid": true
    }},
    "timing.end_time": {{
      "reason_analysis": "The evidence does not provide a duration or explicit end time, so 10:00 is not supported.",
      "is_valid": false
    }}
  }}
}}

[Example 2]
[Example Input]
state_key: "preferences_state:community_involvement_type"
state_value: {{
  "statement": "Prefers outcome-oriented civic activities like infrastructure projects over social-only community mixers"
}}
candidate_field_paths: ["statement"]
evidence_logs: [
  {{
    "app_log_id": "log_0201",
    "api_name": "ReplyEmail",
    "request": {{
      "body": "I'm going to pass on the social-only neighborhood mixer. I am very interested in contributing to the drainage and sidewalk infrastructure improvement committee, and I prefer to focus my volunteer time where I can provide technical value to the community."
    }}
  }}
]

[Example Output]
{{
  "field_verdicts": {{
    "statement": {{
      "reason_analysis": "The evidence does not use the exact target wording, but it clearly supports the core preference: practical infrastructure-oriented community work over a social-only mixer.",
      "is_valid": true
    }}
  }}
}}

[Example 3]
[Example Input]
state_key: "user_attributes_state:industry_software_skills"
state_value: "Microsoft Project (advanced level for timeline and resource management)"
candidate_field_paths: ["current_value"]
evidence_logs: [
  {{
    "app_log_id": "log_0301",
    "api_name": "ReplyEmail",
    "request": {{
      "body": "I've updated the Microsoft Project timeline for the 2024 spring bridge coating cycle, adjusted the resource leveling for the Monongahela and Liberty Bridge phases, and revised the critical path to account for crew availability."
    }}
  }}
]

[Example Output]
{{
  "field_verdicts": {{
    "current_value": {{
      "reason_analysis": "The evidence does not say 'advanced level' verbatim, but the user performs complex Microsoft Project operations including timeline updates, resource leveling, and critical-path revision, which implicitly supports advanced use for timeline and resource management.",
      "is_valid": true
    }}
  }}
}}

[Input]
{{
  "state_key": {state_key},
  "state_value": {state_value},
  "candidate_field_paths": {askable_fields},
  "evidence_logs": {evidence_logs}
}}

[Output JSON ONLY]
{output_template}
""".format(
        state_key=state_key,
        state_value=json.dumps(state_value, ensure_ascii=False),
        askable_fields=json.dumps(normalized_fields, ensure_ascii=False),
        evidence_logs=json.dumps(list(evidence_logs or []), ensure_ascii=False),
        output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        schedule_date_encoding_bullets=SCHEDULE_DATE_ENCODING_BULLETS,
    )


def build_change_reason_validation_prompt(
    *,
    state_key: str,
    state_value: Any,
    change_reason: str,
    evidence_logs: List[Dict[str, Any]],
) -> str:
    return """[Task Instruction]
Validate whether the provided change reason is supported by evidence.

[Definitions]
- state_key: the state item whose change reason is being validated.
- state_value: the target value for this state_key at checkpoint time.
- change_reason: the provided reason text for why the state changed.
- evidence_logs: related app logs up to checkpoint time; these are the primary evidence.
- change_reason_verdict: one decision with `{{reason_analysis, is_valid}}`.

[Constraints]
1. Judge only whether `change_reason` is evidence-supported.
2. Compare the claim in `change_reason` against `state_value` and `evidence_logs`.
3. Do not rewrite the change reason.
4. Set `is_valid=true` only when the evidence logs support this change reason with high confidence.
5. If evidence is missing, indirect, ambiguous, contradicted, or supports only a weaker/generalized explanation, set `is_valid=false`.
6. Explain the evidence match or gap in `reason_analysis`.
7. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "habits_state:morning_walk"
state_value: {{
  "timing": {{
    "start_time": "07:00"
  }}
}}
change_reason: "Routine shifted later after the user's morning schedule changed."
evidence_logs: [
  {{
    "app_log_id": "log_0201",
    "api_name": "scheduleService",
    "response": "Updated the morning walk reminder to 07:00 because the user's morning meeting moved later."
  }}
]

[Example Output]
{{
  "change_reason_verdict": {{
    "reason_analysis": "The evidence explicitly says the morning walk reminder moved to 07:00 because the user's morning schedule changed, matching the provided change reason.",
    "is_valid": true
  }}
}}

[Input]
{{
  "state_key": {state_key},
  "state_value": {state_value},
  "change_reason": {change_reason},
  "evidence_logs": {evidence_logs}
}}

[Output JSON ONLY]
{{
  "change_reason_verdict": {{
    "reason_analysis": "<why this change reason is or is not evidence-supported>",
    "is_valid": "<bool>"
  }}
}}
""".format(
        state_key=state_key,
        state_value=json.dumps(state_value, ensure_ascii=False),
        change_reason=json.dumps(str(change_reason or ""), ensure_ascii=False),
        evidence_logs=json.dumps(list(evidence_logs or []), ensure_ascii=False),
    )


def build_rq3_apply_rewrite_prompt(
    *,
    state_key: str,
    state_value: Any,
    service_category: str = "",
    question: str = "",
    reference_answer: str = "",
    apply_scenario: str = "",
    apply_question: str = "",
    apply_reference_answer: str = "",
    failed_rules: List[str],
    semantic_criteria: List[Dict[str, Any]],
) -> str:
    service_category = str(service_category or "").strip()
    question = str(question or apply_question or "").strip()
    reference_answer = str(reference_answer or apply_reference_answer or "").strip()
    return """[Task Instruction]
Rewrite the invalid service-decision item so it becomes strong and answerable.
Use the failed rules and validation feedback to fix the item directly.
Do not merely paraphrase the invalid item.

[Definitions]
- failed_rules: the names of the checks that failed and must be fixed.
- validation feedback: feedback for each criterion explaining what failed and why.
- strong apply item: an item where the correct answer materially depends on the user's state, asks for a concrete service action, requires the assistant to use that state to decide what to do, and has a specific answer that can be scored with points.

[Constraints]
1. Keep third-person wording about the user; never use "you", "your", or "yours".
2. Keep exactly one service decision question in one sentence.
3. Keep the revised reference answer short, specific, and bounded.
4. Fix every failed rule and every failed semantic criterion; do not preserve the original framing if it caused the failure.
5. If `personalization_necessity` failed, rewrite the question so the correct answer materially depends on the user's state.
6. If `service_decision_quality` failed, rewrite the item so that it asks for a real assistant/service decision rather than raw recall, naked lookup, one-step arithmetic, or an explicit option menu. For habit states, schedule-grounded service operations are allowed when the question asks what the assistant should do around the routine rather than merely recalling the routine time/date.
7. If `answer_groundedness` failed, make the revised answer more specific and bounded without introducing concrete facts, thresholds, tactics, metrics, product specs, or background knowledge that are not explicit in `state_value` or the question.
8. Output JSON ONLY with no extra keys.

[Example]
[Example Input]
failed_rules: ["personalization_necessity", "service_decision_quality"]
semantic_criteria: [
  {{"criterion": "personalization_necessity", "analysis": "The decisive cancellation rule is introduced entirely by the scenario, so the user state is not doing the real work.", "pass": false}},
  {{"criterion": "service_decision_quality", "analysis": "The item mainly asks the model to identify one date from the schedule instead of deciding what the assistant should do.", "pass": false}},
  {{"criterion": "answer_groundedness", "analysis": "The answer is grounded enough once the scenario is improved.", "pass": true}}
]

[Example Output]
{{
  "service_category": "notification strategy",
  "question": "What communication policy should the assistant adopt for the user during recurring technical briefing periods?",
  "reference_answer": "Hold non-urgent update requests until after the user's standing briefing block finishes, but interrupt immediately for urgent escalation requests."
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- invalid QA: object with `service_category`, `question`, `reference_answer`
- failed_rules: string[]
- semantic_criteria: object[]

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- invalid_qa: {{
    "service_category": {service_category},
    "question": {question},
    "reference_answer": {reference_answer}
  }}
- failed_rules: {failed_rules}
- semantic_criteria: {semantic_criteria}

Output JSON ONLY:
{{
  "service_category": "...",
  "question": "...",
  "reference_answer": "..."
}}
""".format(
        state_key=state_key,
        state_value=json.dumps(state_value, ensure_ascii=False),
        service_category=json.dumps(service_category, ensure_ascii=False),
        question=json.dumps(question, ensure_ascii=False),
        reference_answer=json.dumps(reference_answer, ensure_ascii=False),
        failed_rules=json.dumps(list(failed_rules or []), ensure_ascii=False),
        semantic_criteria=json.dumps(list(semantic_criteria or []), ensure_ascii=False),
    )


def build_task_c_v2_rewrite_prompt(
    *,
    state_key: str,
    state_value: Any,
    service_family: str,
    scenario: str,
    task_instruction: str,
    output_template: Any,
    reference_output: Any,
    reference_anchors: Any = None,
    failed_rules: List[str],
    semantic_criteria: List[Dict[str, Any]],
    reference_answer: str = "",
) -> str:
    normalized_family = str(service_family or "").strip()
    criterion_names = {
        "answerability",
        "service_realism",
        "full_field_dependency",
        "low_leakage",
        "output_groundedness",
    }
    structural_issue_labels = {
        "llm_invalid": "The validator response was malformed or incomplete; rewrite conservatively using all available feedback.",
        "missing_scenario": "The item is missing a usable scenario.",
        "missing_task_instruction": "The item is missing a usable task instruction.",
        "missing_reference_answer": "The item is missing a usable reference answer.",
        "missing_output_template": "The item is missing a usable output template.",
        "missing_reference_output": "The item is missing a usable reference output.",
        "output_template_mismatch": "The output template and reference output do not have matching shapes.",
        "not_structured_service_object": "The structured fields are not valid top-level service objects.",
        "raw_state_mirror": "The structured output mirrors the raw state too directly.",
    }
    cleaned_failed_rules: List[str] = []
    structural_issues: List[str] = []
    for raw_rule in list(failed_rules or []):
        rule = str(raw_rule or "").strip()
        if not rule:
            continue
        if rule in criterion_names:
            if rule not in cleaned_failed_rules:
                cleaned_failed_rules.append(rule)
            continue
        issue = structural_issue_labels.get(
            rule,
            "The builder reported an additional structural issue with the item.",
        )
        if issue not in structural_issues:
            structural_issues.append(issue)
    validation_feedback = {
        "failed_rules": cleaned_failed_rules,
        "structural_issues": structural_issues,
        "criteria": list(semantic_criteria or []),
    }
    if normalized_family == "user_communication":
        return """[Task Instruction]
Rewrite the invalid item so it becomes a realistic reminder-message task.
Use the validation_feedback to decide what to change.
Return a JSON delta patch over mutable fields only.

[Definitions]
- validation_feedback: validator feedback containing `failed_rules` and per-criterion `criteria`.
- failed_rules: the names of semantic checks that failed and must be fixed.
- structural_issues: natural-language descriptions of malformed or missing item fields that must also be repaired.
- criteria: feedback for each criterion explaining what passed, what failed, and why.
- answerability: `scenario` plus the fixed `task_instruction` define one clear current-moment communication task.
- service_realism: the item describes a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state.
- full_field_dependency: a good answer depends on all important non-derived field paths in `state_value`.
- low leakage: `scenario` does not restate, paraphrase, or strongly imply personalized habit facts that should come from `state_value`. Current weekday/date/time are allowed as world-background anchors.
- output_groundedness: a short natural-language assistant response whose key personalized content is supported by `state_value`.
{schedule_date_encoding_bullets}

[Repair Instructions]
- If `answerability` failed: rewrite `scenario` so the current moment is clear enough and it is clear what communication should be sent now. For day/date-specific routines, add the current weekday or calendar anchor plus current clock time, using the schedule date encodings.
- If `service_realism` failed: rewrite the item so it becomes a realistic assistant-mediated service action that would feel natural now, not a backend placeholder, arbitrary workflow, contrived form, or task invented only to expose the state.
- If `full_field_dependency` failed: rewrite `scenario` and/or `reference_answer` so a good answer depends on all important state fields.
- If `low_leakage` failed: remove habit identity/action, cadence, stored start/end time, location, modality, or priority from `scenario`; keep current weekday/date/time when they are only world-background anchors.
- If `output_groundedness` failed: rewrite `reference_answer` so its personalized content is grounded in `state_value` without adding unsupported user-specific facts, and fix any weekday/date claim that conflicts with encoded schedule fields.

[Constraints]
1. Rewrite only the mutable item fields:
   - `scenario`
   - `reference_answer`
2. Keep the item in natural-language assistant-response form; do not rewrite it into a structured payload.
3. Fix every failed rule and every failed semantic criterion.
4. Apply the repair instruction for each failed rule shown in validation_feedback, and repair every listed structural issue.
5. Include only the fields you actually changed; omit unchanged fields.
6. Do not add any keys other than:
   - `scenario`
   - `reference_answer`
7. Return JSON only.

[Example]
[Example Input]
state_key: "habits_state:family_dinner"
state_value: {{"schedule": {{"days_of_week": [6]}}, "timing": {{"start_time": "17:30"}}}}
invalid_item: {{
  "scenario": "It is 16:45. This is the user's Sunday family dinner, and nothing has been prepared yet.",
  "task_instruction": "Draft a specific reminder message for the user in this scenario.",
  "reference_answer": "Your Sunday family dinner starts soon, so it is a good time to begin getting things ready."
}}
validation_feedback: {{
  "failed_rules": ["answerability", "low_leakage"],
  "structural_issues": [],
  "criteria": [
    {{"criterion": "answerability", "analysis": "The item never makes clear what should be sent right now.", "pass": false}},
    {{"criterion": "service_realism", "analysis": "The item asks for one communication action.", "pass": true}},
    {{"criterion": "full_field_dependency", "analysis": "The state fields are mostly used.", "pass": true}},
    {{"criterion": "low_leakage", "analysis": "The scenario repeats that this is the user's Sunday family dinner.", "pass": false}},
    {{"criterion": "output_groundedness", "analysis": "The answer stays grounded once the setup is clarified.", "pass": true}}
  ]
}}

[Example Output]
{{
  "scenario": "It is Sunday at 16:45. Everyone is home, and nothing has been prepared yet."
}}

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- invalid_item: {{
    "scenario": {scenario},
    "task_instruction": {task_instruction},
    "reference_answer": {reference_answer}
  }}
- validation_feedback: {validation_feedback}

Output JSON ONLY:
{{
  "<changed_mutable_field>": "..."
}}
""".format(
            state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
            state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
            scenario=json.dumps(str(scenario or ""), ensure_ascii=False),
            task_instruction=json.dumps(str(task_instruction or ""), ensure_ascii=False),
            reference_answer=json.dumps(str(reference_answer or ""), ensure_ascii=False),
            validation_feedback=json.dumps(validation_feedback, ensure_ascii=False, indent=2),
            schedule_date_encoding_bullets=SCHEDULE_DATE_ENCODING_BULLETS,
        )
    if normalized_family == "information_request_construction":
        rewrite_task_instruction = "Rewrite the invalid item so it becomes a realistic search-filter task.\nUse the validation_feedback to decide what to change."
        completion_definition = "the item describes a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state."
        service_repair = "rewrite `scenario`, `output_template`, and/or `reference_output` so the search/filter task feels like a natural user browsing, search, comparison, or planning moment."
        field_repair = "rewrite the search/filter object so it has one or two state-dependent fill leaves. At least one fill leaf must be the field-local core fill for the preference task; a second detail fill is allowed only when grounded and service-useful."
        leakage_repair = "remove any restatement of key preference facts from `scenario`."
        groundedness_repair = "repair `output_template`, `reference_output`, and `reference_anchors` so they form a synthesized search/filter object rather than a raw preference copy, with every output value grounded in `state_value`, exactly one anchor per fill leaf, and anchor roles set to `core` or `detail`."
        object_constraint = "`output_template` and `reference_output` must remain top-level search/filter objects, not final recommendations, ranked lists, backend payloads, or raw preference mirrors."
        example = """[Example]
[Example Input]
state_key: "preferences_state:learning_modality"
state_value: {"statement": "prefers self-paced webinars"}
invalid_item: {
  "scenario": "A training-resource search request is about to run.",
  "task_instruction": "Help the user set the search filters in this scenario.",
  "output_template": {"request_profile": {"preference_statement": "<fill>"}},
  "reference_output": {"request_profile": {"preference_statement": "prefers self-paced webinars"}},
  "reference_anchors": [{"target_path": "request_profile.preference_statement", "role": "core", "state_reference": "prefers self-paced webinars", "anchor_note": "raw statement copy"}]
}
validation_feedback: {
  "failed_rules": ["answerability", "output_groundedness"],
  "structural_issues": [],
  "criteria": [
    {"criterion": "answerability", "analysis": "The item does not make clear what search/filter object should be completed now.", "pass": false},
    {"criterion": "service_realism", "analysis": "The item is structured, but it collapses toward raw preference transfer.", "pass": false},
    {"criterion": "full_field_dependency", "analysis": "The only state field is required.", "pass": true},
    {"criterion": "low_leakage", "analysis": "The scenario does not restate the preference.", "pass": true},
    {"criterion": "output_groundedness", "analysis": "The current object still behaves too much like a raw state copy.", "pass": false}
  ]
}

[Example Output]
{
  "scenario": "The user is browsing training resources for an upcoming professional-development block. The assistant is setting search filters before showing matching options.",
  "output_template": {"content_search_filters": {"preferred_format": "<fill>"}},
  "reference_output": {"content_search_filters": {"preferred_format": "self-paced webinars"}},
  "reference_anchors": [{"target_path": "content_search_filters.preferred_format", "role": "core", "state_reference": "prefers self-paced webinars", "anchor_note": "field-local core training format"}]
}"""
    elif normalized_family == "action_configuration":
        rewrite_task_instruction = "Rewrite the invalid item so it becomes a realistic setup-or-form completion task.\nUse the validation_feedback to decide what to change."
        completion_definition = "the item describes a realistic assistant-mediated service action that a user could naturally be doing now. It should not feel like a backend placeholder, arbitrary workflow, contrived form, or a task invented only to expose the state."
        service_repair = "rewrite `scenario`, `output_template`, and/or `reference_output` so the setup/form/configuration task feels like a natural user product moment."
        field_repair = "rewrite the action object so it has one or two state-dependent fill leaves. At least one fill leaf must be the field-local core fill for the attribute task; a second detail fill is allowed only when grounded, service-useful, and not dependent on an extra user choice."
        leakage_repair = "remove any restatement of key attribute facts from `scenario`."
        groundedness_repair = "repair `output_template`, `reference_output`, and `reference_anchors` so they form an execution-ready configuration object rather than a raw attribute copy, with every output value grounded in `state_value`, exactly one anchor per fill leaf, anchor roles set to `core` or `detail`, and no extra user choice."
        object_constraint = "`output_template` and `reference_output` must remain top-level setup/form/configuration objects, not filtering requests, recommendations, backend dispatches, or raw attribute mirrors."
        example = """[Example]
[Example Input]
state_key: "user_attributes_state:family_sports_gear"
state_value: "Set of 10 practice soccer balls and cones"
invalid_item: {
  "scenario": "The user is registering gear for a local youth league's equipment drive. The assistant is filling the donation details before the form is submitted.",
  "task_instruction": "Help the user complete the setup or form fields in this scenario.",
  "output_template": {"equipment_donation": {"quantity_to_donate": "<fill>", "items": ["<fill>", "<fill>"]}},
  "reference_output": {"equipment_donation": {"quantity_to_donate": 10, "items": ["practice soccer balls", "cones"]}},
  "reference_anchors": [{"target_path": "equipment_donation.quantity_to_donate", "role": "core", "state_reference": "Set of 10 practice soccer balls and cones", "anchor_note": "incorrectly treats owned quantity as donation quantity"}]
}
validation_feedback: {
  "failed_rules": ["answerability", "output_groundedness"],
  "structural_issues": [],
  "criteria": [
    {"criterion": "answerability", "analysis": "The donation form requires choosing what quantity or subset to donate, which is not determined by the state.", "pass": false},
    {"criterion": "service_realism", "analysis": "A deterministic inventory/profile setup would be a better action-configuration task.", "pass": false},
    {"criterion": "full_field_dependency", "analysis": "The gear details are relevant, but the donation quantity is an extra choice.", "pass": false},
    {"criterion": "low_leakage", "analysis": "The scenario does not restate the exact gear details.", "pass": true},
    {"criterion": "output_groundedness", "analysis": "The state says what gear exists, but not what should be donated.", "pass": false}
  ]
}

[Example Output]
{
  "scenario": "The user is setting up a family sports equipment inventory. The assistant is filling the gear details before the equipment profile is saved.",
  "output_template": {"sports_equipment_profile": {"equipment_inventory": "<fill>"}},
  "reference_output": {"sports_equipment_profile": {"equipment_inventory": "set of 10 practice soccer balls and cones"}},
  "reference_anchors": [{"target_path": "sports_equipment_profile.equipment_inventory", "role": "core", "state_reference": "Set of 10 practice soccer balls and cones", "anchor_note": "field-local core inventory configuration"}]
}"""
    else:
        raise ValueError(f"Unsupported Task C v2 service_family: {normalized_family}")

    return """[Task Instruction]
{rewrite_task_instruction}
Return a JSON delta patch over mutable fields only.

[Definitions]
- validation_feedback: validator feedback containing `failed_rules` and per-criterion `criteria`.
- failed_rules: the names of semantic checks that failed and must be fixed.
- structural_issues: natural-language descriptions of malformed or missing item fields that must also be repaired.
- criteria: feedback for each criterion explaining what passed, what failed, and why.
- answerability: `scenario` plus the fixed `task_instruction` define one clear current-moment structured completion task.
- service_realism: {completion_definition}
- full_field_dependency: the structured completion has one or two state-dependent fill leaves. Every fill leaf is necessary for the service task and grounded in `state_value`; at least one fill leaf is the field-local core fill, and any detail fill adds grounded service-relevant precision.
- low leakage: `scenario` does not restate, paraphrase, or strongly imply the user-state facts that should come from `state_value`.
- output_groundedness: `output_template` plus `reference_output` define a task-appropriate service object grounded in `state_value`.
{schedule_date_encoding_bullets}

[Repair Instructions]
- If `answerability` failed: rewrite `scenario` so it is clear what structured object should be completed now; if state_value contains schedule-like weekday/month fields, use the schedule date encodings.
- If `service_realism` failed: {service_repair}
- If `full_field_dependency` failed: {field_repair}
- If `low_leakage` failed: {leakage_repair}
- If `output_groundedness` failed: {groundedness_repair}
- If the item depends on an extra user choice, rewrite it as a deterministic auto-fill task using neutral setup, profile, inventory, connection, or form fields.

[Constraints]
1. Rewrite only the mutable item fields:
   - `scenario`
   - `output_template`
   - `reference_output`
   - `reference_anchors`
2. {object_constraint}
3. Fix every failed rule and every failed semantic criterion.
4. Apply the repair instruction for each failed rule shown in validation_feedback, and repair every listed structural issue.
5. Include only the fields you actually changed; omit unchanged fields.
6. Do not add any keys other than:
   - `scenario`
   - `output_template`
   - `reference_output`
   - `reference_anchors`
7. Return JSON only.

{example}

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- invalid_item: {{
    "scenario": {scenario},
    "task_instruction": {task_instruction},
    "output_template": {output_template},
    "reference_output": {reference_output},
    "reference_anchors": {reference_anchors}
  }}
- validation_feedback: {validation_feedback}

Output JSON ONLY:
{{
  "<changed_mutable_field>": "..."
}}
    """.format(
        rewrite_task_instruction=rewrite_task_instruction,
        completion_definition=completion_definition,
        service_repair=service_repair,
        field_repair=field_repair,
        leakage_repair=leakage_repair,
        groundedness_repair=groundedness_repair,
        object_constraint=object_constraint,
        example=example,
        schedule_date_encoding_bullets=SCHEDULE_DATE_ENCODING_BULLETS,
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
        scenario=json.dumps(str(scenario or ""), ensure_ascii=False),
        task_instruction=json.dumps(str(task_instruction or ""), ensure_ascii=False),
        output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        reference_output=json.dumps(reference_output, ensure_ascii=False, indent=2),
        reference_anchors=json.dumps(reference_anchors if reference_anchors is not None else [], ensure_ascii=False, indent=2),
        validation_feedback=json.dumps(validation_feedback, ensure_ascii=False, indent=2),
    )


# Task A complex-text scoring-point generation prompt.
# Use this when a single state leaf is too semantic to score as one field point.
_VALUE_RUBRIC_FAILURE_DEFINITIONS = """- unsupported: the scoring point requires content that is not stated or clearly entailed by the source text.
- drift: the scoring point changes, overextends, weakens, strengthens, or reverses the source meaning.
- not_atomic: the scoring point combines multiple independently checkable requirements or cannot be judged as one binary hit.
- redundant: the scoring point repeats the same requirement or meaning as another point in the same set.
- over_specific: the scoring point depends on unnecessary exact model / spec / parameter / exact wording; use exact identifiers only when they are canonical facts or needed to disambiguate."""

_VALUE_RUBRIC_SET_FAILURE_DEFINITIONS = """- coverage_gap: the set omits a major scoreable meaning unit from the source text, so the rubric would not adequately test the source meaning."""

_VALUE_RUBRIC_VALIDITY_REQUIREMENTS = """- supported: each point must be directly supported by the source text; this avoids `unsupported`.
- faithful: each point must preserve the source meaning without stronger preferences, motivations, implications, reversals, or weaker/generalized claims; this avoids `drift`.
- atomic: each point must express one independently judgeable requirement on the shared binary `0/1` hit scale; this avoids `not_atomic`.
- distinct: points in the same set must not repeat the same meaning or all hinge on the same narrow identifier, model name, spec, parameter, or exact wording; this avoids `redundant`.
- stable: prefer stable semantic requirements over brittle exact surface forms; use an exact identifier only when it is the canonical memory fact or needed to disambiguate; this avoids `over_specific`.
- covered: the set should cover the major scoreable meaning units in the source text within the {max_points}-point limit; this avoids `coverage_gap`."""

_VALUE_RUBRIC_REPAIR_INSTRUCTIONS = """- unsupported -> delete the unsupported claim or replace it with a claim directly supported by the source text.
- drift -> rewrite the point so it preserves the source meaning without strengthening, weakening, reversing, or adding unstated implications.
- not_atomic -> split the point into separate atomic points when the {max_points}-point limit allows; otherwise keep the most central scoreable meaning.
- redundant -> merge or delete repeated meaning so each remaining point tests a distinct requirement.
- over_specific -> rewrite to a stable semantic requirement unless the exact identifier is the canonical memory fact or needed to disambiguate.
- coverage_gap -> add or revise points so the final set covers the major scoreable meaning units in the source text within the {max_points}-point limit."""


def build_value_micro_points_prompt(
    *,
    state_key: str,
    text_value: str,
    max_points: int,
) -> str:
    max_points = max(1, min(3, int(max_points)))
    template = '[\n    "<one scoring point>",\n    ...,\n    "<one scoring point>"\n  ]'
    return """[Task Instruction]
Break one complex state field into 1 to {max_points} scoring points.
Each scoring point should capture one scoreable meaning unit that a model prediction should express.

[Definitions]
- complex state field: one statement-like field whose meaning should not be judged as one holistic blob.
- scoring point: one scoring requirement.
- scoreable scoring point: one concrete aspect of correctness that can be judged independently on the shared binary `0/1` hit scale.
- Invalid scoring point categories:
{failure_definitions}
- Invalid set-level categories:
{set_failure_definitions}
- Valid scoring-point set requirements:
{validity_requirements}

[Constraints]
1. Generate 1 to {max_points} scoring points.
2. Keep all scoring points positive.
3. Follow every valid scoring-point set requirement: `supported`, `faithful`, `atomic`, `distinct`, `stable`, `covered`.
4. Avoid every invalid point-level category: `unsupported`, `drift`, `not_atomic`, `redundant`, `over_specific`.
5. Avoid the set-level category: `coverage_gap`.
6. Output JSON only.

[Example]
[Example Input]
- state_key: "preferences_state:learning_modality"
- text_value: "Prefers self-paced white papers and webinars over large conferences."

[Example Output]
{{
  "rubric": [
    "The answer states that self-paced white papers or webinars are the preferred format.",
    "The answer does not recommend large conferences as the preferred format."
  ]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- text_value: {text_value}

Output JSON ONLY:
{{
  "rubric": {template}
}}
""".format(
        max_points=max_points,
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        text_value=json.dumps(str(text_value or ""), ensure_ascii=False),
        template=template,
        failure_definitions=_VALUE_RUBRIC_FAILURE_DEFINITIONS,
        set_failure_definitions=_VALUE_RUBRIC_SET_FAILURE_DEFINITIONS,
        validity_requirements=_VALUE_RUBRIC_VALIDITY_REQUIREMENTS.format(max_points=max_points),
    )


# Task A complex-text scoring-point validation prompt.
# Use this to verify that generated micro scoring points stay grounded and independently scoreable.
def build_value_rubric_validation_prompt(
    *,
    state_key: str,
    text_value: str,
    points: List[Dict[str, Any]],
) -> str:
    return """[Task Instruction]
Validate one generated scoring-point set for a complex state field.
Judge whether each scoring point stays grounded in the source text without drift.

[Definitions]
- source text: the original state text that the rubric-point set must stay faithful to.
- scoring point: one scoreable meaning unit.
- point_text: the scoring statement that the evaluator will judge independently.
- scoring-point requirement: one point that checks exactly one concrete aspect of correctness and can be scored independently.
- Invalid scoring point categories:
{failure_definitions}
- Invalid set-level categories:
{set_failure_definitions}

[Constraints]
1. Validate every point independently.
2. Mark a point as failed if it matches any invalid scoring point category.
3. Use only these fail reasons when needed: `unsupported`, `drift`, `not_atomic`, `redundant`, `over_specific`.
4. Use only these set-level failures when needed: `coverage_gap`.
5. `set_pass` can be true only if every point passes and there is no set-level failure.
6. `set_analysis` must briefly explain the set-level decision, especially any `coverage_gap`.
7. Keep each point `analysis` and the `set_analysis` concise and specific.
8. Output JSON only.

[Example]
[Example Input]
- state_key: "preferences_state:learning_modality"
- text_value: "Prefers self-paced white papers and webinars over large conferences."
- points: [
  {{
    "point_id": "scp_pref_p1",
    "point_text": "The answer states that self-paced white papers or webinars are preferred learning formats."
  }},
  {{
    "point_id": "scp_pref_p2",
    "point_text": "The answer says the user dislikes all in-person events."
  }}
]

[Example Output]
{{
  "points": [
    {{
      "point_id": "scp_pref_p1",
      "analysis": "This point is directly supported by the source text and captures the preferred self-paced formats.",
      "pass": true,
      "fail_reasons": []
    }},
    {{
      "point_id": "scp_pref_p2",
      "analysis": "The source only contrasts with large conferences; it does not say the user dislikes all in-person events.",
      "pass": false,
      "fail_reasons": ["drift", "unsupported"]
    }}
  ],
  "set_analysis": "The set has one valid point for the preferred formats, but no valid point captures that large conferences are not preferred.",
  "set_pass": false,
  "set_failures": ["coverage_gap"]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- text_value: {text_value}
- points: {points}

Output JSON ONLY:
{{
  "points": [
    {{
      "point_id": "<point_id from input>",
      "analysis": "<brief validation analysis>",
      "pass": "<bool>",
      "fail_reasons": []
    }}
  ],
  "set_analysis": "<brief set-level analysis; explain coverage_gap when present>",
  "set_pass": "<bool>",
  "set_failures": []
}}
""".format(
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        text_value=json.dumps(str(text_value or ""), ensure_ascii=False),
        points=json.dumps(points, ensure_ascii=False, indent=2),
        failure_definitions=_VALUE_RUBRIC_FAILURE_DEFINITIONS,
        set_failure_definitions=_VALUE_RUBRIC_SET_FAILURE_DEFINITIONS,
    )


# Task A complex-text scoring-point rewrite prompt.
# Use this when validation fails and the micro scoring points need a grounded rewrite.
def build_value_rubric_rewrite_prompt(
    *,
    state_key: str,
    text_value: str,
    validation_feedback: Dict[str, Any],
    max_points: int,
) -> str:
    max_points = max(1, min(3, int(max_points)))
    template = '[\n    "<one corrected scoring point>",\n    ...,\n    "<one corrected scoring point>"\n  ]'
    return """[Task Instruction]
Rewrite an invalid scoring-point set for one complex state field.
Return a corrected scoring-point set that stays faithful to the source text.

[Definitions]
- scoring point: one scoreable meaning unit.
- point_text: one scoring statement that the evaluator will judge independently.
- faithful rewrite: a rewrite that removes drift, unsupported content, and redundancy while preserving the source meaning.
- robust rewrite: a rewrite that avoids unnecessary exact model / spec / parameter wording when a broader semantic fact is sufficient.
- full-set rewrite: return a complete replacement rubric for the whole set, not a patch and not only the failed points.
- Invalid scoring point categories:
{failure_definitions}
- Invalid set-level categories:
{set_failure_definitions}
- Repair instructions:
{repair_instructions}

[Constraints]
1. Return 1 to {max_points} positive scoring points.
2. Return a complete replacement `rubric` for the whole set.
3. Apply the repair instruction for each failed point reason and set-level failure shown in validation feedback.
4. Preserve any valid source meaning from the prior set, but rewrite freely when needed to make the final set valid.
5. Do not copy failed wording just to preserve an old point.
6. Output JSON only.

[Example]
[Example Input]
- state_key: "preferences_state:learning_modality"
- text_value: "Prefers self-paced white papers and webinars over large conferences."
- validation_feedback:
  {{
    "points": [
      {{
        "point_id": "scp_pref_p1",
        "point_text": "The answer states that self-paced white papers or webinars are preferred learning formats.",
        "analysis": "This point is directly supported by the source text and captures the preferred self-paced formats.",
        "pass": true,
        "fail_reasons": []
      }},
      {{
        "point_id": "scp_pref_p2",
        "point_text": "The answer says the user dislikes all in-person events.",
        "analysis": "The source only contrasts with large conferences; it does not say the user dislikes all in-person events.",
        "pass": false,
        "fail_reasons": ["drift", "unsupported"]
      }}
    ],
    "set_analysis": "The set has one valid point for the preferred formats, but no valid point captures that large conferences are not preferred.",
    "set_failures": ["coverage_gap"]
  }}

[Example Output]
{{
  "rubric": [
    "The answer states that self-paced white papers or webinars are preferred learning formats.",
    "The answer states that large conferences are not the preferred learning format."
  ]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- text_value: {text_value}
- validation_feedback: {validation_feedback}

Output JSON ONLY:
{{
  "rubric": {template}
}}
""".format(
        max_points=max_points,
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        text_value=json.dumps(str(text_value or ""), ensure_ascii=False),
        validation_feedback=json.dumps(validation_feedback or {}, ensure_ascii=False, indent=2),
        template=template,
        failure_definitions=_VALUE_RUBRIC_FAILURE_DEFINITIONS,
        set_failure_definitions=_VALUE_RUBRIC_SET_FAILURE_DEFINITIONS,
        repair_instructions=_VALUE_RUBRIC_REPAIR_INSTRUCTIONS.format(max_points=max_points),
    )


def build_change_reason_scoring_points_prompt(
    *,
    state_key: str,
    before_value: Any,
    after_value: Any,
    reference_change_reason: str,
    max_points: int,
) -> str:
    max_points = max(1, min(3, int(max_points)))
    template = ["<one atomic fact about the gold change reason>" for _ in range(max_points)]
    return """[Task Instruction]
Generate atomic facts for evaluating a predicted change reason.
Each atomic fact should capture one atomic part of the gold change reason text.

[Definitions]
- gold change reason: the canonical original explanation for why this state changed.
- atomic fact: one atomic scoring requirement that can be checked independently.
- transition context: the `before` and `after` values that anchor what changed. The transition context is grounding context, not a substitute for the gold change reason text.
- brittle atomic fact: a fact that depends on unnecessary exact model / spec / parameter / exact wording instead of stable transition meaning.

[Constraints]
1. Generate 1 to {max_points} atomic facts.
2. Each atomic fact must be independently judgeable on the shared binary `0/1` hit scale.
3. Keep all atomic facts positive.
4. Every atomic fact must be directly supported by the gold change reason text.
5. Use the transition context only to disambiguate the gold reason text.
6. Do not restate unchanged facts unless they are explicitly part of the gold change reason text.
7. Do not add new causes, motives, or implications that are not stated.
8. Prefer stable semantic transition facts over brittle exact surface forms.
9. Do not generate multiple points that all hinge on the same narrow identifier, model name, spec, parameter, or exact wording.
10. Use an exact identifier only when it is itself the canonical changed fact or is necessary to disambiguate the transition meaning.
11. Do not repeat the same meaning in multiple atomic facts.
12. Output JSON only.

[Example]
Input transition:
- before_value: "commutes by car"
- after_value: "commutes by train"
- reference_change_reason: "Switched to train commuting after downtown parking fees increased."

Output JSON ONLY:
{{
  "rubric": [
    "The answer explains that commuting changed because downtown parking became more expensive.",
    "The answer ties the transition specifically to switching away from car commuting."
  ]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- before_value: {before_value}
- after_value: {after_value}
- reference_change_reason: {reference_change_reason}

Output JSON ONLY:
{{
  "rubric": {template}
}}
""".format(
        max_points=max_points,
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        before_value=json.dumps(before_value, ensure_ascii=False),
        after_value=json.dumps(after_value, ensure_ascii=False),
        reference_change_reason=json.dumps(str(reference_change_reason or ""), ensure_ascii=False),
        template=json.dumps(template, ensure_ascii=False, indent=2),
    )


def build_change_reason_rubric_validation_prompt(
    *,
    state_key: str,
    before_value: Any,
    after_value: Any,
    reference_change_reason: str,
    points: List[Dict[str, Any]],
) -> str:
    return """[Task Instruction]
Validate one generated atomic-fact set for a state change reason.
Judge whether each atomic fact is grounded in the gold change reason text and stays faithful to the actual transition.

[Definitions]
- gold change reason: the canonical original explanation for why the state changed.
- transition context: the concrete `before_value -> after_value` change that grounds the gold reason.
- atomic fact: one atomic scoring fact for a change explanation.
- unsupported fact: a fact that is not justified by the gold change reason plus transition context.
- drift: a fact that changes or overextends the meaning of the gold change reason.
- atomic fact requirement: one fact that checks exactly one concrete reason aspect and can be scored independently.
- over_specific: a fact that depends on unnecessary exact model / spec / parameter / exact wording rather than stable transition meaning.

[Constraints]
1. Validate every point independently.
2. Mark a point as failed if it is unsupported, drifts, is not atomic, is redundant, or is over-specific.
3. Use only these fail reasons when needed: `unsupported`, `drift`, `not_atomic`, `redundant`, `over_specific`.
4. Use only these set-level failures when needed: `coverage_gap`.
5. `set_pass` can be true only if every point passes and there is no set-level failure.
6. Keep each `analysis` concise and specific.
7. Output JSON only.

[Example]
Input transition:
- before_value: "commutes by car"
- after_value: "commutes by train"
- reference_change_reason: "Switched to train commuting after downtown parking fees increased."

Candidate points:
{{
  "points": [
    {{
      "point_id": "crp_commute_p1",
      "point_text": "The change happened because the user rejected all driving forever."
    }}
  ]
}}

Output JSON ONLY:
{{
  "points": [
    {{
      "point_id": "crp_commute_p1",
      "analysis": "The point exaggerates the gold change reason and adds a broader unsupported claim.",
      "pass": false,
      "fail_reasons": ["drift", "unsupported"]
    }}
  ],
  "set_pass": false,
  "set_failures": ["coverage_gap"]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- before_value: {before_value}
- after_value: {after_value}
- reference_change_reason: {reference_change_reason}
- points: {points}

Output JSON ONLY:
{{
  "points": [
    {{
      "point_id": "<point_id from input>",
      "analysis": "<brief validation analysis>",
      "pass": "<bool>",
      "fail_reasons": []
    }}
  ],
  "set_pass": "<bool>",
  "set_failures": []
}}
""".format(
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        before_value=json.dumps(before_value, ensure_ascii=False),
        after_value=json.dumps(after_value, ensure_ascii=False),
        reference_change_reason=json.dumps(str(reference_change_reason or ""), ensure_ascii=False),
        points=json.dumps(points, ensure_ascii=False, indent=2),
    )


def build_change_reason_rubric_rewrite_prompt(
    *,
    state_key: str,
    before_value: Any,
    after_value: Any,
    reference_change_reason: str,
    points: List[Dict[str, Any]],
    validation_points: List[Dict[str, Any]],
    set_failures: List[str],
    max_points: int,
) -> str:
    max_points = max(1, min(3, int(max_points)))
    template = ["<one corrected atomic fact about the gold change reason>" for _ in range(max_points)]
    return """[Task Instruction]
Rewrite an invalid atomic-fact set for a state change reason.
Return a corrected atomic-fact set that stays faithful to the gold change reason text and the actual transition.

[Definitions]
- atomic fact: one atomic scoring fact for a change explanation.
- faithful rewrite: a rewrite that removes drift, unsupported claims, redundancy, and non-atomic wording while preserving the original gold change reason.
- robust rewrite: a rewrite that avoids unnecessary exact model / spec / parameter wording when a broader transition fact is sufficient.

[Constraints]
1. Return 1 to {max_points} positive atomic facts.
2. Each atomic fact must be independently judgeable on the shared binary `0/1` hit scale.
3. Every point must stay directly supported by the gold change reason text.
4. Use the transition context only to disambiguate the gold reason text.
5. Do not add unsupported causes, motives, or broader claims.
6. Keep the set non-redundant.
7. Repair any `over_specific` or brittle point by rewriting it into a broader, more stable transition fact unless the exact identifier is truly necessary.
8. Output JSON only.

[Example]
Input transition:
- before_value: "commutes by car"
- after_value: "commutes by train"
- reference_change_reason: "Switched to train commuting after downtown parking fees increased."

Invalid points:
{{
  "points": [
    {{
      "point_id": "crp_commute_p1",
      "point_text": "The change happened because the user rejected all driving forever."
    }}
  ]
}}

Validation feedback:
{{
  "points": [
    {{
      "point_id": "crp_commute_p1",
      "analysis": "The point exaggerates the gold change reason and adds unsupported meaning.",
      "pass": false,
      "fail_reasons": ["drift", "unsupported"]
    }}
  ],
  "set_failures": ["coverage_gap"]
}}

Output JSON ONLY:
{{
  "rubric": [
    "The answer explains that parking costs increased.",
    "The answer ties the commuting change to switching away from car travel."
  ]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- before_value: {before_value}
- after_value: {after_value}
- reference_change_reason: {reference_change_reason}
- invalid_points: {points}
- validation_points: {validation_points}
- set_failures: {set_failures}

Output JSON ONLY:
{{
  "rubric": {template}
}}
""".format(
        max_points=max_points,
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        before_value=json.dumps(before_value, ensure_ascii=False),
        after_value=json.dumps(after_value, ensure_ascii=False),
        reference_change_reason=json.dumps(str(reference_change_reason or ""), ensure_ascii=False),
        points=json.dumps(points, ensure_ascii=False, indent=2),
        validation_points=json.dumps(validation_points, ensure_ascii=False, indent=2),
        set_failures=json.dumps(list(set_failures or []), ensure_ascii=False),
        template=json.dumps(template, ensure_ascii=False, indent=2),
    )


def build_apply_rubric_validation_prompt(
    *,
    state_key: str,
    state_value: Any,
    service_category: str,
    question: str,
    reference_answer: str,
    points: List[Dict[str, Any]],
) -> str:
    return """[Task Instruction]
Validate one generated atomic-fact set for a service-decision item.
Judge whether each atomic fact is grounded in the reference answer, faithful to the user state, and atomic enough for stable scoring.

[Definitions]
- atomic fact: one atomic scoring fact for the apply answer.
- grounded fact: a fact directly supported by the reference answer and compatible with the user state.
- drift: a fact that changes or overextends the intended answer.
- atomic fact requirement: one fact that checks exactly one concrete aspect of correctness and can be scored independently.
- over_specific: a fact that depends on unnecessary exact model / spec / parameter / exact wording and is therefore too brittle.

[Constraints]
1. Validate every point independently.
2. Mark a point as failed if it is unsupported, drifts, is not atomic, is redundant, or is over-specific.
3. Use only these fail reasons when needed: `unsupported`, `drift`, `not_atomic`, `redundant`, `over_specific`.
4. Use only these set-level failures when needed: `coverage_gap`.
5. `set_pass` can be true only if every point passes and there is no set-level failure.
6. Keep each `analysis` concise and specific.
7. Treat `state_value` as grounding context, not optional metadata; a point that fits the answer wording but ignores the user state can still be invalid.
8. Output JSON only.

[Example]
Candidate QA:
{{
  "question": "What communication policy should the assistant adopt for the user during recurring technical briefing periods?",
  "reference_answer": "Hold non-urgent update requests until after the user's standing briefing block finishes, but interrupt immediately for urgent escalation requests."
}}

Candidate points:
{{
  "points": [
    {{
      "point_id": "aqp_briefing_p1",
      "point_text": "The answer disables all automated handling for briefing-related requests."
    }}
  ]
}}

Output JSON ONLY:
{{
  "points": [
    {{
      "point_id": "aqp_briefing_p1",
      "analysis": "The point conflicts with the reference answer and is not a faithful scoring criterion.",
      "pass": false,
      "fail_reasons": ["drift", "unsupported"]
    }}
  ],
  "set_pass": false,
  "set_failures": ["coverage_gap"]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- state_value: {state_value}
- service_category: {service_category}
- question: {question}
- reference_answer: {reference_answer}
- points: {points}

Output JSON ONLY:
{{
  "points": [
    {{
      "point_id": "<point_id from input>",
      "analysis": "<brief validation analysis>",
      "pass": "<bool>",
      "fail_reasons": []
    }}
  ],
  "set_pass": "<bool>",
  "set_failures": []
}}
""".format(
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False),
        service_category=json.dumps(str(service_category or ""), ensure_ascii=False),
        question=json.dumps(str(question or ""), ensure_ascii=False),
        reference_answer=json.dumps(str(reference_answer or ""), ensure_ascii=False),
        points=json.dumps(points, ensure_ascii=False, indent=2),
    )


def build_apply_rubric_rewrite_prompt(
    *,
    state_key: str,
    state_value: Any,
    service_category: str,
    question: str,
    reference_answer: str,
    points: List[Dict[str, Any]],
    validation_points: List[Dict[str, Any]],
    set_failures: List[str],
    max_points: int,
) -> str:
    template = ["<one corrected atomic fact for the reference answer>" for _ in range(max(1, int(max_points)))]
    return """[Task Instruction]
Rewrite an invalid atomic-fact set for one service-decision item.
Return a corrected atomic-fact set that stays faithful to the reference answer and compatible with the user state.

[Definitions]
- atomic fact: one atomic scoring fact for the apply answer.
- faithful rewrite: a rewrite that removes drift, unsupported content, non-atomic wording, and redundancy while preserving the intended answer.
- robust rewrite: a rewrite that avoids unnecessary exact model / spec / parameter wording when a broader semantic fact is sufficient.

[Constraints]
1. Return 1 to {max_points} atomic facts.
2. Each atomic fact must be independently judgeable on the shared binary `0/1` hit scale.
3. Every point must stay directly supported by the reference answer.
4. Use the user state only to disambiguate or bound the reference answer; do not add new facts beyond the answer.
5. Fix every invalid point and any set-level coverage gap.
6. Keep the set non-redundant.
7. Repair any `over_specific` or brittle point by rewriting it into a broader, more stable semantic fact unless the exact identifier is truly necessary.
11. Output JSON only.

[Example]
Candidate QA:
{{
  "question": "What communication policy should the assistant adopt for the user during recurring technical briefing periods?",
  "reference_answer": "Hold non-urgent update requests until after the user's standing briefing block finishes, but interrupt immediately for urgent escalation requests."
}}

Invalid points:
{{
  "points": [
    {{
      "point_id": "aqp_briefing_p1",
      "point_text": "The answer disables all automated handling for briefing-related requests."
    }}
  ]
}}

Validation feedback:
{{
  "points": [
    {{
      "point_id": "aqp_briefing_p1",
      "analysis": "The point conflicts with the reference answer and introduces unsupported meaning.",
      "pass": false,
      "fail_reasons": ["drift", "unsupported"]
    }}
  ],
  "set_failures": ["coverage_gap"]
}}

[Example Output]
{{
  "rubric": [
    "The answer delays non-urgent requests until after the standing briefing period.",
    "The answer still allows urgent escalation requests to interrupt immediately."
  ]
}}

[Input/Output Format]
Input:
- state_key: {state_key}
- state_value: {state_value}
- service_category: {service_category}
- question: {question}
- reference_answer: {reference_answer}
- invalid_points: {points}
- validation_points: {validation_points}
- set_failures: {set_failures}

Output JSON ONLY:
{{
  "rubric": {template}
}}
""".format(
        max_points=max(1, int(max_points)),
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False),
        service_category=json.dumps(str(service_category or ""), ensure_ascii=False),
        question=json.dumps(str(question or ""), ensure_ascii=False),
        reference_answer=json.dumps(str(reference_answer or ""), ensure_ascii=False),
        points=json.dumps(points, ensure_ascii=False, indent=2),
        validation_points=json.dumps(validation_points, ensure_ascii=False, indent=2),
        set_failures=json.dumps(list(set_failures or []), ensure_ascii=False),
        template=json.dumps(template, ensure_ascii=False, indent=2),
    )


def build_apply_answer_scoring_points_prompt(
    *,
    state_key: str,
    state_value: Any,
    apply_scenario: str,
    apply_question: str,
    apply_reference_answer: str,
    max_points: int,
) -> str:
    template = [
        {
            "polarity": "positive",
            "point_text": "<one required answer point or one forbidden mistake>",
            "reference_value": "<short canonical phrase>",
        }
        for _ in range(max(1, int(max_points)))
    ]
    return """[Task Instruction]
Generate scoreable atomic facts for one service-decision item.
You may include both positive atomic facts and negative atomic facts.

[Definitions]
- positive atomic fact: a meaning unit the predicted answer should include or satisfy.
- negative atomic fact: a mistake, forbidden recommendation, or ignored hard constraint that should lower the score if violated.
- grounding context: the combination of `state_value`, `apply_question`, and `apply_reference_answer`; atomic facts must be faithful to all three, not just the question wording.
- brittle atomic fact: a fact that depends on unnecessary exact model / spec / parameter / exact wording instead of stable semantic meaning.

[Constraints]
1. Generate 1 to {max_points} atomic facts.
2. At least one point must be positive.
3. Negative points are allowed when they make scoring safer.
4. Each atomic fact must be independently judgeable on the shared binary `0/1` hit scale.
5. Keep `reference_value` short and canonical.
6. Use `state_value` together with `apply_question` and `apply_reference_answer` as grounding context.
7. Prefer stable semantic requirements over brittle exact surface forms.
8. Do not generate multiple points that all hinge on the same narrow identifier, model name, spec, parameter, or exact wording.
9. Use an exact identifier only when it is itself the canonical memory fact or is necessary to disambiguate the intended answer.
10. Do not repeat the same meaning in multiple points.
11. Output JSON only.

[Input/Output Format]
Input:
- state_key: {state_key}
- state_value: {state_value}
- apply_scenario: {apply_scenario}
- apply_question: {apply_question}
- apply_reference_answer: {apply_reference_answer}

Output JSON ONLY:
{{
  "points": {template}
}}
""".format(
        max_points=max(1, int(max_points)),
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False),
        apply_scenario=json.dumps(str(apply_scenario or ""), ensure_ascii=False),
        apply_question=json.dumps(str(apply_question or ""), ensure_ascii=False),
        apply_reference_answer=json.dumps(str(apply_reference_answer or ""), ensure_ascii=False),
        template=json.dumps(template, ensure_ascii=False, indent=2),
    )
