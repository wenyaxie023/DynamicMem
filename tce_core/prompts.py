import json
from typing import Any, Dict, List, Optional

from tce_contracts import CURRENT_TASK_CONTRACT_VERSION


TASK_C_V2_USER_COMMUNICATION_TASK_INSTRUCTION = (
    "As the assistant, what single message should be sent to the user right now? "
    "Make it complete for this moment by using the user's routine details, not a generic reminder."
)
TASK_C_V2_INFORMATION_REQUEST_TASK_INSTRUCTION = (
    "As the assistant, complete the filtering parameters that should be sent right now. "
    "Use the user's preference statement to shape the filters, and do not write the final recommendation."
)
TASK_C_V2_ACTION_CONFIGURATION_TASK_INSTRUCTION = (
    "As the assistant, complete the action configuration that should be sent right now. "
    "Use the user's known attributes to fill the required execution fields, and do not write a message or recommendation."
)


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
{context}"""


def _render_agent_memory_section() -> str:
    return """[Memory]
    Use your memory about the user to answer the question."""


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
    snapshot_template = {k: target_value_templates.get(k, "<fill the blank>") for k in target_keys}
    evidence_template = {
        k: [{"app_log_id": "<app_log_id>", "evidence_content": "<supporting snippet>"}]
        for k in target_keys
    }
    fill_template = {
        "snapshot_state": snapshot_template,
        "evidence": evidence_template,
    }
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
  "snapshot_state": {{
    "<key>": "<value or nested object following the template>",
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

Template:
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
   - for weekly routines: include weekday + clock time;
   - for monthly or date-like routines: include calendar anchor + clock time.
8. scenario may include only:
   - the current moment,
   - whether something has or has not happened yet,
   - whether something has or has not been prepared,
   - at most one additional situational fact that plausibly matters right now.
9. scenario must not restate or paraphrase the routine action, frequency, start time, end time, location, or any other fact already present in state_value, except that it may state the current day/date/time as part of the world background.
10. reference_answer must be exactly one natural assistant-to-user message, not a meta description.
11. reference_answer must be complete enough that a fully correct answer would use every terminal field in state_value.
12. Before finalizing, silently confirm the item would later pass the same five semantic validation criteria:
   - answerability: scenario plus the fixed task_instruction define one clear current-moment communication task.
   - service_completion_quality: the item asks for one concrete assistant communication rather than recall or raw state restatement.
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
Generate exactly one preference-conditioned filtering task for an assistant that prepares structured retrieval or recommendation parameters.

Each item contains:
- `scenario`: synthesize this field.
- `task_instruction`: copy the fixed string exactly.
- `output_template`: synthesize this field.
- `reference_output`: synthesize this field as the intended correct filled object.

The task should require the assistant to use the user's preference state to fill filtering parameters, not only the scenario.
The reference_output must fill a structured filtering-parameter object for a downstream retrieval, screening, or recommendation system.
It must not be a final recommendation, a ranked list, or a free-form explanation.

[Key design goal]
This is a filtering task, not a copy-the-statement task.
The generated item should require the answering assistant to translate the user's preference statement into semantically meaningful filtering parameters.

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

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "output_template"
   - "reference_output"
4. Copy task_instruction exactly as this fixed string:
   {fixed_task_instruction}
5. scenario must be short, natural, and written as world background.
6. scenario must not use first-person or second-person wording such as "I", "we", "you", "your", or "you've".
7. scenario must make the filtering situation feel like a plausible user or product moment, not like a backend log line.
8. Prefer natural situations such as:
   - the user is about to browse options,
   - a shortlist is being prepared,
   - candidate results are being narrowed before display.
9. Avoid robotic phrasing such as:
   - "a filtering step is about to run"
   - "a screening request is about to be sent"
   - "a downstream module will execute now"
10. scenario may include only:
   - the immediate user goal or option space,
   - the fact that a shortlist or filtering pass is being prepared,
   - at most one additional situational fact that plausibly matters right now.
11. scenario must not restate or paraphrase the user's actual preference content.
12. output_template and reference_output must both be top-level JSON objects.
13. output_template and reference_output must have exactly the same nested shape.
14. Every leaf in output_template must be the string "<fill>".
15. Do not use a fixed universal key like only "preference_statement". Instead, synthesize request-facing keys and grouping that fit the domain implied by the preference statement.
16. The synthesized schema should decompose the preference into meaningful filtering dimensions when appropriate, such as:
   - preferred types or formats,
   - desired attributes,
   - required features,
   - avoided or deprioritized options,
   - priorities or goals.
17. reference_output must be a coherent canonical fill of output_template.
18. Before finalizing, silently confirm the item would later pass the same five semantic validation criteria:
   - answerability: scenario plus the fixed task_instruction define one clear current-moment structured completion task.
   - service_completion_quality: the item defines a real structured filtering task rather than free-form QA or a raw state dump.
   - full_field_dependency: a fully correct reference_output needs all non-derived field paths in state_value.
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
        "The user is deciding how to spend the next professional-development block. "
        "A shortlist of learning options is being prepared before anything is shown."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "content_acquisition_filters": {
            "preferred_modalities": ["<fill>", "<fill>"],
            "content_characteristics": ["<fill>", "<fill>", "<fill>"],
            "deprioritized_formats": ["<fill>"]
        }
    },
    "reference_output": {
        "content_acquisition_filters": {
            "preferred_modalities": [
                "technical white papers",
                "webinars"
            ],
            "content_characteristics": [
                "in-depth",
                "self-paced",
                "technical"
            ],
            "deprioritized_formats": [
                "large live conferences"
            ]
        }
    },
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
        "Candidate strategies are being narrowed before anything is surfaced."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "investment_filters": {
            "primary_objectives": ["<fill>", "<fill>"],
            "time_horizon": "<fill>",
            "deprioritized_strategies": ["<fill>"]
        }
    },
    "reference_output": {
        "investment_filters": {
            "primary_objectives": [
                "capital preservation",
                "tax-efficient growth"
            ],
            "time_horizon": "long-term",
            "deprioritized_strategies": [
                "high-risk speculative trading"
            ]
        }
    },
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
        "Nearby coffee-shop options are being narrowed before results are shown."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "venue_filters": {
            "preferred_ambience": ["<fill>"],
            "preferred_venue_types": ["<fill>"],
            "required_features": ["<fill>"],
            "deprioritized_venue_types": ["<fill>"]
        }
    },
    "reference_output": {
        "venue_filters": {
            "preferred_ambience": [
                "quiet"
            ],
            "preferred_venue_types": [
                "neighborhood coffee shops"
            ],
            "required_features": [
                "table seating"
            ],
            "deprioritized_venue_types": [
                "loud chain cafes"
            ]
        }
    },
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
- The output collapses the entire preference into one copied statement instead of a filtering-oriented decomposition.

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
        "<synthesized_filter_key>": "<fill or nested fills>"
      }}
    }},
    "reference_output": {{
      "<same synthesized shape as output_template>": "..."
    }}
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
Generate exactly one attribute-conditioned action-configuration task for an assistant that prepares structured tool or workflow payloads.

Each item contains:
- `scenario`: synthesize this field.
- `task_instruction`: copy the fixed string exactly.
- `output_template`: synthesize this field.
- `reference_output`: synthesize this field as the intended correct filled object.

The task should require the assistant to use the user's attribute state to fill execution fields, not only the scenario.
The reference_output must fill a structured action-configuration object for a downstream tool, workflow, form, or executable service.
It must not be a user-facing message, a retrieval request, or a free-form explanation.

[Key design goal]
This is an execution-configuration task, not a copy-the-attribute task.
The generated item should require the answering assistant to translate the user's known attributes into the specific fields needed to carry out an action.

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

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "output_template"
   - "reference_output"
4. Copy task_instruction exactly as this fixed string:
   {fixed_task_instruction}
5. scenario must be short, natural, and written as world background.
6. scenario must not use first-person or second-person wording such as "I", "we", "you", "your", or "you've".
7. scenario must make the execution moment feel like a plausible user or product moment, not like a backend log line.
8. Prefer natural situations such as:
   - the user is completing checkout,
   - a setup flow is being finished,
   - a profile or form is being prepared before submission,
   - a device or account connection is being configured.
9. Avoid robotic phrasing such as:
   - "an action configuration is about to be sent"
   - "a downstream workflow will execute now"
   - "a payload is being prepared for a module"
10. scenario may include only:
   - the immediate user goal or action being completed,
   - the fact that a form, setup, or execution payload is being prepared,
   - at most one additional situational fact that plausibly matters right now.
11. scenario must not restate or paraphrase the user's actual attribute values.
12. output_template and reference_output must both be top-level JSON objects.
13. output_template and reference_output must have exactly the same nested shape.
14. Every leaf in output_template must be the string "<fill>".
15. Prefer configuration-facing schemas that decompose compound attribute strings into execution-relevant fields when the decomposition is directly supported by state_value.
16. Do not invent facts that are not directly stated in state_value.
17. reference_output must preserve all grounded attribute facts needed by the synthesized configuration schema.
18. For list-valued state_value, preserve source order when the configuration represents per-item entries.
19. Before finalizing, silently confirm the item would later pass the same five semantic validation criteria:
   - answerability: scenario plus the fixed task_instruction define one clear current-moment structured completion task.
   - service_completion_quality: the item defines a real structured action-configuration task rather than free-form QA or a raw state dump.
   - full_field_dependency: a fully correct reference_output needs all non-derived field paths in state_value.
   - low_leakage: scenario does not restate or strongly imply the attribute facts that should come from state_value.
   - output_groundedness: output_template plus reference_output define a task-appropriate action-configuration object grounded in state_value rather than a raw state copy or unsupported content.

[Good Example A Input]
state_key: {json.dumps("user_attributes_state:primary_job_role", ensure_ascii=False)}
state_value: {json.dumps("Senior Coatings Consultant at PPG Industries (specializing in heavy-duty infrastructure and marine protection)", ensure_ascii=False, indent=2)}

[Good Example A Output]
{json.dumps({"item": {
    "scenario": (
        "A registration form for a technical industry symposium is being finalized. "
        "The professional credential section is being completed before attendee details are submitted."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "symposium_registration": {
            "professional_profile": {
                "job_title": "<fill>",
                "organization": "<fill>",
                "specialization_areas": ["<fill>", "<fill>"]
            }
        }
    },
    "reference_output": {
        "symposium_registration": {
            "professional_profile": {
                "job_title": "Senior Coatings Consultant",
                "organization": "PPG Industries",
                "specialization_areas": [
                    "heavy-duty infrastructure",
                    "marine protection"
                ]
            }
        }
    },
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
        "A wellness app setup is being finalized. "
        "Connected-device sources are being configured before health data syncing starts."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "wearable_sync_setup": {
            "connected_sources": [
                {
                    "device_model": "<fill>",
                    "device_variant": "<fill>",
                    "enabled_metrics": ["<fill>", "<fill>"]
                },
                {
                    "device_model": "<fill>",
                    "device_variant": "<fill>",
                    "enabled_metrics": ["<fill>", "<fill>"]
                }
            ]
        }
    },
    "reference_output": {
        "wearable_sync_setup": {
            "connected_sources": [
                {
                    "device_model": "Apple Watch Series 9",
                    "device_variant": "Midnight aluminum",
                    "enabled_metrics": [
                        "daily heart rate",
                        "step tracking"
                    ]
                },
                {
                    "device_model": "Oura Ring Gen3",
                    "device_variant": "Stealth finish",
                    "enabled_metrics": [
                        "sleep staging",
                        "recovery metrics"
                    ]
                }
            ]
        }
    },
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
        "A unified content and services hub is being connected for the user. "
        "Subscription entitlements are being prepared before linked services are shown."
    ),
    "task_instruction": fixed_task_instruction,
    "output_template": {
        "subscription_entitlements": {
            "linked_services": [
                {
                    "service_name": "<fill>",
                    "plan_or_bundle": "<fill>",
                    "usage_context": "<fill>"
                },
                {
                    "service_name": "<fill>",
                    "plan_or_bundle": "<fill>",
                    "usage_context": "<fill>"
                },
                {
                    "service_name": "<fill>",
                    "plan_or_bundle": "<fill>",
                    "usage_context": "<fill>"
                }
            ]
        }
    },
    "reference_output": {
        "subscription_entitlements": {
            "linked_services": [
                {
                    "service_name": "Audible",
                    "plan_or_bundle": "Premium Plus",
                    "usage_context": "listening to non-fiction during 45-minute commutes"
                },
                {
                    "service_name": "Disney Bundle",
                    "plan_or_bundle": "including Hulu and ESPN+",
                    "usage_context": "family entertainment and sports coverage"
                },
                {
                    "service_name": "MasterClass",
                    "plan_or_bundle": "annual subscription",
                    "usage_context": "learning technical crafting and cooking skills"
                }
            ]
        }
    },
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
        "<synthesized_execution_field>": "<fill or nested fills>"
      }}
    }},
    "reference_output": {{
      "<same synthesized shape as output_template>": "..."
    }}
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
- Fill the structured `output` object using the memory and the provided scenario.
- Preserve the required nested structure exactly.
- Do not add extra fields.
- `evidence` must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only.

[Output format]
{{
  "output": {output_template},
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
- Write one short natural-language assistant response that best fits the scenario using the memory.
- Do not return a structured payload or bullet list.
- `evidence` must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only.

[Output format]
{{
  "answer": "<one short assistant response>",
  "evidence": [
    {{
      "app_log_id": "<app_log_id>",
      "evidence_content": "<supporting snippet from the same log>"
    }}
  ]
}}"""
    return "{header}\n\n{memory_section}\n\n{instructions}\n".format(
        header=task_body,
        memory_section=memory_section,
        instructions=instructions,
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
                "criterion": "service_completion_quality",
                "analysis": "<whether the item defines a real assistant communication task instead of raw state recall>",
                "pass": "<bool>",
            },
            {
                "criterion": "full_field_dependency",
                "analysis": "<whether answering well requires all non-derived fields in the provided state>",
                "pass": "<bool>",
            },
            {
                "criterion": "low_leakage",
                "analysis": "<whether scenario and task_instruction avoid restating the key user-state facts>",
                "pass": "<bool>",
            },
            {
                "criterion": "output_groundedness",
                "analysis": "<whether each personalized part of reference_answer is grounded by the relevant fields in the provided state>",
                "pass": "<bool>",
            },
        ]
        return """[Task Instruction]
Validate whether this item is a strong current-moment assistant message task.
Judge it using the five required criteria and give one short analysis for each.
Do not output an overall verdict; only return the per-criterion judgments.

[Definitions]
- answerability: `scenario` plus `task_instruction` must define one clear communication task for the current moment. Judge this by checking whether the current moment is anchored well enough and whether the assistant can tell what kind of message should be sent now.
- service_completion_quality: the item must ask the assistant to produce one concrete user-facing communication, not merely restate the habit or answer a recall question. Judge this by checking whether the task is a real assistant action rather than state restatement.
- full_field_dependency: answering well should require all non-derived fields in `state_value`; dropping an important field path should make the communication materially weaker or incorrect. Judge this by checking which field paths in `state_value` are actually needed by the ideal message.
- low_leakage: `scenario` and `task_instruction` should describe only the local situation and current communication task; they must not restate or strongly imply the key habit facts that should come from `state_value`. Judge this by comparing `scenario` and `task_instruction` against the field paths in `state_value`.
- output_groundedness: `reference_answer` must be a short natural-language assistant response whose personalized content is grounded in `state_value`. Judge this by checking which parts of `reference_answer` are supported by which state fields, and fail if it adds unsupported user-specific facts.

[Constraints]
1. Evaluate exactly the five required criteria in this fixed order: answerability, service_completion_quality, full_field_dependency, low_leakage, output_groundedness.
2. Output exactly one object for each required criterion.
3. Use the criterion names exactly as given; do not rename, reorder, omit, or add criteria.
4. Set `pass` to true only when the criterion is clearly satisfied.
5. Keep each `analysis` concise but specific.
6. Mark `answerability` as failed if the current-moment task is vague, underspecified, or unclear about what the assistant should send now.
7. Mark `service_completion_quality` as failed if the item mainly behaves like a recall question, a raw state restatement, or a generic check-in rather than one concrete assistant communication.
8. Mark `full_field_dependency` as failed if one or more important field paths in `state_value` are unused, optionalized, or collapsible without materially changing the ideal message.
9. Mark `low_leakage` as failed if `scenario` or `task_instruction` restates or paraphrases the habit action, cadence, scheduled day, timing, location, or priority.
10. Mark `output_groundedness` as failed if `reference_answer` introduces unsupported tactics, thresholds, or user-specific facts absent from `state_value`.
11. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "habits_state:client_technical_briefing"
state_value: {{"schedule": {{"frequency_type": "weekly", "days_of_week": [0]}}, "timing": {{"start_time": "10:00"}}, "location": "regional corporate headquarters"}}
candidate_item: {{
  "scenario": "It is Monday at 09:20. Nothing has been started yet this morning.",
  "task_instruction": "Write the short reminder message the assistant should send right now.",
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
      "criterion": "service_completion_quality",
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
      "analysis": "The scenario gives only the local current moment and does not restate the routine details from the state.",
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
state_key: "habits_state:evening_walk"
state_value: {{"timing": {{"start_time": "19:00"}}, "schedule": {{"days_of_week": [2, 4, 6]}}}}
candidate_item: {{
  "scenario": "The user has an evening routine sometime this week.",
  "task_instruction": "Write the short reminder message the assistant should send right now.",
  "reference_answer": "Send a reminder that this is one of the user's regular evening walk windows and that it normally starts at 19:00."
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "answerability",
      "analysis": "The current moment is not anchored well enough because the item does not say whether it is a scheduled walk day or close to the send time now.",
      "pass": false
    }},
    {{
      "criterion": "service_completion_quality",
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
      "analysis": "The scenario stays local and does not restate the actual walk schedule details from the state.",
      "pass": true
    }},
    {{
      "criterion": "output_groundedness",
      "analysis": "The reference answer is short and grounded in the provided state without adding unsupported user facts.",
      "pass": true
    }}
  ]
}}

[Example 2]
[Example Input]
state_key: "preferences_state:learning_modality"
state_value: {{"statement": "prefers self-paced webinars"}}
candidate_item: {{
  "scenario": "A training-related workflow may run later.",
  "task_instruction": "Fill the structured request payload for the next product step.",
  "output_template": {{"request_profile": {{"preferred_format": "<fill>"}}}},
  "reference_output": {{"request_profile": {{"preferred_format": "self-paced webinars"}}}}
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "answerability",
      "analysis": "The item does not anchor a clear enough current workflow moment or specify which concrete step is being completed now.",
      "pass": false
    }},
    {{
      "criterion": "service_completion_quality",
      "analysis": "It is still framed as completing one structured service object rather than answering a free-form recall question.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "The only state field is still needed to fill the preferred format in the payload.",
      "pass": true
    }},
    {{
      "criterion": "low_leakage",
      "analysis": "The scenario stays abstract and does not restate the user's actual learning preference.",
      "pass": true
    }},
    {{
      "criterion": "output_groundedness",
      "analysis": "The structured output is task-appropriate and the filled value is grounded in the preference statement.",
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
        )
    criteria_template = [
        {
            "criterion": "answerability",
            "analysis": "<whether scenario plus task_instruction define one clear current-moment structured completion task that can be answered from the provided state>",
            "pass": "<bool>",
        },
        {
            "criterion": "service_completion_quality",
            "analysis": "<whether the scenario and task instruction define a real structured service-completion task>",
            "pass": "<bool>",
        },
        {
            "criterion": "full_field_dependency",
            "analysis": "<whether the item requires all non-derived fields in the provided state>",
            "pass": "<bool>",
        },
        {
            "criterion": "low_leakage",
            "analysis": "<whether scenario and task_instruction avoid restating the key user-state facts that should come from state_value>",
            "pass": "<bool>",
        },
        {
            "criterion": "output_groundedness",
            "analysis": "<whether each required part of reference_output is grounded by the relevant fields in the provided state and forms a task-appropriate service object>",
            "pass": "<bool>",
        },
    ]
    return """[Task Instruction]
Validate whether this item is a strong structured completion task.
Judge it using the five required criteria and give one short analysis for each.
Do not output an overall verdict; only return the per-criterion judgments.

[Definitions]
- answerability: `scenario` plus `task_instruction` must define one clear structured completion task for the current moment. Judge this by checking whether the current product moment is clear enough and whether the assistant can tell what object should be completed now.
- service_completion_quality: the item must define a real structured service-completion task, not a free-form recall question or a raw state dump. Judge this by checking whether the task really asks for a service object to be completed.
- full_field_dependency: the item must require all non-derived fields in `state_value`; dropping any required field path should make the service object incomplete or materially worse. Judge this by checking which field paths in `state_value` are actually needed by the ideal `reference_output`.
- low_leakage: `scenario` and `task_instruction` should describe only the local product/service situation and current completion task; they must not restate or strongly imply the key user-state facts that should come from `state_value`. Judge this by comparing `scenario` and `task_instruction` against the field paths in `state_value`.
- output_groundedness: `output_template` plus `reference_output` must define one task-appropriate structured service object grounded in `state_value`. Judge this by checking which required parts of `reference_output` are supported by which state fields, and fail if the item merely copies the raw state schema or invents unsupported content.

[Constraints]
1. Evaluate exactly the five required criteria in this fixed order: answerability, service_completion_quality, full_field_dependency, low_leakage, output_groundedness.
2. Output exactly one object for each required criterion.
3. Use the criterion names exactly as given; do not rename, reorder, omit, or add criteria.
4. Set `pass` to true only when the criterion is clearly satisfied.
5. Keep each `analysis` concise but specific.
6. Mark `answerability` as failed if the current-moment task is vague, underspecified, or unclear about what structured object should be completed now.
7. Mark `service_completion_quality` as failed if the item mainly behaves like a free-form QA question rather than a structured completion task.
8. Mark `full_field_dependency` as failed if some required part of `state_value` is unused, optionalized, or collapsed away.
9. Mark `low_leakage` as failed if `scenario` or `task_instruction` restates or paraphrases the key preference or attribute facts that should come from `state_value`.
10. Mark `output_groundedness` as failed if the item simply mirrors the raw state schema, fails to produce a top-level service object, or invents unsupported state content in `reference_output`.
11. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "preferences_state:learning_modality"
state_value: {{"statement": "Prefers in-depth, self-paced technical white papers and webinars over large live conferences"}}
candidate_item: {{
  "scenario": "The user is reviewing learning options for the next professional-development block. A shortlist is being prepared before results are shown.",
  "task_instruction": "Fill the structured request payload before the search is sent.",
  "output_template": {{"content_acquisition_filters": {{"preferred_modalities": ["<fill>", "<fill>"], "content_characteristics": ["<fill>", "<fill>", "<fill>"], "deprioritized_formats": ["<fill>"]}}}},
  "reference_output": {{"content_acquisition_filters": {{"preferred_modalities": ["technical white papers", "webinars"], "content_characteristics": ["in-depth", "self-paced", "technical"], "deprioritized_formats": ["large live conferences"]}}}}
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "answerability",
      "analysis": "The current workflow moment and the structured filtering step are clear enough that one bounded payload can be completed now.",
      "pass": true
    }},
    {{
      "criterion": "service_completion_quality",
      "analysis": "The item is framed as completing one structured filtering object rather than answering a free-form recall question.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "The preference statement is needed to derive the preferred modalities, desired content characteristics, and deprioritized formats.",
      "pass": true
    }},
    {{
      "criterion": "low_leakage",
      "analysis": "The scenario describes the local product moment without restating the user's actual learning preferences.",
      "pass": true
    }},
    {{
      "criterion": "output_groundedness",
      "analysis": "The structured output is a synthesized filtering object rather than a raw state copy, and each filled value is grounded in the preference statement.",
      "pass": true
    }}
  ]
}}

[Example 2]
[Example Input]
state_key: "preferences_state:learning_modality"
state_value: {{"statement": "prefers self-paced webinars"}}
candidate_item: {{
  "scenario": "A training-related workflow may run later.",
  "task_instruction": "Fill the structured request payload for the next product step.",
  "output_template": {{"request_profile": {{"preferred_format": "<fill>"}}}},
  "reference_output": {{"request_profile": {{"preferred_format": "self-paced webinars"}}}}
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "answerability",
      "analysis": "The item does not anchor a clear enough current workflow moment or specify which concrete step is being completed now.",
      "pass": false
    }},
    {{
      "criterion": "service_completion_quality",
      "analysis": "It is still framed as completing one structured service object rather than answering a free-form recall question.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "The only state field is still needed to fill the preferred format in the payload.",
      "pass": true
    }},
    {{
      "criterion": "low_leakage",
      "analysis": "The scenario stays abstract and does not restate the user's actual learning preference.",
      "pass": true
    }},
    {{
      "criterion": "output_groundedness",
      "analysis": "The structured output is task-appropriate and the filled value is grounded in the preference statement.",
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
    "output_template": {output_template},
    "reference_output": {reference_output}
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
        output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        reference_output=json.dumps(reference_output, ensure_ascii=False, indent=2),
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
                "reason_analysis": "<why this exact field value is or is not inferable from evidence>",
                "is_valid": "<bool>",
            }
            for path in normalized_fields
        }
    }
    return """[Task Instruction]
Validate whether this state is inferable from evidence.
Evaluate field-level inferability.
Success means each candidate field is marked valid only when the target value for that field is supported by evidence.

[Definitions]
- state_key: the state item being validated.
- state_value: the structured target value for this state_key at checkpoint time.
- candidate_field_paths: field paths to evaluate for inferability. The target value for each path is found inside state_value.
- evidence_logs: related app logs up to checkpoint time; these are the primary evidence.
- field_verdicts: an object keyed exactly by candidate field path. Each value is one field-level decision with `{{reason_analysis, is_valid}}`.

[Constraints]
1. Evaluate each path in `candidate_field_paths` independently.
2. Output exactly one `field_verdicts` entry per candidate field path.
3. Use exactly the field keys shown in the output template; do not add, remove, or rename field keys.
4. For each candidate field, compare the target value in `state_value` against the evidence logs.
5. Set `is_valid=true` only if evidence logs support inferring that exact field value with high confidence.
6. If evidence is missing, ambiguous, contradicted, or supports only a weaker/generalized value, set `is_valid=false` and explain why in `reason_analysis`.
7. Output JSON ONLY with no markdown and no extra keys.

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
      "reason_analysis": "The evidence explicitly says the budget review reminder is recurring on Sunday, matching days_of_week=[6].",
      "is_valid": true
    }},
    "timing.start_time": {{
      "reason_analysis": "The evidence explicitly gives the reminder time as 09:00, matching the target start_time.",
      "is_valid": true
    }},
    "timing.end_time": {{
      "reason_analysis": "The evidence does not provide a duration or explicit end time, so 10:00 is not supported.",
      "is_valid": false
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
    failed_rules: List[str],
    semantic_criteria: List[Dict[str, Any]],
    reference_answer: str = "",
) -> str:
    normalized_family = str(service_family or "").strip()
    validation_feedback = {
        "failed_rules": list(failed_rules or []),
        "criteria": list(semantic_criteria or []),
    }
    if normalized_family == "user_communication":
        return """[Task Instruction]
Rewrite the invalid item using the validation_feedback.
Return a JSON delta patch over mutable fields only.

[Definitions]
- validation_feedback: validator feedback containing `failed_rules` and per-criterion `criteria`.
- failed_rules: the names of the checks that failed and must be fixed.
- criteria: feedback for each criterion explaining what passed, what failed, and why.
- answerability: `scenario` plus the fixed `task_instruction` define one clear current-moment communication task.
- service_completion_quality: the item asks for one concrete assistant communication rather than a recall question, a raw state restatement, or a generic check-in.
- full_field_dependency: a good answer depends on all important non-derived field paths in `state_value`.
- low leakage: `scenario` does not restate, paraphrase, or strongly imply the habit facts that should come from `state_value`.
- output_groundedness: a short natural-language assistant response whose key personalized content is supported by `state_value`.

[Repair Instructions]
- If `answerability` failed: rewrite `scenario` so the current moment is clear enough and it is clear what communication should be sent now.
- If `service_completion_quality` failed: rewrite the item so it asks for one concrete assistant communication rather than a recall question, raw state restatement, or generic check-in.
- If `full_field_dependency` failed: rewrite `scenario` and/or `reference_answer` so a good answer depends on all important state fields.
- If `low_leakage` failed: remove any restatement of the habit action, cadence, scheduled day, timing, location, or priority from `scenario`.
- If `output_groundedness` failed: rewrite `reference_answer` so its personalized content is grounded in `state_value` without adding unsupported user-specific facts.

[Constraints]
1. Rewrite only the mutable item fields:
   - `scenario`
   - `reference_answer`
2. Keep the item in natural-language assistant-response form; do not rewrite it into a structured payload.
3. Fix every failed rule and every failed semantic criterion.
4. Apply the repair instruction for each failed rule shown in validation_feedback.
5. Include only the fields you actually changed; omit unchanged fields.
6. Do not add any keys other than:
   - `scenario`
   - `reference_answer`
7. Return JSON only.

[Example]
[Example Input]
state_key: "habits_state:family_dinner"
state_value: {{"schedule": {{"days_of_week": [0]}}, "timing": {{"start_time": "17:30"}}}}
invalid_item: {{
  "scenario": "It is 16:45. This is the user's Sunday family dinner, and nothing has been prepared yet.",
  "task_instruction": "Write the short reminder message the assistant should send right now.",
  "reference_answer": "Your Sunday family dinner starts soon, so it is a good time to begin getting things ready."
}}
validation_feedback: {{
  "failed_rules": ["answerability", "low_leakage"],
  "criteria": [
    {{"criterion": "answerability", "analysis": "The item never makes clear what should be sent right now.", "pass": false}},
    {{"criterion": "service_completion_quality", "analysis": "The item asks for one communication action.", "pass": true}},
    {{"criterion": "full_field_dependency", "analysis": "The state fields are mostly used.", "pass": true}},
    {{"criterion": "low_leakage", "analysis": "The scenario repeats that this is the user's Sunday family dinner.", "pass": false}},
    {{"criterion": "output_groundedness", "analysis": "The answer stays grounded once the setup is clarified.", "pass": true}}
  ]
}}

[Example Output]
{{
  "scenario": "It is 16:45. Everyone is home, and nothing has been prepared yet."
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
        )
    return """[Task Instruction]
Rewrite the invalid item using the validation_feedback.
Return a JSON delta patch over mutable fields only.

[Definitions]
- validation_feedback: validator feedback containing `failed_rules` and per-criterion `criteria`.
- failed_rules: the names of the checks that failed and must be fixed.
- criteria: feedback for each criterion explaining what passed, what failed, and why.
- answerability: `scenario` plus the fixed `task_instruction` define one clear current-moment structured completion task.
- service_completion_quality: the item defines a real structured service-completion task rather than a free-form QA question or raw state dump.
- full_field_dependency: every required non-derived field path in `state_value` is needed by the ideal structured completion.
- low leakage: `scenario` does not restate, paraphrase, or strongly imply the user-state facts that should come from `state_value`.
- output_groundedness: `output_template` plus `reference_output` define a task-appropriate service object grounded in `state_value`.

[Repair Instructions]
- If `answerability` failed: rewrite `scenario` so it is clear what structured object should be completed now.
- If `service_completion_quality` failed: rewrite `scenario`, `output_template`, and/or `reference_output` so the item becomes a real structured service-completion task.
- If `full_field_dependency` failed: rewrite the service object so every required part of `state_value` is needed by the ideal structured completion.
- If `low_leakage` failed: remove any restatement of key preference or attribute facts from `scenario`.
- If `output_groundedness` failed: repair `output_template` and `reference_output` so they form a task-appropriate service object rather than a raw state copy, with every required output value grounded in `state_value`.

[Constraints]
1. Rewrite only the mutable item fields:
   - `scenario`
   - `output_template`
   - `reference_output`
2. `output_template` and `reference_output` must remain top-level structured service objects appropriate for this task type.
3. Fix every failed rule and every failed semantic criterion.
4. Apply the repair instruction for each failed rule shown in validation_feedback.
5. Include only the fields you actually changed; omit unchanged fields.
6. Do not add any keys other than:
   - `scenario`
   - `output_template`
   - `reference_output`
7. Return JSON only.

[Example]
[Example Input]
state_key: "preferences_state:learning_modality"
state_value: {{"statement": "prefers self-paced webinars"}}
invalid_item: {{
  "scenario": "A training-resource search request is about to run.",
  "task_instruction": "Fill the structured request payload before the search is sent.",
  "output_template": {{"request_profile": {{"preference_statement": "<fill>"}}}},
  "reference_output": {{"request_profile": {{"preference_statement": "prefers self-paced webinars"}}}}
}}
validation_feedback: {{
  "failed_rules": ["answerability", "output_groundedness"],
  "criteria": [
    {{"criterion": "answerability", "analysis": "The item does not make clear what payload should be completed now.", "pass": false}},
    {{"criterion": "service_completion_quality", "analysis": "The item is already framed as a structured completion task.", "pass": true}},
    {{"criterion": "full_field_dependency", "analysis": "The only state field is required.", "pass": true}},
    {{"criterion": "low_leakage", "analysis": "The scenario does not restate the preference.", "pass": true}},
    {{"criterion": "output_groundedness", "analysis": "The current payload still behaves too much like a raw state copy.", "pass": false}}
  ]
}}

[Example Output]
{{
  "output_template": {{"request_profile": {{"preferred_format": "<fill>"}}}},
  "reference_output": {{"request_profile": {{"preferred_format": "self-paced webinars"}}}}
}}

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- invalid_item: {{
    "scenario": {scenario},
    "task_instruction": {task_instruction},
    "output_template": {output_template},
    "reference_output": {reference_output}
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
        output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        reference_output=json.dumps(reference_output, ensure_ascii=False, indent=2),
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
