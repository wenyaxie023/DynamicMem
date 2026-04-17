import json
from typing import Any, Dict, List, Optional

from tce_contracts import CURRENT_TASK_CONTRACT_VERSION, normalize_task_c_source_value


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
    if str(state_type or "").strip() != "habit":
        raise ValueError("Task C v2 user_communication prompt requires state_type='habit'.")

    fixed_task_instruction = (
        "As the assistant, what single message should be sent to the user right now? "
        "Make it complete for this moment by using the user's routine details, not a generic reminder."
    )

    return """[Task]
Generate exactly one low-leakage benchmark item for a habit-conditioned User Communication task.

The answering assistant will see:
- state_value
- scenario
- task_instruction

Your job is to create an item where a fully correct answer requires using the habit information in state_value, not just the scenario.

[What the answering assistant must do]
The answer must be exactly one proactive assistant-to-user message for this moment.
It must be free-form natural language.

[Definitions]
- terminal field: one leaf field in state_value whose value is scalar or array-valued and not further decomposed.
- leaf path: the path to a terminal field using dot notation, for example: schedule.days_of_week or timing.start_time.
- low leakage: scenario does not restate, paraphrase, or strongly imply user-state facts that should instead be recovered from state_value.
- world-background scenario: a short third-person description of what is true right now in the world. It is not spoken by the assistant, not spoken by the user, and not written from the user's point of view.

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "reference_answer"
   - "scoring_rubric"
4. task_instruction must be exactly this string:
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
9. scenario must not restate or paraphrase the routine action, frequency, scheduled day, start time, end time, location, priority, or any other fact already present in state_value, except that it may state the current day/date/time as part of the world background.
10. reference_answer must be exactly one natural assistant-to-user message, not a meta description.
11. reference_answer must be complete enough that a fully correct answer would satisfy all rubric criteria.
12. scoring_rubric must be a JSON object with exactly one key: "criteria".
13. scoring_rubric.criteria must be a list of criterion objects.
14. Each criterion object must have exactly these keys:
   - "id"
   - "description"
15. There must be exactly one correctness criterion for each terminal field in state_value.
16. Each criterion must explicitly name the corresponding leaf path in its id or description.
17. Do not add global style, naturalness, service-family-fit, or generic quality criteria.
18. Before finalizing, silently check:
   - if state_value were hidden, the task would become substantially underdetermined;
   - every terminal field in state_value is necessary for full credit;
   - the scenario does not leak the state;
   - the current moment is anchored clearly enough to make the item answerable.

[Good Example A Input]
state_key: "habits_state:home_strength_training"
state_value: {{
  "action": "home_strength_training",
  "frequency": "3 times per week",
  "week_of_day": ["Monday", "Wednesday", "Friday"],
  "start_time": "07:00"
}}

[Good Example A Output]
{{
  "item": {{
    "scenario": "It is Wednesday at 06:50. Nothing has been started yet this morning.",
    "task_instruction": {fixed_task_instruction},
    "reference_answer": "Your home strength session starts at 7:00, and Wednesday is one of your Monday-Wednesday-Friday workout days. Getting started on time will keep this three-times-a-week routine on track.",
    "scoring_rubric": {{
      "criteria": [
        {{
          "id": "action",
          "description": "The message correctly uses state_value.action by identifying the routine as home_strength_training rather than a different or generic activity."
        }},
        {{
          "id": "frequency",
          "description": "The message correctly uses state_value.frequency by making clear that this is a three-times-per-week routine."
        }},
        {{
          "id": "week_of_day",
          "description": "The message correctly uses state_value.week_of_day by making clear that the current Wednesday moment matches the Monday-Wednesday-Friday schedule."
        }},
        {{
          "id": "start_time",
          "description": "The message correctly uses state_value.start_time by relating the current 06:50 moment to the 07:00 start."
        }}
      ]
    }}
  }}
}}

[Good Example B Input]
state_key: "habits_state:sunday_family_dinner"
state_value: {{
  "action": "sunday_family_dinner",
  "frequency": "weekly",
  "week_of_day": ["Sunday"],
  "start_time": "18:30"
}}

[Good Example B Output]
{{
  "item": {{
    "scenario": "It is Sunday at 16:45. Everyone is home, and nothing has been prepared yet.",
    "task_instruction": {fixed_task_instruction},
    "reference_answer": "It’s Sunday, and your weekly family dinner is at 6:30 tonight. Now is a good time to get the plan settled so the evening stays easy.",
    "scoring_rubric": {{
      "criteria": [
        {{
          "id": "action",
          "description": "The message correctly uses state_value.action by identifying the routine as sunday_family_dinner rather than a generic meal or unrelated event."
        }},
        {{
          "id": "frequency",
          "description": "The message correctly uses state_value.frequency by making clear that this is a weekly routine."
        }},
        {{
          "id": "week_of_day",
          "description": "The message correctly uses state_value.week_of_day by making clear that the current Sunday moment is the scheduled day for the routine."
        }},
        {{
          "id": "start_time",
          "description": "The message correctly uses state_value.start_time by connecting the current 16:45 moment to the 18:30 dinner time."
        }}
      ]
    }}
  }}
}}

[Good Example C Input]
state_key: "habits_state:investment_transfer"
state_value: {{
  "action": "investment_transfer",
  "frequency": "once per month",
  "week_of_day": ["day 1 of each month"],
  "start_time": "09:00"
}}

[Good Example C Output]
{{
  "item": {{
    "scenario": "It is the 2nd day of the month at 08:30. Nothing was completed yesterday during the expected window.",
    "task_instruction": {fixed_task_instruction},
    "reference_answer": "Yesterday’s 9:00 investment transfer for your first-of-the-month routine didn’t happen. It’s worth taking care of it this morning so the monthly habit doesn’t slip.",
    "scoring_rubric": {{
      "criteria": [
        {{
          "id": "action",
          "description": "The message correctly uses state_value.action by identifying the routine as investment_transfer rather than a generic finance task."
        }},
        {{
          "id": "frequency",
          "description": "The message correctly uses state_value.frequency by making clear that this is a once-per-month routine."
        }},
        {{
          "id": "week_of_day",
          "description": "The message correctly uses state_value.week_of_day by making clear that the missed routine belongs to day 1 of each month."
        }},
        {{
          "id": "start_time",
          "description": "The message correctly uses state_value.start_time by referring to the missed 09:00 transfer window."
        }}
      ]
    }}
  }}
}}

[Bad Example — Do Not Imitate]
{{
  "item": {{
    "scenario": "It is 07:50, and you've just sat down at your desk for the morning.",
    "task_instruction": {fixed_task_instruction},
    "reference_answer": "Your weekly review starts soon.",
    "scoring_rubric": {{
      "criteria": [
        {{
          "id": "frequency",
          "description": "The message mentions that the routine is weekly."
        }}
      ]
    }}
  }}
}}

Why the bad example fails:
- The scenario is written as if the assistant were inhabiting the user's perspective.
- The current moment is weakly grounded.
- The reference answer is too generic.
- The rubric is incomplete and does not cover every terminal field.

[Input]
- checkpoint_timestamp: {checkpoint_timestamp}
- state_key: {state_key}
- state_value: {state_value}

[Output JSON ONLY]
{{
  "item": {{
    "scenario": "...",
    "task_instruction": {fixed_task_instruction},
    "reference_answer": "...",
    "scoring_rubric": {{
      "criteria": [
        {{
          "id": "...",
          "description": "..."
        }}
      ]
    }}
  }}
}}
""".format(
        checkpoint_timestamp=json.dumps(str(checkpoint_timestamp or ""), ensure_ascii=False),
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
        fixed_task_instruction=json.dumps(fixed_task_instruction, ensure_ascii=False),
    )
def _build_task_c_v2_information_request_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_type: str,
    state_key: str,
    state_value: Any,
    service_family: str,
) -> str:
    if str(state_type or "").strip() != "preference":
        raise ValueError("Task C v2 information_request_construction prompt requires state_type='preference'.")

    fixed_task_instruction = (
        "As the assistant, complete the filtering parameters that should be sent right now. "
        "Use the user's preference statement to shape the filters, and do not write the final recommendation."
    )

    example_state = {
        "statement": "Prefers in-depth, self-paced technical white papers and webinars over large live conferences"
    }
    example_output_template = {
        "content_acquisition_filters": {
            "preferred_modalities": ["<fill>", "<fill>"],
            "content_characteristics": ["<fill>", "<fill>", "<fill>"],
            "deprioritized_formats": ["<fill>"]
        }
    }
    example_reference_output = {
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
    }

    example_alt_state = {
        "statement": "Prefers long-term capital preservation and tax-efficient growth over high-risk speculative trading"
    }
    example_alt_output_template = {
        "investment_filters": {
            "primary_objectives": ["<fill>", "<fill>"],
            "time_horizon": "<fill>",
            "deprioritized_strategies": ["<fill>"]
        }
    }
    example_alt_reference_output = {
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
    }

    example_third_state = {
        "statement": "Prefers quiet neighborhood coffee shops with table seating over loud chain cafes"
    }
    example_third_output_template = {
        "venue_filters": {
            "preferred_ambience": ["<fill>"],
            "preferred_venue_types": ["<fill>"],
            "required_features": ["<fill>"],
            "deprioritized_venue_types": ["<fill>"]
        }
    }
    example_third_reference_output = {
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
    }

    return """[Task]
Generate exactly one low-leakage benchmark item for a preference-conditioned Information Request Construction task.

The answering assistant will see:
- state_value
- scenario
- task_instruction
- output_template

Your job is to create an item where a fully correct answer requires using the preference information in state_value, not just the scenario.

[What the answering assistant must do]
The answer must fill a structured filtering-parameter object for a downstream retrieval, screening, or recommendation system.
It must not be a final recommendation, a ranked list, or a free-form explanation.

[Key design goal]
This is a filtering task, not a copy-the-statement task.
The generated item should require the answering assistant to translate the user's preference statement into semantically meaningful filtering parameters.

Schema synthesis is allowed.
Scoring-unit synthesis is not allowed.

That means:
- output_template should use synthesized, request-facing keys;
- reference_output should be one coherent canonical fill of that template;
- scoring_rubric must align one-to-one with the scalar leaves in reference_output after flattening arrays and objects.

[Definitions]
- preference statement: the value in state_value.statement.
- low leakage: scenario does not restate, paraphrase, or strongly imply the user's actual preference content.
- world-background scenario: a short third-person or neutral description of what is happening right now in the product or assistant context. It is not spoken by the assistant, not spoken by the user, and not written from the user's point of view.
- scalar leaf: one scalar value in reference_output after recursively flattening nested objects and arrays.
- canonical reference_output: one valid answer, but not necessarily the only valid answer.
- one-to-one rubric alignment: every scalar leaf in reference_output must have exactly one corresponding scoring criterion, and every scoring criterion must correspond to exactly one scalar leaf in reference_output.

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "output_template"
   - "reference_output"
   - "scoring_rubric"
4. task_instruction must be exactly this string:
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
18. scoring_rubric must be a JSON object with exactly one key: "criteria".
19. scoring_rubric.criteria must be a list of criterion objects.
20. Each criterion object must have exactly these keys:
   - "path"
   - "canonical_value"
   - "description"
21. scoring_rubric must satisfy one-to-one rubric alignment:
   - every scalar leaf in reference_output must appear exactly once in scoring_rubric.criteria;
   - every criterion must correspond to exactly one scalar leaf in reference_output.
22. path must be the exact path to the corresponding scalar leaf in reference_output, including array indices when needed.
23. canonical_value must be exactly the scalar value found at that path in reference_output.
24. description must explain how this canonical_value links back to the original preference statement and what semantic role it plays in filtering.
25. description must not merely restate canonical_value mechanically. It should explain why this slot exists as a decomposition of the user's stated preference.
26. Do not add grouped semantic criteria that cover multiple reference_output leaves at once.
27. Do not add global style, naturalness, usefulness, or generic quality criteria.
28. Before finalizing, silently check:
   - if state_value were hidden, the task would become substantially underdetermined;
   - the scenario does not leak the preference;
   - the scenario feels like a natural product moment;
   - the synthesized schema is domain-appropriate rather than generic;
   - the rubric aligns exactly one-to-one with reference_output leaves;
   - each description explains the semantic link between the leaf and the original preference statement.

[Good Example A Input]
state_key: "preferences_state:learning_modality"
state_value: {example_state}

[Good Example A Output]
{{
  "item": {{
    "scenario": "The user is deciding how to spend the next professional-development block. A shortlist of learning options is being prepared before anything is shown.",
    "task_instruction": {fixed_task_instruction},
    "output_template": {example_output_template},
    "reference_output": {example_reference_output},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "content_acquisition_filters.preferred_modalities[0]",
          "canonical_value": "technical white papers",
          "description": "This leaf captures one of the specific formats the user actively prefers, separating preferred learning modalities from broader qualities like depth or pacing."
        }},
        {{
          "path": "content_acquisition_filters.preferred_modalities[1]",
          "canonical_value": "webinars",
          "description": "This leaf captures another preferred modality from the statement, showing that the filter should include webinar-style options rather than collapsing everything into a single generic content preference."
        }},
        {{
          "path": "content_acquisition_filters.content_characteristics[0]",
          "canonical_value": "in-depth",
          "description": "This leaf represents the desired level of substance in the content, translating the user's preference into a filtering characteristic rather than a format choice."
        }},
        {{
          "path": "content_acquisition_filters.content_characteristics[1]",
          "canonical_value": "self-paced",
          "description": "This leaf represents the user's preference for content that can be consumed on their own schedule, which is a pacing constraint rather than a modality."
        }},
        {{
          "path": "content_acquisition_filters.content_characteristics[2]",
          "canonical_value": "technical",
          "description": "This leaf captures the technical nature of the desired material, linking the preference statement to filtering for more specialized content."
        }},
        {{
          "path": "content_acquisition_filters.deprioritized_formats[0]",
          "canonical_value": "large live conferences",
          "description": "This leaf captures the option type that should be filtered out or pushed down because the statement explicitly contrasts the user's preferred formats against large live conferences."
        }}
      ]
    }}
  }}
}}

[Good Example B Input]
state_key: "preferences_state:capital_allocation"
state_value: {example_alt_state}

[Good Example B Output]
{{
  "item": {{
    "scenario": "The user is reviewing investment options for an upcoming planning session. Candidate strategies are being narrowed before anything is surfaced.",
    "task_instruction": {fixed_task_instruction},
    "output_template": {example_alt_output_template},
    "reference_output": {example_alt_reference_output},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "investment_filters.primary_objectives[0]",
          "canonical_value": "capital preservation",
          "description": "This leaf captures one of the user's main portfolio objectives, separating preservation from other investment goals that might otherwise dominate the shortlist."
        }},
        {{
          "path": "investment_filters.primary_objectives[1]",
          "canonical_value": "tax-efficient growth",
          "description": "This leaf captures the growth objective the user wants, but specifically in a tax-efficient form, so the filter does more than represent generic growth-seeking behavior."
        }},
        {{
          "path": "investment_filters.time_horizon",
          "canonical_value": "long-term",
          "description": "This leaf captures the temporal framing implied by the statement, which should affect which candidate strategies are considered relevant."
        }},
        {{
          "path": "investment_filters.deprioritized_strategies[0]",
          "canonical_value": "high-risk speculative trading",
          "description": "This leaf captures the kind of strategy that should be screened out or pushed down because the user's preference explicitly rejects speculative, high-risk approaches."
        }}
      ]
    }}
  }}
}}

[Good Example C Input]
state_key: "preferences_state:coffee_shop_style"
state_value: {example_third_state}

[Good Example C Output]
{{
  "item": {{
    "scenario": "The user is choosing a place for a casual conversation later today. Nearby coffee-shop options are being narrowed before results are shown.",
    "task_instruction": {fixed_task_instruction},
    "output_template": {example_third_output_template},
    "reference_output": {example_third_reference_output},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "venue_filters.preferred_ambience[0]",
          "canonical_value": "quiet",
          "description": "This leaf captures the atmosphere the user wants the shortlist to favor, separating ambience from venue type or seating requirements."
        }},
        {{
          "path": "venue_filters.preferred_venue_types[0]",
          "canonical_value": "neighborhood coffee shops",
          "description": "This leaf captures the kind of venue the user prefers, turning the statement into a venue-type filter rather than a generic coffee-shop preference."
        }},
        {{
          "path": "venue_filters.required_features[0]",
          "canonical_value": "table seating",
          "description": "This leaf captures the seating feature that should materially affect which venues survive filtering, rather than being treated as incidental detail."
        }},
        {{
          "path": "venue_filters.deprioritized_venue_types[0]",
          "canonical_value": "loud chain cafes",
          "description": "This leaf captures the class of venues that should be screened out or pushed down because the user's preference explicitly contrasts against them."
        }}
      ]
    }}
  }}
}}

[Bad Example — Do Not Imitate]
{{
  "item": {{
    "scenario": "The user prefers quiet neighborhood coffee shops over loud chain cafes, and a shortlist is being prepared.",
    "task_instruction": {fixed_task_instruction},
    "output_template": {{
      "filtering_params": {{
        "preference_statement": "<fill>"
      }}
    }},
    "reference_output": {{
      "filtering_params": {{
        "preference_statement": "Prefers quiet neighborhood coffee shops with table seating over loud chain cafes"
      }}
    }},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "filtering_params.preference_statement",
          "canonical_value": "Prefers quiet neighborhood coffee shops with table seating over loud chain cafes",
          "description": "This leaf repeats the whole statement."
        }}
      ]
    }}
  }}
}}

Why the bad example fails:
- The scenario leaks the user's actual preference.
- The schema is not synthesized.
- The rubric description does not explain a semantic decomposition; it just notes that the full statement was repeated.

[Input]
- checkpoint_timestamp: {checkpoint_timestamp}
- state_key: {state_key}
- state_value: {state_value}

[Output JSON ONLY]
{{
  "item": {{
    "scenario": "...",
    "task_instruction": {fixed_task_instruction},
    "output_template": {{
      "<synthesized_request_key>": {{
        "<synthesized_filter_key>": "<fill or nested fills>"
      }}
    }},
    "reference_output": {{
      "<same synthesized shape as output_template>": "..."
    }},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "...",
          "canonical_value": "...",
          "description": "..."
        }}
      ]
    }}
  }}
}}
""".format(
        checkpoint_timestamp=json.dumps(str(checkpoint_timestamp or ""), ensure_ascii=False),
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
        fixed_task_instruction=json.dumps(fixed_task_instruction, ensure_ascii=False),
        example_state=json.dumps(example_state, ensure_ascii=False, indent=2),
        example_output_template=json.dumps(example_output_template, ensure_ascii=False, indent=2),
        example_reference_output=json.dumps(example_reference_output, ensure_ascii=False, indent=2),
        example_alt_state=json.dumps(example_alt_state, ensure_ascii=False, indent=2),
        example_alt_output_template=json.dumps(example_alt_output_template, ensure_ascii=False, indent=2),
        example_alt_reference_output=json.dumps(example_alt_reference_output, ensure_ascii=False, indent=2),
        example_third_state=json.dumps(example_third_state, ensure_ascii=False, indent=2),
        example_third_output_template=json.dumps(example_third_output_template, ensure_ascii=False, indent=2),
        example_third_reference_output=json.dumps(example_third_reference_output, ensure_ascii=False, indent=2),
    )
def _build_task_c_v2_action_configuration_question_pack_prompt(
    *,
    checkpoint_timestamp: str,
    state_type: str,
    state_key: str,
    state_value: Any,
    service_family: str,
) -> str:
    if str(state_type or "").strip() != "attribute":
        raise ValueError("Task C v2 action_configuration prompt requires state_type='attribute'.")

    # Kept for call-site compatibility; intentionally unused in the prompt.
    _ = service_family

    fixed_task_instruction = (
        "As the assistant, complete the action configuration that should be sent right now. "
        "Use the user's known attributes to fill the required execution fields, and do not write a message or recommendation."
    )

    example_state = "Senior Coatings Consultant at PPG Industries (specializing in heavy-duty infrastructure and marine protection)"
    example_output_template = {
        "symposium_registration": {
            "professional_profile": {
                "job_title": "<fill>",
                "organization": "<fill>",
                "specialization_areas": ["<fill>", "<fill>"]
            }
        }
    }
    example_reference_output = {
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
    }

    example_alt_state = [
        "Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)",
        "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)"
    ]
    example_alt_output_template = {
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
    }
    example_alt_reference_output = {
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
    }

    example_third_state = [
        "Audible Premium Plus (used for listening to non-fiction during 45-minute commutes)",
        "Disney Bundle including Hulu and ESPN+ (family entertainment and sports coverage)",
        "MasterClass (annual subscription used for learning technical crafting and cooking skills)"
    ]
    example_third_output_template = {
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
    }
    example_third_reference_output = {
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
    }

    return """[Task]
Generate exactly one low-leakage benchmark item for an attribute-conditioned Action Configuration task.

The answering assistant will see:
- state_value
- scenario
- task_instruction
- output_template

Your job is to create an item where a fully correct answer requires using the attribute information in state_value, not just the scenario.

[What the answering assistant must do]
The answer must fill a structured action-configuration object for a downstream tool, workflow, form, or executable service.
It must not be a user-facing message, a retrieval request, or a free-form explanation.

[Key design goal]
This is an execution-configuration task, not a copy-the-attribute task.
The generated item should require the answering assistant to translate the user's known attributes into the specific fields needed to carry out an action.

Schema synthesis is allowed.
Scoring-unit synthesis is not allowed.

That means:
- output_template should use synthesized, configuration-facing keys;
- reference_output should be one coherent canonical fill of that template;
- scoring_rubric must align one-to-one with the scalar leaves in reference_output after flattening arrays and objects.

[Definitions]
- attribute value: the information contained in state_value.
- low leakage: scenario does not restate, paraphrase, or strongly imply the actual attribute values in state_value.
- world-background scenario: a short third-person or neutral description of what is happening right now in the product or assistant context. It is not spoken by the assistant, not spoken by the user, and not written from the user's point of view.
- scalar leaf: one scalar value in reference_output after recursively flattening nested objects and arrays.
- canonical reference_output: one valid answer, but not necessarily the only valid answer.
- grounded decomposition: a raw attribute string may be split into multiple configuration leaves only when each resulting leaf is directly supported by the wording of state_value and serves a distinct execution role.
- one-to-one rubric alignment: every scalar leaf in reference_output must have exactly one corresponding scoring criterion, and every scoring criterion must correspond to exactly one scalar leaf in reference_output.

[Hard Constraints]
1. Generate exactly one item.
2. Output JSON only with exactly one top-level key: "item".
3. item must contain exactly these keys:
   - "scenario"
   - "task_instruction"
   - "output_template"
   - "reference_output"
   - "scoring_rubric"
4. task_instruction must be exactly this string:
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
19. scoring_rubric must be a JSON object with exactly one key: "criteria".
20. scoring_rubric.criteria must be a list of criterion objects.
21. Each criterion object must have exactly these keys:
   - "path"
   - "canonical_value"
   - "description"
22. scoring_rubric must satisfy one-to-one rubric alignment:
   - every scalar leaf in reference_output must appear exactly once in scoring_rubric.criteria;
   - every criterion must correspond to exactly one scalar leaf in reference_output.
23. path must be the exact path to the corresponding scalar leaf in reference_output, including array indices when needed.
24. canonical_value must be exactly the scalar value found at that path in reference_output.
25. description must explain how this canonical_value links back to the original attribute and what execution role it plays in the configuration.
26. description must not merely restate canonical_value mechanically.
27. Do not add grouped criteria that cover multiple reference_output leaves at once.
28. Do not add global style, naturalness, usefulness, or generic quality criteria.
29. Before finalizing, silently check:
   - if state_value were hidden, the task would become substantially underdetermined;
   - the scenario does not leak the attribute values;
   - the scenario feels like a natural product moment;
   - the synthesized schema is configuration-facing rather than a raw copy of state_value;
   - the rubric aligns exactly one-to-one with reference_output leaves;
   - each description explains the semantic link between the leaf and the original attribute.

[Good Example A Input]
state_key: "user_attributes_state:primary_job_role"
state_value: {example_state}

[Good Example A Output]
{{
  "item": {{
    "scenario": "A registration form for a technical industry symposium is being finalized. The professional credential section is being completed before attendee details are submitted.",
    "task_instruction": {fixed_task_instruction},
    "output_template": {example_output_template},
    "reference_output": {example_reference_output},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "symposium_registration.professional_profile.job_title",
          "canonical_value": "Senior Coatings Consultant",
          "description": "This leaf carries the user's current job title into the role field used to describe the attendee professionally."
        }},
        {{
          "path": "symposium_registration.professional_profile.organization",
          "canonical_value": "PPG Industries",
          "description": "This leaf carries the employer name into the organization field needed for the registration profile."
        }},
        {{
          "path": "symposium_registration.professional_profile.specialization_areas[0]",
          "canonical_value": "heavy-duty infrastructure",
          "description": "This leaf captures one specialization area directly stated in the role attribute so the registration can reflect the attendee's technical focus."
        }},
        {{
          "path": "symposium_registration.professional_profile.specialization_areas[1]",
          "canonical_value": "marine protection",
          "description": "This leaf captures another specialization area from the attribute, separating technical focus areas instead of collapsing them into a single summary."
        }}
      ]
    }}
  }}
}}

[Good Example B Input]
state_key: "user_attributes_state:fitness_technology"
state_value: {example_alt_state}

[Good Example B Output]
{{
  "item": {{
    "scenario": "A wellness app setup is being finalized. Connected-device sources are being configured before health data syncing starts.",
    "task_instruction": {fixed_task_instruction},
    "output_template": {example_alt_output_template},
    "reference_output": {example_alt_reference_output},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "wearable_sync_setup.connected_sources[0].device_model",
          "canonical_value": "Apple Watch Series 9",
          "description": "This leaf carries the first known device model into the source configuration for syncing."
        }},
        {{
          "path": "wearable_sync_setup.connected_sources[0].device_variant",
          "canonical_value": "Midnight aluminum",
          "description": "This leaf carries the distinguishing variant details for the first device so the setup reflects the specific hardware entry stated in the attribute."
        }},
        {{
          "path": "wearable_sync_setup.connected_sources[0].enabled_metrics[0]",
          "canonical_value": "daily heart rate",
          "description": "This leaf carries one tracking use directly stated for the first device into the metrics enabled for sync."
        }},
        {{
          "path": "wearable_sync_setup.connected_sources[0].enabled_metrics[1]",
          "canonical_value": "step tracking",
          "description": "This leaf carries another tracking use stated for the first device into the configured metric list."
        }},
        {{
          "path": "wearable_sync_setup.connected_sources[1].device_model",
          "canonical_value": "Oura Ring Gen3",
          "description": "This leaf carries the second known device model into the source configuration for syncing."
        }},
        {{
          "path": "wearable_sync_setup.connected_sources[1].device_variant",
          "canonical_value": "Stealth finish",
          "description": "This leaf carries the distinguishing finish of the second device into the configuration, grounded in the attribute text."
        }},
        {{
          "path": "wearable_sync_setup.connected_sources[1].enabled_metrics[0]",
          "canonical_value": "sleep staging",
          "description": "This leaf carries one directly stated sleep-related use for the second device into the sync setup."
        }},
        {{
          "path": "wearable_sync_setup.connected_sources[1].enabled_metrics[1]",
          "canonical_value": "recovery metrics",
          "description": "This leaf carries another stated use for the second device into the configured metric list."
        }}
      ]
    }}
  }}
}}

[Good Example C Input]
state_key: "user_attributes_state:digital_subscriptions"
state_value: {example_third_state}

[Good Example C Output]
{{
  "item": {{
    "scenario": "A unified content and services hub is being connected for the user. Subscription entitlements are being prepared before linked services are shown.",
    "task_instruction": {fixed_task_instruction},
    "output_template": {example_third_output_template},
    "reference_output": {example_third_reference_output},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "subscription_entitlements.linked_services[0].service_name",
          "canonical_value": "Audible",
          "description": "This leaf extracts the service name from the first subscription so the entitlement config can refer to the linked service explicitly."
        }},
        {{
          "path": "subscription_entitlements.linked_services[0].plan_or_bundle",
          "canonical_value": "Premium Plus",
          "description": "This leaf carries the plan tier from the first subscription, separating the plan detail from the base service name."
        }},
        {{
          "path": "subscription_entitlements.linked_services[0].usage_context",
          "canonical_value": "listening to non-fiction during 45-minute commutes",
          "description": "This leaf carries the stated use context for the first subscription so the configuration reflects how the service is actually used."
        }},
        {{
          "path": "subscription_entitlements.linked_services[1].service_name",
          "canonical_value": "Disney Bundle",
          "description": "This leaf extracts the named service bundle from the second subscription entry."
        }},
        {{
          "path": "subscription_entitlements.linked_services[1].plan_or_bundle",
          "canonical_value": "including Hulu and ESPN+",
          "description": "This leaf carries the bundle composition detail for the second subscription so the entitlement config preserves what is included."
        }},
        {{
          "path": "subscription_entitlements.linked_services[1].usage_context",
          "canonical_value": "family entertainment and sports coverage",
          "description": "This leaf captures the stated household use context for the second subscription."
        }},
        {{
          "path": "subscription_entitlements.linked_services[2].service_name",
          "canonical_value": "MasterClass",
          "description": "This leaf extracts the service name from the third subscription entry."
        }},
        {{
          "path": "subscription_entitlements.linked_services[2].plan_or_bundle",
          "canonical_value": "annual subscription",
          "description": "This leaf carries the subscription term stated for the third service so the entitlement config reflects the user's actual access type."
        }},
        {{
          "path": "subscription_entitlements.linked_services[2].usage_context",
          "canonical_value": "learning technical crafting and cooking skills",
          "description": "This leaf captures the stated learning use context for the third subscription."
        }}
      ]
    }}
  }}
}}

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
    }},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "devices[0]",
          "canonical_value": "Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)",
          "description": "This leaf repeats the first device string."
        }},
        {{
          "path": "devices[1]",
          "canonical_value": "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)",
          "description": "This leaf repeats the second device string."
        }}
      ]
    }}
  }}
}}

Why the bad example fails:
- The scenario leaks the attribute values.
- The task_instruction asks for a free-form recommendation instead of a structured action configuration.
- The schema fails to decompose execution-relevant parts of the attribute strings into meaningful configuration fields.

[Input]
- checkpoint_timestamp: {checkpoint_timestamp}
- state_key: {state_key}
- state_value: {state_value}

[Output JSON ONLY]
{{
  "item": {{
    "scenario": "...",
    "task_instruction": {fixed_task_instruction},
    "output_template": {{
      "<synthesized_configuration_key>": {{
        "<synthesized_execution_field>": "<fill or nested fills>"
      }}
    }},
    "reference_output": {{
      "<same synthesized shape as output_template>": "..."
    }},
    "scoring_rubric": {{
      "criteria": [
        {{
          "path": "...",
          "canonical_value": "...",
          "description": "..."
        }}
      ]
    }}
  }}
}}
""".format(
        checkpoint_timestamp=json.dumps(str(checkpoint_timestamp or ""), ensure_ascii=False),
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
        fixed_task_instruction=json.dumps(fixed_task_instruction, ensure_ascii=False),
        example_state=json.dumps(example_state, ensure_ascii=False),
        example_output_template=json.dumps(example_output_template, ensure_ascii=False, indent=2),
        example_reference_output=json.dumps(example_reference_output, ensure_ascii=False, indent=2),
        example_alt_state=json.dumps(example_alt_state, ensure_ascii=False, indent=2),
        example_alt_output_template=json.dumps(example_alt_output_template, ensure_ascii=False, indent=2),
        example_alt_reference_output=json.dumps(example_alt_reference_output, ensure_ascii=False, indent=2),
        example_third_state=json.dumps(example_third_state, ensure_ascii=False, indent=2),
        example_third_output_template=json.dumps(example_third_output_template, ensure_ascii=False, indent=2),
        example_third_reference_output=json.dumps(example_third_reference_output, ensure_ascii=False, indent=2),
    )

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
            state_value=state_value,
            service_family=normalized_family,
        )
    if normalized_family == "action_configuration":
        return _build_task_c_v2_action_configuration_question_pack_prompt(
            checkpoint_timestamp=checkpoint_timestamp,
            state_type=state_type,
            state_key=state_key,
            state_value=state_value,
            service_family=normalized_family,
        )
    raise ValueError(f"Unsupported Task C v2 service_family: {normalized_family}")


def _describe_task_c_v2_service_family(service_family: str) -> str:
    normalized = str(service_family or "").strip()
    if normalized == "user_communication":
        return "Habit-Conditioned User Communication"
    if normalized == "information_request_construction":
        return "Preference-Conditioned Filtering Parameter Completion"
    if normalized == "action_configuration":
        return "Attribute-Conditioned Action Configuration"
    return normalized

def _build_service_application_prompt(
    *,
    memory_section: str,
    question_text: str,
) -> str:
    return """{question_text}

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
  "answer": "<concise answer>",
  "evidence": [
    {{
      "app_log_id": "<app_log_id>",
      "evidence_content": "<supporting snippet from the same log>"
    }}
  ]
}}

""".format(
        question_text=question_text,
        memory_section=memory_section,
    )


def _build_structured_service_completion_prompt(
    *,
    memory_section: str,
    service_family: str,
    scenario: str,
    task_instruction: str,
    output_template: Any,
) -> str:
    return """[Scenario]
{scenario}

[Task Instruction]
{task_instruction}

[Task Type]
{service_family_description}

[Required Output Object]
{output_template}

{memory_section}

[Instructions]
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
}}
""".format(
        scenario=str(scenario or "").strip(),
        task_instruction=str(task_instruction or "").strip(),
        service_family_description=_describe_task_c_v2_service_family(service_family),
        output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        memory_section=memory_section,
    )


def _build_task_c_v2_user_communication_answer_prompt(
    *,
    memory_section: str,
    scenario: str,
    task_instruction: str,
) -> str:
    return """[Scenario]
{scenario}

[Task Instruction]
{task_instruction}

{memory_section}

[Instructions]
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
}}
""".format(
        scenario=str(scenario or "").strip(),
        task_instruction=str(task_instruction or "").strip(),
        memory_section=memory_section,
    )


def build_service_application_prompt_with_inline_memory(
    *,
    question_text: str,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    return _build_service_application_prompt(
        memory_section=_render_inline_memory_section(
            inline_memory_blocks=_coerce_inline_memory_blocks(
                inline_memory_blocks=inline_memory_blocks,
                context_logs=context_logs,
                log_to_text=log_to_text,
            )
        ),
        question_text=question_text,
    )


def build_service_application_prompt_with_agent_memory(
    *,
    question_text: str,
    context_logs: Optional[List[Dict[str, Any]]],
    log_to_text,
    inline_memory_blocks: Optional[List[str]] = None,
) -> str:
    del context_logs, log_to_text, inline_memory_blocks
    return _build_service_application_prompt(
        memory_section=_render_agent_memory_section(),
        question_text=question_text,
    )


def build_structured_service_completion_prompt_with_inline_memory(
    *,
    service_family: str,
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
        service_family=service_family,
        scenario=scenario,
        task_instruction=task_instruction,
        output_template=output_template,
    )


def build_structured_service_completion_prompt_with_agent_memory(
    *,
    service_family: str,
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
        service_family=service_family,
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
Validate whether this apply-service QA item is a strong personalized service decision item.
Judge the item by three fixed semantic criteria and provide one explicit analysis for each criterion.
Do not output an overall verdict; only output the structured criterion-level judgments.

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
                "criterion": "answer_groundedness",
                "analysis": "<whether reference_answer is short, natural, and grounded in the provided state>",
                "pass": "<bool>",
            },
        ]
        return """[Task Instruction]
Validate whether this Task C v2 item is a strong proactive user-communication task.
Judge the item by four fixed semantic criteria and provide one explicit analysis for each criterion.
Do not output an overall verdict; only output the structured criterion-level judgments.

[Definitions]
- service_completion_quality: the item must ask the assistant to produce one concrete user-facing communication, not merely restate the habit or answer a recall question.
- full_field_dependency: answering well should require all non-derived fields in `state_value`; dropping an important field should make the communication materially weaker or incorrect.
- low_leakage: `scenario` and `task_instruction` should describe only the local situation and current communication task; they must not restate or strongly imply the key habit facts that should come from `state_value`.
- answer_groundedness: `reference_answer` must be a short natural-language assistant response that is specific, state-grounded, and free of unsupported user-specific details.

[Constraints]
1. Evaluate exactly the four required criteria in this fixed order: service_completion_quality, full_field_dependency, low_leakage, answer_groundedness.
2. Output exactly one object for each required criterion.
3. Use the criterion names exactly as given; do not rename, reorder, omit, or add criteria.
4. Set `pass` to true only when the criterion is clearly satisfied.
5. Keep each `analysis` concise but specific.
6. Mark `low_leakage` as failed if `scenario` or `task_instruction` restates or paraphrases the habit action, cadence, scheduled day, timing, location, or priority.
7. Mark `answer_groundedness` as failed if `reference_answer` introduces unsupported tactics, thresholds, or user-specific facts absent from `state_value`.
8. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "habits_state:morning_run"
state_value: {{"timing": {{"start_time": "06:30"}}, "schedule": {{"days_of_week": [1, 3, 5]}}}}
candidate_item: {{
  "service_family": "user_communication",
  "scenario": "It is 06:10. Nothing has been logged yet today.",
  "task_instruction": "Write the short reminder message the assistant should send right now.",
  "reference_answer": "Send a reminder that this is one of the user's regular morning run windows and that the run normally starts at 06:30."
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "service_completion_quality",
      "analysis": "The item asks for one concrete assistant message rather than raw recall.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "The timing and recurring-day pattern are both needed for a good reminder.",
      "pass": true
    }},
    {{
      "criterion": "low_leakage",
      "analysis": "The scenario describes only the current moment and does not restate the routine details.",
      "pass": true
    }},
    {{
      "criterion": "answer_groundedness",
      "analysis": "The answer is short, natural, and grounded in the state without adding unsupported facts.",
      "pass": true
    }}
  ]
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- candidate item: object with `service_family`, `scenario`, `task_instruction`, `reference_answer`

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- candidate_item: {{
    "service_family": {service_family},
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
            service_family=json.dumps(normalized_family, ensure_ascii=False),
            scenario=json.dumps(str(scenario or ""), ensure_ascii=False),
            task_instruction=json.dumps(str(task_instruction or ""), ensure_ascii=False),
            reference_answer=json.dumps(str(reference_answer or ""), ensure_ascii=False),
        )
    criteria_template = [
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
            "criterion": "schema_groundedness",
            "analysis": "<whether the item uses a family-appropriate service object while preserving one source leaf per required output leaf>",
            "pass": "<bool>",
        },
        {
            "criterion": "point_pairability",
            "analysis": "<whether the item can be scored with one paired field-correctness point per required output field>",
            "pass": "<bool>",
        },
    ]
    return """[Task Instruction]
Validate whether this Task C v2 item is a strong structured proactive personalized-service completion item.
Judge the item by four fixed semantic criteria and provide one explicit analysis for each criterion.
Do not output an overall verdict; only output the structured criterion-level judgments.

[Definitions]
- service_completion_quality: the item must define a real structured service-completion task, not a free-form recall question.
- full_field_dependency: the item must require all non-derived fields in `state_value`; dropping any required field should make the service object incomplete.
- schema_groundedness: `output_template` and `reference_output` must form a family-appropriate top-level service object. They may rename and regroup fields, but they must preserve every source leaf value exactly once and must not merely copy the raw state schema.
- point_pairability: the item must be scorable with deterministic paired field-correctness points, where the required output leaves appear in the same order as the source leaves and each output leaf can be paired with one source field.

[Constraints]
1. Evaluate exactly the four required criteria in this fixed order: service_completion_quality, full_field_dependency, schema_groundedness, point_pairability.
2. Output exactly one object for each required criterion.
3. Use the criterion names exactly as given; do not rename, reorder, omit, or add criteria.
4. Set `pass` to true only when the criterion is clearly satisfied.
5. Keep each `analysis` concise but specific.
6. Mark `service_completion_quality` as failed if the item mainly behaves like a free-form QA question rather than a structured completion task.
7. Mark `full_field_dependency` as failed if some required part of `state_value` is unused, optionalized, or collapsed away.
8. Mark `schema_groundedness` as failed if the item simply mirrors the raw state schema, fails to produce a top-level service object, drops source values, or adds extra unpaired required leaves.
9. Mark `point_pairability` as failed if the item would require fuzzy whole-answer judgment rather than one-to-one leaf pairing in source-leaf order.
10. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "habits_state:morning_run"
state_value: {{"timing": {{"start_time": "06:30"}}}}
candidate_item: {{
  "service_family": "user_communication",
  "scenario": "The assistant is preparing the morning routine-support communication object.",
  "task_instruction": "Fill the user-communication payload for the scheduled routine.",
  "output_template": {{"communication_payload": {{"scheduled_start_time": "<fill>"}}}},
  "reference_output": {{"communication_payload": {{"scheduled_start_time": "06:30"}}}}
}}

[Example Output]
{{
  "criteria": [
    {{
      "criterion": "service_completion_quality",
      "analysis": "The item asks for one structured service object rather than a free-form answer.",
      "pass": true
    }},
    {{
      "criterion": "full_field_dependency",
      "analysis": "The only state field is required to complete the payload.",
      "pass": true
    }},
    {{
      "criterion": "schema_groundedness",
      "analysis": "The template is a communication payload rather than a raw state copy, and it preserves the source value once in a service-facing field.",
      "pass": true
    }},
    {{
      "criterion": "point_pairability",
      "analysis": "The output can be scored with one direct paired field-correctness point from timing.start_time to communication_payload.scheduled_start_time.",
      "pass": true
    }}
  ]
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- candidate item: object with `service_family`, `scenario`, `task_instruction`, `output_template`, `reference_output`

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- candidate_item: {{
    "service_family": {service_family},
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
        service_family=json.dumps(str(service_family or ""), ensure_ascii=False),
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
    state_observability: Dict[str, Any],
    evidence_logs: List[Dict[str, Any]],
) -> str:
    return """[Task Instruction]
Validate whether this state is inferable from evidence and question-worthy for apply QA generation.
Evaluate field-level inferability first, then produce a key-level decision.
Success means only evidence-supported fields are marked valid.

[Definitions]
- state_key: the state item being validated.
- state_value: the structured target value for this state_key at checkpoint time.
- candidate_field_paths: field paths to evaluate for inferability.
- state_observability: observability metadata for this state key (for example `is_valid`, `evidence_app_log_ids`, `last_app_log_id`).
- evidence_logs: related app logs up to checkpoint time; these are the primary evidence.
- field_verdict: one field-level decision with `{{field_name, reason_analysis, is_valid}}`.
- key-level `is_questionable`: true only when at least one field verdict has `is_valid=true`.

[Constraints]
1. Evaluate each path in `candidate_field_paths` independently.
2. Output exactly one `field_verdicts` item per candidate field path.
3. Use field names exactly from `candidate_field_paths`; do not invent paths.
4. Set `is_valid=true` only if evidence logs support inferring that field with high confidence.
5. If evidence is missing or ambiguous, set `is_valid=false` and explain why in `reason_analysis`.
6. Set top-level `is_questionable=true` only when at least one field has `is_valid=true`.
7. Keep `reason_codes` concise and machine-friendly (snake_case preferred).
8. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "habits_state:budget_review"
candidate_field_paths: ["schedule.days_of_week", "timing.start_time", "timing.end_time"]
evidence_logs: [{{"app_log_id":"log_0101","api_name":"scheduleService","response":"...Sunday 09:00 reminder..."}}]

[Example Output]
{{
  "is_questionable": true,
  "reason_codes": ["partial_field_support"],
  "field_verdicts": [
    {{
      "field_name": "schedule.days_of_week",
      "reason_analysis": "Weekly Sunday pattern is explicit in multiple logs.",
      "is_valid": true
    }},
    {{
      "field_name": "timing.start_time",
      "reason_analysis": "Start action repeatedly appears at 09:00.",
      "is_valid": true
    }},
    {{
      "field_name": "timing.end_time",
      "reason_analysis": "No evidence provides a duration or explicit end time.",
      "is_valid": false
    }}
  ]
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- candidate_field_paths: string[]
- state_observability: object
- evidence_logs: object[]

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- candidate_field_paths: {askable_fields}
- state_observability: {state_observability}
- evidence_logs: {evidence_logs}

Output JSON ONLY:
{{
  "is_questionable": "<bool>",
  "reason_codes": ["<short_code>", "..."],
  "field_verdicts": [
    {{
      "field_name": "<field path from candidate_field_paths>",
      "reason_analysis": "<why inferable or not inferable from evidence>",
      "is_valid": "<bool>"
    }}
  ]
}}
""".format(
        state_key=state_key,
        state_value=json.dumps(state_value, ensure_ascii=False),
        askable_fields=json.dumps(list(askable_fields or []), ensure_ascii=False),
        state_observability=json.dumps(state_observability or {}, ensure_ascii=False),
        evidence_logs=json.dumps(list(evidence_logs or []), ensure_ascii=False),
    )


def build_change_reason_validation_prompt(
    *,
    state_key: str,
    state_value: Any,
    change_reason: str,
    state_observability: Dict[str, Any],
    evidence_logs: List[Dict[str, Any]],
) -> str:
    return """[Task Instruction]
Validate whether one canonical gold change reason is sufficiently supported by evidence logs to be used as a Task B gold reason.
This is a metadata validation task, not a field-level state validation task.

[Definitions]
- state_key: the state item whose change reason is being validated.
- state_value: the structured target value for this state_key at checkpoint time.
- change_reason: the canonical gold reason text from upstream data construction.
- state_observability: observability metadata for this state key (for example `is_valid`, `evidence_app_log_ids`, `last_app_log_id`, `last_change_reason`).
- evidence_logs: related app logs up to checkpoint time; these are the primary evidence.
- valid change reason: a reason text that is explicitly supported, or strongly and conservatively inferable, from the evidence logs and does not conflict with the state change implied by the evidence.

[Constraints]
1. Judge only whether `change_reason` is evidence-supported enough to be used as gold metadata for Task B.
2. Be conservative: if evidence is missing, indirect, or ambiguous, mark the change reason invalid.
3. Do not rewrite the change reason.
4. `exists` must be `true` when a non-empty `change_reason` is provided.
5. Set `is_valid=true` only when the evidence logs support this change reason with high confidence.
6. If invalid, explain the evidence gap or mismatch in `reason_analysis`.
7. Keep `reason_codes` concise and machine-friendly (snake_case preferred).
8. Output JSON ONLY with no markdown and no extra keys.

[Example]
[Example Input]
state_key: "profile_state:commute_mode"
change_reason: "Switched to train commuting after downtown parking fees increased."
evidence_logs: [{{"app_log_id":"log_0201","api_name":"commutePlanner","response":"...parking fees downtown rose again... train commute selected for weekdays..."}}]

[Example Output]
{{
  "exists": true,
  "reason_analysis": "The evidence explicitly mentions higher downtown parking fees and the switch to train commuting.",
  "is_valid": true,
  "reason_codes": []
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- change_reason: string
- state_observability: object
- evidence_logs: object[]

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- change_reason: {change_reason}
- state_observability: {state_observability}
- evidence_logs: {evidence_logs}

Output JSON ONLY:
{{
  "exists": "<bool>",
  "reason_analysis": "<why the change reason is or is not evidence-supported>",
  "is_valid": "<bool>",
  "reason_codes": ["<short_code>", "..."]
}}
""".format(
        state_key=state_key,
        state_value=json.dumps(state_value, ensure_ascii=False),
        change_reason=json.dumps(str(change_reason or ""), ensure_ascii=False),
        state_observability=json.dumps(state_observability or {}, ensure_ascii=False),
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
Rewrite one invalid apply-service QA item so that it becomes a strong personalized service decision item.
Use the failed rules and semantic-criterion feedback to directly fix the item.
Do not merely paraphrase the invalid item.

[Definitions]
- failed_rules: the programmatic failure codes and failed criterion names that must be fixed.
- semantic criteria feedback: criterion-level judgments explaining which properties failed and why.
- strong apply item: an item where the correct answer materially depends on the user's state, asks for a concrete service action, requires the assistant to use that state to decide what to do, and has a specific answer that can be scored with points.

[Constraints]
1. Keep third-person wording about the user; never use "you", "your", or "yours".
2. Keep exactly one service decision question in one sentence.
3. Keep the revised reference answer short, specific, and deterministic.
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
    if normalized_family == "user_communication":
        return """[Task Instruction]
Rewrite one invalid Task C v2 user-communication item so that it becomes a strong proactive personalized-service task.
Use the failed rules and semantic-criterion feedback to directly fix the item.
Do not change the required service family.

[Definitions]
- failed_rules: the programmatic failure codes and failed criterion names that must be fixed.
- semantic criteria feedback: criterion-level judgments explaining which properties failed and why.
- low leakage: `scenario` and `task_instruction` do not restate, paraphrase, or strongly imply the habit facts that should come from `state_value`.
- grounded answer: a short natural-language assistant response whose key personalized content is supported by `state_value`.

[Constraints]
1. Keep `service_family = {service_family}` exactly.
2. Keep the item in natural-language assistant-response form; do not rewrite it into a structured payload.
3. Keep exactly these fields in the rewritten item: `service_family`, `scenario`, `task_instruction`, `reference_answer`.
4. Fix every failed rule and every failed semantic criterion.
5. If `service_completion_quality` failed, rewrite the item so it asks for one concrete assistant communication rather than a recall question.
6. If `full_field_dependency` failed, rewrite the item so a good answer depends on all important state fields.
7. If `low_leakage` failed, remove any restatement of action, cadence, scheduled day, timing, location, or priority from `scenario` and `task_instruction`.
8. If `answer_groundedness` failed, make the revised `reference_answer` more state-grounded without adding unsupported user-specific facts.
9. Output JSON ONLY with no extra keys.

[Example]
[Example Input]
failed_rules: ["low_leakage", "answer_groundedness"]
semantic_criteria: [
  {{"criterion": "service_completion_quality", "analysis": "The item asks for one communication action.", "pass": true}},
  {{"criterion": "full_field_dependency", "analysis": "The state fields are mostly used.", "pass": true}},
  {{"criterion": "low_leakage", "analysis": "The scenario repeats that this is the user's Sunday family dinner.", "pass": false}},
  {{"criterion": "answer_groundedness", "analysis": "The answer adds unsupported preparation advice.", "pass": false}}
]

[Example Output]
{{
  "service_family": "user_communication",
  "scenario": "It is 16:45. Everyone is home, and nothing has been prepared yet.",
  "task_instruction": "Write the short reminder message the assistant should send right now.",
  "reference_answer": "Send a high-priority reminder about the family's recurring dinner window at the family home dining room."
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- invalid item: object with `service_family`, `scenario`, `task_instruction`, `reference_answer`
- failed_rules: string[]
- semantic_criteria: object[]

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- invalid_item: {{
    "service_family": {service_family},
    "scenario": {scenario},
    "task_instruction": {task_instruction},
    "reference_answer": {reference_answer}
  }}
- failed_rules: {failed_rules}
- semantic_criteria: {semantic_criteria}

Output JSON ONLY:
{{
  "service_family": {service_family},
  "scenario": "...",
  "task_instruction": "...",
  "reference_answer": "..."
}}
""".format(
            state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
            state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
            service_family=json.dumps(normalized_family, ensure_ascii=False),
            scenario=json.dumps(str(scenario or ""), ensure_ascii=False),
            task_instruction=json.dumps(str(task_instruction or ""), ensure_ascii=False),
            reference_answer=json.dumps(str(reference_answer or ""), ensure_ascii=False),
            failed_rules=json.dumps(list(failed_rules or []), ensure_ascii=False),
            semantic_criteria=json.dumps(list(semantic_criteria or []), ensure_ascii=False, indent=2),
        )
    return """[Task Instruction]
Rewrite one invalid Task C v2 item so that it becomes a strong structured proactive personalized-service completion item.
Use the failed rules and semantic-criterion feedback to directly fix the item.
Do not change the required service family.

[Definitions]
- failed_rules: the programmatic failure codes and failed criterion names that must be fixed.
- semantic criteria feedback: criterion-level judgments explaining which properties failed and why.
- family-appropriate service object: a top-level structured payload that matches the required service family and is not merely a raw copy of the source state.
- order-preserving pairing: the required output leaves must appear in the same order as the source leaves so the item can be scored deterministically one-to-one.

[Constraints]
1. Keep `service_family = {service_family}` exactly.
2. Keep `output_template` and `reference_output` as top-level structured service objects.
3. Fix every failed rule and every failed semantic criterion.
4. If `service_completion_quality` failed, rewrite the scenario and task instruction so the item becomes a real structured service-completion task.
5. If `full_field_dependency` failed, rewrite the service object so every source field is required.
6. If `schema_groundedness` failed, repair the item so it becomes a family-appropriate service object rather than a raw state copy, while preserving every source leaf exactly once.
7. If `point_pairability` failed, rewrite the item so the required output leaves can be paired one-to-one with source leaves in source-leaf order.
8. Do not add extra unpaired required output leaves.
9. Output JSON ONLY with no extra keys.

[Example]
[Example Input]
failed_rules: ["service_completion_quality", "point_pairability"]
semantic_criteria: [
  {{"criterion": "service_completion_quality", "analysis": "The item still behaves like a free-form answer request.", "pass": false}},
  {{"criterion": "full_field_dependency", "analysis": "The only state field is required.", "pass": true}},
  {{"criterion": "schema_groundedness", "analysis": "The schema mirrors the state correctly.", "pass": true}},
  {{"criterion": "point_pairability", "analysis": "The task instruction is too vague for direct field scoring.", "pass": false}}
]

[Example Output]
{{
  "service_family": "user_communication",
  "scenario": "The assistant is preparing the proactive routine-support object for this scheduled activity.",
  "task_instruction": "Fill the user-communication payload so every required routine field is available for the assistant's next action.",
  "output_template": {{"communication_payload": {{"scheduled_start_time": "<fill>"}}}},
  "reference_output": {{"communication_payload": {{"scheduled_start_time": "06:30"}}}}
}}

[Input/Output Format]
Input:
- state_key: string
- state_value: object
- invalid item: object with `service_family`, `scenario`, `task_instruction`, `output_template`, `reference_output`
- failed_rules: string[]
- semantic_criteria: object[]

Input Payload:
- state_key: {state_key}
- state_value: {state_value}
- invalid_item: {{
    "service_family": {service_family},
    "scenario": {scenario},
    "task_instruction": {task_instruction},
    "output_template": {output_template},
    "reference_output": {reference_output}
  }}
- failed_rules: {failed_rules}
- semantic_criteria: {semantic_criteria}

Output JSON ONLY:
{{
  "service_family": {service_family},
  "scenario": "...",
  "task_instruction": "...",
  "output_template": {output_template},
  "reference_output": {reference_output}
}}
""".format(
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        state_value=json.dumps(state_value, ensure_ascii=False, indent=2),
        service_family=json.dumps(str(service_family or ""), ensure_ascii=False),
        scenario=json.dumps(str(scenario or ""), ensure_ascii=False),
        task_instruction=json.dumps(str(task_instruction or ""), ensure_ascii=False),
        output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        reference_output=json.dumps(reference_output, ensure_ascii=False, indent=2),
        failed_rules=json.dumps(list(failed_rules or []), ensure_ascii=False),
        semantic_criteria=json.dumps(list(semantic_criteria or []), ensure_ascii=False, indent=2),
    )


def build_value_micro_points_prompt(
    *,
    state_key: str,
    text_value: str,
    max_points: int,
) -> str:
    max_points = max(1, min(3, int(max_points)))
    template = ["<one atomic scoring point>" for _ in range(max_points)]
    return """[Task Instruction]
Break one complex state field into 1 to {max_points} atomic facts.
Each atomic fact should capture one atomic meaning unit that a model prediction should express.

[Definitions]
- complex state field: one statement-like field whose meaning should not be judged as one holistic blob.
- atomic fact: one atomic scoring requirement.
- scoreable atomic fact: one concrete aspect of correctness that can be judged independently on the shared binary `0/1` hit scale.
- brittle atomic fact: a fact that depends on unnecessary exact model / spec / parameter / exact wording instead of stable semantic meaning.

[Constraints]
1. Generate 1 to {max_points} atomic facts.
2. Each atomic fact must be independently judgeable on the shared binary `0/1` hit scale.
3. Keep all atomic facts positive.
4. Every atomic fact must be directly supported by the source text.
5. Do not add stronger preferences, motivations, or implications that are not stated.
6. Do not repeat the same meaning in multiple points.
7. Prefer stable semantic requirements over brittle exact surface forms.
8. Do not generate multiple points that all hinge on the same narrow identifier, model name, spec, parameter, or exact wording.
9. Use an exact identifier only when it is itself the canonical memory fact or is necessary to disambiguate two otherwise-confusable meanings.
10. Output JSON only.

[Example]
Input source text:
"Prefers self-paced white papers and webinars over large conferences."

Output JSON ONLY:
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
        template=json.dumps(template, ensure_ascii=False, indent=2),
    )


def build_value_rubric_validation_prompt(
    *,
    state_key: str,
    text_value: str,
    points: List[Dict[str, Any]],
) -> str:
    return """[Task Instruction]
Validate one generated atomic-fact set for a complex state field.
Judge whether each atomic fact stays grounded in the source text without drift.

[Definitions]
- source text: the original state text that the rubric-point set must stay faithful to.
- atomic fact: one scoreable meaning unit.
- point_text: the atomic scoring statement that the evaluator will judge independently.
- drift: the atomic fact changes, overextends, or reverses the meaning of the source text.
- atomic fact requirement: one fact that checks exactly one concrete aspect of correctness and can be scored independently.
- over_specific: the atomic fact depends on unnecessary exact model / spec / parameter / exact wording and is therefore too brittle.

[Constraints]
1. Validate every point independently.
2. Mark a point as failed if it is unsupported, drifts, is redundant, is over-specific, or is not atomic enough to score independently.
3. Use only these fail reasons when needed: `unsupported`, `drift`, `not_atomic`, `redundant`, `over_specific`.
4. Use only these set-level failures when needed: `coverage_gap`.
5. `set_pass` can be true only if every point passes and there is no set-level failure.
6. Keep each `analysis` concise and specific.
7. Output JSON only.

[Example]
Source text:
"Prefers self-paced white papers and webinars over large conferences."

Candidate points:
{{
  "points": [
    {{
      "point_id": "scp_pref_p1",
      "point_text": "The user only learns effectively through live conferences."
    }}
  ]
}}

Output JSON ONLY:
{{
  "points": [
    {{
      "point_id": "scp_pref_p1",
      "analysis": "The point reverses the stated preference and is not supported by the source text.",
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
  "set_pass": "<bool>",
  "set_failures": []
}}
""".format(
        state_key=json.dumps(str(state_key or ""), ensure_ascii=False),
        text_value=json.dumps(str(text_value or ""), ensure_ascii=False),
        points=json.dumps(points, ensure_ascii=False, indent=2),
    )


def build_value_rubric_rewrite_prompt(
    *,
    state_key: str,
    text_value: str,
    points: List[Dict[str, Any]],
    validation_points: List[Dict[str, Any]],
    set_failures: List[str],
    max_points: int,
) -> str:
    max_points = max(1, min(3, int(max_points)))
    template = ["<one atomic scoring point>" for _ in range(max_points)]
    return """[Task Instruction]
Rewrite an invalid atomic-fact set for one complex state field.
Return a corrected atomic-fact set that stays faithful to the source text.

[Definitions]
- atomic fact: one scoreable meaning unit.
- point_text: one atomic scoring statement that the evaluator will judge independently.
- faithful rewrite: a rewrite that removes drift, unsupported content, and redundancy while preserving the source meaning.
- robust rewrite: a rewrite that avoids unnecessary exact model / spec / parameter wording when a broader semantic fact is sufficient.

[Constraints]
1. Return 1 to {max_points} positive atomic facts.
2. Each atomic fact must be independently judgeable on the shared binary `0/1` hit scale.
3. Fix every invalid point so the final set is directly supported by the source text.
4. Do not add stronger preferences, motivations, or implications that are not stated.
5. Keep every point atomic and non-redundant.
6. Repair any `over_specific` or brittle point by rewriting it into a broader, more stable semantic fact unless the exact identifier is truly necessary.
7. Output JSON only.

[Example]
Source text:
"Prefers self-paced white papers and webinars over large conferences."

Invalid points:
{{
  "points": [
    {{
      "point_id": "scp_pref_p1",
      "point_text": "The user only learns effectively through live conferences."
    }}
  ]
}}

Validation feedback:
{{
  "points": [
    {{
      "point_id": "scp_pref_p1",
      "analysis": "The point reverses the stated preference and introduces unsupported meaning.",
      "pass": false,
      "fail_reasons": ["drift", "unsupported"]
    }}
  ],
  "set_failures": ["coverage_gap"]
}}

Output JSON ONLY:
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
        text_value=json.dumps(str(text_value or ""), ensure_ascii=False),
        points=json.dumps(points, ensure_ascii=False, indent=2),
        validation_points=json.dumps(validation_points, ensure_ascii=False, indent=2),
        set_failures=json.dumps(list(set_failures or []), ensure_ascii=False),
        template=json.dumps(template, ensure_ascii=False, indent=2),
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
Validate one generated atomic-fact set for an apply-service QA item.
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
Rewrite an invalid atomic-fact set for one apply-service QA item.
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
Generate scoreable atomic facts for one apply-service QA item.
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
