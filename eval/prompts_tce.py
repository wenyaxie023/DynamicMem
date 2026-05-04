import json
from typing import Any, Dict, List


def _slot_criterion(slot: Dict[str, Any]) -> str:
    point_text = str(slot.get("point_text") or "").strip()
    if point_text:
        return point_text
    reference_value = slot.get("reference_value")
    target_path = str(slot.get("target_path") or "").strip()
    polarity = str(slot.get("polarity") or "").strip().lower()
    reference_text = json.dumps(reference_value, ensure_ascii=False, sort_keys=True)
    if target_path:
        if polarity == "negative":
            return f"The prediction should avoid `{target_path}` with value {reference_text}."
        return f"The prediction should satisfy `{target_path}` with value {reference_text}."
    if polarity == "negative":
        return f"The prediction should avoid the prohibited value {reference_text}."
    return f"The prediction should satisfy the required value {reference_text}."


def prompt_point_ids_for_slots(slots: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for slot in slots:
        if isinstance(slot, dict) and str(slot.get("point_id") or "").strip():
            out.append(str(len(out) + 1))
    return out


def _checklist_from_slots(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    checklist: List[Dict[str, Any]] = []
    prompt_point_ids = prompt_point_ids_for_slots(slots)
    prompt_idx = 0
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        point_id = str(slot.get("point_id") or "").strip()
        if not point_id:
            continue
        item: Dict[str, Any] = {
            "point_id": prompt_point_ids[prompt_idx],
            "criterion": _slot_criterion(slot),
        }
        prompt_idx += 1
        checklist.append(item)
    return checklist


def _slot_judgment_template(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt_point_ids = prompt_point_ids_for_slots(slots)
    return [
        {
            "point_id": prompt_point_id,
            "analysis": "<brief checklist-item judgment>",
            "correct": "<bool>",
        }
        for prompt_point_id in prompt_point_ids
    ]


def _render_slot_judge_prompt(
    *,
    task_label: str,
    unit_label: str,
    slots: List[Dict[str, Any]],
    prediction: Any,
    extra_definitions: str = "",
    extra_constraints: str = "",
    example_input: Dict[str, Any],
    example_output: Dict[str, Any],
) -> str:
    checklist = _checklist_from_slots(slots)
    return """[Task Instruction]
You are evaluating {task_label}.
Read the full prediction and decide whether it satisfies each checklist item.
Return exactly one boolean judgment for every short `point_id`.

[Definitions]
- prediction
  - the full model output being judged
- checklist item
  - one required fact or constraint to judge against the full prediction
- `point_id`
  - a short checklist identifier such as "1", "2", or "3"; copy it exactly from the checklist
- `correct`
  - `true` when the prediction satisfies the checklist item's core idea or practical value; otherwise `false`
- core idea
  - the essential meaning, value, or constraint required by the checklist item. If the full prediction communicates this correctly, mark it correct even when wording, formatting, field names, or detail organization differ.
- semantic equivalent
  - the same practical value despite harmless formatting or wording differences, such as casing, punctuation, spacing, common time formats, weekday name/index encodings where 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, and 6=Sunday, or obvious synonyms
{extra_definitions}

[Constraints]
1. Judge each checklist item independently.
2. Return every provided `point_id` exactly once.
3. Do not add or drop checklist items.
4. `correct` must be boolean, not a score.
5. If the prediction gets the checklist item's core idea or practical value right, mark it correct; do not require exact wording, formatting, field names, or identical detail organization unless the checklist item explicitly requires them.
6. Broad topical overlap alone is not enough: mark an item incorrect when the prediction omits the core idea, contradicts it, is unrelated, is too vague to establish it, or adds conflicting content.
7. Treat the checklist as unordered unless a checklist item explicitly requires order.
8. Count clear semantic equivalents as correct, but do not move a value from a different field or indexed position unless the checklist item itself treats that list as unordered.
9. In every judgment object, write `analysis` before `correct`.
{extra_constraints}

[Example]
[Example Input]
{example_input}

[Example Output]
{example_output}

[Input/Output Format]
Input:
- one {unit_label}
- `prediction`: the full model output for this unit
- `checklist`: unordered list of checklist objects
- every checklist object includes:
  - `point_id`
  - `criterion`

Output JSON ONLY:
{{
  "judgments": [
    {{
      "point_id": "<point_id>",
      "analysis": "<brief checklist-item judgment>",
      "correct": "<bool>"
    }}
  ]
}}
[Prediction]
{prediction}

[Checklist]
{checklist}

Fill this exact judgments template (do not add/drop point_ids):
{judgment_template}
""".format(
        task_label=task_label,
        unit_label=unit_label,
        extra_definitions=extra_definitions,
        extra_constraints=extra_constraints,
        example_input=json.dumps(example_input, ensure_ascii=False, indent=2),
        example_output=json.dumps(example_output, ensure_ascii=False, indent=2),
        prediction=json.dumps(prediction, ensure_ascii=False, indent=2),
        checklist=json.dumps(checklist, ensure_ascii=False, indent=2),
        judgment_template=json.dumps(_slot_judgment_template(slots), ensure_ascii=False, indent=2),
    )


def _strip_slot_polarity(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stripped: List[Dict[str, Any]] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        copied = dict(slot)
        copied.pop("polarity", None)
        stripped.append(copied)
    return stripped


def _prediction_from_slots(slots: List[Dict[str, Any]]) -> Any:
    values = [slot.get("predicted_value") for slot in slots if isinstance(slot, dict) and "predicted_value" in slot]
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    by_point: Dict[str, Any] = {}
    for idx, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        key = str(slot.get("point_id") or idx)
        by_point[key] = slot.get("predicted_value")
    return by_point


def _build_task_a_slot_judge_prompt(
    *,
    state_key: str,
    slots: List[Dict[str, Any]],
    prediction: Any = None,
) -> str:
    prompt_slots = _strip_slot_polarity(slots)
    example_input = {
        "prediction": {
            "current_value": "webinar",
            "excluded_format": "live conference",
        },
        "checklist": [
            {
                "point_id": "1",
                "criterion": "The prediction should satisfy `current_value` with value \"webinar\".",
            },
            {
                "point_id": "2",
                "criterion": "The answer avoids a live conference recommendation.",
            },
        ],
    }
    example_output = {
        "judgments": [
            {
                "point_id": "1",
                "analysis": "The predicted value matches the required preferred format.",
                "correct": True,
            },
            {
                "point_id": "2",
                "analysis": "The prediction explicitly avoids live conferences.",
                "correct": True,
            },
        ]
    }
    prompt = _render_slot_judge_prompt(
        task_label="a predicted user state",
        unit_label="user-state prediction",
        slots=prompt_slots,
        prediction=_prediction_from_slots(prompt_slots) if prediction is None else prediction,
        extra_constraints="10. For list-like user-state predictions, do not require the prediction order to match the checklist order unless a checklist item explicitly says order matters.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt


def _snapshot_holistic_category(state_key: str) -> str:
    normalized = str(state_key or "").strip()
    if normalized.startswith("habits_state:"):
        return "habit"
    if normalized.startswith("preferences_state:"):
        return "preference"
    if normalized.startswith("user_attributes_state:"):
        return "attribute"
    return "profile"


def _snapshot_holistic_category_prompt_parts(category: str) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    if category == "habit":
        guidance = """- How to identify core vs detail for habit fields
  - Use `state_key` to identify the routine or recurring activity being evaluated, such as a morning walk, board meeting, medication routine, or commute habit
  - For `schedule.*` fields, core is the recurrence pattern or scheduled day/time concept; exact JSON encoding, wording, or weekday index format is detail
  - For `timing.*` fields, core is the intended time window or time point for the same routine; if the reference is an exact time, a prediction within 5 minutes can still count as the same timing core, while larger offsets usually change the practical timing
  - For exact `timing.*` details, use `detail_quality=2` for the exact time, `detail_quality=1` for a time within 10 minutes, and `detail_quality=0` for a larger offset unless the field is explicitly an approximate time window
  - For `location` fields, core is the same venue, route, or place category for the routine; compatible added specificity preserves core, such as "sofa in living room" for "sofa"
  - For `location` details, full address, room, landmark, branch-level specificity, or compatible added specificity can improve detail quality; missing required specificity lowers detail quality
  - For other habit fields, core is the routine component named by `field_path`; qualifiers, exact labels, and formatting are detail
- habit core_correct examples
  - `core_correct=true`: 06:33 for a 06:30 start time is within 5 minutes and still identifies the same routine-time field
  - `core_correct=true`: "every Monday, Wednesday, and Friday" identifies the same scheduled weekdays as [0, 2, 4]
  - `core_correct=true`: "sofa in the living room" preserves the same location core as "sofa" with compatible extra specificity
  - `core_correct=false`: 06:45 for a 06:30 exact start time is too far away for the same timing core
  - `core_correct=false`: "evening run" does not identify the same morning-walk routine field
- habit detail_quality examples
  - `detail_quality=2`: the requested field's exact days, exact time, or full compatible location specificity is present
  - `detail_quality=1`: timing is within 10 minutes but not exact, one scheduled weekday is missing, or the location has the right core but misses required specificity, such as "sofa" for "sofa in living room"
  - `detail_quality=0`: timing differs by more than 10 minutes, or the requested field's detail is missing, unusable, or contradictory"""
        example_input = {
            "state_key": "habits_state:morning_walk",
            "golden": {
                "schedule": {"frequency_type": "weekly", "days_of_week": [0, 2, 4]},
                "timing": {"start_time": "06:30"},
                "location": "lakefront trail",
            },
            "predicted": {
                "schedule": {"frequency_type": "weekly", "days_of_week": "Monday, Wednesday, and Friday"},
                "timing": {"start_time": "06:33"},
                "location": "lakefront trail entrance",
            },
            "fields_to_judge": [
                "schedule.frequency_type",
                "schedule.days_of_week",
                "timing.start_time",
                "location",
            ],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "schedule.frequency_type",
                    "analysis": "The prediction describes a weekly routine.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
                {
                    "field_path": "schedule.days_of_week",
                    "analysis": "Monday, Wednesday, and Friday exactly match weekday indexes 0, 2, and 4.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
                {
                    "field_path": "timing.start_time",
                    "analysis": "The prediction gives 06:33, which is within 5 minutes of 06:30 but not exact.",
                    "core_correct": True,
                    "detail_quality": 1,
                },
                {
                    "field_path": "location",
                    "analysis": "The prediction keeps the lakefront trail location and adds compatible entrance-level specificity.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
            ]
        }
        return guidance, example_input, example_output

    if category == "preference":
        guidance = """- How to identify core vs detail for preference fields
  - Use `state_key` to identify what preference is being evaluated, such as learning format, travel style, investment style, diet, communication, or service choices
  - For `statement`, core is the preference direction plus the main object or constraint: what the user primarily prefers, avoids, prioritizes, or requires
  - Secondary constraints, qualifiers, tradeoffs, scope limits, named examples, intensity, and explicit contrast options are details unless they define the preference itself
  - If the reference says "X over Y", preserving X is usually core; preserving the contrast with Y is usually detail unless Y is the main decision boundary
  - If the reference is mainly an avoidance preference, then the avoided option is core
  - If the prediction only mentions the broad topic but not the user's preference direction, the core is missing
- preference core_correct examples
  - `core_correct=true`: "prefers webinars and written materials" preserves the main preferred learning formats
  - `core_correct=true`: "avoids live conferences" preserves the core when the field is mainly about avoiding live conferences
  - `core_correct=false`: "prefers live conferences" reverses the preference
- preference detail_quality examples
  - `detail_quality=2`: all major qualifiers, constraints, scope limits, and avoided options from the reference are present
  - `detail_quality=1`: one or more details are missing, such as the self-paced qualifier or the avoidance of live conferences
  - `detail_quality=0`: the requested field's preference details are missing, unusable, or contradictory"""
        example_input = {
            "state_key": "preferences_state:learning_format",
            "golden": {
                "statement": "Prefers self-paced webinars and written guides over live conferences for professional learning"
            },
            "predicted": {
                "statement": "The user prefers webinars and written materials for professional learning."
            },
            "fields_to_judge": ["statement"],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "statement",
                    "analysis": "The prediction captures the preference for webinars and written materials, but omits the self-paced requirement and the avoidance of live conferences.",
                    "core_correct": True,
                    "detail_quality": 1,
                }
            ]
        }
        return guidance, example_input, example_output

    if category == "attribute":
        guidance = """- How to identify core vs detail for attribute fields
  - Use `state_key` to identify what stable personal fact is being evaluated, such as a subscription, relationship, role, membership, tool, organization, place, skill, or contribution
  - For `field_path = value`, judge the whole scalar or list attribute entry
  - Core is the main factual entity, relationship, role, object, organization, place, tool, or stable fact named by the requested field
  - Tiers, versions, branch names, addresses, departments, dates, scope, qualifiers, and extra specificity are details unless they change the identity of the fact
  - If the prediction gives a different entity, relationship, role, or object, the core is wrong even if the broad topic overlaps
- attribute core_correct examples
  - `core_correct=true`: "Spotify" identifies the same service as "Spotify Premium", though the tier is missing
  - `core_correct=true`: the same organization is named, even if the prediction omits a department, branch, or address
  - `core_correct=false`: a different service, organization, relationship, or role is given
- attribute detail_quality examples
  - `detail_quality=2`: important qualifiers such as tier, version, role, branch, address, scope, or plan type are present
  - `detail_quality=1`: one or more qualifiers are missing, such as Premium or family-plan details for a subscription
  - `detail_quality=0`: the requested field's attribute details are missing, unusable, or contradictory"""
        example_input = {
            "state_key": "user_attributes_state:music_subscription",
            "golden": "Spotify Premium family plan",
            "predicted": "The user has Spotify.",
            "fields_to_judge": ["value"],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "value",
                    "analysis": "The field path `value` refers to the whole scalar attribute entry. The prediction identifies Spotify as the same subscription service, but omits Premium and family-plan details.",
                    "core_correct": True,
                    "detail_quality": 1,
                }
            ]
        }
        return guidance, example_input, example_output

    guidance = """- How to identify core vs detail for profile fields
  - Use `state_key`, `field_path`, and `golden` to identify the requested field's main personal fact or pattern
  - Core is the requested field's central meaning; details are exact names, values, qualifiers, scope, and constraints"""
    example_input = {
        "state_key": "profile:example",
        "golden": {"value": "Uses a named service for a recurring personal activity"},
        "predicted": {"value": "Uses the same named service"},
        "fields_to_judge": ["value"],
    }
    example_output = {
        "field_judgments": [
            {
                "field_path": "value",
                "analysis": "The prediction captures the main value but omits the recurring personal-activity scope.",
                "core_correct": True,
                "detail_quality": 1,
            }
        ]
    }
    return guidance, example_input, example_output


def build_snapshot_holistic_judge_prompt(
    *,
    state_key: str,
    golden_state_value: Any,
    predicted_state_value: Any,
    fields_to_judge: List[Dict[str, Any]],
) -> str:
    category = _snapshot_holistic_category(state_key)
    category_guidance, example_input, example_output = _snapshot_holistic_category_prompt_parts(category)
    return """[Task Instruction]
You are evaluating whether a predicted personal profile entry matches a reference profile entry.
Use a Core + Detail field evaluation method: judge each requested field separately, using the full prediction as context.
For every field, first identify the field's core meaning and supporting details, then score core correctness and detail quality separately.

[Definitions]
- state_key
  - a label for the profile entry being evaluated; use it to understand the entry type and topic before deciding what is core
- golden
  - the reference profile entry to evaluate against
- predicted
  - the model's predicted profile entry
- fields_to_judge
  - the exact field paths to judge; return one judgment for each field path
  - if `golden` is a scalar or list instead of an object, the field path `value` means the entire profile entry
- field_path
  - the requested field identifier; copy it exactly into the output
  - `value` is a special field path for judging the whole scalar or list entry
- core meaning
  - the requested field's central meaning after considering `state_key`, `field_path`, and `golden`
  - core is the part that must be present for the field to be practically the same profile information
- detail
  - supporting precision beyond the core, such as exact wording, exact time, exact encoding, qualifiers, constraints, tier/version, branch/address, examples, or scope
- core_correct
  - the score for whether `predicted` captures the requested field's core meaning
- detail_quality
  - the score for how completely and accurately `predicted` captures the field's details
  - `2`: key details are complete and accurate
  - `1`: important details are missing, vague, or slightly imprecise
  - `0`: details are mostly missing, wrong, contradictory, or unsupported
- semantic equivalent
  - the same practical meaning despite harmless wording or formatting differences, such as weekday names versus weekday indexes where 0=Monday and 6=Sunday
{category_guidance}

[Constraints]
1. Do not use task-pack point criteria, point ids, or per-field point accounting.
2. For each field, first identify its core meaning from `state_key`, `field_path`, and `golden`; then score `predicted`.
3. Judge every requested `field_path` independently, but use the full predicted profile entry as context.
4. Return every requested `field_path` exactly once.
5. Set `core_correct` to `true` when the requested field's core meaning is correct even if details are incomplete.
6. Set `core_correct` to `false` when the prediction omits the field, contradicts the field, gives a different core value, or is too vague to identify the same field meaning.
7. Use `detail_quality` only for detail completeness and precision; do not use it to override `core_correct`.
8. Do not require exact wording, JSON field names, key order, or identical formatting when the meaning is semantically equivalent.
9. If `core_correct` is `false`, `detail_quality` should normally be `0` unless the prediction contains some accurate but non-core details.
10. In every field judgment, write `analysis` before `core_correct` and `detail_quality`.

[Example]
[Example Input]
{example_input}

[Example Output]
{example_output}

[Input/Output Format]
Input:
- `state_key`
- `golden`
- `predicted`
- `fields_to_judge`

Output JSON ONLY:
{{
  "field_judgments": [
    {{
      "field_path": "<field_path>",
      "analysis": "<brief analysis before the labels>",
      "core_correct": "<bool>",
      "detail_quality": "<0|1|2 integer>"
    }}
  ]
}}

[Input]
{actual_input}
""".format(
        category_guidance=category_guidance,
        example_input=json.dumps(example_input, ensure_ascii=False, indent=2),
        example_output=json.dumps(example_output, ensure_ascii=False, indent=2),
        actual_input=json.dumps(
            {
                "state_key": str(state_key or ""),
                "golden": golden_state_value,
                "predicted": predicted_state_value,
                "fields_to_judge": [
                    str(field.get("field_path") or "")
                    for field in fields_to_judge
                    if isinstance(field, dict) and str(field.get("field_path") or "")
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def _build_task_b_change_slot_judge_prompt(
    *,
    state_key: str,
    slots: List[Dict[str, Any]],
) -> str:
    example_input = {
        "prediction": {
            "before": {"timing": {"start_time": "06:30"}},
            "after": {"timing": {"start_time": "07:00"}},
            "change_reason": "The routine shifted later after a schedule change.",
        },
        "checklist": [
            {
                "point_id": "1",
                "criterion": "The previous routine start time is 06:30.",
            },
            {
                "point_id": "2",
                "criterion": "The new routine start time is 07:00.",
            },
            {
                "point_id": "3",
                "criterion": "The change reason says the routine shifted later.",
            },
        ],
    }
    example_output = {
        "judgments": [
            {
                "point_id": "1",
                "analysis": "The predicted previous value matches 06:30.",
                "correct": True,
            },
            {
                "point_id": "2",
                "analysis": "The predicted new value matches 07:00.",
                "correct": True,
            },
            {
                "point_id": "3",
                "analysis": "The reason correctly describes a later shift.",
                "correct": True,
            },
        ]
    }
    prompt = _render_slot_judge_prompt(
        task_label="a predicted state change",
        unit_label="state-change prediction",
        slots=slots,
        prediction=_prediction_from_slots(slots),
        extra_constraints="10. Judge previous value, new value, and change-reason checklist items independently; a strong change reason does not make wrong previous/new values correct.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt


def _build_task_c_v2_structured_slot_judge_prompt(
    *,
    state_key: str,
    qa_id: str,
    service_family: str,
    scenario: str,
    task_instruction: str,
    reference_output: Any,
    predicted_output: Any,
    slots: List[Dict[str, Any]],
) -> str:
    prompt_slots = _strip_slot_polarity(slots)
    normalized_family = str(service_family or "").strip()
    if normalized_family == "action_configuration":
        task_label = "a structured action setup output"
        unit_label = "structured action setup output"
        example_input = {
            "prediction": {
                "membership_directory_submission": {
                    "affiliation": {
                        "organization_name": "American Coatings Association",
                        "community_type": "industry-focused professional networking community",
                    }
                }
            },
            "checklist": [
                {
                    "point_id": "1",
                    "criterion": "The structured output correctly sets `membership_directory_submission.affiliation.organization_name` to American Coatings Association.",
                },
                {
                    "point_id": "2",
                    "criterion": "The structured output describes the affiliation as an industry-focused professional networking community.",
                },
            ],
        }
        example_output = {
            "judgments": [
                {
                    "point_id": "1",
                    "analysis": "The predicted structured output uses American Coatings Association as the organization name.",
                    "correct": True,
                },
                {
                    "point_id": "2",
                    "analysis": "The predicted structured output identifies the community as industry-focused professional networking.",
                    "correct": True,
                },
            ]
        }
    else:
        task_label = "a structured information request output"
        unit_label = "structured information request output"
        example_input = {
            "prediction": {
                "allocation_parameters": {
                    "time_frame": "long-term",
                    "desired_outcomes": ["tax-efficient growth"],
                }
            },
            "checklist": [
                {
                    "point_id": "1",
                    "criterion": "The structured output correctly sets `allocation_parameters.time_frame` to a long-term horizon.",
                },
                {
                    "point_id": "2",
                    "criterion": "The structured output includes tax-efficient growth as a desired outcome.",
                },
            ],
        }
        example_output = {
            "judgments": [
                {
                    "point_id": "1",
                    "analysis": "The predicted structured output uses a long-term time frame.",
                    "correct": True,
                },
                {
                    "point_id": "2",
                    "analysis": "The predicted structured output includes tax-efficient growth in desired outcomes.",
                    "correct": True,
                },
            ]
        }
    prompt = _render_slot_judge_prompt(
        task_label=task_label,
        unit_label=unit_label,
        slots=prompt_slots,
        prediction=predicted_output,
        extra_constraints="10. Judge each checklist item against the full predicted structured output; do not give credit for overall usefulness unless the checklist item is satisfied.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt


def _build_task_c_v2_user_communication_slot_judge_prompt(
    *,
    state_key: str,
    qa_id: str,
    service_family: str,
    scenario: str,
    task_instruction: str,
    reference_answer: str,
    predicted_answer: str,
    slots: List[Dict[str, Any]],
) -> str:
    prompt_slots = _strip_slot_polarity(slots)
    example_input = {
        "prediction": "Morning walk starts at 06:30 on the lakefront trail, so I am sending your reminder now.",
        "checklist": [
            {
                "point_id": "1",
                "criterion": "The message should mention that the walk starts at 06:30.",
            },
            {
                "point_id": "2",
                "criterion": "The message should mention the lakefront trail location.",
            },
        ],
    }
    example_output = {
        "judgments": [
            {
                "point_id": "1",
                "analysis": "The predicted message explicitly mentions the 06:30 start time.",
                "correct": True,
            },
            {
                "point_id": "2",
                "analysis": "The predicted message explicitly mentions the lakefront trail.",
                "correct": True,
            },
        ]
    }
    prompt = _render_slot_judge_prompt(
        task_label="an assistant message",
        unit_label="assistant message",
        slots=prompt_slots,
        prediction=predicted_answer,
        extra_constraints="10. Judge each checklist item against the full predicted message; do not require wording overlap beyond what the checklist item itself requires.\n11. If a checklist item asks whether the message is about the targeted routine, mark it incorrect when the message is mainly about a different routine or unrelated task, even if it mentions overlapping details such as day or time.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt


def _build_legacy_task_c_apply_slot_judge_prompt(
    *,
    state_key: str,
    qa_id: str,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    slots: List[Dict[str, Any]],
) -> str:
    example_input = {
        "prediction": "Use self-paced white papers and webinars rather than a live conference.",
        "checklist": [
            {
                "point_id": "1",
                "criterion": "The answer recommends a self-paced learning format.",
            },
            {
                "point_id": "2",
                "criterion": "The answer avoids a live conference recommendation.",
            },
        ],
    }
    example_output = {
        "judgments": [
            {
                "point_id": "1",
                "analysis": "The predicted answer explicitly recommends a self-paced format.",
                "correct": True,
            },
            {
                "point_id": "2",
                "analysis": "The predicted answer explicitly avoids a live conference recommendation.",
                "correct": True,
            },
        ]
    }
    prompt = _render_slot_judge_prompt(
        task_label="a personalized assistant answer",
        unit_label="assistant answer",
        slots=slots,
        prediction=predicted_answer,
        extra_constraints="10. Judge each checklist item independently against the full prediction.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt


def _apply_holistic_service_type(service_family: str) -> str:
    normalized = str(service_family or "").strip()
    if normalized == "user_communication":
        return "assistant_message"
    if normalized == "action_configuration":
        return "structured_action_setup"
    if normalized == "information_request_construction":
        return "structured_information_request"
    return "personalized_assistant_answer"


def _apply_holistic_prompt_parts(service_type: str) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    if service_type == "assistant_message":
        guidance = """- How to identify core vs detail for assistant-message checklist fields
  - Judge each requested field against the full reference message, full predicted message, and service context
  - Core is the checklist field's main practical requirement, such as the targeted routine, schedule, time, location, or action
  - Details are exact times, dates, locations, constraints, names, reminders, and other grounded specifics
  - If the requested field checks the targeted routine and the predicted message is about a different routine, task, person, or service moment, the core is wrong even if some details overlap
- assistant-message core_correct examples
  - `core_correct=true`: a reminder about the same morning walk preserves the core even if one route detail is missing
  - `core_correct=true`: a field asking for the start time is satisfied when the predicted message includes the correct start time
  - `core_correct=false`: a message about an evening run does not preserve the core of a morning-walk reminder
- assistant-message detail_quality examples
  - `detail_quality=2`: all important timing, location, and action details from the reference are present
  - `detail_quality=1`: one important detail is missing or slightly vague, such as omitting the route while keeping the correct routine and time
  - `detail_quality=0`: details are mostly missing, wrong, contradictory, or unsupported"""
        example_input = {
            "scenario": "It is Monday morning and the assistant is preparing a routine reminder.",
            "task_instruction": "Write the assistant message to send now.",
            "reference": "Your morning walk starts at 06:30 on the lakefront trail, so it is time to head out.",
            "predicted": "Your morning walk starts around 06:30. I am sending the reminder now.",
            "fields_to_judge": [
                {
                    "field_path": "timing.start_time",
                    "criterion": "The message should include the 06:30 start time.",
                    "reference_value": "06:30",
                },
                {
                    "field_path": "location",
                    "criterion": "The message should include the lakefront trail location.",
                    "reference_value": "lakefront trail",
                },
            ],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "timing.start_time",
                    "analysis": "The predicted message includes the 06:30 start time.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
                {
                    "field_path": "location",
                    "analysis": "The predicted message omits the lakefront trail location.",
                    "core_correct": False,
                    "detail_quality": 0,
                },
            ]
        }
        return guidance, example_input, example_output

    if service_type == "structured_action_setup":
        guidance = """- How to identify core vs detail for structured action setup fields
  - Use the scenario and task instruction to understand what real action or setup the assistant is completing
  - For each requested structured field, core is the correct action-relevant entity, choice, relationship, or value for that field
  - Details are tier, plan, address, branch, role, scope, version, notes, dates, and other precision inside the same action setup
  - A broadly useful but different setup value is not core-correct for the requested field
- structured action core_correct examples
  - `core_correct=true`: "Tawuniya" preserves the provider core for "Premium home insurance policy from Tawuniya"
  - `core_correct=true`: the same coverage type is selected even if the tier or scope detail is shorter
  - `core_correct=false`: a different bank, organization, role, or configuration choice is selected
- structured action detail_quality examples
  - `detail_quality=2`: all important qualifiers needed by the field are present
  - `detail_quality=1`: one or more qualifiers are missing, such as Premium tier, full coverage scope, branch, account purpose, or role scope
  - `detail_quality=0`: field details are missing, unusable, contradictory, or unsupported"""
        example_input = {
            "scenario": "The user is updating a property management dashboard. The assistant is filling the insurance section of the asset profile before saving the changes.",
            "task_instruction": "Fill the setup or form fields that should be applied now to complete the configuration.",
            "reference": {
                "property_insurance_config": {
                    "policy_provider": "Premium home insurance policy from Tawuniya",
                    "coverage_scope": "comprehensive coverage for villa and contents",
                }
            },
            "predicted": {
                "property_insurance_config": {
                    "policy_provider": "Tawuniya",
                    "coverage_scope": "Comprehensive Villa Insurance",
                }
            },
            "fields_to_judge": ["property_insurance_config.policy_provider", "property_insurance_config.coverage_scope"],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "property_insurance_config.policy_provider",
                    "analysis": "The prediction preserves the Tawuniya provider core but omits the premium home-insurance policy qualifier.",
                    "core_correct": True,
                    "detail_quality": 1,
                },
                {
                    "field_path": "property_insurance_config.coverage_scope",
                    "analysis": "The prediction captures comprehensive villa insurance but is less explicit about contents coverage.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
            ]
        }
        return guidance, example_input, example_output

    guidance = """- How to identify core vs detail for structured search/filter fields
  - Use the scenario and task instruction to understand what search, filter, comparison, or planning request the assistant is completing
  - For each requested structured field, core is the correct search/filter intent, preferred option, avoided option, constraint, or desired outcome for that field
  - Details are qualifiers, intensity, named examples, exclusions, scope limits, ordering, and format requirements inside the same information request
  - If the prediction gives the opposite preference direction or a different filter value, the core is wrong
- structured search/filter core_correct examples
  - `core_correct=true`: "home gym" preserves the indoor-environment core for climate-controlled exercise
  - `core_correct=true`: "avoid outdoor activities" preserves the core when the field is specifically an exclusion filter
  - `core_correct=false`: leaving an exclusion blank omits the excluded-environment core
- structured search/filter detail_quality examples
  - `detail_quality=2`: all important qualifiers, exclusions, and scope limits for the field are present
  - `detail_quality=1`: a qualifier or exclusion is missing, such as climate-controlled or outdoor-activity contrast
  - `detail_quality=0`: field details are missing, unusable, contradictory, or unsupported"""
    example_input = {
        "scenario": "The user is exploring local fitness facilities and workout classes. The assistant is configuring search parameters to narrow down the available options.",
        "task_instruction": "Fill the search filters the assistant should apply now before showing matches.",
        "reference": {
            "fitness_search_criteria": {
                "preferred_environment": "climate-controlled indoor exercise environments",
                "excluded_environment": "outdoor activities",
            }
        },
        "predicted": {
            "fitness_search_criteria": {
                "preferred_environment": "Home Gym",
                "excluded_environment": "",
            }
        },
        "fields_to_judge": ["fitness_search_criteria.preferred_environment", "fitness_search_criteria.excluded_environment"],
    }
    example_output = {
        "field_judgments": [
            {
                "field_path": "fitness_search_criteria.preferred_environment",
                "analysis": "The prediction keeps the indoor exercise-environment core, but narrows it to home gyms and omits the climate-controlled qualifier.",
                "core_correct": True,
                "detail_quality": 1,
            },
            {
                "field_path": "fitness_search_criteria.excluded_environment",
                "analysis": "The prediction leaves the excluded environment blank, so it does not exclude outdoor activities.",
                "core_correct": False,
                "detail_quality": 0,
            },
        ]
    }
    return guidance, example_input, example_output


def _apply_holistic_prompt_json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _apply_holistic_render_fields_for_prompt(fields_to_judge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rendered: List[Dict[str, Any]] = []
    for field in fields_to_judge:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path") or "").strip()
        if not field_path:
            continue
        item: Dict[str, Any] = {"field_path": field_path}
        if field.get("criterion"):
            item["criterion"] = str(field.get("criterion") or "")
        if "reference_value" in field:
            item["reference_value"] = _apply_holistic_prompt_json_safe(field.get("reference_value"))
        elif "golden_field_value" in field:
            item["reference_value"] = _apply_holistic_prompt_json_safe(field.get("golden_field_value"))
        if "predicted_field_value_at_same_path" in field:
            item["predicted_value"] = _apply_holistic_prompt_json_safe(field.get("predicted_field_value_at_same_path"))
        rendered.append(item)
    return rendered


def build_apply_holistic_judge_prompt(
    *,
    state_key: str,
    qa_id: str,
    service_family: str,
    scenario: str,
    task_instruction: str,
    reference_answer: str = "",
    predicted_answer: str = "",
    reference_output: Any = None,
    predicted_output: Any = None,
    fields_to_judge: List[Dict[str, Any]],
) -> str:
    service_type = _apply_holistic_service_type(service_family)
    guidance, example_input, example_output = _apply_holistic_prompt_parts(service_type)
    reference = reference_answer if service_type == "assistant_message" or reference_output is None else reference_output
    predicted = predicted_answer if service_type == "assistant_message" or reference_output is None else predicted_output
    return """[Task Instruction]
You are evaluating whether a predicted assistant response matches the reference response for the same personalized service situation.
Use a Core + Detail field evaluation method for service fields: judge each requested assistant-message or structured-output field separately, using the full service context.
For every field, first identify the field's core service value and supporting service details, then score core correctness and detail quality separately.

[Definitions]
- scenario
  - the concrete service moment or user situation
- task_instruction
  - what the assistant is supposed to complete
- reference
  - the reference assistant message or structured service output
- predicted
  - the model's predicted assistant message or structured service output
- fields_to_judge
  - the exact field paths to judge; return one judgment for each field path
  - if `reference` is a scalar instead of an object, `value` means the entire response
- field_path
  - the requested field identifier; copy it exactly into the output
- core service value
  - the requested field's central service meaning after considering `scenario`, `task_instruction`, `field_path`, `fields_to_judge`, and `reference`
  - core is the part that must be present for the predicted response to be practically useful for the same personalized service moment
  - core belongs to the service output field being judged, not to a raw source record
- detail
  - supporting precision beyond the core, such as exact time, date, place, cadence, exact encoding, qualifiers, exclusions, constraints, tier/version, branch/address, examples, or scope
- core_correct
  - the score for whether `predicted` captures the requested field's core service value
- detail_quality
  - the score for how completely and accurately `predicted` captures the field's details
  - `2`: key details are complete and accurate
  - `1`: important details are missing, vague, or slightly imprecise
  - `0`: details are mostly missing, wrong, contradictory, or unsupported
- semantic equivalent
  - the same practical service meaning despite harmless wording, formatting, key naming, or ordering differences
{guidance}

[Constraints]
1. Do not use scoring points, point ids, rubrics, or per-point accounting.
2. For each field, first identify its core service value from `scenario`, `task_instruction`, `field_path`, `fields_to_judge`, and `reference`; then score `predicted`.
3. Judge every requested `field_path` independently, but use the full predicted response as context.
4. Return every requested `field_path` exactly once.
5. Set `core_correct` to `true` when the requested field's core service value is correct even if details are incomplete.
6. Set `core_correct` to `false` when the prediction omits the field, contradicts the field, gives a different core value, targets a different service moment, or is too vague to identify the same field meaning.
7. Use `detail_quality` only for detail completeness and precision; do not use it to override `core_correct`.
8. Do not require exact wording, JSON field names, key order, or identical formatting when the meaning is semantically equivalent.
9. Do not penalize the prediction for not restating source records when the service output is correct, but do penalize missing details needed for the service response to be specific and actionable.
10. If `core_correct` is `false`, `detail_quality` should normally be `0` unless the prediction contains some accurate but non-core details.
11. In every field judgment, write `analysis` before `core_correct` and `detail_quality`.

[Example]
[Example Input]
{example_input}

[Example Output]
{example_output}

[Input/Output Format]
Input:
- `scenario`
- `task_instruction`
- `reference`
- `predicted`
- `fields_to_judge`

Output JSON ONLY:
{{
  "field_judgments": [
    {{
      "field_path": "<field_path>",
      "analysis": "<brief analysis before the labels>",
      "core_correct": "<bool>",
      "detail_quality": "<0|1|2 integer>"
    }}
  ]
}}

[Input]
{actual_input}
""".format(
        guidance=guidance,
        example_input=json.dumps(example_input, ensure_ascii=False, indent=2),
        example_output=json.dumps(example_output, ensure_ascii=False, indent=2),
        actual_input=json.dumps(
            {
                "scenario": str(scenario or ""),
                "task_instruction": str(task_instruction or ""),
                "reference": reference,
                "predicted": predicted,
                "fields_to_judge": _apply_holistic_render_fields_for_prompt(fields_to_judge),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def build_snapshot_slot_judge_prompt(
    *,
    state_key: str,
    slots: List[Dict[str, Any]],
    prediction: Any = None,
) -> str:
    return _build_task_a_slot_judge_prompt(
        state_key=state_key,
        slots=slots,
        prediction=prediction,
    )


def build_change_slot_judge_prompt(
    *,
    state_key: str,
    slots: List[Dict[str, Any]],
) -> str:
    return _build_task_b_change_slot_judge_prompt(
        state_key=state_key,
        slots=slots,
    )


def build_apply_slot_judge_prompt(
    *,
    state_key: str,
    qa_id: str,
    question: str = "",
    reference_answer: str = "",
    predicted_answer: str = "",
    service_family: str = "",
    scenario: str = "",
    task_instruction: str = "",
    reference_output: Any = None,
    predicted_output: Any = None,
    slots: List[Dict[str, Any]],
) -> str:
    normalized_family = str(service_family or "").strip()
    structured_v2 = bool(
        (service_family or scenario or task_instruction or reference_output is not None or predicted_output is not None)
        and normalized_family != "user_communication"
    )
    if structured_v2:
        return _build_task_c_v2_structured_slot_judge_prompt(
            state_key=state_key,
            qa_id=qa_id,
            service_family=service_family,
            scenario=scenario,
            task_instruction=task_instruction,
            reference_output=reference_output,
            predicted_output=predicted_output,
            slots=slots,
        )

    if normalized_family == "user_communication":
        return _build_task_c_v2_user_communication_slot_judge_prompt(
            state_key=state_key,
            qa_id=qa_id,
            service_family=normalized_family,
            scenario=scenario,
            task_instruction=task_instruction,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer,
            slots=slots,
        )

    return _build_legacy_task_c_apply_slot_judge_prompt(
        state_key=state_key,
        qa_id=qa_id,
        question=question,
        reference_answer=reference_answer,
        predicted_answer=predicted_answer,
        slots=slots,
    )
