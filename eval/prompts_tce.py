import json
from typing import Any, Dict, List


def _slot_judgment_template(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "point_id": str(slot.get("point_id") or ""),
            "analysis": "<brief slot-level judgment>",
            "correct": "<bool>",
        }
        for slot in slots
    ]


def _build_slot_judge_prompt(
    *,
    task_label: str,
    unit_label: str,
    slots: List[Dict[str, Any]],
    extra_definitions: str = "",
    extra_constraints: str = "",
    example_input: Dict[str, Any],
    example_output: Dict[str, Any],
) -> str:
    return """[Task Instruction]
You are evaluating {task_label} with slot-level atomic-fact judgment.
For this {unit_label}, judge each slot independently and decide whether the prediction satisfies that slot.
Success means returning strict boolean correctness for every provided `point_id`.

[Definitions]
- slot
  - one benchmark scoring point / atomic fact
- `correct = true`
  - the prediction satisfies that slot faithfully enough to count as correct
- `correct = false`
  - the prediction misses the slot, contradicts it, is too vague to satisfy it, or relies on unsupported content
- For `field` slots:
  - use `reference_value` and `predicted_value` as the main comparison anchors
  - indexed list paths such as `schedule_dates.0` are ordinary ordered field slots
- If a legacy `list_item` slot appears in older artifacts:
  - use `reference_value` and `predicted_value` as the main comparison anchors
- For `micro` slots:
  - use `point_text` as the atomic fact and compare it against `predicted_value`
- For negative slots:
  - `correct = true` only if the prediction avoids the prohibited content or behavior
{extra_definitions}

[Constraints]
1. Judge each slot independently.
2. Return every provided `point_id` exactly once.
3. Do not add or drop slots.
4. `correct` must be boolean, not a score.
5. Be strict: broad topical overlap is not enough when the slot requires a more specific value or constraint.
6. Unsupported additions should make the affected slot incorrect when they conflict with or overstep the atomic fact.
7. In every judgment object, write `analysis` before `correct`.
{extra_constraints}

[Example]
[Example Input]
{example_input}

[Example Output]
{example_output}

[Input/Output Format]
Input:
- one {unit_label}
- `slots`: list of slot objects
- every slot includes:
  - `point_id`
  - `point_type`
  - `polarity`
  - `predicted_value`
- `field` / legacy `list_item` slots additionally include:
  - `reference_value`
- `micro` slots additionally include:
  - `point_text`
- optional task-specific fields may include:
  - `target_path`
  - `slot_group`
  - `list_index`

Output JSON ONLY:
{{
  "judgments": [
    {{
      "point_id": "<point_id>",
      "analysis": "<brief slot-level judgment>",
      "correct": "<bool>"
    }}
  ]
}}

[Slots]
{slots}

Fill this exact judgments template (do not add/drop point_ids):
{judgment_template}
""".format(
        task_label=task_label,
        unit_label=unit_label,
        extra_definitions=extra_definitions,
        extra_constraints=extra_constraints,
        example_input=json.dumps(example_input, ensure_ascii=False, indent=2),
        example_output=json.dumps(example_output, ensure_ascii=False, indent=2),
        slots=json.dumps(slots, ensure_ascii=False, indent=2),
        judgment_template=json.dumps(_slot_judgment_template(slots), ensure_ascii=False, indent=2),
    )


def build_snapshot_slot_judge_prompt(
    *,
    state_key: str,
    slots: List[Dict[str, Any]],
) -> str:
    example_input = {
        "state_key": "preferences_state:learning_modality",
        "slots": [
            {
                "point_id": "scp_learning_p1",
                "point_type": "field",
                "polarity": "positive",
                "target_path": "current_value",
                "reference_value": "webinar",
                "predicted_value": "webinar",
            },
            {
                "point_id": "scp_learning_p2",
                "point_type": "micro",
                "polarity": "positive",
                "point_text": "The answer avoids a live conference recommendation.",
                "predicted_value": "Prefers webinar-based continuing education instead of live conferences.",
            },
        ],
    }
    example_output = {
        "judgments": [
            {
                "point_id": "scp_learning_p1",
                "analysis": "The predicted value matches the required preferred format.",
                "correct": True,
            },
            {
                "point_id": "scp_learning_p2",
                "analysis": "The prediction explicitly avoids live conferences.",
                "correct": True,
            },
        ]
    }
    prompt = _build_slot_judge_prompt(
        task_label="TCE Task A state completion",
        unit_label=f"state key `{state_key}`",
        slots=slots,
        extra_constraints="8. For indexed list paths, judge the provided `predicted_value` at that same ordered position only; do not search other positions for a better match.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt + "\n\n[Unit Context]\n" + json.dumps(
        {"state_key": state_key},
        ensure_ascii=False,
        indent=2,
    )


def build_change_slot_judge_prompt(
    *,
    state_key: str,
    slots: List[Dict[str, Any]],
) -> str:
    example_input = {
        "state_key": "habits_state:morning_walk",
        "slots": [
            {
                "point_id": "chg_walk_before_p1",
                "slot_group": "before",
                "point_type": "field",
                "polarity": "positive",
                "target_path": "timing.start_time",
                "reference_value": "06:30",
                "predicted_value": "06:30",
            },
            {
                "point_id": "chg_walk_after_p1",
                "slot_group": "after",
                "point_type": "field",
                "polarity": "positive",
                "target_path": "timing.start_time",
                "reference_value": "07:00",
                "predicted_value": "07:00",
            },
            {
                "point_id": "chg_walk_reason_p1",
                "slot_group": "change_reason",
                "point_type": "micro",
                "polarity": "positive",
                "point_text": "The change reason says the routine shifted later.",
                "predicted_value": "The routine shifted later after a schedule change.",
            },
        ],
    }
    example_output = {
        "judgments": [
            {
                "point_id": "chg_walk_before_p1",
                "analysis": "The predicted previous value matches 06:30.",
                "correct": True,
            },
            {
                "point_id": "chg_walk_after_p1",
                "analysis": "The predicted new value matches 07:00.",
                "correct": True,
            },
            {
                "point_id": "chg_walk_reason_p1",
                "analysis": "The reason correctly describes a later shift.",
                "correct": True,
            },
        ]
    }
    prompt = _build_slot_judge_prompt(
        task_label="TCE Task B change tracking",
        unit_label=f"changed state key `{state_key}`",
        slots=slots,
        extra_definitions="- `slot_group`\n  - indicates whether the slot belongs to `before`, `after`, or `change_reason`",
        extra_constraints="8. Judge `before`, `after`, and `change_reason` slots on their own evidence; a strong change reason does not make wrong before/after slots correct.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt + "\n\n[Unit Context]\n" + json.dumps(
        {"state_key": state_key},
        ensure_ascii=False,
        indent=2,
    )


def build_apply_slot_judge_prompt(
    *,
    state_key: str,
    qa_id: str,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    slots: List[Dict[str, Any]],
) -> str:
    example_input = {
        "state_key": "preferences_state:learning_modality",
        "qa_id": "q1",
        "question": "What training format should the assistant arrange?",
        "reference_answer": "Arrange self-paced white papers and webinars instead of a live conference.",
        "predicted_answer": "Use self-paced white papers and webinars rather than a live conference.",
        "slots": [
            {
                "point_id": "aqp_learning_p1",
                "point_type": "micro",
                "polarity": "positive",
                "point_text": "The answer recommends a self-paced learning format.",
                "predicted_value": "Use self-paced white papers and webinars rather than a live conference.",
            },
            {
                "point_id": "aqp_learning_p2",
                "point_type": "micro",
                "polarity": "positive",
                "point_text": "The answer avoids a live conference recommendation.",
                "predicted_value": "Use self-paced white papers and webinars rather than a live conference.",
            },
        ],
    }
    example_output = {
        "judgments": [
            {
                "point_id": "aqp_learning_p1",
                "analysis": "The predicted answer explicitly recommends a self-paced format.",
                "correct": True,
            },
            {
                "point_id": "aqp_learning_p2",
                "analysis": "The predicted answer explicitly avoids a live conference recommendation.",
                "correct": True,
            },
        ]
    }
    prompt = _build_slot_judge_prompt(
        task_label="TCE Task C personalized service QA",
        unit_label=f"Task C item `{state_key}::{qa_id}`",
        slots=slots,
        extra_definitions="- `reference_answer`\n  - provided as context only; do not grade the whole answer globally\n- `question`\n  - gives the service-decision context for the atomic facts",
        extra_constraints="8. Use the question and reference answer only as context for understanding the slots; still judge each slot independently.",
        example_input=example_input,
        example_output=example_output,
    )
    return prompt + "\n\n[Item Context]\n" + json.dumps(
        {
            "state_key": state_key,
            "qa_id": qa_id,
            "question": question,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
        },
        ensure_ascii=False,
        indent=2,
    )
