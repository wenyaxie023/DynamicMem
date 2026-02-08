from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, StrictUndefined

from .prompts import DOUBLECHECK_PROMPT, JUDGE_PROMPT, TYPE1, TYPE2, TYPE3, TYPE4, TYPE5, TYPE6


TYPE_TEMPLATES = {
    1: TYPE1,
    2: TYPE2,
    3: TYPE3,
    4: TYPE4,
    5: TYPE5,
    6: TYPE6,
}

_JINJA_ENV = Environment(
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
)


def _normalize_qtype(qtype: Any) -> int:
    if isinstance(qtype, int):
        return qtype
    if isinstance(qtype, str):
        value = qtype.strip().lower()
        if value.startswith("t") and value[1:].isdigit():
            return int(value[1:])
        if value.isdigit():
            return int(value)
    return 1


def render_qa_prompt(
    *,
    task: Any,
    context: Any,
    qtype: Any = None,
) -> str:
    qtype_num = _normalize_qtype(qtype)
    template = TYPE_TEMPLATES.get(qtype_num, TYPE1)

    task_json = json.dumps(task, ensure_ascii=False, indent=2)
    context_json = json.dumps(context, ensure_ascii=False, indent=2)

    return _JINJA_ENV.from_string(template).render(
        TASK_JSON=task_json,
        CONTEXT_TEXT=context_json,
    )


def render_judge_prompt(
    *,
    question: Any,
    answer: Any,
    context: Any,
) -> str:
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return _JINJA_ENV.from_string(JUDGE_PROMPT).render(
        QUESTION="" if question is None else str(question),
        ANSWER="" if answer is None else str(answer),
        CONTEXT_JSON=context_json,
    )


def render_doublecheck_prompt(
    *,
    question: Any,
    answer: Any,
    context: Any,
) -> str:
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return _JINJA_ENV.from_string(DOUBLECHECK_PROMPT).render(
        QUESTION="" if question is None else str(question),
        ANSWER="" if answer is None else str(answer),
        CONTEXT_JSON=context_json,
    )
