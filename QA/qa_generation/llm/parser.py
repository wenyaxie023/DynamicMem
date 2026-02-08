from __future__ import annotations

from typing import Any, Dict


def parse_qa_response(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, Exception):
        raise resp

    obj = resp
    if isinstance(obj, list):
        if not obj:
            raise ValueError("Expected non-empty list response")
        first_dict = next((x for x in obj if isinstance(x, dict)), None)
        if first_dict is None:
            raise ValueError("Expected at least one object in response list")
        obj = first_dict

    if not isinstance(obj, dict):
        raise ValueError("Expected object or list[object] from LLM")

    question = obj.get("question")
    answer = obj.get("answer")
    evidence = obj.get("evidence")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing valid 'question' field")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Missing valid 'answer' field")

    return {
        "query": question,
        "reference": answer,
        "prediction": "",
        "metadata": {
            "reference_evidence": evidence,
            "draft_question": obj.get("draft1", ""),
            "draft_answer": obj.get("draft2", ""),
        },
    }
