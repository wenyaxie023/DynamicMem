from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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


def parse_judge_response(resp: Any) -> Tuple[str, int, Optional[Dict[str, Any]]]:
    if isinstance(resp, Exception):
        raise resp
    if not isinstance(resp, dict):
        raise ValueError("Expected object for judge response")

    rationale = resp.get("rationale")
    judge = resp.get("judge")
    refine_info = resp.get("refine_info")

    if not isinstance(rationale, str):
        raise ValueError("Missing valid 'rationale' field")

    if isinstance(judge, str) and judge.isdigit():
        judge = int(judge)
    if not isinstance(judge, int):
        raise ValueError("Missing valid 'judge' field")

    normalized_judge = 1 if judge == 1 else 0
    if not isinstance(refine_info, dict):
        return rationale, normalized_judge, None
    return rationale, normalized_judge, refine_info


def parse_doublecheck_response(resp: Any) -> Tuple[str, int]:
    rationale, judge, _ = parse_judge_response(resp)
    return rationale, judge
