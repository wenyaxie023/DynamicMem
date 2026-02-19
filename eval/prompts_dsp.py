import json
from typing import Any, Dict


def build_llm_judge_prompt(value_pairs: Dict[str, Dict[str, Any]]) -> str:
    judgment_template = [
        {"key": k, "reason": "<very short>", "score": 0}
        for k in sorted(value_pairs.keys())
    ]
    return """You are evaluating dynamic state prediction value quality.

Task:
For each key independently, decide whether predicted_value is correct vs expected_value.

[Per-key Value Pairs]
{value_pairs}

Fill this exact judgments template (do not add/drop keys):
{judgment_template}

Output JSON ONLY:
{{
  "judgments": [
    {{"key": "<state_key>", "reason": "<very short>", "score": 0-10}}
  ]
}}

Rules:
1. Include each key exactly once in judgments.
2. Judge only value correctness for each key.
3. Semantic equivalence is sufficient: if predicted and expected mean the same thing, assign a score near 10 even if wording differs.
4. Do NOT penalize for paraphrases, minor phrasing differences, or missing stylistic words (e.g., "set", adjective wording) when core meaning matches.
5. Penalize only factual mismatches, missing critical facts, contradictions, or clearly less specific content that changes meaning.
6. Use a 0-10 score scale for each key:
   - 10: fully correct and complete.
   - 7-9: mostly correct with minor missing details.
   - 4-6: partially correct; some important details missing or slightly wrong.
   - 1-3: mostly incorrect but with small overlap.
   - 0: completely incorrect or contradictory.
7. For each judgment, write reason first, then assign score.
8. If uncertain, use a conservative score.
""".format(
        value_pairs=json.dumps(value_pairs, ensure_ascii=False),
        judgment_template=json.dumps(judgment_template, ensure_ascii=False),
    )
