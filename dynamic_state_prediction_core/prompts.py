import json
from typing import Any, Dict, List, Optional


def build_generation_prompt(
    *,
    context_logs: List[Dict[str, Any]],
    target_keys: List[str],
    target_value_templates: Dict[str, Any],
    log_to_text,
    retrieval_query: Optional[str] = None,
    context_note: str = "Context app logs",
) -> str:
    context = "\n<->\n".join(log_to_text(log) for log in context_logs)
    snapshot_template = {k: target_value_templates.get(k, "<fill the blank>") for k in target_keys}
    evidence_template = {k: ["<app_log_id>"] for k in target_keys}
    fill_template = {
        "snapshot_state": snapshot_template,
        "evidence": evidence_template,
    }
    fill_template_block = json.dumps(fill_template, ensure_ascii=False, indent=2)

    return f"""Based on user memory, predict the values for the following state keys:

User memory:
{context}

Fill this dict template:
{fill_template_block}

Output JSON ONLY with this schema:
{{
  "snapshot_state": {{
    "<key>": "<value or nested object following the template>",
    "...": "<same structure as template>"
  }},
  "evidence": {{
    "<key>": ["<app_log_id>", "..."],
    "...": []
  }}
}}

Rules:
1. Use only evidence implied by logs.
2. MUST include every key in the template exactly once in snapshot_state and evidence.
3. Keep exactly the same nested key structure in snapshot_state. Only fill leaf values.
4. If a leaf value is unresolvable from logs, use null.
5. evidence for each key must be a list of app_log_id strings from provided user memory.
6. If evidence is unknown, use empty list.
7. app_log_id strings in evidence must exactly match the IDs shown in user memory (e.g., "log_00028" must stay "log_00028", never "log_28").
8. No markdown. No extra keys.
"""
