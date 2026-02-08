from __future__ import annotations

from typing import Any, Dict, List

from qa_generation.io import read_json_list, write_json
from qa_generation.resolvers import EvidenceResolver
from shared.config import GenerationConfig


def _normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str) and x]


def _build_qa_context_rows(
    *,
    qa_rows: List[Dict[str, Any]],
    resolver: EvidenceResolver,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in qa_rows:
        uid = row.get("uid")
        if not isinstance(uid, str) or not uid:
            continue
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            context_logs: List[Dict[str, Any]] = []
        else:
            log_ids = _normalize_str_list(metadata.get("reference_evidence"))
            context_logs = resolver.context_logs_from_log_ids(log_ids)
        output.append({"uid": uid, "context": context_logs})
    return output


def run_build_qa_context(
    *,
    config: GenerationConfig,
) -> int:
    qa_rows = read_json_list(config.qa_output_path, allow_missing=False)
    raw_rows = read_json_list(config.raw_path, allow_missing=False)
    resolver = EvidenceResolver(raw_rows)
    output = _build_qa_context_rows(qa_rows=qa_rows, resolver=resolver)
    write_json(config.qa_context_path, output)
    print(f"Saved QA context: {config.qa_context_path} (count={len(output)})")
    return 0

