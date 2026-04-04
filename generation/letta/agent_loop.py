import json
import os
import time
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

INGEST_PROMPT_PREFIX = "Ingest this user app log into your memory."


def sort_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(log: Dict[str, Any]):
        return (
            str(log.get("timestamp", "")),
            str(log.get("app_log_id", "")),
        )

    return sorted((x for x in logs if isinstance(x, dict)), key=_key)


def parse_json_response_text(text: Any) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    candidates = [s]
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(s[start : end + 1].strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _letta_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type", "")).strip().lower() != "text":
                continue
            text_val = part.get("text")
            if isinstance(text_val, str) and text_val.strip():
                parts.append(text_val.strip())
        return "\n".join(parts).strip()
    return ""


def estimate_usage_cost_usd(
    usage: Dict[str, Any],
    *,
    prompt_cost_per_1m: float,
    completion_cost_per_1m: float,
    cached_input_cost_per_1m: Optional[float] = None,
    cache_write_cost_per_1m: Optional[float] = None,
) -> Dict[str, float]:
    prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
    completion_tokens = max(0, int(usage.get("completion_tokens") or 0))
    cached_input_tokens = max(0, int(usage.get("cached_input_tokens") or 0))
    cache_write_tokens = max(0, int(usage.get("cache_write_tokens") or 0))
    uncached_prompt_tokens = max(0, prompt_tokens - cached_input_tokens)

    prompt_cost = uncached_prompt_tokens / 1_000_000.0 * float(prompt_cost_per_1m)
    completion_cost = completion_tokens / 1_000_000.0 * float(completion_cost_per_1m)
    cached_input_cost = (
        cached_input_tokens / 1_000_000.0 * float(cached_input_cost_per_1m)
        if cached_input_cost_per_1m is not None
        else 0.0
    )
    cache_write_cost = (
        cache_write_tokens / 1_000_000.0 * float(cache_write_cost_per_1m)
        if cache_write_cost_per_1m is not None
        else 0.0
    )
    total_cost = prompt_cost + completion_cost + cached_input_cost + cache_write_cost
    return {
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "cached_input_cost_usd": cached_input_cost,
        "cache_write_cost_usd": cache_write_cost,
        "total_cost_usd": total_cost,
    }


def _normalize_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _normalize_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    ivalue = int(value)
    if ivalue <= 0:
        return None
    return ivalue


class _LocalFallbackAgent:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def ingest_log(self, log: Dict[str, Any]) -> None:
        self.logs.append(dict(log))

    def ask_json(self, question: str) -> Dict[str, Any]:
        template = {
            "snapshot_state": {},
            "evidence": {},
        }
        marker = "Return JSON only with this exact top-level shape:\n"
        rules_marker = "\nRules:"
        if marker in question and rules_marker in question:
            try:
                raw = question.split(marker, 1)[1].split(rules_marker, 1)[0].strip()
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    template = parsed
            except Exception:
                pass
        snapshot_state = template.get("snapshot_state")
        evidence = template.get("evidence")
        if not isinstance(snapshot_state, dict):
            snapshot_state = {}
        if not isinstance(evidence, dict):
            evidence = {}
        return {
            "snapshot_state": {str(k): None for k in snapshot_state.keys()},
            "evidence": {str(k): [] for k in evidence.keys()},
        }

    def ask_text(self, question: str) -> str:
        return json.dumps(self.ask_json(question), ensure_ascii=False)


class LettaAgentLoop:
    def __init__(
        self,
        user_namespace: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        embedding: Optional[str] = None,
        answer_temperature: Optional[float] = 0.0,
        answer_top_p: Optional[float] = 1.0,
        answer_top_k: Optional[int] = None,
        mode: str = "sdk",
        allow_local_fallback: bool = False,
        persona: Optional[str] = None,
        human: Optional[str] = None,
        create_agent: bool = True,
        lease_registry_path: Optional[Path] = None,
    ):
        self._user_namespace = user_namespace
        self._lease_registry_path = lease_registry_path
        self._llm_provider = str(llm_provider or "openai")
        self._llm_model = str(llm_model or os.getenv("LETTA_MODEL", "openai/gpt-5-mini"))
        self._embedding = str(embedding or os.getenv("LETTA_EMBEDDING", "openai/text-embedding-3-large"))
        self._answer_temperature = _normalize_optional_float(answer_temperature)
        self._answer_top_p = _normalize_optional_float(answer_top_p)
        self._answer_top_k = _normalize_optional_int(answer_top_k)
        self._mode = str(mode or "sdk").strip().lower()
        self._allow_local_fallback = bool(allow_local_fallback)
        self._persona = str(persona).strip() if persona else "A professional log analyst."
        self._human = (
            str(human).strip()
            if human
            else "Analyzing user application logs for patterns and facts."
        )
        self._fallback = False
        self._fallback_agent: Optional[_LocalFallbackAgent] = None
        self.client = None
        self.agent: Optional[Any] = None
        self._usage_records: List[Dict[str, Any]] = []
        self._init_backend()
        if create_agent:
            self._create_agent()

    def create_agent(self) -> str:
        self._create_agent()
        return self._require_agent_id()

    def _init_backend(self) -> None:
        if self._mode == "local":
            self._activate_fallback()
            return
        try:
            from letta_client import Letta

            api_key = os.getenv("LETTA_API_KEY")
            base_url = os.getenv("LETTA_BASE_URL")
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = Letta(**kwargs)
        except Exception:
            if not self._allow_local_fallback:
                raise
            self._activate_fallback()

    def _activate_fallback(self) -> None:
        self._fallback = True
        self._fallback_agent = _LocalFallbackAgent()
        self.client = None

    def _create_agent(self) -> None:
        if self._fallback:
            self.agent = SimpleNamespace(id="local-fallback")
            return

        assert self.client is not None
        instruction = ""
        model_settings: Dict[str, Any] = {
            "provider_type": self._llm_provider,
            "response_format": {
                "type": "json_object",
            },
        }
        if self._answer_temperature is not None:
            model_settings["temperature"] = self._answer_temperature
        if self._answer_top_p is not None:
            model_settings["top_p"] = self._answer_top_p
        if self._answer_top_k is not None:
            model_settings["top_k"] = self._answer_top_k
        self.agent = self.client.agents.create(
            agent_type="letta_v1_agent",
            model=self._llm_model,
            embedding=self._embedding,
            context_window_limit=32000,
            memory_blocks=[
                {
                    "label": "persona",
                    "value": self._persona + instruction,
                    "description": "Core assistant persona and memory policy. Try to use tools to help manage memory efficiently. User logs can be very long, so you need keep the balance between core memory and archival memory.",
                },
                {
                    "label": "human",
                    "value": self._human,
                    "description": "User context for the active task.",
                },
            ],
            system="""You are a stateful agent processing long-term user logs. You should both accurately remember details and maintain an efficient, abstracted memory or state representation. Here are some guidelines to help you manage your memory and reasoning:
                        State Modeling:
                        User logs may implicitly reflect evolving latent states over time.
                        You may infer and maintain such states if useful,
                        You can freely organize the structure of core memory as you see fit
                        It is good to classify different types of inferred information into separate memory blocks if that helps you stay organized.
                        Retrieval Policy:
                        - Use archival memory for historical details.
                        - Use core memory only for compact, persistent abstractions.
                        - Avoid overloading core memory. User logs can be very long, so you need keep the balance between core memory and archival memory.
                        Your objective is to reason accurately over long time spans while maintaining memory efficiency.""",
            tools=[
                "memory",
                "send_message",
                "memory_apply_patch",
                "core_memory_append",
                "core_memory_replace",
                "archival_memory_insert",
                "archival_memory_search",
                "memory_rethink",
            ],
            model_settings=model_settings,
        )

    def _require_agent_id(self) -> str:
        if self._fallback:
            return "local-fallback"
        if self.agent is None:
            self._create_agent()
        assert self.agent is not None
        return str(self.agent.id)

    def active_agent_id(self) -> Optional[str]:
        if self.agent is None:
            return None
        return str(getattr(self.agent, "id", "") or "").strip() or None

    def active_agent_name(self) -> Optional[str]:
        if self.agent is None:
            return None
        name = str(getattr(self.agent, "name", "") or "").strip()
        return name or None

    def attach_agent(self, agent_id: str) -> str:
        agent_id = str(agent_id or "").strip()
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        if self._fallback:
            self.agent = SimpleNamespace(id=agent_id, name="local-fallback")
            return agent_id
        assert self.client is not None
        remote = self.client.agents.retrieve(agent_id=agent_id)
        self.agent = SimpleNamespace(
            id=str(getattr(remote, "id", agent_id)),
            name=getattr(remote, "name", None),
        )
        return str(self.agent.id)

    def _read_leases(self) -> List[str]:
        if self._lease_registry_path is None or not self._lease_registry_path.exists():
            return []
        try:
            raw = json.loads(self._lease_registry_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for x in raw:
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    def _write_leases(self, ids: List[str]) -> None:
        if self._lease_registry_path is None:
            return
        self._lease_registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lease_registry_path.write_text(
            json.dumps(ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _lease_add(self, agent_id: str) -> None:
        if self._lease_registry_path is None:
            return
        ids = self._read_leases()
        if agent_id in ids:
            return
        ids.append(agent_id)
        self._write_leases(ids)

    def _lease_remove(self, agent_id: str) -> None:
        if self._lease_registry_path is None:
            return
        ids = [x for x in self._read_leases() if x != agent_id]
        self._write_leases(ids)

    def _extract_agent_ids(self, response: Any) -> List[str]:
        if response is None:
            return []
        if isinstance(response, dict):
            raw = response.get("agent_ids") or []
            return [str(x) for x in raw if x]
        raw = getattr(response, "agent_ids", None)
        if raw is None:
            return []
        return [str(x) for x in raw if x]

    def _extract_assistant_text(self, response: Any) -> str:
        messages = getattr(response, "messages", None)
        if messages is None and isinstance(response, dict):
            messages = response.get("messages")
        if not isinstance(messages, list):
            return "No text response received from agent."

        for msg in reversed(messages):
            msg_dict = msg.to_dict() if hasattr(msg, "to_dict") else msg
            if not isinstance(msg_dict, dict):
                continue
            message_type = str(msg_dict.get("message_type", "")).strip().lower()
            if message_type != "assistant_message":
                continue
            text = _letta_content_to_text(msg_dict.get("content"))
            if text:
                return text
        return "No text response received from agent."

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        if response is None:
            return {}
        if hasattr(response, "to_dict"):
            payload = response.to_dict()
            if isinstance(payload, dict):
                return payload
        if isinstance(response, dict):
            return response
        return {}

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        payload = self._response_to_dict(response)
        raw_usage = payload.get("usage")
        if hasattr(raw_usage, "to_dict"):
            raw_usage = raw_usage.to_dict()
        if not isinstance(raw_usage, dict):
            raw_usage = {}
        record: Dict[str, Any] = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
            "reasoning_tokens": int(raw_usage.get("reasoning_tokens") or 0),
            "cached_input_tokens": int(raw_usage.get("cached_input_tokens") or 0),
            "cache_write_tokens": int(raw_usage.get("cache_write_tokens") or 0),
            "context_tokens": int(raw_usage.get("context_tokens") or 0),
            "total_tokens": int(raw_usage.get("total_tokens") or 0),
            "step_count": int(raw_usage.get("step_count") or 0),
            "run_ids": list(raw_usage.get("run_ids") or []),
        }
        stop_reason = payload.get("stop_reason")
        if hasattr(stop_reason, "to_dict"):
            stop_reason = stop_reason.to_dict()
        if isinstance(stop_reason, dict):
            record["stop_reason"] = str(stop_reason.get("stop_reason") or "") or None
        else:
            record["stop_reason"] = None
        return record

    def _record_usage(self, *, phase: str, agent_id: str, response: Any) -> Dict[str, Any]:
        usage = self._extract_usage(response)
        usage["phase"] = str(phase or "")
        usage["agent_id"] = str(agent_id or "")
        usage["backend"] = self.effective_backend()
        usage["timestamp_unix"] = time.time()
        self._usage_records.append(usage)
        return usage

    def usage_records(self) -> List[Dict[str, Any]]:
        return [dict(x) for x in self._usage_records]

    def usage_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "turn_count": len(self._usage_records),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "context_tokens": 0,
            "total_tokens": 0,
            "step_count": 0,
            "run_ids": [],
        }
        seen_run_ids = set()
        for record in self._usage_records:
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_input_tokens",
                "cache_write_tokens",
                "context_tokens",
                "total_tokens",
                "step_count",
            ):
                summary[key] += int(record.get(key) or 0)
            for run_id in record.get("run_ids") or []:
                run_id = str(run_id or "").strip()
                if not run_id or run_id in seen_run_ids:
                    continue
                seen_run_ids.add(run_id)
                summary["run_ids"].append(run_id)
        return summary

    def _format_log_as_dialogue(self, log: dict) -> str:
        return (
            f"{INGEST_PROMPT_PREFIX} "
            "Do not ask follow-up questions; just absorb it.\n"
            # f"app_log_id: {log.get('app_log_id')}\n"
            # f"timestamp: {log.get('timestamp')}\n"
            f"payload: {json.dumps(log, ensure_ascii=False)}"
        )

    def ingest_log(self, log: dict):
        if self._fallback:
            assert self._fallback_agent is not None
            self._fallback_agent.ingest_log(log)
            return None
        assert self.client is not None
        agent_id = self._require_agent_id()
        response = self.client.agents.messages.create(
            agent_id=agent_id,
            messages=[{
                "role": "user",
                "content": self._format_log_as_dialogue(log),
            }],
        )
        self._record_usage(phase="ingest_log", agent_id=agent_id, response=response)
        return response

    def ingest_logs_dialogue(self, logs: list):
        for log in logs:
            self.ingest_log(log)
        print(f"Successfully ingested {len(logs)} logs via dialogue turns.")

    def ask(self, question: str, expect_json: bool = False):
        return self.ask_with_agent(
            agent_id=self._require_agent_id(),
            question=question,
            expect_json=expect_json,
        )

    def ask_with_agent(self, agent_id: str, question: str, expect_json: bool = False):
        if self._fallback:
            assert self._fallback_agent is not None
            return self._fallback_agent.ask_text(question)

        assert self.client is not None
        if expect_json:
            question += """ Your final response must be valid JSON. 
            like {"answer": "your answer here", "reasoning": "your reasoning here"}
            """

        response = self.client.agents.messages.create(
            agent_id=agent_id,
            messages=[{
                "role": "user",
                "content": question,
            }],
        )
        self._record_usage(phase="ask", agent_id=agent_id, response=response)
        return self._extract_assistant_text(response)

    def ask_json(self, question: str):
        if self._fallback:
            assert self._fallback_agent is not None
            return self._fallback_agent.ask_json(question)
        text = self.ask(question, expect_json=True)
        parsed = parse_json_response_text(text)
        if parsed is not None:
            return parsed
        return {"_raw_text": text}

    def ask_json_with_agent(self, agent_id: str, question: str):
        if self._fallback:
            assert self._fallback_agent is not None
            return self._fallback_agent.ask_json(question)
        text = self.ask_with_agent(agent_id=agent_id, question=question, expect_json=True)
        parsed = parse_json_response_text(text)
        if parsed is not None:
            return parsed
        return {"_raw_text": text}

    def effective_backend(self) -> str:
        return "local_fallback" if self._fallback else "sdk"

    def list_messages(
        self,
        agent_id: Optional[str] = None,
        *,
        limit: int = 100,
        order: str = "desc",
        include_err: bool = True,
    ) -> List[Dict[str, Any]]:
        if self._fallback:
            return []
        assert self.client is not None
        target_agent_id = str(agent_id or self._require_agent_id()).strip()
        if not target_agent_id:
            return []
        return [
            self._response_to_dict(x)
            for x in self.client.agents.messages.list(
                agent_id=target_agent_id,
                limit=max(1, int(limit)),
                order=str(order or "desc"),
                include_err=bool(include_err),
            )
        ]

    def export_agent_file(self, agent_id: Optional[str] = None) -> str:
        if self._fallback:
            raise RuntimeError("Local fallback mode does not support agent export.")
        assert self.client is not None
        target_agent_id = str(agent_id or self._require_agent_id())
        return self.client.agents.export_file(agent_id=target_agent_id)

    def save_agent_file(self, output_path: Path, agent_id: Optional[str] = None) -> Path:
        schema = self.export_agent_file(agent_id=agent_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(schema, encoding="utf-8")
        return output_path

    def import_agent_file(self, file_bytes: bytes, activate: bool = False) -> str:
        if self._fallback:
            raise RuntimeError("Local fallback mode does not support agent import.")
        assert self.client is not None
        response = self.client.agents.import_file(file=file_bytes)
        agent_ids = self._extract_agent_ids(response)
        if not agent_ids:
            raise RuntimeError("Letta import_file returned no agent_ids.")
        imported_agent_id = agent_ids[0]
        self._lease_add(imported_agent_id)
        for i in range(10):
            try:
                self.client.agents.retrieve(imported_agent_id)
                break
            except Exception:
                if i == 9:
                    raise
                time.sleep(0.2)
        if activate:
            self.agent = SimpleNamespace(id=imported_agent_id)
        return imported_agent_id

    def load_agent_file(self, file_path: Path, activate: bool = False) -> str:
        return self.import_agent_file(file_path.read_bytes(), activate=activate)

    def delete_agent(self, agent_id: str, ignore_missing: bool = True) -> bool:
        if self._fallback:
            return False
        assert self.client is not None
        if not hasattr(self.client.agents, "delete"):
            if ignore_missing:
                self._lease_remove(agent_id)
                return False
            raise AttributeError("Letta client has no agents.delete method")
        try:
            self.client.agents.delete(agent_id)
            self._lease_remove(agent_id)
            return True
        except Exception:
            if not ignore_missing:
                raise
            self._lease_remove(agent_id)
            return False

    def cleanup_leased_agents(self, max_delete: Optional[int] = None) -> Dict[str, int]:
        ids = self._read_leases()
        if max_delete is not None and max_delete >= 0:
            ids = ids[:max_delete]
        attempted = 0
        deleted = 0
        for agent_id in ids:
            attempted += 1
            ok = self.delete_agent(agent_id, ignore_missing=True)
            if ok:
                deleted += 1
        remaining = len(self._read_leases())
        return {
            "attempted": attempted,
            "deleted": deleted,
            "remaining_leases": remaining,
        }

    def close(self):
        return None
