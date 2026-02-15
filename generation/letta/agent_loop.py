import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from generation.letta.client import LLMClient


def _parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(ts, fmt)
            if dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
    except ValueError:
        return datetime.max


def sort_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        logs,
        key=lambda x: (
            _parse_ts(str(x.get("timestamp", "9999-12-31 23:59:59"))),
            str(x.get("app_log_id", "")),
        ),
    )


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("output_text", "text", "response", "message", "content"):
            if key in value:
                out = _extract_text(value[key])
                if out:
                    return out
        for key in ("messages", "outputs", "items", "results"):
            if key in value and isinstance(value[key], list):
                texts = [_extract_text(v) for v in value[key]]
                texts = [t for t in texts if t]
                if texts:
                    return "\n".join(texts)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        texts = [_extract_text(v) for v in value]
        texts = [t for t in texts if t]
        return "\n".join(texts)
    if hasattr(value, "model_dump"):
        try:
            return _extract_text(value.model_dump(mode="json"))
        except Exception:
            return _extract_text(value.model_dump())
    return str(value)


class LettaAgentLoop:
    """
    Agent-loop adapter for Letta.

    - Ingest stage: each app log is sent as one message, response ignored.
    - Query stage: send a question/instruction and parse JSON from response.
    """

    def __init__(
        self,
        *,
        user_namespace: str,
        llm_provider: str,
        llm_model: str,
        llm_max_workers: int,
        mode: str = "sdk",
        allow_local_fallback: bool = True,
        persona: Optional[str] = None,
        human: Optional[str] = None,
    ):
        self.user_namespace = user_namespace
        self.mode = mode
        self.allow_local_fallback = allow_local_fallback
        self.persona = persona or (
            "You are a long-horizon memory agent. Build and maintain accurate memory from "
            "chronological app logs. Prefer factual consistency over speculation."
        )
        self.human = human or (
            "You serve one user. Inputs are chronological app logs. Maintain robust memory. "
            "When asked questions, answer with JSON only and cite app_log_id evidence."
        )
        self._local_llm = LLMClient(
            provider=llm_provider,
            model_name=llm_model,
            max_workers=llm_max_workers,
        )
        self._local_memory_messages: List[str] = []
        self._sdk_client = None
        self._sdk_agent_id = None
        self._sdk_ready = False
        self._inited = False

    def initialize(self) -> None:
        if self._inited:
            return
        if self.mode == "sdk":
            self._sdk_ready = self._init_sdk_agent()
            if not self._sdk_ready and not self.allow_local_fallback:
                raise RuntimeError(
                    "Letta SDK init failed and local fallback disabled. "
                    "Please verify LETTA SDK installation/API."
                )
        self._inited = True

    def ingest_log(self, log: Dict[str, Any]) -> None:
        self.initialize()
        payload = {
            "type": "app_log",
            "app_log_id": log.get("app_log_id"),
            "timestamp": log.get("timestamp"),
            "app_name": log.get("app_name"),
            "api_name": log.get("api_name"),
            "request": log.get("request", {}),
            "response": log.get("response", {}),
        }
        msg = (
            "Ingest this new app log into memory. "
            "Update memory structures as needed. "
            "No explanation needed.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        self._send_message(msg)

    def ask_json(self, prompt: str) -> Dict[str, Any]:
        self.initialize()
        raw = self._send_message(prompt)
        text = _extract_text(raw).strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
            return {}

    def close(self) -> None:
        self._local_llm.close()

    def _send_message(self, message: str) -> Any:
        if self._sdk_ready:
            try:
                return self._send_sdk_message(message)
            except Exception:
                if not self.allow_local_fallback:
                    raise
        return self._send_local_message(message)

    def _send_local_message(self, message: str) -> Any:
        # Local fallback: append memory messages and ask base LLM with a bounded memory prefix.
        self._local_memory_messages.append(message)
        memory_tail = self._local_memory_messages[-200:]
        composed = (
            f"System persona:\n{self.persona}\n\n"
            f"Human profile:\n{self.human}\n\n"
            "Conversation memory (recent tail):\n"
            + "\n\n".join(memory_tail)
            + "\n\nRespond to the LAST message only."
        )
        return self._local_llm.ask(composed, response_type="text")

    def _init_sdk_agent(self) -> bool:
        candidates = [
            ("letta_client", "Letta"),
            ("letta_client", "Client"),
            ("letta", "Letta"),
            ("letta", "Client"),
        ]
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name, None)
                if cls is None:
                    continue
                client = self._instantiate_client(cls)
                if client is None:
                    continue
                agent_id = self._get_or_create_agent(client)
                if not agent_id:
                    continue
                self._sdk_client = client
                self._sdk_agent_id = agent_id
                return True
            except Exception:
                continue
        return False

    def _instantiate_client(self, cls: Any) -> Any:
        base_url = os.getenv("LETTA_BASE_URL")
        api_key = os.getenv("LETTA_API_KEY")
        trials = [{}]
        if base_url:
            trials.extend([{"base_url": base_url}, {"url": base_url}, {"endpoint": base_url}])
        if api_key:
            trials.extend([{"api_key": api_key}, {"token": api_key}, {"bearer_token": api_key}])
        if base_url and api_key:
            trials.extend(
                [
                    {"base_url": base_url, "api_key": api_key},
                    {"url": base_url, "token": api_key},
                    {"endpoint": base_url, "bearer_token": api_key},
                ]
            )
        for kwargs in trials:
            try:
                return cls(**kwargs)
            except Exception:
                continue
        return None

    def _get_or_create_agent(self, client: Any) -> Optional[str]:
        name = f"muse_{self.user_namespace}"

        # get existing
        find_calls = [
            lambda: client.get_agent(name=name),
            lambda: client.get_agent(agent_name=name),
            lambda: client.agents.get(name=name),
            lambda: client.agents.retrieve(name=name),
        ]
        for call in find_calls:
            try:
                out = call()
                agent_id = self._extract_agent_id(out)
                if agent_id:
                    return agent_id
            except Exception:
                continue

        # create
        create_calls = [
            lambda: client.create_agent(name=name, persona=self.persona, human=self.human),
            lambda: client.create_agent(agent_name=name, persona=self.persona, human=self.human),
            lambda: client.agents.create(name=name, persona=self.persona, human=self.human),
            lambda: client.agents.create(agent_name=name, persona=self.persona, human=self.human),
        ]
        for call in create_calls:
            try:
                out = call()
                agent_id = self._extract_agent_id(out)
                if agent_id:
                    return agent_id
            except Exception:
                continue
        return None

    def _extract_agent_id(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (str, int)):
            out = str(value).strip()
            return out or None
        if isinstance(value, dict):
            for key in ("agent_id", "id", "uuid"):
                if value.get(key) is not None:
                    out = str(value.get(key)).strip()
                    if out:
                        return out
            for key in ("agent", "data", "result"):
                if key in value:
                    out = self._extract_agent_id(value[key])
                    if out:
                        return out
            return None
        if hasattr(value, "model_dump"):
            try:
                return self._extract_agent_id(value.model_dump(mode="json"))
            except Exception:
                return self._extract_agent_id(value.model_dump())
        return None

    def _send_sdk_message(self, message: str) -> Any:
        if self._sdk_client is None or self._sdk_agent_id is None:
            raise RuntimeError("SDK client or agent not initialized")
        aid = self._sdk_agent_id
        calls = [
            lambda: self._sdk_client.send_message(agent_id=aid, message=message),
            lambda: self._sdk_client.send_message(agent_id=aid, content=message),
            lambda: self._sdk_client.create_message(agent_id=aid, message=message),
            lambda: self._sdk_client.agents.messages.create(agent_id=aid, message=message),
            lambda: self._sdk_client.agents.messages.create(agent_id=aid, content=message),
            lambda: self._sdk_client.agents.send_message(agent_id=aid, message=message),
        ]
        for call in calls:
            try:
                return call()
            except Exception:
                continue
        raise RuntimeError("No supported Letta SDK send-message method found for this client version")

