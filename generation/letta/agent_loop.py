import os
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from letta_client import Letta # Updated SDK client for 2026
from dotenv import load_dotenv

load_dotenv()

class LettaAgentLoop:
    def __init__(
        self,
        user_namespace: Optional[str] = None,
        create_agent: bool = True,
        lease_registry_path: Optional[Path] = None,
    ):
        # Initialize the Letta client with environment variables
        api_key = os.getenv("LETTA_API_KEY")
        self.client = Letta(
            api_key=api_key,
#            base_url=os.getenv("LETTA_BASE_URL")
        )
        self._user_namespace = user_namespace
        self._lease_registry_path = lease_registry_path
        self.agent: Optional[Any] = None
        if create_agent:
            self._create_agent()

    def _create_agent(self) -> None:
        default_persona = "A professional log analyst."
        instruction = ""
        self.agent = self.client.agents.create(
            agent_type="letta_v1_agent",
            model=os.getenv("LETTA_MODEL", "my-openai-key/gpt-5-mini"),
            embedding=os.getenv("LETTA_EMBEDDING", "openai/text-embedding-3-large"),
            context_window_limit=32000,
            memory_blocks=[
                {
                    "label": "persona",
                    "value": default_persona + instruction,
                    "description": "Core assistant persona and memory policy. Try to use tools to help manage memory efficiently. User logs can be very long, so you need keep the balance between core memory and archival memory.",
                },
                {
                    "label": "human",
                    "value": "Analyzing user application logs for patterns and facts.",
                    "description": "User context for the active task.",
                }
            ],
            system='''You are a stateful agent processing long-term user logs. You should both accurately remember details and maintain an efficient, abstracted memory or state representation. Here are some guidelines to help you manage your memory and reasoning:
                        State Modeling:
                        User logs may implicitly reflect evolving latent states over time.
                        You may infer and maintain such states if useful,
                        You can freely organize the structure of core memory as you see fit
                        It is good to classify different types of inferred information into separate memory blocks if that helps you stay organized.
                        Retrieval Policy:
                        - Use archival memory for historical details.
                        - Use core memory only for compact, persistent abstractions.
                        - Avoid overloading core memory. User logs can be very long, so you need keep the balance between core memory and archival memory.
                        Your objective is to reason accurately over long time spans while maintaining memory efficiency.''',
            tools = [
                "memory",
                "send_message",
                "memory_apply_patch",
                "core_memory_append",
                "core_memory_replace",
                "archival_memory_insert",
                "archival_memory_search",
                "memory_rethink",
            ],
            model_settings={
                "provider_type": "openai",
                "temperature": 0.2,
                "response_format": {
                    "type": "json_object",
                },
            }
        )

    def _require_agent_id(self) -> str:
        if self.agent is None:
            self._create_agent()
        assert self.agent is not None
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

        def _letta_content_to_text(content: Any) -> str:
            # Letta assistant_message currently returns content as JSON string.
            if isinstance(content, str):
                return content.strip()
            # Keep lightweight support for structured text blocks.
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

    def _format_log_as_dialogue(self, log: dict) -> str:
        return (
            "Ingest this user app log into your memory. "
            "Do not ask follow-up questions; just absorb it.\n"
            f"app_log_id: {log.get('app_log_id')}\n"
            f"timestamp: {log.get('timestamp')}\n"
            f"payload: {json.dumps(log, ensure_ascii=False)}"
        )

    def ingest_log(self, log: dict):
        """
        Ingest one log through a normal user->assistant message turn.
        This matches the "round-by-round dialogue" ingestion style.
        """
        response = self.client.agents.messages.create(
            agent_id=self._require_agent_id(),
            messages=[{
                "role": "user",
                "content": self._format_log_as_dialogue(log)
            }]
        )
        return response

    def ingest_logs_dialogue(self, logs: list):
        """
        Ingest logs as sequential dialogue turns (instead of passages API).
        """
        for log in logs:
            self.ingest_log(log)
        print(f"Successfully ingested {len(logs)} logs via dialogue turns.")

    def ask(self, question: str, expect_json: bool = False):
        """
        Send a query to the agent.
        The agent will autonomously decide to search archival memory if needed.
        """
        return self.ask_with_agent(
            agent_id=self._require_agent_id(),
            question=question,
            expect_json=expect_json,
        )

    def ask_with_agent(self, agent_id: str, question: str, expect_json: bool = False):
        """
        Ask question against a specific agent id.
        Useful when multiple imported checkpoint agents are used in one run.
        """
        if expect_json:
            question += ''' Your final response must be valid JSON. 
            like {"answer": "your answer here", "reasoning": "your reasoning here"}
            '''

        response = self.client.agents.messages.create(
            agent_id=agent_id,
            messages=[{
                "role": "user",
                "content": question
            }]
        )
        return self._extract_assistant_text(response)

    def ask_json(self, question: str):
        return self.ask(question, expect_json=True)

    def export_agent_file(self, agent_id: Optional[str] = None) -> str:
        target_agent_id = str(agent_id or self._require_agent_id())
        return self.client.agents.export_file(agent_id=target_agent_id)

    def save_agent_file(self, output_path: Path, agent_id: Optional[str] = None) -> Path:
        schema = self.export_agent_file(agent_id=agent_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(schema, encoding="utf-8")
        return output_path

    def import_agent_file(self, file_bytes: bytes, activate: bool = False) -> str:
        response = self.client.agents.import_file(file=file_bytes)
        agent_ids = self._extract_agent_ids(response)
        if not agent_ids:
            raise RuntimeError("Letta import_file returned no agent_ids.")
        imported_agent_id = agent_ids[0]
        self._lease_add(imported_agent_id)
        # Ensure imported agent is retrievable before messaging it.
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
        # Keep explicit close hook for caller compatibility.
        return None
