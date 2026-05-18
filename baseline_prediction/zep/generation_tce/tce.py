#!/usr/bin/env python3
"""Zep (Graphiti) TCE baseline using shared TCE hooks and a conservative live builder."""

import asyncio
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DIR = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIR))

GRAPHITI_DIR = SCRIPT_DIR.parent / "graphiti"
if str(GRAPHITI_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPHITI_DIR))

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from openai import APIStatusError, AsyncOpenAI, RateLimitError

from generation.common.llm_client import LLMClient
from generation.common.provider_config import load_repo_dotenv, resolve_openai_compatible_credentials
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import normalize_app_logs, observed_logs_for_checkpoint, run_pipeline, to_log_text


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _load_builder_progress(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _builder_progress_index(payload: Dict[str, Any]) -> int:
    raw = payload.get("confirmed_last_log_idx")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return int(raw)
        except Exception:
            return -1
    return -1


def _resume_state_is_usable(
    builder_progress: Dict[str, Any],
    *,
    app_logs_path: Path,
    all_logs: List[Dict[str, Any]],
    db_path: str,
) -> bool:
    if not builder_progress:
        return False
    if not os.path.exists(db_path):
        return False
    progress_logs_path = str(builder_progress.get("app_logs_path") or "").strip()
    if progress_logs_path and progress_logs_path != str(app_logs_path):
        return False
    confirmed_idx = _builder_progress_index(builder_progress)
    if confirmed_idx < -1:
        return False
    if confirmed_idx >= len(all_logs):
        return False
    return True


def _load_indexed_log_ids_from_db(db_path: str) -> Set[str]:
    if not db_path or not os.path.exists(db_path):
        return set()

    fake_state_path = Path(db_path) / "fake_graphiti_state.json"
    if fake_state_path.exists():
        try:
            payload = json.loads(fake_state_path.read_text(encoding="utf-8"))
        except Exception:
            return set()
        logs = payload.get("logs")
        if not isinstance(logs, list):
            return set()
        return {
            str(log.get("app_log_id") or "").strip()
            for log in logs
            if isinstance(log, dict) and str(log.get("app_log_id") or "").strip()
        }

    try:
        import kuzu  # type: ignore
    except Exception:
        return set()

    src_path = Path(db_path)
    snapshot_holder = tempfile.TemporaryDirectory(prefix="zep_resume_snapshot_")
    snapshot_dir = Path(snapshot_holder.name)
    snapshot_path = snapshot_dir / src_path.name
    try:
        _copy_db_artifact(src_path, snapshot_path)
        db = kuzu.Database(str(snapshot_path))
        conn = kuzu.Connection(db)
        rows = list(conn.execute("MATCH (e:Episodic) RETURN e.name AS name;").rows_as_dict())
        return {
            str(row.get("name") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        }
    finally:
        snapshot_holder.cleanup()


def _reconcile_resume_state(
    builder_progress: Dict[str, Any],
    *,
    all_logs: List[Dict[str, Any]],
    db_path: str,
) -> Dict[str, Any]:
    progress_confirmed_idx = _builder_progress_index(builder_progress)
    db_log_ids = _load_indexed_log_ids_from_db(db_path)
    if not db_log_ids:
        return {
            "usable": progress_confirmed_idx >= -1,
            "confirmed_idx": progress_confirmed_idx,
            "indexed_log_ids": {
                str(log.get("app_log_id") or "").strip()
                for log in all_logs[: progress_confirmed_idx + 1]
                if str(log.get("app_log_id") or "").strip()
            },
            "db_episode_count": 0,
            "db_contiguous_prefix_idx": -1,
            "resume_progress_lag": 0,
        }

    contiguous_prefix_idx = -1
    db_known_indices: List[int] = []
    for idx, log in enumerate(all_logs):
        app_log_id = str(log.get("app_log_id") or "").strip()
        if not app_log_id:
            break
        if app_log_id in db_log_ids:
            db_known_indices.append(idx)
            if idx == contiguous_prefix_idx + 1:
                contiguous_prefix_idx = idx

    has_noncontiguous_future = any(idx > contiguous_prefix_idx for idx in db_known_indices)
    usable = progress_confirmed_idx <= contiguous_prefix_idx and not has_noncontiguous_future
    confirmed_idx = contiguous_prefix_idx if usable else progress_confirmed_idx
    indexed_log_ids = {
        str(log.get("app_log_id") or "").strip()
        for log in all_logs[: confirmed_idx + 1]
        if str(log.get("app_log_id") or "").strip()
    }
    return {
        "usable": usable,
        "confirmed_idx": confirmed_idx,
        "indexed_log_ids": indexed_log_ids,
        "db_episode_count": len(db_log_ids),
        "db_contiguous_prefix_idx": contiguous_prefix_idx,
        "resume_progress_lag": max(0, contiguous_prefix_idx - progress_confirmed_idx),
        "has_noncontiguous_future": has_noncontiguous_future,
    }


def _safe_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "checkpoint"
    chars: List[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    cooked = "".join(chars).strip("_")
    return cooked or "checkpoint"


def _snapshot_root(output_path: Path) -> Path:
    return output_path.parent / "checkpoint_snapshots"


def _snapshot_manifest_path(snapshot_root: Path) -> Path:
    return snapshot_root / "manifest.json"


def _load_snapshot_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        return []
    return [dict(x) for x in snapshots if isinstance(x, dict)]


def _snapshot_db_path(snapshot_root: Path, checkpoint_id: str) -> Path:
    return snapshot_root / "snapshots" / _safe_name(checkpoint_id)


def _db_wal_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".wal")


def _remove_db_artifact(db_path: Path) -> None:
    if db_path.is_dir():
        shutil.rmtree(db_path, ignore_errors=True)
    elif db_path.exists():
        db_path.unlink(missing_ok=True)
    wal_path = _db_wal_path(db_path)
    if wal_path.exists():
        wal_path.unlink(missing_ok=True)


def _copy_db_artifact(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path)
        return
    shutil.copy2(src_path, dst_path)
    src_wal_path = _db_wal_path(src_path)
    if src_wal_path.exists():
        shutil.copy2(src_wal_path, _db_wal_path(dst_path))


def _usage_cost_sidecar_path(output_path: Path) -> Path:
    return output_path.parent / "usage_cost.json"


def _merge_usage_summary(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "request_count": 0,
        "chat_request_count": 0,
        "embedding_request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "context_tokens": 0,
        "total_tokens": 0,
        "turn_count": 0,
        "by_model": [],
    }
    by_model: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for source in (base, delta):
        if not isinstance(source, dict):
            continue
        for key in (
            "request_count",
            "chat_request_count",
            "embedding_request_count",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "context_tokens",
            "total_tokens",
            "turn_count",
        ):
            merged[key] += int(source.get(key) or 0)
        for bucket in source.get("by_model") or []:
            if not isinstance(bucket, dict):
                continue
            bucket_key = (str(bucket.get("request_kind") or ""), str(bucket.get("model") or ""))
            dst = by_model.setdefault(
                bucket_key,
                {
                    "request_kind": bucket_key[0],
                    "model": bucket_key[1],
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            for field in ("request_count", "prompt_tokens", "completion_tokens", "total_tokens"):
                dst[field] += int(bucket.get(field) or 0)
    merged["by_model"] = list(by_model.values())
    return merged


def _write_usage_cost_sidecar(
    *,
    sidecar_path: Path,
    output_path: Path,
    llm_provider: str,
    llm_model: str,
    retriever_provider: str,
    embedding_model_name: str,
    build_duration_s: float,
    generation_duration_s: float,
    total_duration_s: float,
    build_memory_usage: Dict[str, Any],
    retrieval_usage: Dict[str, Any],
    answer_llm_usage: Dict[str, Any],
) -> None:
    existing_payload: Dict[str, Any] = {}
    if sidecar_path.exists():
        try:
            raw_existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            raw_existing = {}
        if isinstance(raw_existing, dict):
            existing_payload = raw_existing

    existing_timing = existing_payload.get("timing") if isinstance(existing_payload.get("timing"), dict) else {}
    timing = {
        "build_memory_duration_s": float(existing_timing.get("build_memory_duration_s") or 0.0) + float(build_duration_s),
        "generation_duration_s": float(existing_timing.get("generation_duration_s") or 0.0) + float(generation_duration_s),
        "total_duration_s": float(existing_timing.get("total_duration_s") or 0.0) + float(total_duration_s),
    }
    existing_usage = existing_payload.get("usage") if isinstance(existing_payload.get("usage"), dict) else {}
    usage = {
        "build_memory": _merge_usage_summary(
            existing_usage.get("build_memory") if isinstance(existing_usage.get("build_memory"), dict) else {},
            build_memory_usage if isinstance(build_memory_usage, dict) else {},
        ),
        "retrieval": _merge_usage_summary(
            existing_usage.get("retrieval") if isinstance(existing_usage.get("retrieval"), dict) else {},
            retrieval_usage if isinstance(retrieval_usage, dict) else {},
        ),
        "answer_llm": _merge_usage_summary(
            existing_usage.get("answer_llm") if isinstance(existing_usage.get("answer_llm"), dict) else {},
            answer_llm_usage if isinstance(answer_llm_usage, dict) else {},
        ),
    }
    payload = {
        "prediction_path": str(output_path),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retriever_provider": retriever_provider,
        "embedding_model": embedding_model_name,
        "usage": usage,
        "timing": timing,
        "generated_at": datetime.now().isoformat(),
    }
    _atomic_write_json(sidecar_path, payload)


class _UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []

    def _usage_to_dict(self, raw_usage: Any) -> Dict[str, Any]:
        if hasattr(raw_usage, "model_dump"):
            raw_usage = raw_usage.model_dump(mode="json", by_alias=True)
        elif hasattr(raw_usage, "to_dict"):
            raw_usage = raw_usage.to_dict()
        if isinstance(raw_usage, dict):
            return raw_usage
        return {}

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        raw_usage = self._usage_to_dict(getattr(response, "usage", None))
        input_details = self._usage_to_dict(raw_usage.get("input_tokens_details"))
        output_details = self._usage_to_dict(raw_usage.get("output_tokens_details"))
        prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": int(raw_usage.get("reasoning_tokens") or output_details.get("reasoning_tokens") or 0),
            "cached_input_tokens": int(raw_usage.get("cached_input_tokens") or input_details.get("cached_tokens") or 0),
            "cache_write_tokens": int(raw_usage.get("cache_write_tokens") or input_details.get("cache_write_tokens") or 0),
            "context_tokens": int(raw_usage.get("context_tokens") or 0),
            "total_tokens": total_tokens,
        }

    def record_response(self, *, phase: str, request_kind: str, response: Any, model: str = "") -> None:
        usage = self._extract_usage(response)
        usage["phase"] = str(phase or "")
        usage["request_kind"] = str(request_kind or "")
        usage["model"] = str(model or getattr(response, "model", "") or "")
        usage["timestamp_unix"] = time.time()
        with self._lock:
            self._records.append(dict(usage))

    def usage_summary(self, *, phase: Optional[str] = None) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "request_count": 0,
            "chat_request_count": 0,
            "embedding_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "context_tokens": 0,
            "total_tokens": 0,
            "turn_count": 0,
            "by_model": [],
        }
        with self._lock:
            records = [dict(x) for x in self._records]
        if phase is not None:
            records = [x for x in records if str(x.get("phase") or "") == str(phase or "")]
        summary["request_count"] = len(records)
        summary["turn_count"] = len(records)
        by_model: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in records:
            request_kind = str(record.get("request_kind") or "")
            if request_kind == "embedding":
                summary["embedding_request_count"] += 1
            else:
                summary["chat_request_count"] += 1
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_input_tokens",
                "cache_write_tokens",
                "context_tokens",
                "total_tokens",
            ):
                summary[key] += int(record.get(key) or 0)
            bucket_key = (request_kind, str(record.get("model") or ""))
            bucket = by_model.setdefault(
                bucket_key,
                {
                    "request_kind": bucket_key[0],
                    "model": bucket_key[1],
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            bucket["request_count"] += 1
            bucket["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
            bucket["completion_tokens"] += int(record.get("completion_tokens") or 0)
            bucket["total_tokens"] += int(record.get("total_tokens") or 0)
        summary["by_model"] = list(by_model.values())
        return summary


class _TrackedAsyncOpenAIProxy:
    def __init__(
        self,
        target: Any,
        *,
        tracker: _UsageTracker,
        phase_getter: Callable[[], str],
        path: str = "",
    ):
        self._target = target
        self._tracker = tracker
        self._phase_getter = phase_getter
        self._path = path

    def _subpath(self, name: str) -> str:
        return "{}.{}".format(self._path, name) if self._path else name

    def _should_record(self, path: str) -> bool:
        return path.endswith(".create") or path.endswith(".parse")

    def _request_kind(self, path: str) -> str:
        lowered = str(path or "").lower()
        if "embedding" in lowered:
            return "embedding"
        return "response"

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        path = self._subpath(name)
        if inspect.iscoroutinefunction(attr):
            if not self._should_record(path):
                return attr

            async def _wrapped(*args, **kwargs):
                result = await attr(*args, **kwargs)
                self._tracker.record_response(
                    phase=self._phase_getter(),
                    request_kind=self._request_kind(path),
                    response=result,
                    model=str(kwargs.get("model") or getattr(result, "model", "") or ""),
                )
                return result

            return _wrapped
        if isinstance(attr, (str, int, float, bool, bytes, bytearray, type(None), list, dict, set, tuple)):
            return attr
        if callable(attr):
            return attr
        return _TrackedAsyncOpenAIProxy(
            attr,
            tracker=self._tracker,
            phase_getter=self._phase_getter,
            path=path,
        )


class _AsyncLoopThread:
    def __init__(self, name: str):
        self._name = name
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            loop.run_forever()
            loop.close()

        self._thread = threading.Thread(target=_runner, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait()

    def run(self, coro: Any) -> Any:
        self.start()
        if self._loop is None:
            raise RuntimeError("Async loop is not available.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        if self._loop is None or self._thread is None:
            return
        loop = self._loop
        thread = self._thread
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        self._loop = None
        self._thread = None
        self._ready.clear()


def async_retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    rate_limit_delay: float = 60.0,
):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except RateLimitError:
                    delay = rate_limit_delay
                    print("[Zep] Rate limit hit (429). Waiting {}s before retry...".format(delay))
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise
                except APIStatusError as exc:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print("[Zep] API error (status {}). Waiting {}s before retry...".format(exc.status_code, delay))
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise
                except Exception as exc:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print("[Zep] Error: {}: {}. Waiting {}s before retry...".format(type(exc).__name__, exc, delay))
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise
            return None

        return wrapper

    return decorator


async def _await_maybe(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class _AzureCompatibleOpenAIGenericClient(OpenAIGenericClient):
    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        return str(model or "").strip().lower().startswith(("gpt-5", "o1", "o3"))

    async def _generate_response(
        self,
        messages: List[Any],
        response_model: Any = None,
        max_tokens: int = 16384,
        model_size: Any = None,
    ) -> Dict[str, Any]:
        model_name = str(self.model or "gpt-4.1-mini")
        if not self._uses_max_completion_tokens(model_name):
            return await super()._generate_response(
                messages,
                response_model=response_model,
                max_tokens=max_tokens,
                model_size=model_size,
            )
        if not getattr(self, "_zep_logged_azure_compat", False):
            print("[Zep] Azure-compatible Graphiti LLM client using max_completion_tokens")
            self._zep_logged_azure_compat = True

        openai_messages: List[Dict[str, str]] = []
        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role in {"user", "system"}:
                openai_messages.append({"role": message.role, "content": message.content})

        response_format: Dict[str, Any] = {"type": "json_object"}
        if response_model is not None:
            schema_name = getattr(response_model, "__name__", "structured_response")
            json_schema = response_model.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                },
            }

        response = await self.client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            temperature=self.temperature,
            max_completion_tokens=max_tokens or self.max_tokens,
            response_format=response_format,
        )
        result = response.choices[0].message.content or ""
        return json.loads(result)


class _AzureCompatibleOpenAIRerankerClient(OpenAIRerankerClient):
    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        return str(model or "").strip().lower().startswith(("gpt-5", "o1", "o3"))

    async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        message_batches: List[List[Dict[str, str]]] = [
            [
                {
                    "role": "system",
                    "content": "You are an expert tasked with determining whether the passage is relevant to the query",
                },
                {
                    "role": "user",
                    "content": (
                        'Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.\n'
                        "<PASSAGE>\n"
                        f"{passage}\n"
                        "</PASSAGE>\n"
                        "<QUERY>\n"
                        f"{query}\n"
                        "</QUERY>\n"
                    ),
                },
            ]
            for passage in passages
        ]
        model_name = str(self.config.model or "gpt-4.1-nano")
        request_key = "max_completion_tokens" if self._uses_max_completion_tokens(model_name) else "max_tokens"
        responses = await asyncio.gather(
            *[
                self.client.chat.completions.create(
                    model=model_name,
                    messages=message_batch,
                    temperature=0,
                    logit_bias={"6432": 1, "7983": 1},
                    logprobs=True,
                    top_logprobs=2,
                    **{request_key: 1},
                )
                for message_batch in message_batches
            ]
        )

        scores: List[float] = []
        for response in responses:
            if response.choices[0].logprobs is None or response.choices[0].logprobs.content is None:
                continue
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            if len(top_logprobs) == 0:
                continue
            probability = math.exp(top_logprobs[0].logprob)
            if top_logprobs[0].token.strip().split(" ")[0].lower() == "true":
                scores.append(probability)
            else:
                scores.append(1 - probability)

        ranked = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
        ranked.sort(reverse=True, key=lambda item: item[1])
        return ranked


@dataclass
class GraphitiConfig:
    db_path: str
    embedding_provider: str
    embedding_model: str
    graphiti_llm_provider: str
    graphiti_llm_model: str
    batch_size: int
    max_coroutines: int
    max_tokens: int = 16384


class ZepGraphitiIndex:
    """Thread-safe sync wrapper around Graphiti using a dedicated async loop thread."""

    def __init__(self, config: GraphitiConfig, *, usage_tracker: Optional[_UsageTracker] = None):
        self.config = config
        self.graphiti: Optional[Graphiti] = None
        self.indexed_log_ids: Set[str] = set()
        self._runner = _AsyncLoopThread("zep-graphiti-loop")
        self._driver: Any = None
        self._clients: List[Any] = []
        self._usage_tracker = usage_tracker or _UsageTracker()
        self._usage_phase = "idle"

    def initialize(self, *, reset_db: bool, indexed_log_ids: Optional[Set[str]] = None) -> None:
        if reset_db and os.path.exists(self.config.db_path):
            _remove_db_artifact(Path(self.config.db_path))
            print("[Zep] Cleaned existing database: {}".format(self.config.db_path))
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.indexed_log_ids = set(indexed_log_ids or set())
        self._runner.run(self._initialize_async())

    async def _initialize_async(self) -> None:
        if self.graphiti is not None:
            return

        embedder_api_key, embedder_base_url = resolve_openai_compatible_credentials(
            self.config.embedding_provider,
            require_api_key=True,
        )
        graphiti_api_key, graphiti_base_url = resolve_openai_compatible_credentials(
            self.config.graphiti_llm_provider,
            require_api_key=True,
        )

        print("[Zep] Initializing Graphiti with Kuzu database: {}".format(self.config.db_path))
        print("[Zep] Embedding provider/model: {}/{}".format(self.config.embedding_provider, self.config.embedding_model))
        print(
            "[Zep] Graphiti LLM provider/model: {}/{}".format(
                self.config.graphiti_llm_provider,
                self.config.graphiti_llm_model,
            )
        )
        print("[Zep] Max coroutines: {}".format(self.config.max_coroutines))

        embedder_openai_client = _TrackedAsyncOpenAIProxy(
            AsyncOpenAI(api_key=embedder_api_key, base_url=embedder_base_url),
            tracker=self._usage_tracker,
            phase_getter=lambda: self._usage_phase,
            path="async_openai",
        )
        if (
            self.config.embedding_provider == self.config.graphiti_llm_provider
            and embedder_api_key == graphiti_api_key
            and embedder_base_url == graphiti_base_url
        ):
            llm_openai_client = embedder_openai_client
            self._clients = [embedder_openai_client]
        else:
            llm_openai_client = _TrackedAsyncOpenAIProxy(
                AsyncOpenAI(api_key=graphiti_api_key, base_url=graphiti_base_url),
                tracker=self._usage_tracker,
                phase_getter=lambda: self._usage_phase,
                path="async_openai",
            )
            self._clients = [embedder_openai_client, llm_openai_client]

        embedder_config = OpenAIEmbedderConfig(
            api_key=embedder_api_key,
            embedding_model=self.config.embedding_model,
            base_url=embedder_base_url,
        )
        embedder = OpenAIEmbedder(config=embedder_config, client=embedder_openai_client)

        llm_config = LLMConfig(
            api_key=graphiti_api_key,
            model=self.config.graphiti_llm_model,
            base_url=graphiti_base_url,
        )
        llm_client_cls = _AzureCompatibleOpenAIGenericClient if self.config.graphiti_llm_provider == "azure" else OpenAIGenericClient
        reranker_cls = _AzureCompatibleOpenAIRerankerClient if self.config.graphiti_llm_provider == "azure" else OpenAIRerankerClient
        llm_client = llm_client_cls(
            config=llm_config,
            client=llm_openai_client,
            max_tokens=self.config.max_tokens,
        )
        cross_encoder = reranker_cls(config=llm_config, client=llm_openai_client)
        print("[Zep] Graphiti llm client class: {}".format(type(llm_client).__name__))
        print("[Zep] Graphiti reranker class: {}".format(type(cross_encoder).__name__))

        graph_driver = KuzuDriver(db=self.config.db_path)
        self._driver = graph_driver
        self.graphiti = Graphiti(
            uri=None,
            user=None,
            password=None,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            store_raw_episode_content=True,
            graph_driver=graph_driver,
            max_coroutines=self.config.max_coroutines,
        )

        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices

        for query in get_fulltext_indices(GraphProvider.KUZU):
            try:
                await graph_driver.execute_query(query)
            except Exception as exc:
                if "already exists" not in str(exc):
                    raise
        print("[Zep] Graphiti index initialized")

    def close(self) -> None:
        if self.graphiti is not None:
            try:
                self._runner.run(self._close_async())
            finally:
                self.graphiti = None
                self._driver = None
                self._clients = []
        self._runner.stop()

    async def _close_async(self) -> None:
        if self.graphiti is not None:
            await self.graphiti.close()
            print("[Zep] Graphiti connection closed")
        if self._driver is not None:
            close_fn = getattr(self._driver, "close", None)
            if callable(close_fn):
                await _await_maybe(close_fn())
        seen = set()
        for client in self._clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            close_fn = getattr(client, "close", None) or getattr(client, "aclose", None)
            if callable(close_fn):
                await _await_maybe(close_fn())

    def add_episode(self, episode: Dict[str, Any]) -> None:
        self._runner.run(self._add_episode_async(episode))

    @async_retry_with_backoff(max_attempts=5, base_delay=10.0, max_delay=120.0, rate_limit_delay=60.0)
    async def _add_episode_with_retry(
        self,
        *,
        episode_id: str,
        episode_body: Dict[str, Any],
        reference_time: datetime,
    ) -> None:
        if self.graphiti is None:
            raise RuntimeError("Graphiti not initialized")
        serialized_episode = json.dumps(episode_body, ensure_ascii=False)
        try:
            await _await_maybe(
                self.graphiti.add_episode(
                    name=episode_id,
                    episode_body=serialized_episode,
                    source_description="App log entry",
                    source=EpisodeType.json,
                    reference_time=reference_time,
                )
            )
        except TypeError:
            raw_episode = RawEpisode(
                name=episode_id,
                content=serialized_episode,
                source_description="App log entry",
                source=EpisodeType.json,
                reference_time=reference_time,
            )
            try:
                await _await_maybe(self.graphiti.add_episode(raw_episode))
            except TypeError:
                await _await_maybe(self.graphiti.add_episode(episode=raw_episode))

    async def _add_episode_async(self, episode: Dict[str, Any]) -> None:
        if self.graphiti is None:
            raise RuntimeError("Graphiti not initialized")
        episode_id = str(episode.get("app_log_id") or "").strip()
        if not episode_id:
            episode_id = "ep_{}".format(len(self.indexed_log_ids))
        if episode_id in self.indexed_log_ids:
            return
        timestamp = str(episode.get("timestamp") or "").strip()
        reference_time = datetime.now(timezone.utc)
        if timestamp:
            try:
                reference_time = _parse_timestamp(timestamp)
            except ValueError:
                pass
        prev_phase = self._usage_phase
        self._usage_phase = "build_memory"
        try:
            await self._add_episode_with_retry(
                episode_id=episode_id,
                episode_body=episode,
                reference_time=reference_time,
            )
        finally:
            self._usage_phase = prev_phase
        self.indexed_log_ids.add(episode_id)

    def add_episodes_bulk(self, episodes: List[Dict[str, Any]], batch_size: int = 1) -> None:
        self._runner.run(self._add_episodes_bulk_async(episodes, batch_size=batch_size))

    @async_retry_with_backoff(max_attempts=5, base_delay=10.0, max_delay=120.0, rate_limit_delay=60.0)
    async def _add_batch_with_retry(self, raw_episodes: List[RawEpisode]) -> None:
        if self.graphiti is None:
            raise RuntimeError("Graphiti not initialized")
        await self.graphiti.add_episode_bulk(raw_episodes)

    async def _add_episodes_bulk_async(self, episodes: List[Dict[str, Any]], batch_size: int = 1) -> None:
        if self.graphiti is None:
            raise RuntimeError("Graphiti not initialized")
        if not episodes:
            return

        prev_phase = self._usage_phase
        self._usage_phase = "build_memory"
        try:
            print("[Zep] Adding {} episodes".format(len(episodes)))
            for i in range(0, len(episodes), max(1, int(batch_size))):
                batch = episodes[i : i + max(1, int(batch_size))]
                raw_episodes: List[RawEpisode] = []
                for j, episode in enumerate(batch):
                    episode_id = str(episode.get("app_log_id") or "ep_{}_{}".format(i, j))
                    timestamp = str(episode.get("timestamp") or "").strip()
                    reference_time = datetime.now(timezone.utc)
                    if timestamp:
                        try:
                            reference_time = _parse_timestamp(timestamp)
                        except ValueError:
                            pass
                    raw_episodes.append(
                        RawEpisode(
                            name=episode_id,
                            content=json.dumps(episode, ensure_ascii=False),
                            source_description="App log entry",
                            source=EpisodeType.json,
                            reference_time=reference_time,
                        )
                    )
                await self._add_batch_with_retry(raw_episodes)
                for raw_episode in raw_episodes:
                    self.indexed_log_ids.add(str(raw_episode.name))
        finally:
            self._usage_phase = prev_phase

    def search(self, query: str, top_k: int = 5, checkpoint_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._runner.run(self._search_async(query=query, top_k=top_k, checkpoint_timestamp=checkpoint_timestamp))

    @async_retry_with_backoff(max_attempts=5, base_delay=10.0, max_delay=120.0, rate_limit_delay=60.0)
    async def _search_async(
        self,
        query: str,
        top_k: int = 5,
        checkpoint_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del checkpoint_timestamp
        if self.graphiti is None:
            raise RuntimeError("Graphiti not initialized")

        prev_phase = self._usage_phase
        self._usage_phase = "retrieval"
        try:
            results = await self.graphiti.search_(query=query)
        finally:
            self._usage_phase = prev_phase
        retrieved_episodes: List[Dict[str, Any]] = []

        if hasattr(results, "episodes") and results.episodes:
            for episode in results.episodes[:top_k]:
                try:
                    episode_data = json.loads(episode.content)
                except Exception:
                    continue
                if isinstance(episode_data, dict):
                    retrieved_episodes.append(episode_data)

        if hasattr(results, "edges") and results.edges:
            seen_episode_ids = {
                str(item.get("app_log_id") or "").strip()
                for item in retrieved_episodes
                if isinstance(item, dict)
            }
            for edge in results.edges[:top_k]:
                if not hasattr(edge, "source_episode"):
                    continue
                try:
                    episode_data = json.loads(edge.source_episode)
                except Exception:
                    continue
                if not isinstance(episode_data, dict):
                    continue
                app_log_id = str(episode_data.get("app_log_id") or "").strip()
                if app_log_id and app_log_id in seen_episode_ids:
                    continue
                if app_log_id:
                    seen_episode_ids.add(app_log_id)
                retrieved_episodes.append(episode_data)

        return retrieved_episodes

    def usage_summary(self, *, phase: Optional[str] = None) -> Dict[str, Any]:
        return self._usage_tracker.usage_summary(phase=phase)


def _parse_timestamp(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(ts, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise ValueError("Unsupported timestamp format: {}".format(ts)) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    retrieval_top_k: int,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    answer_temperature: Optional[float],
    answer_top_p: Optional[float],
    answer_top_k: Optional[int],
    retriever_provider: str,
    retriever_model: str,
    retriever_batch_size: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    max_coroutines: int = 5,
    graphiti_llm_provider: Optional[str] = None,
    graphiti_llm_model: Optional[str] = None,
    graphiti_max_tokens: int = 16384,
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    rq3_apply_retrieval_top_k: Optional[int] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    enable_final_qa: bool = False,
    final_qa_path: Optional[str] = None,
    final_qa_output_path: Optional[str] = None,
    final_qa_retrieval_top_k: Optional[int] = None,
    final_qa_save_prompt_and_raw: bool = False,
) -> Dict[str, Any]:
    load_repo_dotenv(REPO_ROOT_DIR)
    run_t0 = time.time()

    print("[Zep] === Configuration ===")
    print("[Zep] Answer LLM Provider/Model: {}/{}".format(llm_provider, llm_model))
    print("[Zep] Retriever Provider/Model: {}/{}".format(retriever_provider, retriever_model))
    print(
        "[Zep] Graphiti LLM Provider/Model: {}/{}".format(
            graphiti_llm_provider or llm_provider,
            graphiti_llm_model or llm_model,
        )
    )
    print("[Zep] Retrieval Top-K: {}".format(retrieval_top_k))
    print("[Zep] Resume: {}".format(resume))
    print("[Zep] ====================")

    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=answer_temperature,
        top_p=answer_top_p,
        top_k=answer_top_k,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = str(output_path.parent / "zep_graphiti_db")
    progress_path = output_path.parent / "builder_progress.json"
    snapshot_root = _snapshot_root(output_path)
    snapshot_manifest_path = _snapshot_manifest_path(snapshot_root)

    all_logs = normalize_app_logs(json.loads(app_logs_path.read_text(encoding="utf-8")))
    print("[Zep] Loaded {} app logs".format(len(all_logs)))

    config = GraphitiConfig(
        db_path=db_path,
        embedding_provider=str(retriever_provider or "openai"),
        embedding_model=str(retriever_model or "text-embedding-3-large"),
        graphiti_llm_provider=str(graphiti_llm_provider or llm_provider or "openai"),
        graphiti_llm_model=str(graphiti_llm_model or llm_model or "gpt-5-mini"),
        batch_size=max(1, int(retriever_batch_size)),
        max_coroutines=max(1, int(max_coroutines)),
        max_tokens=max(1, int(graphiti_max_tokens)),
    )
    graphiti_usage_tracker = _UsageTracker()
    index = ZepGraphitiIndex(config, usage_tracker=graphiti_usage_tracker)
    builder_progress: Dict[str, Any] = {}
    snapshot_manifest: List[Dict[str, Any]] = _load_snapshot_manifest(snapshot_manifest_path)
    answer_llm_usage: Dict[str, Any] = {}
    phase_timing = {"build_memory_duration_s": 0.0}

    if resume:
        candidate_progress = _load_builder_progress(progress_path)
        if _resume_state_is_usable(
            candidate_progress,
            app_logs_path=app_logs_path,
            all_logs=all_logs,
            db_path=db_path,
        ):
            resume_state = _reconcile_resume_state(
                candidate_progress,
                all_logs=all_logs,
                db_path=db_path,
            )
            try:
                confirmed_idx = int(resume_state.get("confirmed_idx", -1))
            except Exception:
                confirmed_idx = -1
            indexed_log_ids = set(resume_state.get("indexed_log_ids") or set())
            if not resume_state.get("usable", False):
                print(
                    "[Zep] Resume DB state is inconsistent with confirmed progress; rebuilding Graphiti index from scratch"
                )
                index.initialize(reset_db=True, indexed_log_ids=set())
            else:
                index.initialize(reset_db=False, indexed_log_ids=indexed_log_ids)
                builder_progress = dict(candidate_progress)
                if confirmed_idx != _builder_progress_index(candidate_progress):
                    print(
                        "[Zep] Advancing resume progress from confirmed log index {} to {} based on DB prefix".format(
                            _builder_progress_index(candidate_progress),
                            confirmed_idx,
                        )
                    )
                    confirmed_app_log_id = ""
                    if 0 <= confirmed_idx < len(all_logs):
                        confirmed_app_log_id = str(all_logs[confirmed_idx].get("app_log_id") or "").strip()
                    builder_progress["confirmed_last_log_idx"] = int(confirmed_idx)
                    builder_progress["confirmed_app_log_id"] = confirmed_app_log_id
                    builder_progress["indexed_log_count"] = len(indexed_log_ids)
                    builder_progress["updated_at"] = _now_iso()
                    builder_progress["status"] = "resume_reconciled"
                    builder_progress.pop("last_error", None)
                    _atomic_write_json(progress_path, builder_progress)
                else:
                    builder_progress = dict(candidate_progress)
                print(
                    "[Zep] Resuming from confirmed log index {} (DB episodes: {}, contiguous prefix idx: {})".format(
                        confirmed_idx,
                        int(resume_state.get("db_episode_count") or 0),
                        int(resume_state.get("db_contiguous_prefix_idx", -1)),
                    )
                )
        else:
            if candidate_progress or os.path.exists(db_path):
                print("[Zep] Resume state invalid or incomplete; rebuilding Graphiti index from scratch")
            index.initialize(reset_db=True, indexed_log_ids=set())
    else:
        index.initialize(reset_db=True, indexed_log_ids=set())

    builder_lock = threading.RLock()
    snapshot_lock = threading.Lock()

    def _persist_builder_progress(
        *,
        confirmed_last_log_idx: Optional[int] = None,
        status: Optional[str] = None,
        last_error: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> None:
        current_idx = _builder_progress_index(builder_progress)
        if confirmed_last_log_idx is None:
            trusted_idx = current_idx
        else:
            trusted_idx = max(current_idx, int(confirmed_last_log_idx))
        if trusted_idx >= len(all_logs):
            trusted_idx = len(all_logs) - 1

        confirmed_app_log_id = ""
        if 0 <= trusted_idx < len(all_logs):
            confirmed_app_log_id = str(all_logs[trusted_idx].get("app_log_id") or "").strip()

        builder_progress["db_path"] = db_path
        builder_progress["app_logs_path"] = str(app_logs_path)
        builder_progress["confirmed_last_log_idx"] = int(trusted_idx)
        builder_progress["confirmed_app_log_id"] = confirmed_app_log_id
        builder_progress["indexed_log_count"] = len(index.indexed_log_ids)
        builder_progress["updated_at"] = _now_iso()
        if status is not None:
            builder_progress["status"] = str(status or "")
        elif "status" not in builder_progress:
            builder_progress["status"] = "initialized"
        if checkpoint_id is not None:
            builder_progress["checkpoint_id"] = str(checkpoint_id or "")
        if last_error:
            builder_progress["last_error"] = str(last_error)
        elif last_error == "":
            builder_progress.pop("last_error", None)
        _atomic_write_json(progress_path, builder_progress)

    _persist_builder_progress(status=str(builder_progress.get("status") or "initialized"))

    def _snapshot_entry_for_checkpoint(cp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        checkpoint_id = str(cp.get("checkpoint_id") or "").strip()
        checkpoint_app_log_id = str((cp.get("as_of") or {}).get("app_log_id") or "").strip()
        with snapshot_lock:
            for entry in snapshot_manifest:
                entry_checkpoint_id = str(entry.get("checkpoint_id") or "").strip()
                entry_app_log_id = str(entry.get("checkpoint_app_log_id") or "").strip()
                snapshot_db_rel = str(entry.get("snapshot_db_path") or "").strip()
                if not snapshot_db_rel:
                    continue
                if checkpoint_id and entry_checkpoint_id == checkpoint_id:
                    snapshot_db_abs = snapshot_root / snapshot_db_rel
                    if snapshot_db_abs.exists():
                        return dict(entry)
                if checkpoint_app_log_id and entry_app_log_id == checkpoint_app_log_id:
                    snapshot_db_abs = snapshot_root / snapshot_db_rel
                    if snapshot_db_abs.exists():
                        return dict(entry)
        return None

    def _upsert_snapshot_entry(entry: Dict[str, Any]) -> None:
        checkpoint_id = str(entry.get("checkpoint_id") or "").strip()
        checkpoint_app_log_id = str(entry.get("checkpoint_app_log_id") or "").strip()
        with snapshot_lock:
            replaced = False
            for idx, existing in enumerate(snapshot_manifest):
                existing_checkpoint_id = str(existing.get("checkpoint_id") or "").strip()
                existing_app_log_id = str(existing.get("checkpoint_app_log_id") or "").strip()
                if checkpoint_id and existing_checkpoint_id == checkpoint_id:
                    snapshot_manifest[idx] = dict(entry)
                    replaced = True
                    break
                if checkpoint_app_log_id and existing_app_log_id == checkpoint_app_log_id:
                    snapshot_manifest[idx] = dict(entry)
                    replaced = True
                    break
            if not replaced:
                snapshot_manifest.append(dict(entry))
            payload = {
                "snapshots": sorted(
                    [dict(x) for x in snapshot_manifest],
                    key=lambda item: (
                        int(item.get("confirmed_last_log_idx", -1)) if isinstance(item.get("confirmed_last_log_idx", -1), int) else -1,
                        str(item.get("checkpoint_id", "")),
                    ),
                )
            }
            _atomic_write_json(snapshot_manifest_path, payload)

    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)

    def close() -> None:
        nonlocal answer_llm_usage
        usage_summary_fn = getattr(client, "usage_summary", None)
        if callable(usage_summary_fn):
            raw_usage = usage_summary_fn()
            if isinstance(raw_usage, dict):
                answer_llm_usage = raw_usage
        client.close()
        try:
            index.close()
        except Exception as exc:
            print("[Zep] Warning: Error closing Graphiti: {}".format(exc))

    def _advance_builder_to_checkpoint(cp: Dict[str, Any]) -> Dict[str, Any]:
        observed_logs, _cp_dt, _cp_ts = observed_logs_for_checkpoint(cp, all_logs)
        target_idx = len(observed_logs) - 1
        checkpoint_id = str(cp.get("checkpoint_id") or "")

        with builder_lock:
            current_idx = _builder_progress_index(builder_progress)
            if target_idx <= current_idx:
                _persist_builder_progress(status="ready", checkpoint_id=checkpoint_id, last_error="")
                return {
                    "confirmed_last_log_idx": current_idx,
                    "new_indexed_count": 0,
                    "target_log_idx": target_idx,
                }

            _persist_builder_progress(status="ingesting", checkpoint_id=checkpoint_id, last_error="")
            new_count = 0
            for idx in range(current_idx + 1, target_idx + 1):
                log = all_logs[idx]
                try:
                    step_t0 = time.time()
                    index.add_episode(log)
                    phase_timing["build_memory_duration_s"] += time.time() - step_t0
                except Exception as exc:
                    _persist_builder_progress(
                        confirmed_last_log_idx=idx - 1,
                        status="ingest_failed",
                        checkpoint_id=checkpoint_id,
                        last_error=str(exc),
                    )
                    raise
                new_count += 1
                _persist_builder_progress(
                    confirmed_last_log_idx=idx,
                    status="ingesting",
                    checkpoint_id=checkpoint_id,
                    last_error="",
                )

            _persist_builder_progress(
                confirmed_last_log_idx=target_idx,
                status="ready",
                checkpoint_id=checkpoint_id,
                last_error="",
            )
            print(
                "[Zep] CP {}: indexed {} new logs, total: {}".format(
                    checkpoint_id,
                    new_count,
                    len(index.indexed_log_ids),
                )
            )
            return {
                "confirmed_last_log_idx": target_idx,
                "new_indexed_count": new_count,
                "target_log_idx": target_idx,
            }

    def _open_snapshot_query_index(snapshot_entry: Dict[str, Any]) -> ZepGraphitiIndex:
        snapshot_db_rel = str(snapshot_entry.get("snapshot_db_path") or "").strip()
        if not snapshot_db_rel:
            raise FileNotFoundError("Zep snapshot entry missing snapshot_db_path.")
        snapshot_db_abs = snapshot_root / snapshot_db_rel
        if not snapshot_db_abs.exists():
            raise FileNotFoundError("Zep checkpoint snapshot not found: {}".format(snapshot_db_abs))
        snapshot_config = GraphitiConfig(
            db_path=str(snapshot_db_abs),
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            graphiti_llm_provider=config.graphiti_llm_provider,
            graphiti_llm_model=config.graphiti_llm_model,
            batch_size=config.batch_size,
            max_coroutines=config.max_coroutines,
            max_tokens=config.max_tokens,
        )
        snapshot_index = ZepGraphitiIndex(snapshot_config, usage_tracker=graphiti_usage_tracker)
        confirmed_idx_raw = snapshot_entry.get("confirmed_last_log_idx", -1)
        try:
            confirmed_idx = int(confirmed_idx_raw)
        except Exception:
            confirmed_idx = -1
        indexed_log_ids = {
            str(log.get("app_log_id") or "").strip()
            for log in all_logs[: confirmed_idx + 1]
            if str(log.get("app_log_id") or "").strip()
        }
        snapshot_index.initialize(reset_db=False, indexed_log_ids=indexed_log_ids)
        return snapshot_index

    def _ensure_checkpoint_snapshot(cp: Dict[str, Any], progress_info: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint_id = str(cp.get("checkpoint_id") or "").strip()
        checkpoint_app_log_id = str((cp.get("as_of") or {}).get("app_log_id") or "").strip()
        checkpoint_timestamp = str((cp.get("as_of") or {}).get("timestamp") or "")
        try:
            target_log_idx = int(progress_info.get("target_log_idx", -1))
        except Exception:
            target_log_idx = -1
        existing = _snapshot_entry_for_checkpoint(cp)
        if existing is not None:
            existing_idx_raw = existing.get("confirmed_last_log_idx", -1)
            try:
                existing_idx = int(existing_idx_raw)
            except Exception:
                existing_idx = -1
            if existing_idx >= target_log_idx:
                return existing

        with builder_lock:
            current_indexed_ids = set(index.indexed_log_ids)
            snapshot_id = checkpoint_id or checkpoint_app_log_id or "cp_{}".format(max(target_log_idx, 0))
            snapshot_db_abs = _snapshot_db_path(snapshot_root, snapshot_id)
            index.close()
            try:
                if snapshot_db_abs.exists() or _db_wal_path(snapshot_db_abs).exists():
                    _remove_db_artifact(snapshot_db_abs)
                _copy_db_artifact(Path(db_path), snapshot_db_abs)
            finally:
                index.initialize(reset_db=False, indexed_log_ids=current_indexed_ids)

        entry = {
            "snapshot_id": snapshot_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": checkpoint_app_log_id,
            "checkpoint_timestamp": checkpoint_timestamp,
            "confirmed_last_log_idx": target_log_idx,
            "indexed_log_count": len(current_indexed_ids),
            "snapshot_db_path": str(snapshot_db_abs.relative_to(snapshot_root)),
            "created_at": _now_iso(),
        }
        _upsert_snapshot_entry(entry)
        return dict(entry)

    def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
        progress_info = _advance_builder_to_checkpoint(cp)
        snapshot_entry = _ensure_checkpoint_snapshot(cp, progress_info)
        snapshot_index = _open_snapshot_query_index(snapshot_entry)
        return CheckpointHandle(
            checkpoint_id=str(cp.get("checkpoint_id") or ""),
            state_kind="zep_checkpoint_snapshot",
            state_ref={
                "entry": dict(snapshot_entry),
                "query_index": snapshot_index,
            },
            metadata={
                "retrieval_mode": "zep_graphiti_snapshot",
                "query_isolation_mode": "checkpoint_snapshot_copy",
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "checkpoint_app_log_id": str((cp.get("as_of") or {}).get("app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
                "graphiti_indexed_count": len(snapshot_index.indexed_log_ids),
                "builder_confirmed_last_log_idx": int(progress_info.get("confirmed_last_log_idx", -1)),
                "builder_target_log_idx": int(progress_info.get("target_log_idx", -1)),
                "new_indexed_count": int(progress_info.get("new_indexed_count") or 0),
                "snapshot_id": snapshot_entry.get("snapshot_id"),
                "snapshot_db_path": snapshot_entry.get("snapshot_db_path"),
            },
        )

    def finalize_checkpoint_state(checkpoint_handle: CheckpointHandle) -> None:
        state_ref = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        snapshot_index = state_ref.get("query_index")
        close_fn = getattr(snapshot_index, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                print("[Zep] Warning: Error closing checkpoint snapshot index: {}".format(exc))

    def retrieve_context_for_query(
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        state_ref = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        snapshot_entry = state_ref.get("entry") if isinstance(state_ref.get("entry"), dict) else {}
        snapshot_index = state_ref.get("query_index")
        if snapshot_index is None:
            raise RuntimeError("Zep checkpoint snapshot handle is missing query_index.")
        retrieval_query = str(query_spec.retrieval_query_text or "").strip()
        if not retrieval_query:
            raise ValueError("Zep requires shared QuerySpec.retrieval_query_text.")

        top_k_for_call = int(retrieval_top_k)
        top_k_override = retrieval_options.common.get("top_k")
        if isinstance(top_k_override, int):
            try:
                top_k_for_call = int(top_k_override)
            except Exception:
                top_k_for_call = int(retrieval_top_k)

        indexed_log_ids = set(getattr(snapshot_index, "indexed_log_ids", set()) or set())

        if not indexed_log_ids or top_k_for_call == 0:
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "zep_graphiti_snapshot",
                    "query_isolation_mode": "checkpoint_snapshot_copy",
                    "retrieval_query": retrieval_query,
                    "retrieval_top_k": top_k_for_call,
                    "num_retrieved_logs": 0,
                    "retrieved_app_log_ids": [],
                    "retriever_provider": retriever_provider,
                    "retriever_model": retriever_model,
                    "graphiti_llm_provider": config.graphiti_llm_provider,
                    "graphiti_llm_model": config.graphiti_llm_model,
                    "graphiti_indexed_count": len(indexed_log_ids),
                    "builder_confirmed_last_log_idx": int(snapshot_entry.get("confirmed_last_log_idx", -1)),
                    "checkpoint_state_kind": checkpoint_handle.state_kind,
                    "snapshot_id": snapshot_entry.get("snapshot_id"),
                    "snapshot_db_path": snapshot_entry.get("snapshot_db_path"),
                },
            )

        allowed_log_ids = set()
        for i, log in enumerate(memory_pool):
            app_log_id = log.get("app_log_id")
            key = str(app_log_id).strip() if app_log_id is not None else "pool_{}".format(i)
            allowed_log_ids.add(key)

        search_top_k = len(indexed_log_ids) if top_k_for_call <= 0 else min(
            len(indexed_log_ids),
            max(top_k_for_call, top_k_for_call * 5),
        )
        retrieved = snapshot_index.search(
            retrieval_query,
            top_k=max(1, search_top_k),
            checkpoint_timestamp=str(query_spec.checkpoint_timestamp or ""),
        )

        selected_logs: List[Dict[str, Any]] = []
        selected_ids: List[str] = []
        seen_ids: Set[str] = set()
        for log in retrieved:
            if not isinstance(log, dict):
                continue
            app_log_id = str(log.get("app_log_id") or "").strip()
            if not app_log_id or app_log_id in seen_ids or app_log_id not in allowed_log_ids:
                continue
            seen_ids.add(app_log_id)
            selected_ids.append(app_log_id)
            selected_logs.append(log)
            if top_k_for_call > 0 and len(selected_logs) >= top_k_for_call:
                break

        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in selected_logs],
            debug_metadata={
                "retrieval_mode": "zep_graphiti_snapshot",
                "query_isolation_mode": "checkpoint_snapshot_copy",
                "retrieval_query": retrieval_query,
                "retrieval_top_k": top_k_for_call,
                "num_retrieved_logs": len(selected_logs),
                "retrieved_app_log_ids": selected_ids,
                "retriever_provider": retriever_provider,
                "retriever_model": retriever_model,
                "graphiti_llm_provider": config.graphiti_llm_provider,
                "graphiti_llm_model": config.graphiti_llm_model,
                "graphiti_indexed_count": len(indexed_log_ids),
                "builder_confirmed_last_log_idx": int(snapshot_entry.get("confirmed_last_log_idx", -1)),
                "checkpoint_state_kind": checkpoint_handle.state_kind,
                "snapshot_id": snapshot_entry.get("snapshot_id"),
                "snapshot_db_path": snapshot_entry.get("snapshot_db_path"),
            },
        )

    print("[Zep] Starting TCE pipeline")
    result = run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=client.supports_structured_response(),
        close=close,
        prepare_checkpoint_state=prepare_checkpoint_state,
        finalize_checkpoint_state=finalize_checkpoint_state,
        retrieve_context_for_query=retrieve_context_for_query,
        baseline_name="zep",
        memory_prompt_mode="inline_memory",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
        enable_change_reasoning=enable_change_reasoning,
        enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=rq3_apply_retrieval_top_k,
        retrieval_options_backend={
            "retriever_provider": retriever_provider,
            "retriever_model": retriever_model,
            "graphiti_llm_provider": config.graphiti_llm_provider,
            "graphiti_llm_model": config.graphiti_llm_model,
        },
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
        enable_final_qa=enable_final_qa,
        final_qa_path=final_qa_path,
        final_qa_output_path=final_qa_output_path,
        final_qa_retrieval_top_k=final_qa_retrieval_top_k,
        final_qa_save_prompt_and_raw=final_qa_save_prompt_and_raw,
    )
    total_duration_s = time.time() - run_t0
    build_duration_s = float(phase_timing.get("build_memory_duration_s") or 0.0)
    generation_duration_s = max(0.0, total_duration_s - build_duration_s)
    if not answer_llm_usage:
        usage_summary_fn = getattr(client, "usage_summary", None)
        if callable(usage_summary_fn):
            raw_usage = usage_summary_fn()
            if isinstance(raw_usage, dict):
                answer_llm_usage = raw_usage
    build_memory_usage = graphiti_usage_tracker.usage_summary(phase="build_memory")
    retrieval_usage = graphiti_usage_tracker.usage_summary(phase="retrieval")
    _write_usage_cost_sidecar(
        sidecar_path=_usage_cost_sidecar_path(output_path),
        output_path=output_path,
        llm_provider=llm_provider,
        llm_model=llm_model,
        retriever_provider=retriever_provider,
        embedding_model_name=retriever_model,
        build_duration_s=build_duration_s,
        generation_duration_s=generation_duration_s,
        total_duration_s=total_duration_s,
        build_memory_usage=build_memory_usage,
        retrieval_usage=retrieval_usage,
        answer_llm_usage=answer_llm_usage,
    )
    if isinstance(result, dict):
        result["answer_llm_usage"] = dict(answer_llm_usage)
        result["graphiti_build_usage"] = dict(build_memory_usage)
        result["graphiti_retrieval_usage"] = dict(retrieval_usage)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Zep (Graphiti) baseline generation for TCE.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--retriever-provider", type=str, default="openai")
    parser.add_argument("--retriever-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--retriever-batch-size", type=int, default=10)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument("--max-coroutines", type=int, default=5)
    parser.add_argument("--graphiti-llm-provider", type=str, default=None)
    parser.add_argument("--graphiti-llm-model", type=str, default=None)
    parser.add_argument("--graphiti-max-tokens", type=int, default=16384)

    args = parser.parse_args()
    run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        retrieval_top_k=args.retrieval_top_k,
        max_visible_logs=None,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        answer_temperature=0.0,
        answer_top_p=1.0,
        answer_top_k=None,
        retriever_provider=args.retriever_provider,
        retriever_model=args.retriever_model,
        retriever_batch_size=args.retriever_batch_size,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        max_coroutines=args.max_coroutines,
        graphiti_llm_provider=args.graphiti_llm_provider,
        graphiti_llm_model=args.graphiti_llm_model,
        graphiti_max_tokens=args.graphiti_max_tokens,
    )


if __name__ == "__main__":
    main()
