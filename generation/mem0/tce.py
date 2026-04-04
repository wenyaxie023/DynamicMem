#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from generation.common.llm_client import LLMClient
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import normalize_app_logs, run_pipeline, to_log_text


_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(value)).strip("._-")
    return cleaned or "default"


def _load_mem0_class():
    return importlib.import_module("mem0").Memory


def _patch_mem0_openai_llm_for_gpt5() -> None:
    try:
        openai_mod = importlib.import_module("mem0.llms.openai")
    except Exception:
        return

    openai_llm_cls = getattr(openai_mod, "OpenAILLM", None)
    if openai_llm_cls is None:
        return
    if getattr(openai_llm_cls, "_dynamicmem_gpt5_patch_applied", False):
        return

    extract_json = getattr(importlib.import_module("mem0.memory.utils"), "extract_json")

    def _parse_response(self, response, tools):
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        return response.choices[0].message.content

    def _generate_response(self, messages, response_format=None, tools=None, tool_choice="auto"):
        model_name = str(getattr(self.config, "model", "") or "").strip()
        is_gpt5_model = model_name.startswith("gpt-5")
        params = {
            "model": model_name,
            "messages": messages,
        }
        if not is_gpt5_model:
            params["temperature"] = self.config.temperature
            params["top_p"] = self.config.top_p
        max_tokens = getattr(self.config, "max_tokens", None)
        if max_tokens is not None and not is_gpt5_model:
            params["max_tokens"] = max_tokens

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return _parse_response(self, response, tools)

    openai_llm_cls.generate_response = _generate_response
    openai_llm_cls._dynamicmem_gpt5_patch_applied = True


def _normalize_mem0_provider(provider: Optional[str]) -> str:
    normalized = str(provider or "").strip().lower()
    if not normalized:
        return "openai"
    if normalized == "azure":
        return "azure_openai"
    return normalized


def _apply_mem0_provider_config(
    *,
    node: Dict[str, Any],
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
    api_base: Optional[str],
    azure_deployment: Optional[str],
    azure_api_version: Optional[str],
) -> None:
    node["provider"] = provider
    config = node.setdefault("config", {})
    if model:
        config["model"] = model
    if provider == "azure_openai":
        azure_kwargs = config.setdefault("azure_kwargs", {})
        if api_key:
            azure_kwargs["api_key"] = api_key
        if api_base:
            azure_kwargs["azure_endpoint"] = api_base
        if azure_api_version:
            azure_kwargs["api_version"] = azure_api_version
        resolved_deployment = azure_deployment or (str(model).strip() if model else "")
        if resolved_deployment:
            azure_kwargs["azure_deployment"] = resolved_deployment
        config.pop("api_key", None)
        config.pop("openai_base_url", None)
    else:
        if api_key:
            config["api_key"] = api_key
        if api_base:
            config["openai_base_url"] = api_base
        config.pop("azure_kwargs", None)


def build_mem0_config(
    collection_name: str,
    host: str,
    port: int,
    embedder_provider: str = "openai",
    embedder_api_key: Optional[str] = None,
    embedder_api_base: Optional[str] = None,
    embedder_azure_deployment: Optional[str] = None,
    embedder_azure_api_version: Optional[str] = None,
    llm_provider: str = "openai",
    llm_api_key: Optional[str] = None,
    llm_api_base: Optional[str] = None,
    llm_azure_deployment: Optional[str] = None,
    llm_azure_api_version: Optional[str] = None,
    embedder_model: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    config = {
        "embedder": {"provider": "openai", "config": {"model": embedder_model or "text-embedding-3-small"}},
        "llm": {"provider": "openai", "config": {"model": llm_model or "gpt-4o-mini"}},
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "host": host,
                "port": port,
            },
        },
    }
    _apply_mem0_provider_config(
        node=config["embedder"],
        provider=_normalize_mem0_provider(embedder_provider),
        model=embedder_model or "text-embedding-3-small",
        api_key=embedder_api_key,
        api_base=embedder_api_base,
        azure_deployment=embedder_azure_deployment,
        azure_api_version=embedder_azure_api_version,
    )
    _apply_mem0_provider_config(
        node=config["llm"],
        provider=_normalize_mem0_provider(llm_provider),
        model=llm_model or "gpt-4o-mini",
        api_key=llm_api_key,
        api_base=llm_api_base,
        azure_deployment=llm_azure_deployment,
        azure_api_version=llm_azure_api_version,
    )
    return config


def load_mem0_config(
    config_path: Optional[str],
    collection_name: str,
    host: str,
    port: int,
    embedder_provider: str,
    embedder_api_key: Optional[str],
    embedder_api_base: Optional[str],
    embedder_azure_deployment: Optional[str],
    embedder_azure_api_version: Optional[str],
    llm_provider: str,
    llm_api_key: Optional[str],
    llm_api_base: Optional[str],
    llm_azure_deployment: Optional[str],
    llm_azure_api_version: Optional[str],
    embedder_model: Optional[str],
    llm_model: Optional[str],
) -> Dict[str, Any]:
    if config_path:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    else:
        config = build_mem0_config(
            collection_name=collection_name,
            host=host,
            port=port,
            embedder_provider=embedder_provider,
            embedder_api_key=embedder_api_key,
            embedder_api_base=embedder_api_base,
            embedder_azure_deployment=embedder_azure_deployment,
            embedder_azure_api_version=embedder_azure_api_version,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_azure_deployment=llm_azure_deployment,
            llm_azure_api_version=llm_azure_api_version,
            embedder_model=embedder_model,
            llm_model=llm_model,
        )

    _apply_mem0_provider_config(
        node=config.setdefault("embedder", {}),
        provider=_normalize_mem0_provider(embedder_provider),
        model=embedder_model,
        api_key=embedder_api_key,
        api_base=embedder_api_base,
        azure_deployment=embedder_azure_deployment,
        azure_api_version=embedder_azure_api_version,
    )
    _apply_mem0_provider_config(
        node=config.setdefault("llm", {}),
        provider=_normalize_mem0_provider(llm_provider),
        model=llm_model,
        api_key=llm_api_key,
        api_base=llm_api_base,
        azure_deployment=llm_azure_deployment,
        azure_api_version=llm_azure_api_version,
    )
    config.setdefault("vector_store", {}).setdefault("config", {})["collection_name"] = collection_name
    config["vector_store"]["config"]["host"] = host
    config["vector_store"]["config"]["port"] = port
    return config


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _checkpoint_root(checkpoint_dir: Optional[str], user_id: str, collection_name: str) -> Path:
    base = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir
        else Path(__file__).resolve().parent / "checkpoints"
    )
    return base / _safe_name(user_id) / _safe_name(collection_name)


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _progress_path(root: Path) -> Path:
    return root / "builder_progress.json"


def _load_manifest_entries(root: Path) -> List[Dict[str, Any]]:
    path = _manifest_path(root)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    return [x for x in entries if isinstance(x, dict)]


def _load_builder_progress(root: Path) -> Dict[str, Any]:
    path = _progress_path(root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _builder_progress_index(payload: Dict[str, Any]) -> int:
    raw = payload.get("confirmed_last_event_idx")
    return int(raw) if isinstance(raw, int) or (isinstance(raw, str) and raw.strip()) else -1


def _write_builder_progress(
    root: Path,
    *,
    collection_name: str,
    confirmed_last_event_idx: int,
    confirmed_app_log_id: Optional[str],
    status: str,
) -> None:
    _atomic_write_json(
        _progress_path(root),
        {
            "builder_collection_name": collection_name,
            "confirmed_last_event_idx": int(confirmed_last_event_idx),
            "confirmed_app_log_id": str(confirmed_app_log_id or "").strip(),
            "status": str(status or ""),
            "updated_at": datetime.now().isoformat(),
        },
    )


def _snapshot_path(root: Path, checkpoint_id: str) -> Path:
    return root / "snapshots" / "{}.json".format(_safe_name(checkpoint_id))


def _snapshot_exists_for_entry(root: Path, entry: Dict[str, Any]) -> bool:
    snapshot_path_raw = str(entry.get("snapshot_path", "")).strip()
    if not snapshot_path_raw:
        return False
    return (root / snapshot_path_raw).exists()


def _latest_manifest_entry(root: Path) -> Optional[Dict[str, Any]]:
    entries = [entry for entry in _load_manifest_entries(root) if _snapshot_exists_for_entry(root, entry)]
    if not entries:
        return None
    return max(
        entries,
        key=lambda entry: (
            int(entry.get("last_event_idx", -1)),
            str(entry.get("checkpoint_id", "")).strip(),
        ),
    )


def _collection_name_for_query(base_collection_name: str, checkpoint_id: str) -> str:
    return "{}__query__{}".format(_safe_name(base_collection_name), _safe_name(checkpoint_id))


def _checkpoint_cut_index(cp: Dict[str, Any], app_logs: List[Dict[str, Any]]) -> int:
    as_of = cp.get("as_of")
    as_of = as_of if isinstance(as_of, dict) else {}
    log_index = as_of.get("log_index")
    if isinstance(log_index, int) and log_index >= 0:
        return min(log_index, max(len(app_logs) - 1, 0))
    app_log_id = as_of.get("app_log_id")
    if app_log_id is not None:
        app_log_id = str(app_log_id).strip()
        for idx, log in enumerate(app_logs):
            if str(log.get("app_log_id", "")).strip() == app_log_id:
                return idx
    raise ValueError(
        "Checkpoint must provide as_of.log_index or as_of.app_log_id for mem0 snapshot materialization: {}".format(
            cp.get("checkpoint_id")
        )
    )


def _add_app_log_to_memory(memory: Any, log: Dict[str, Any], user_id: str) -> None:
    memory.add(
        [{"role": "user", "content": "{}".format(to_log_text(log))}],
        metadata={
            "app_log_id": str(log.get("app_log_id", "")),
            "timestamp": str(log.get("timestamp", "")),
        },
        user_id=user_id,
    )


def _materialize_checkpoint_snapshots(
    *,
    benchmark_path: Path,
    app_logs_path: Path,
    user_id: str,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    config_path: Optional[str],
    checkpoint_dir: Optional[str],
    retriever_provider: str,
    embedder_api_key: Optional[str],
    embedder_api_base: Optional[str],
    embedder_azure_deployment: Optional[str],
    embedder_azure_api_version: Optional[str],
    mem0_llm_provider: str,
    llm_api_key: Optional[str],
    llm_api_base: Optional[str],
    llm_azure_deployment: Optional[str],
    llm_azure_api_version: Optional[str],
    embedder_model: Optional[str],
    answer_llm_model: str,
    resume: bool,
    max_checkpoints: Optional[int],
    reset_collections: bool,
) -> Path:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    checkpoints = [cp for cp in benchmark.get("checkpoints", []) if isinstance(cp, dict)]
    if max_checkpoints is not None:
        checkpoints = checkpoints[: max(0, int(max_checkpoints))]

    app_logs = normalize_app_logs(json.loads(app_logs_path.read_text(encoding="utf-8")))
    root = _checkpoint_root(checkpoint_dir, user_id, collection_name)
    required_checkpoint_ids = {
        str(cp.get("checkpoint_id", "")).strip()
        for cp in checkpoints
        if str(cp.get("checkpoint_id", "")).strip()
    }
    existing_entries_by_id = {
        str(entry.get("checkpoint_id", "")).strip(): entry
        for entry in _load_manifest_entries(root)
        if str(entry.get("checkpoint_id", "")).strip() and _snapshot_exists_for_entry(root, entry)
    }
    existing_ids = {
        checkpoint_id
        for checkpoint_id in existing_entries_by_id
        if checkpoint_id
    }
    if not reset_collections and required_checkpoint_ids and required_checkpoint_ids.issubset(existing_ids):
        return root

    Memory = _load_mem0_class()
    builder = Memory.from_config(
        load_mem0_config(
            config_path=config_path,
            collection_name=collection_name,
            host=qdrant_host,
            port=qdrant_port,
            embedder_provider=retriever_provider,
            embedder_api_key=embedder_api_key,
            embedder_api_base=embedder_api_base,
            embedder_azure_deployment=embedder_azure_deployment,
            embedder_azure_api_version=embedder_azure_api_version,
            llm_provider=mem0_llm_provider,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_azure_deployment=llm_azure_deployment,
            llm_azure_api_version=llm_azure_api_version,
            embedder_model=embedder_model,
            llm_model=answer_llm_model,
        )
    )
    if reset_collections:
        try:
            builder.delete_all_memories()
        except Exception:
            pass
        existing_entries_by_id = {}
        _write_builder_progress(
            root,
            collection_name=collection_name,
            confirmed_last_event_idx=-1,
            confirmed_app_log_id=None,
            status="reset",
        )

    progress = _load_builder_progress(root) if resume and not reset_collections else {}
    progress_collection_name = str(progress.get("builder_collection_name", "")).strip()
    has_progress = progress_collection_name == collection_name and _builder_progress_index(progress) >= 0
    last_event_idx = min(_builder_progress_index(progress), len(app_logs) - 1) if has_progress else -1

    if not reset_collections and resume and not has_progress:
        latest_entry = _latest_manifest_entry(root)
        if latest_entry is not None:
            try:
                builder.delete_all_memories()
            except Exception:
                pass
            latest_payload = _load_snapshot_payload(root, latest_entry)
            restored_logs = [
                log
                for log in (latest_payload.get("app_logs") or [])
                if isinstance(log, dict)
            ]
            for log in restored_logs:
                _add_app_log_to_memory(builder, log, user_id)
            last_event_idx = int(latest_payload.get("last_event_idx", latest_entry.get("last_event_idx", -1)))
            restored_app_log_id = None
            if 0 <= last_event_idx < len(app_logs):
                restored_app_log_id = str(app_logs[last_event_idx].get("app_log_id", "")).strip() or None
            _write_builder_progress(
                root,
                collection_name=collection_name,
                confirmed_last_event_idx=last_event_idx,
                confirmed_app_log_id=restored_app_log_id,
                status="restored_from_snapshot",
            )
    elif not reset_collections and not resume:
        try:
            builder.delete_all_memories()
        except Exception:
            pass
        existing_entries_by_id = {}
        last_event_idx = -1
        _write_builder_progress(
            root,
            collection_name=collection_name,
            confirmed_last_event_idx=-1,
            confirmed_app_log_id=None,
            status="initialized",
        )

    manifest_entries: List[Dict[str, Any]] = []
    for cp in checkpoints:
        checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
        if not checkpoint_id:
            continue
        as_of = cp.get("as_of")
        as_of = as_of if isinstance(as_of, dict) else {}
        cut_index = _checkpoint_cut_index(cp, app_logs)
        existing_entry = existing_entries_by_id.get(checkpoint_id)
        if existing_entry is not None and cut_index <= last_event_idx:
            manifest_entries.append(existing_entry)
            continue

        for idx in range(last_event_idx + 1, cut_index + 1):
            log = app_logs[idx]
            _add_app_log_to_memory(builder, log, user_id)
            last_event_idx = idx
            _write_builder_progress(
                root,
                collection_name=collection_name,
                confirmed_last_event_idx=last_event_idx,
                confirmed_app_log_id=str(log.get("app_log_id", "")).strip() or None,
                status="confirmed",
            )

        snapshot_path = _snapshot_path(root, checkpoint_id)
        _atomic_write_json(
            snapshot_path,
            {
                "snapshot_id": checkpoint_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_app_log_id": str(as_of.get("app_log_id", "")).strip(),
                "last_event_idx": cut_index,
                "builder_collection_name": collection_name,
                "app_logs": app_logs[: cut_index + 1],
                "updated_at": datetime.now().isoformat(),
            },
        )
        entry = {
            "snapshot_id": checkpoint_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": str(as_of.get("app_log_id", "")).strip(),
            "snapshot_path": str(snapshot_path.relative_to(root)),
            "builder_collection_name": collection_name,
            "last_event_idx": cut_index,
            "updated_at": datetime.now().isoformat(),
        }
        existing_entries_by_id[checkpoint_id] = entry
        manifest_entries.append(entry)

    missing = sorted(required_checkpoint_ids - {str(x.get("checkpoint_id", "")).strip() for x in manifest_entries})
    if missing:
        raise RuntimeError(
            "Failed to materialize mem0 checkpoint snapshots. Missing checkpoint ids: {}".format(", ".join(missing))
        )
    _atomic_write_json(_manifest_path(root), {"checkpoints": manifest_entries})
    completed_app_log_id = None
    if 0 <= last_event_idx < len(app_logs):
        completed_app_log_id = str(app_logs[last_event_idx].get("app_log_id", "")).strip() or None
    _write_builder_progress(
        root,
        collection_name=collection_name,
        confirmed_last_event_idx=last_event_idx,
        confirmed_app_log_id=completed_app_log_id,
        status="complete",
    )
    return root


def _resolve_manifest_entry(root: Path, cp: Dict[str, Any]) -> Dict[str, Any]:
    entries = _load_manifest_entries(root)
    if not entries:
        raise FileNotFoundError("mem0 checkpoint manifest not found or empty: {}".format(_manifest_path(root)))

    checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
    as_of = cp.get("as_of")
    as_of = as_of if isinstance(as_of, dict) else {}
    checkpoint_app_log_id = str(as_of.get("app_log_id", "")).strip()

    for entry in entries:
        if checkpoint_id and str(entry.get("checkpoint_id", "")).strip() == checkpoint_id:
            return entry
    for entry in entries:
        if checkpoint_app_log_id and str(entry.get("checkpoint_app_log_id", "")).strip() == checkpoint_app_log_id:
            return entry
    return entries[-1]


def _iter_search_results(results: Any) -> List[Any]:
    if isinstance(results, dict):
        if isinstance(results.get("results"), list):
            return list(results["results"])
        if isinstance(results.get("memories"), list):
            return list(results["memories"])
        return [results]
    if isinstance(results, list):
        return list(results)
    return []


def _extract_app_log_id(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            app_log_id = metadata.get("app_log_id")
            if app_log_id is not None and str(app_log_id).strip():
                return str(app_log_id).strip()
        for key in ("memory", "content", "text", "value"):
            value = entry.get(key)
            if isinstance(value, str) and "[APP_LOG]" in value:
                payload_text = value.split("[APP_LOG]", 1)[1].strip().splitlines()[0].strip()
                try:
                    payload = json.loads(payload_text)
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("app_log_id") is not None:
                    return str(payload.get("app_log_id")).strip()
    return None


def _load_snapshot_payload(root: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_path_raw = str(entry.get("snapshot_path", "")).strip()
    if not snapshot_path_raw:
        raise FileNotFoundError("mem0 snapshot entry missing snapshot_path for checkpoint {}".format(entry.get("checkpoint_id")))
    snapshot_path = root / snapshot_path_raw
    if not snapshot_path.exists():
        raise FileNotFoundError("mem0 snapshot payload not found: {}".format(snapshot_path))
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mem0 snapshot payload must be an object: {}".format(snapshot_path))
    return payload


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    user_id: str,
    retrieval_top_k: int,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    config_path: Optional[str],
    checkpoint_dir: Optional[str],
    embedder_api_key: Optional[str],
    embedder_api_base: Optional[str],
    llm_api_key: Optional[str],
    llm_api_base: Optional[str],
    embedder_model: Optional[str],
    answer_llm_model: Optional[str],
    reset_collections: bool,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    retriever_provider: str = "openai",
    retriever_model: Optional[str] = None,
    embedder_azure_deployment: Optional[str] = None,
    embedder_azure_api_version: Optional[str] = None,
    llm_azure_deployment: Optional[str] = None,
    llm_azure_api_version: Optional[str] = None,
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
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
    _patch_mem0_openai_llm_for_gpt5()
    resolved_embedder_model = embedder_model or retriever_model
    resolved_mem0_llm_model = answer_llm_model or llm_model

    manifest_root = _materialize_checkpoint_snapshots(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        user_id=user_id,
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        retriever_provider=retriever_provider,
        embedder_api_key=embedder_api_key,
        embedder_api_base=embedder_api_base,
        embedder_azure_deployment=embedder_azure_deployment,
        embedder_azure_api_version=embedder_azure_api_version,
        mem0_llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_api_base=llm_api_base,
        llm_azure_deployment=llm_azure_deployment,
        llm_azure_api_version=llm_azure_api_version,
        embedder_model=resolved_embedder_model,
        answer_llm_model=resolved_mem0_llm_model,
        resume=resume,
        max_checkpoints=max_checkpoints,
        reset_collections=reset_collections,
    )

    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=answer_temperature,
        top_p=answer_top_p,
        top_k=answer_top_k,
    )

    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)

    def close() -> None:
        client.close()

    def prepare_checkpoint_state(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
    ) -> CheckpointHandle:
        entry = _resolve_manifest_entry(manifest_root, cp)
        snapshot_payload = _load_snapshot_payload(manifest_root, entry)
        temp_collection_name = "{}__{}".format(
            _collection_name_for_query(collection_name, str(entry.get("checkpoint_id", ""))),
            uuid.uuid4().hex[:8],
        )
        Memory = _load_mem0_class()
        query_memory = Memory.from_config(
            load_mem0_config(
                config_path=config_path,
                collection_name=temp_collection_name,
                host=qdrant_host,
                port=qdrant_port,
                embedder_provider=retriever_provider,
                embedder_api_key=embedder_api_key,
                embedder_api_base=embedder_api_base,
                embedder_azure_deployment=embedder_azure_deployment,
                embedder_azure_api_version=embedder_azure_api_version,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
                llm_azure_deployment=llm_azure_deployment,
                llm_azure_api_version=llm_azure_api_version,
                embedder_model=resolved_embedder_model,
                llm_model=resolved_mem0_llm_model,
            )
        )
        try:
            query_memory.delete_all_memories()
        except Exception:
            pass
        snapshot_logs = [
            log
            for log in (snapshot_payload.get("app_logs") or [])
            if isinstance(log, dict)
        ]
        for log in snapshot_logs:
            _add_app_log_to_memory(query_memory, log, user_id)
        return CheckpointHandle(
            checkpoint_id=str(entry.get("checkpoint_id", "") or cp.get("checkpoint_id") or ""),
            state_kind="mem0_snapshot",
            state_ref={
                "manifest_root": str(manifest_root),
                "entry": dict(entry),
                "query_collection_name": temp_collection_name,
            },
            metadata={
                "retrieval_mode": "mem0_checkpoint_snapshot",
                "builder_collection_name": collection_name,
                "query_collection_name": temp_collection_name,
                "checkpoint_id": entry.get("checkpoint_id"),
                "checkpoint_app_log_id": entry.get("checkpoint_app_log_id"),
                "snapshot_id": entry.get("snapshot_id"),
                "last_event_idx": entry.get("last_event_idx"),
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "available_app_log_ids": [
                    str(log.get("app_log_id")).strip()
                    for log in memory_pool
                    if log.get("app_log_id") is not None
                ],
            },
        )

    def retrieve_context_for_query(
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        state_ref = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        entry = state_ref.get("entry") if isinstance(state_ref.get("entry"), dict) else {}
        retrieval_query = str(query_spec.retrieval_query_text or "").strip()
        if not retrieval_query:
            raise ValueError("mem0 requires shared QuerySpec.retrieval_query_text.")
        top_k_for_call = retrieval_top_k
        top_k_override = retrieval_options.common.get("top_k")
        if isinstance(top_k_override, int):
            top_k_for_call = int(top_k_override)
        k = len(memory_pool) if top_k_for_call <= 0 else min(top_k_for_call, len(memory_pool))

        if k <= 0:
            return RetrievalResult(
                mode="mem0_checkpoint_snapshot",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "mem0_checkpoint_snapshot",
                    "builder_collection_name": collection_name,
                    "query_collection_name": state_ref.get("query_collection_name"),
                    "checkpoint_id": entry.get("checkpoint_id"),
                    "checkpoint_app_log_id": entry.get("checkpoint_app_log_id"),
                    "snapshot_id": entry.get("snapshot_id"),
                    "retrieval_top_k": top_k_for_call,
                    "num_retrieved_logs": 0,
                    "retrieved_app_log_ids": [],
                    "retrieval_query": retrieval_query,
                },
            )

        Memory = _load_mem0_class()
        query_memory = Memory.from_config(
            load_mem0_config(
                config_path=config_path,
                collection_name=str(state_ref.get("query_collection_name", "")),
                host=qdrant_host,
                port=qdrant_port,
                embedder_provider=retriever_provider,
                embedder_api_key=embedder_api_key,
                embedder_api_base=embedder_api_base,
                embedder_azure_deployment=embedder_azure_deployment,
                embedder_azure_api_version=embedder_azure_api_version,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
                llm_azure_deployment=llm_azure_deployment,
                llm_azure_api_version=llm_azure_api_version,
                embedder_model=resolved_embedder_model,
                llm_model=resolved_mem0_llm_model,
            )
        )
        raw_results = query_memory.search(retrieval_query, user_id=user_id)

        memory_pool_by_id: Dict[str, Dict[str, Any]] = {}
        for idx, log in enumerate(memory_pool):
            app_log_id = log.get("app_log_id")
            key = str(app_log_id).strip() if app_log_id is not None else "pool_{}".format(idx)
            if key and key not in memory_pool_by_id:
                memory_pool_by_id[key] = log

        selected_logs: List[Dict[str, Any]] = []
        selected_ids: List[str] = []
        seen_ids: Set[str] = set()
        for result in _iter_search_results(raw_results):
            app_log_id = _extract_app_log_id(result)
            if not app_log_id or app_log_id in seen_ids:
                continue
            log = memory_pool_by_id.get(app_log_id)
            if log is None:
                continue
            seen_ids.add(app_log_id)
            selected_ids.append(app_log_id)
            selected_logs.append(log)
            if len(selected_logs) >= k:
                break

        return RetrievalResult(
            mode="mem0_checkpoint_snapshot",
            inline_memory_blocks=[to_log_text(log) for log in selected_logs],
            debug_metadata={
                "retrieval_mode": "mem0_checkpoint_snapshot",
                "builder_collection_name": collection_name,
                "query_collection_name": state_ref.get("query_collection_name"),
                "checkpoint_id": entry.get("checkpoint_id"),
                "checkpoint_app_log_id": entry.get("checkpoint_app_log_id"),
                "snapshot_id": entry.get("snapshot_id"),
                "retrieval_top_k": top_k_for_call,
                "num_retrieved_logs": len(selected_logs),
                "retrieved_app_log_ids": selected_ids,
                "retrieval_query": retrieval_query,
            },
        )

    def finalize_checkpoint_state(checkpoint_handle: CheckpointHandle) -> None:
        state_ref = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        query_collection_name = str(state_ref.get("query_collection_name", "")).strip()
        if not query_collection_name:
            return
        Memory = _load_mem0_class()
        query_memory = Memory.from_config(
            load_mem0_config(
                config_path=config_path,
                collection_name=query_collection_name,
                host=qdrant_host,
                port=qdrant_port,
                embedder_provider=retriever_provider,
                embedder_api_key=embedder_api_key,
                embedder_api_base=embedder_api_base,
                embedder_azure_deployment=embedder_azure_deployment,
                embedder_azure_api_version=embedder_azure_api_version,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
                llm_azure_deployment=llm_azure_deployment,
                llm_azure_api_version=llm_azure_api_version,
                embedder_model=resolved_embedder_model,
                llm_model=resolved_mem0_llm_model,
            )
        )
        try:
            query_memory.delete_all_memories()
        except Exception:
            return

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=client.supports_structured_response(),
        close=close,
        prepare_checkpoint_state=prepare_checkpoint_state,
        retrieve_context_for_query=retrieve_context_for_query,
        finalize_checkpoint_state=finalize_checkpoint_state,
        baseline_name="mem0",
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
            "builder_collection_name": collection_name,
            "qdrant_host": qdrant_host,
            "qdrant_port": qdrant_port,
            "retriever_provider": retriever_provider,
            "embedder_model": resolved_embedder_model,
            "answer_llm_model": resolved_mem0_llm_model,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="mem0 TCE generation with builder snapshots and restore-based queries")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-id", type=str, required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--max-visible-logs", type=int, default=None)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--embedder-api-key", type=str, default=None)
    parser.add_argument("--embedder-api-base", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-api-base", type=str, default=None)
    parser.add_argument("--embedder-model", type=str, default=None)
    parser.add_argument("--answer-llm-model", type=str, default=None)
    parser.add_argument("--reset-collections", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument("--enable-change-reasoning", action="store_true")
    args = parser.parse_args()

    result = run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        user_id=args.user_id,
        retrieval_top_k=args.retrieval_top_k,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        collection_name=args.collection_name or "membench_mem0_{}".format(args.user_id),
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        config_path=args.config_path,
        checkpoint_dir=args.checkpoint_dir,
        embedder_api_key=args.embedder_api_key,
        embedder_api_base=args.embedder_api_base,
        llm_api_key=args.llm_api_key,
        llm_api_base=args.llm_api_base,
        embedder_model=args.embedder_model,
        answer_llm_model=args.answer_llm_model or args.llm_model,
        reset_collections=args.reset_collections,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        enable_change_reasoning=args.enable_change_reasoning,
    )
    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("predictions", [])))


if __name__ == "__main__":
    main()
