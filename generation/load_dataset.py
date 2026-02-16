import json
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from copy import deepcopy

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer

@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str

@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]

@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]

@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events

@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]

@dataclass
class LoCoMoSample:
    """A single sample from the LoComo dataset"""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]

def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
            
        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)

def parse_conversation(conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.turns:
                    sessions[session_id] = session
    
    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )

def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file, including image-based content by using captions.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []
    
    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0
            
            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    # Check if QA has image evidence
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break
                    
                    if has_image_evidence:
                        sample_image_qa_count += 1
                        
                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1
                    
                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
            
            # Parse conversation
            conversation = parse_conversation(sample["conversation"])
            
            # Parse event summary
            event_summary = EventSummary(events=sample["event_summary"])
            
            # Parse observation
            observation = Observation(observations=sample["observation"])
            
            # Get session summary
            session_summary = sample.get("session_summary", {})
            
            # Create sample object
            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)
            
            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)
            
            # Print statistics for this sample
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")
    
    return samples

def get_dataset_statistics(samples: List[LoCoMoSample]) -> Dict:
    """
    Get basic statistics about the text-only dataset.
    
    Args:
        samples: List of LoCoMoSample objects
        
    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_samples": len(samples),
        "total_qa_pairs": sum(len(sample.qa) for sample in samples),
        "total_sessions": sum(len(sample.conversation.sessions) for sample in samples),
        "total_turns": sum(
            sum(len(session.turns) for session in sample.conversation.sessions.values())
            for sample in samples
        ),
        "qa_with_adversarial": sum(
            sum(1 for qa in sample.qa if qa.adversarial_answer is not None)
            for sample in samples
        )
    }
    return stats

# -----------------------------
# MemBench dataset (app logs)
# -----------------------------

@dataclass
class MemBenchEvent:
    event_id: str
    timestamp: Optional[str]
    app_name: str
    api_name: str
    request: Optional[dict]
    response: Optional[dict]

@dataclass
class MemBenchSample:
    """A single MemBench sample (typically 1 user/app-log file)."""
    sample_id: str
    qa: List[QA]
    app_logs: List[MemBenchEvent]

_TIMESTAMP_KEY_RE = re.compile(r"(timestamp|created_at|updated_at|last_updated)", re.IGNORECASE)
_MEMBENCH_SIZE_ALIASES = {
    "s": "small",
    "small": "small",
    "m": "medium",
    "med": "medium",
    "medium": "medium",
    "l": "large",
    "lg": "large",
    "large": "large",
}

def _normalize_membench_size(size: str) -> str:
    size_key = (size or "small").strip().lower()
    normalized = _MEMBENCH_SIZE_ALIASES.get(size_key)
    if normalized:
        return normalized
    raise ValueError(f"Invalid size '{size}'. Expected small, medium, or large.")

def _extract_user_index(user_id: str) -> Optional[int]:
    token = (user_id or "").strip().lower()
    if not token:
        return None
    if re.fullmatch(r"\d+", token):
        return int(token)
    match = re.fullmatch(r"(\d+)_user_(\d+)", token)
    if match:
        left = int(match.group(1))
        right = int(match.group(2))
        return right if right != left else left
    return None

def _user_id_aliases(user_id: str) -> set[str]:
    token = (user_id or "").strip().lower()
    aliases = {token}
    idx = _extract_user_index(token)
    if idx is not None:
        padded = f"{idx:03d}"
        aliases.add(str(idx))
        aliases.add(padded)
        aliases.add(f"{padded}_user_{padded}")
    return aliases

def _matches_user_id(sample_id: str, user_id: str) -> bool:
    if not sample_id or not user_id:
        return False
    return bool(_user_id_aliases(sample_id) & _user_id_aliases(user_id))

def _resolve_user_app_log_file(
    app_log_path: Path,
    user_id: str,
    size: str,
) -> Tuple[Path, str]:
    if app_log_path.is_file():
        inferred_sample_id = app_log_path.parent.name or app_log_path.stem
        if not _matches_user_id(inferred_sample_id, user_id):
            raise ValueError(
                f"App log file '{app_log_path}' does not match user_id='{user_id}'. "
                f"Inferred sample_id='{inferred_sample_id}'."
            )
        return app_log_path, inferred_sample_id

    if not app_log_path.exists():
        raise FileNotFoundError(f"App log path not found at {app_log_path}")
    if not app_log_path.is_dir():
        raise ValueError(f"Invalid app log path: {app_log_path}")

    pattern = f"app_log_{size}.json"
    matches: List[Tuple[Path, str]] = []
    direct_file = app_log_path / pattern
    if direct_file.exists() and _matches_user_id(app_log_path.name, user_id):
        matches.append((direct_file, app_log_path.name))

    for entry in sorted(app_log_path.iterdir()):
        if not entry.is_dir():
            continue
        candidate = entry / pattern
        if candidate.exists() and _matches_user_id(entry.name, user_id):
            matches.append((candidate, entry.name))

    if not matches:
        raise FileNotFoundError(
            f"No '{pattern}' found for user_id='{user_id}' under {app_log_path}"
        )
    if len(matches) > 1:
        rendered = ", ".join(f"{path} (sample_id={sid})" for path, sid in matches)
        raise ValueError(
            f"Ambiguous app log matches for user_id='{user_id}' under {app_log_path}: {rendered}"
        )
    return matches[0]

def _is_timestamp_key(key: str) -> bool:
    return bool(_TIMESTAMP_KEY_RE.search(key))

def _collect_timestamps(obj: object, out: List[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and _is_timestamp_key(key):
                if isinstance(value, (str, int, float)):
                    out.append(str(value))
            _collect_timestamps(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_timestamps(item, out)

def _strip_timestamp_fields(obj: object) -> object:
    if isinstance(obj, dict):
        filtered = {}
        for key, value in obj.items():
            if isinstance(key, str) and _is_timestamp_key(key):
                continue
            filtered[key] = _strip_timestamp_fields(value)
        return filtered
    if isinstance(obj, list):
        return [_strip_timestamp_fields(item) for item in obj]
    return obj

def build_membench_memory_from_event(event: MemBenchEvent) -> tuple[str, Optional[str]]:
    """
    Convert a MemBench event into memory content + time.

    Rules:
    - The entire event is dumped as JSON for the memory payload.
    - Use the event-level timestamp when present; otherwise fallback to timestamp fields.
    - No fields are stripped from the payload.
    """
    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[1], MemBenchEvent):
        event = event[1]
    if isinstance(event, dict):
        payload = event
        event_timestamp = event.get("timestamp")
    else:
        payload = {
            "app_log_id": event.event_id,
            "timestamp": event.timestamp,
            "app_name": event.app_name,
            "api_name": event.api_name,
            "request": event.request,
            "response": event.response,
        }
        event_timestamp = event.timestamp

    timestamps: List[str] = []
    _collect_timestamps(payload, timestamps)
    time_str = event_timestamp or (timestamps[0] if timestamps else None)
    payload_with_time = dict(payload)
    payload_with_time["memory_timestamp"] = time_str
    content = json.dumps(payload_with_time, ensure_ascii=False, sort_keys=True)
    return content, time_str

def _parse_qa_list(raw_qa: object) -> List[QA]:
    if raw_qa is None:
        return []
    if isinstance(raw_qa, dict) and "qa" in raw_qa:
        raw_qa = raw_qa["qa"]
    if not isinstance(raw_qa, list):
        raise ValueError("Invalid QA format: expected list or object with 'qa' list.")

    qa_list: List[QA] = []
    for qa in raw_qa:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question") or qa.get("query") or ""
        answer = qa.get("answer")
        if answer is None:
            if "prediction" in qa:
                answer = qa.get("prediction")
            else:
                answer = qa.get("reference")
        evidence = qa.get("evidence")
        if evidence is None:
            metadata = qa.get("metadata")
            if isinstance(metadata, dict):
                evidence = metadata.get("reference_evidence")
        evidence_list = list(evidence) if evidence is not None else []
        qa_list.append(
            QA(
                question=str(question),
                answer=answer,
                evidence=evidence_list,
                category=qa.get("category"),
                adversarial_answer=qa.get("adversarial_answer"),
            )
        )
    return qa_list

def _find_user_qa_file(app_log_file: Path, sample_id: str) -> Optional[Path]:
    parent = app_log_file.parent
    direct = parent / f"qa_human_{sample_id}.json"
    if direct.exists():
        return direct
    digits = re.findall(r"\\d+", sample_id)
    if digits:
        candidate = parent / f"qa_human_{digits[-1]}.json"
        if candidate.exists():
            return candidate
    matches = sorted(parent.glob("qa_human_*.json"))
    if len(matches) == 1:
        return matches[0]
    return None

def load_membench_dataset(
    app_log_path: Union[str, Path],
    qa_path: Optional[Union[str, Path]] = None,
    *,
    user_id: str,
    size: str = "small",
    load_ckpts: bool = False,
) -> Union[List[MemBenchSample], Tuple[List[MemBenchSample], List[dict]]]:
    """
    Load one MemBench sample for a specific user from:
    - `app_log_path`: JSON file with an app log list, or a directory containing user subfolders.
      When a directory is provided, a matching user subfolder is expected to contain app_log_{size}.json.
    - `qa_path` (optional): JSON file containing `{"qa": [...]}`.
      If omitted, tries to read per-user `qa_human_*.json` or `qa` from app logs.
    - `user_id` (required): user/sample identifier (supports aliases like 1, 001, 001_user_001).
    - `load_ckpts`: if True, also loads raw benchmark checkpoints.

    Returns:
    - `[MemBenchSample]` when `load_ckpts=False`
    - `([MemBenchSample], checkpoints)` when `load_ckpts=True`
    """
    if not str(user_id or "").strip():
        raise ValueError("user_id is required and cannot be empty.")
    app_log_path = Path(app_log_path)
    size = _normalize_membench_size(size)
    app_log_file, sample_id = _resolve_user_app_log_file(app_log_path, user_id, size)

    qa_list_from_path: Optional[List[QA]] = None
    if qa_path is not None:
        qa_path = Path(qa_path)
        if not qa_path.exists():
            raise FileNotFoundError(f"QA file not found at {qa_path}")
        with open(qa_path, "r", encoding="utf-8") as f:
            raw_qa = json.load(f)
        qa_list_from_path = _parse_qa_list(raw_qa)

    with open(app_log_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_app_logs = raw.get("app_logs") if isinstance(raw, dict) else raw
    if not isinstance(raw_app_logs, list):
        raise ValueError(
            f"Invalid app log format in {app_log_file}: "
            "expected list or object with 'app_logs' list."
        )

    if qa_list_from_path is not None:
        qa_list = list(qa_list_from_path)
    else:
        qa_file = _find_user_qa_file(app_log_file, sample_id)
        if qa_file is not None:
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_list = _parse_qa_list(json.load(f))
        elif isinstance(raw, dict) and "qa" in raw:
            qa_list = _parse_qa_list(raw["qa"])
        else:
            qa_list = []

    events: List[MemBenchEvent] = []
    for ev in raw_app_logs:
        if not isinstance(ev, dict):
            continue
        event_id = ev.get("app_log_id", ev.get("event_id", ""))
        events.append(
            MemBenchEvent(
                event_id=str(event_id),
                timestamp=ev.get("timestamp"),
                app_name=str(ev.get("app_name", "")),
                api_name=str(ev.get("api_name", "")),
                request=ev.get("request"),
                response=ev.get("response"),
            )
        )

    samples = [MemBenchSample(sample_id=sample_id, qa=qa_list, app_logs=events)]
    if not load_ckpts:
        return samples

    benchmark_path = app_log_file.parent / "dynamic_state_prediction_benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Checkpoint benchmark file not found for user '{sample_id}': {benchmark_path}"
        )
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_payload: Any = json.load(f)
    checkpoints = benchmark_payload.get("checkpoints", []) if isinstance(benchmark_payload, dict) else []
    if not isinstance(checkpoints, list):
        raise ValueError(
            f"Invalid checkpoint format in {benchmark_path}: expected top-level 'checkpoints' list."
        )
    return samples, checkpoints

if __name__ == "__main__":
    # Example usage
    dataset_path = Path(__file__).parent / "data" / "locomo10.json"
    try:
        print(f"Loading dataset from: {dataset_path}")
        samples = load_locomo_dataset(dataset_path)
        for sample_idx, sample in enumerate(samples):
            print(f"\nSample {sample_idx}:")
            for _,turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    print(turn)
                    break   
        # stats = get_dataset_statistics(samples)
        # print("\nDataset Statistics (Text-only content):")
        # for key, value in stats.items():
        #     print(f"{key}: {value}")
        # print(len(samples))
        # for sample in samples:
        #     print(sample)
        #     break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
