import json
from typing import Dict, List, Optional, Union
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

_TIMESTAMP_KEY_RE = re.compile(r"(timestamp|created_at|updated_at)", re.IGNORECASE)

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
    - Only `response` content is added as memory payload.
    - Any `*timestamp*` or `created_at` fields are extracted as the memory `time`.
    - Timestamp fields are removed from the text payload; everything else is concatenated.
    """
    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[1], MemBenchEvent):
        event = event[1]
    response_obj = event.response or {}
    timestamps: List[str] = []
    _collect_timestamps(response_obj, timestamps)
    time_str = timestamps[0] if timestamps else None

    response_without_timestamps = _strip_timestamp_fields(deepcopy(response_obj))
    response_text = json.dumps(response_without_timestamps, ensure_ascii=False, sort_keys=True)

    content = (
        f"Event {event.event_id} | App: {event.app_name} | API: {event.api_name} | "
        f"Response: {response_text}"
    )
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
        qa_list.append(
            QA(
                question=str(qa.get("question", "")),
                answer=qa.get("answer"),
                evidence=list(qa.get("evidence", [])) if qa.get("evidence") is not None else [],
                category=qa.get("category"),
                adversarial_answer=qa.get("adversarial_answer"),
            )
        )
    return qa_list

def load_membench_dataset(
    app_log_path: Union[str, Path],
    qa_path: Optional[Union[str, Path]] = None,
) -> MemBenchSample:
    """
    Load a MemBench dataset sample from:
    - `app_log_path`: JSON file containing top-level `app_logs` list (or the list itself).
    - `qa_path` (optional): JSON file containing `{"qa": [...]}`.
      If omitted, tries to read `qa` from `app_log_path` (if present).
    """
    app_log_path = Path(app_log_path)
    if not app_log_path.exists():
        raise FileNotFoundError(f"App log file not found at {app_log_path}")

    with open(app_log_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_app_logs = raw.get("app_logs") if isinstance(raw, dict) else raw
    if not isinstance(raw_app_logs, list):
        raise ValueError("Invalid app log format: expected list or object with 'app_logs' list.")

    if qa_path is None and isinstance(raw, dict) and "qa" in raw:
        raw_qa = raw["qa"]
    elif qa_path is not None:
        qa_path = Path(qa_path)
        if not qa_path.exists():
            raise FileNotFoundError(f"QA file not found at {qa_path}")
        with open(qa_path, "r", encoding="utf-8") as f:
            raw_qa = json.load(f)
    else:
        raw_qa = None

    qa_list = _parse_qa_list(raw_qa)

    events: List[MemBenchEvent] = []
    for ev in raw_app_logs:
        if not isinstance(ev, dict):
            continue
        events.append(
            MemBenchEvent(
                event_id=str(ev.get("event_id", "")),
                app_name=str(ev.get("app_name", "")),
                api_name=str(ev.get("api_name", "")),
                request=ev.get("request"),
                response=ev.get("response"),
            )
        )

    sample_id = app_log_path.stem
    return MemBenchSample(sample_id=sample_id, qa=qa_list, app_logs=events)

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
