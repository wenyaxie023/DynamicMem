from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .json_store import write_json


def write_run_state(
    *,
    path: Path,
    done_uids: Iterable[str],
    pending_uids: Iterable[str] | None = None,
) -> None:
    done_list: List[str] = [uid for uid in done_uids if isinstance(uid, str) and uid]
    payload = {
        "done_count": len(done_list),
        "done_uids": done_list,
    }
    if pending_uids is not None:
        pending_list: List[str] = [uid for uid in pending_uids if isinstance(uid, str) and uid]
        payload["pending_count"] = len(pending_list)
        payload["pending_uids"] = pending_list
    write_json(path, payload)

