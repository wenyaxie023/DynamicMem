import json
from pathlib import Path

import pytest

from bench_core.loader import load_app_logs, load_benchmark, load_prediction


def _write(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_benchmark_ok(tmp_path: Path):
    p = tmp_path / "benchmark.json"
    _write(
        p,
        {
            "user_id": "001_user_001",
            "total_checkpoints": 2,
            "checkpoints": [{"checkpoint_id": "cp_0001"}, {"checkpoint_id": "cp_0002"}],
        },
    )

    bundle = load_benchmark(p)
    assert bundle.user_id == "001_user_001"
    assert bundle.total_checkpoints == 2
    assert len(bundle.checkpoints) == 2


def test_load_benchmark_fallback_total(tmp_path: Path):
    p = tmp_path / "benchmark.json"
    _write(p, {"checkpoints": [{"checkpoint_id": "cp_0001"}]})
    bundle = load_benchmark(p)
    assert bundle.total_checkpoints == 1


def test_load_benchmark_invalid(tmp_path: Path):
    p = tmp_path / "benchmark.json"
    _write(p, [])
    with pytest.raises(ValueError):
        load_benchmark(p)


def test_load_app_logs_list_and_dict(tmp_path: Path):
    p1 = tmp_path / "logs_list.json"
    p2 = tmp_path / "logs_obj.json"
    _write(p1, [{"app_log_id": "log_1"}])
    _write(p2, {"app_logs": [{"app_log_id": "log_1"}, {"app_log_id": "log_2"}]})

    b1 = load_app_logs(p1)
    b2 = load_app_logs(p2)
    assert len(b1.app_logs) == 1
    assert len(b2.app_logs) == 2


def test_load_prediction_list_and_wrapped(tmp_path: Path):
    p1 = tmp_path / "pred_list.json"
    p2 = tmp_path / "pred_wrap.json"
    payload = [{"checkpoint_id": "cp_0001", "snapshot_state": {}}]
    _write(p1, payload)
    _write(p2, {"predictions": payload})

    b1 = load_prediction(p1)
    b2 = load_prediction(p2)
    assert "cp_0001" in b1.predictions_by_id
    assert "cp_0001" in b2.predictions_by_id


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_benchmark(tmp_path / "missing.json")
