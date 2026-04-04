#!/usr/bin/env python3
import json
import tempfile
import time
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.orchestrator_protocol import AnswerExecutionResult, CheckpointHandle, RetrievalResult
from tce_core.pipeline import normalize_evidence_prediction, run_pipeline


def _legacy_inline_pipeline_kwargs(retrieve_context):
    def prepare_checkpoint_state(checkpoint, memory_pool):
        return CheckpointHandle(
            checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
            state_kind="prepared_memory",
            state_ref=checkpoint,
            metadata={
                "checkpoint_timestamp": str((checkpoint.get("as_of") or {}).get("timestamp", "")),
                "checkpoint_app_log_id": str((checkpoint.get("as_of") or {}).get("app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
            },
        )

    def retrieve_context_for_query(checkpoint_handle, query_spec, retrieval_options, memory_pool):
        checkpoint = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        top_k_override = retrieval_options.common.get("top_k")
        ctx_info = retrieve_context(
            checkpoint,
            memory_pool,
            query_spec.target_keys,
            task_text=query_spec.retrieval_query_text,
            retrieval_top_k_override=top_k_override if isinstance(top_k_override, int) else None,
        )
        context_logs = ctx_info.get("context_logs") or []
        metadata = dict(ctx_info.get("metadata") or {})
        metadata.setdefault(
            "retrieved_app_log_ids",
            [
                log.get("app_log_id")
                for log in context_logs
                if isinstance(log, dict) and log.get("app_log_id") is not None
            ],
        )
        metadata.setdefault("retrieval_query", query_spec.retrieval_query_text)
        return RetrievalResult(
            mode=str(metadata.get("retrieval_mode") or "inline_memory"),
            inline_memory_blocks=[json.dumps(log, ensure_ascii=False) for log in context_logs if isinstance(log, dict)],
            debug_metadata=metadata,
        )

    return {
        "prepare_checkpoint_state": prepare_checkpoint_state,
        "retrieve_context_for_query": retrieve_context_for_query,
    }


def _mock_retrieve_context(checkpoint, memory_pool, target_keys, task_text=None, retrieval_top_k_override=None):
    del checkpoint, retrieval_top_k_override
    return {
        "context_logs": memory_pool,
        "retrieval_query": task_text,
        "metadata": {"target_keys_seen": list(target_keys)},
    }


class TcePipelinePackFirstAcceptance(unittest.TestCase):
    def test_unified_pipeline_passes_shared_queryspec_and_inline_memory_blocks_to_hooks(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"profile_state": {"favorite_coffee": "latte"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "item_id": "scp1",
                                "state_key": "profile_state:favorite_coffee",
                                "question_text": "Infer current favorite coffee.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Retrieve favorite coffee evidence.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [
            {
                "app_log_id": "log_0001",
                "timestamp": "2025-01-01 07:00:00",
                "response": {"favorite_coffee": "latte"},
            }
        ]
        seen = {}

        def prepare_checkpoint_state(cp, memory_pool):
            return CheckpointHandle(
                checkpoint_id=str(cp.get("checkpoint_id") or ""),
                state_kind="prepared_test_state",
                state_ref={"memory_pool_size": len(memory_pool)},
                metadata={},
            )

        def retrieve_context_for_query(checkpoint_handle, query_spec, retrieval_options, memory_pool):
            seen["checkpoint_timestamp"] = query_spec.checkpoint_timestamp
            seen["task_query_text"] = query_spec.task_query_text
            seen["retrieval_query_text"] = query_spec.retrieval_query_text
            seen["answer_query_text"] = query_spec.answer_query_text
            seen["state_kind"] = checkpoint_handle.state_kind
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=['{"app_log_id":"log_0001","favorite_coffee":"latte"}'],
                debug_metadata={"retrieval_query": query_spec.retrieval_query_text},
            )

        def answer_query(checkpoint_handle, query_spec, retrieval_result):
            seen["inline_memory_blocks"] = list(retrieval_result.inline_memory_blocks)
            return AnswerExecutionResult(
                raw_output={
                    "snapshot_state": {"profile_state:favorite_coffee": "latte"},
                    "evidence": {
                        "profile_state:favorite_coffee": [
                            {
                                "app_log_id": "log_0001",
                                "evidence_content": "favorite_coffee latte",
                            }
                        ]
                    },
                },
                prompt="hook prompt",
            )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=lambda _: {},
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                prepare_checkpoint_state=prepare_checkpoint_state,
                retrieve_context_for_query=retrieve_context_for_query,
                answer_query=answer_query,
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
            )

        self.assertEqual(seen["checkpoint_timestamp"], "2025-01-01 08:00:00")
        self.assertEqual(seen["task_query_text"], "Infer current favorite coffee.")
        self.assertEqual(seen["retrieval_query_text"], "Retrieve favorite coffee evidence.")
        self.assertEqual(seen["answer_query_text"], "Infer current favorite coffee.")
        self.assertEqual(seen["state_kind"], "prepared_test_state")
        self.assertEqual(
            seen["inline_memory_blocks"],
            ['{"app_log_id":"log_0001","favorite_coffee":"latte"}'],
        )
        self.assertEqual(
            result["predictions"][0]["snapshot_state"]["profile_state:favorite_coffee"],
            "latte",
        )

    def test_normalize_evidence_prediction_keeps_empty_app_log_id_when_content_exists(self):
        normalized = normalize_evidence_prediction(
            {"k": [{"app_log_id": "", "evidence_content": "processed memory summary"}]},
            ["k"],
        )
        self.assertEqual(
            normalized,
            {"k": [{"app_log_id": "", "evidence_content": "processed memory summary"}]},
        )

    def test_pipeline_prefers_state_completion_pack_targets(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}},
                        "profile_state": {"favorite_coffee": "latte"},
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp_mock",
                                "state_key": "habits_state:morning_walk",
                                "question_text": "PACK QUESTION: infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "PACK QUESTION: infer morning walk state.",
                                "pack_source": "computed",
                                "pack_identity": {
                                    "state_key": "habits_state:morning_walk",
                                    "validated_state_value_signature": "{\"timing\": {\"start_time\": \"06:30\"}}",
                                    "pack_version": "v1",
                                },
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [
            {
                "app_log_id": "log_0001",
                "timestamp": "2025-01-01 07:00:00",
                "app_name": "calendar",
                "api_name": "create_event",
                "request": {"title": "Morning walk"},
                "response": {"start_time": "06:30"},
            }
        ]

        def ask_json(prompt: str):
            return {
                "snapshot_state": {
                    "habits_state:morning_walk": {"timing": {"start_time": "06:30"}}
                },
                "evidence": {
                    "habits_state:morning_walk": [
                        {"app_log_id": "log_0001", "evidence_content": "Morning walk start_time 06:30"}
                    ]
                },
            }

        def retrieve_context(checkpoint, memory_pool, target_keys, task_text=None, retrieval_top_k_override=None):
            return {
                "context_logs": memory_pool,
                "retrieval_query": task_text,
                "metadata": {"target_keys_seen": list(target_keys)},
            }

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=ask_json,
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(retrieve_context),
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=False,
            )

        prediction = result["predictions"][0]
        self.assertEqual(
            prediction["metadata"]["target_keys"],
            ["habits_state:morning_walk"],
        )
        self.assertTrue(prediction["metadata"]["state_completion_pack_used"])
        self.assertTrue(prediction["metadata"]["predict_per_key"])
        self.assertIn("PACK QUESTION", prediction["metadata"]["prompt"][0])
        self.assertEqual(
            prediction["metadata"]["per_key_retrieval"][0]["retrieval_query"],
            "PACK QUESTION: infer morning walk state.",
        )
        self.assertEqual(
            prediction["metadata"]["raw_model_output"]["mode"],
            "per_key",
        )
        self.assertEqual(
            prediction["snapshot_state"],
            {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
        )
        self.assertEqual(
            prediction["evidence"]["habits_state:morning_walk"][0]["app_log_id"],
            "log_0001",
        )

    def test_pipeline_prefers_apply_pack_retrieval_query_and_prompt_uses_scenario(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "preferences_state": {"learning_modality": {"statement": "prefers self-paced webinars"}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "preferences_state:learning_modality": {
                                "item_id": "scp_pref",
                                "state_key": "preferences_state:learning_modality",
                                "question_text": "Infer the user's current learning modality preference.",
                                "answer_template": {"statement": "<fill the blank>"},
                                "retrieval_query": "Infer the user's current learning modality preference.",
                                "pack_source": "computed",
                                "pack_identity": {"state_key": "preferences_state:learning_modality"},
                            }
                        },
                    },
                    "change_tracking_pack": {"version": "v1", "pack_authoring": "deterministic", "keys": {}},
                    "rq3_apply_service_qa": {
                        "version": "v1",
                        "keys": {
                            "preferences_state:learning_modality": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "apply_scenario": "The team can fund exactly one option. Candidate actions: A. conference B. webinars C. defer.",
                                        "apply_question": "Which candidate action should the assistant recommend?",
                                        "apply_reference_answer": "B. webinars",
                                        "retrieval_query": "Service scenario:\nThe team can fund exactly one option.\n\nQuestion:\nWhich candidate action should the assistant recommend?",
                                    }
                                ]
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [
            {
                "app_log_id": "log_0001",
                "timestamp": "2025-01-01 07:00:00",
                "app_name": "calendar",
                "api_name": "note",
                "request": {"title": "Learning preference"},
                "response": {"text": "prefers self-paced webinars"},
            }
        ]
        seen_task_text = {"value": None}

        def ask_json(prompt: str):
            if '"snapshot_state"' in prompt:
                return {"snapshot_state": {}, "evidence": {}}
            return {
                "answer": "B. webinars",
                "evidence": [{"app_log_id": "log_0001", "evidence_content": "prefers self-paced webinars"}],
            }

        def retrieve_context(checkpoint, memory_pool, target_keys, task_text=None, retrieval_top_k_override=None):
            seen_task_text["value"] = task_text
            return {
                "context_logs": memory_pool,
                "retrieval_query": task_text,
                "metadata": {"target_keys_seen": list(target_keys)},
            }

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=ask_json,
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(retrieve_context),
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=True,
            )

        prediction = result["predictions"][0]
        self.assertEqual(
            seen_task_text["value"],
            "Service scenario:\nThe team can fund exactly one option.\n\nQuestion:\nWhich candidate action should the assistant recommend?",
        )
        rq3_records = prediction["metadata"]["rq3_apply"]["records"]
        self.assertIn("Service scenario:", rq3_records[0]["prompt"])
        self.assertIn("Question:", rq3_records[0]["prompt"])
        self.assertEqual(
            prediction["rq3_apply_answers"]["preferences_state:learning_modality"]["items"][0]["evidence"][0]["evidence_content"],
            "prefers self-paced webinars",
        )

    def test_pipeline_requires_state_completion_pack(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}},
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Task A requires state_completion_pack"):
                run_pipeline(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    ask_json=lambda prompt: {"snapshot_state": {}, "evidence": {}},
                    ask_structured=None,
                    use_structured_response=False,
                    close=lambda: None,
                    **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                    baseline_name="rag",
                    memory_prompt_mode="inline_memory",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    predict_per_key=True,
                    enable_change_reasoning=False,
                    enable_rq3_apply_service_qa=False,
                )

    def test_pipeline_rejects_blank_state_completion_retrieval_query(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}},
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp1",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Task A requires a non-empty prebuilt retrieval_query"):
                run_pipeline(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    ask_json=lambda prompt: {"snapshot_state": {}, "evidence": {}},
                    ask_structured=None,
                    use_structured_response=False,
                    close=lambda: None,
                    **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                    baseline_name="rag",
                    memory_prompt_mode="inline_memory",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    predict_per_key=True,
                    enable_change_reasoning=False,
                    enable_rq3_apply_service_qa=False,
                )

    def test_pipeline_requires_change_tracking_pack_when_task_b_enabled(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}},
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp1",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Task B requires change_tracking_pack"):
                run_pipeline(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    ask_json=lambda prompt: {"snapshot_state": {}, "evidence": {}},
                    ask_structured=None,
                    use_structured_response=False,
                    close=lambda: None,
                    **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                    baseline_name="rag",
                    memory_prompt_mode="inline_memory",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    predict_per_key=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=False,
                )

    def test_pipeline_rejects_blank_change_tracking_retrieval_query(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}},
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp1",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            }
                        },
                    },
                    "change_tracking_pack": {
                        "version": "v1",
                        "previous_cutoff_ts": "2024-12-01 08:00:00",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "ctp1",
                                "question_text": "Infer the change for morning walk.",
                                "before_template": {"timing": {"start_time": "<fill the blank>"}},
                                "after_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Task B requires a non-empty prebuilt retrieval_query"):
                run_pipeline(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    ask_json=lambda prompt: {"snapshot_state": {}, "evidence": {}},
                    ask_structured=None,
                    use_structured_response=False,
                    close=lambda: None,
                    **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                    baseline_name="rag",
                    memory_prompt_mode="inline_memory",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    predict_per_key=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=False,
                )

    def test_pipeline_allows_empty_apply_pack_when_task_c_enabled(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "preferences_state": {"learning_modality": {"statement": "prefers self-paced webinars"}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "preferences_state:learning_modality": {
                                "item_id": "scp1",
                                "question_text": "Infer learning modality.",
                                "answer_template": {"statement": "<fill the blank>"},
                                "retrieval_query": "Infer learning modality.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=lambda prompt: {"snapshot_state": {}, "evidence": {}},
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=False,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=True,
            )

        prediction = result["predictions"][0]
        self.assertEqual(prediction["rq3_apply_answers"], {})
        self.assertEqual(prediction["metadata"]["rq3_apply"], {"enabled": True})

    def test_pipeline_rejects_blank_apply_retrieval_query(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "preferences_state": {"learning_modality": {"statement": "prefers self-paced webinars"}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "preferences_state:learning_modality": {
                                "item_id": "scp1",
                                "question_text": "Infer learning modality.",
                                "answer_template": {"statement": "<fill the blank>"},
                                "retrieval_query": "Infer learning modality.",
                            }
                        },
                    },
                    "rq3_apply_service_qa": {
                        "version": "v1",
                        "keys": {
                            "preferences_state:learning_modality": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "apply_scenario": "A learning resource must be chosen.",
                                        "apply_question": "Which resource should the assistant recommend?",
                                        "apply_reference_answer": "Recommend webinars",
                                        "retrieval_query": "",
                                    }
                                ]
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Task C requires a non-empty prebuilt retrieval_query"):
                run_pipeline(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    ask_json=lambda prompt: {"snapshot_state": {}, "evidence": {}},
                    ask_structured=None,
                    use_structured_response=False,
                    close=lambda: None,
                    **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                    baseline_name="rag",
                    memory_prompt_mode="inline_memory",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    predict_per_key=True,
                    enable_change_reasoning=False,
                    enable_rq3_apply_service_qa=True,
                )

    def test_pipeline_rejects_legacy_combined_task_a_mode(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp_m",
                                "state_key": "habits_state:morning_walk",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "per-key Task A state-completion"):
                run_pipeline(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    ask_json=lambda prompt: {"snapshot_state": {}, "evidence": {}},
                    ask_structured=None,
                    use_structured_response=False,
                    close=lambda: None,
                    **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                    baseline_name="rag",
                    memory_prompt_mode="inline_memory",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    predict_per_key=False,
                    enable_change_reasoning=False,
                    enable_rq3_apply_service_qa=False,
                )

    def test_pipeline_writes_partial_checkpoint_after_each_key(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}},
                            "budget_review": {"timing": {"start_time": "09:30"}},
                        }
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp_b",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer budget review state.",
                            },
                            "habits_state:morning_walk": {
                                "item_id": "scp_m",
                                "state_key": "habits_state:morning_walk",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            },
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]
        call_count = {"n": 0}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            def ask_json(prompt: str):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    saved = json.loads(output_path.read_text(encoding="utf-8"))
                    partial = saved["predictions"][0]
                    self.assertFalse(partial["metadata"]["_checkpoint_complete"])
                    self.assertEqual(set(partial["snapshot_state"].keys()), {"habits_state:budget_review"})
                if "budget review" in prompt.lower():
                    return {
                        "snapshot_state": {"habits_state:budget_review": {"timing": {"start_time": "09:30"}}},
                        "evidence": {
                            "habits_state:budget_review": [
                                {"app_log_id": "log_0001", "evidence_content": "budget review 09:30"}
                            ]
                        },
                    }
                return {
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "morning walk 06:30"}
                        ]
                    },
                }

            run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=ask_json,
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=False,
                save_every_generation_keys=1,
            )

    def test_pipeline_resume_ignores_incomplete_checkpoint(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp_m",
                                "state_key": "habits_state:morning_walk",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]
        calls = {"n": 0}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            output_path.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "checkpoint_id": "cp1",
                                "snapshot_state": {"habits_state:morning_walk": {}},
                                "evidence": {},
                                "metadata": {"_checkpoint_complete": False},
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            def ask_json(prompt: str):
                calls["n"] += 1
                return {
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "morning walk 06:30"}
                        ]
                    },
                }

            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=ask_json,
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=True,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=False,
            )

        self.assertEqual(calls["n"], 1)
        self.assertTrue(result["predictions"][0]["metadata"]["_checkpoint_complete"])

    def test_pipeline_processes_checkpoints_concurrently(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp_m1",
                                "state_key": "habits_state:morning_walk",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            }
                        },
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"budget_review": {"timing": {"start_time": "09:30"}}}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp_b2",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer budget review state.",
                            }
                        },
                    },
                },
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        def ask_json(prompt: str):
            time.sleep(0.25)
            if "budget review" in prompt.lower():
                return {
                    "snapshot_state": {"habits_state:budget_review": {"timing": {"start_time": "09:30"}}},
                    "evidence": {"habits_state:budget_review": [{"app_log_id": "log_0001", "evidence_content": "budget review 09:30"}]},
                }
            return {
                "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                "evidence": {"habits_state:morning_walk": [{"app_log_id": "log_0001", "evidence_content": "morning walk 06:30"}]},
            }

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            start = time.perf_counter()
            run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=ask_json,
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=False,
                max_checkpoints=2,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=False,
                checkpoint_workers=2,
            )
            elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.45)

    def test_pipeline_processes_keys_concurrently_within_checkpoint(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}},
                            "budget_review": {"timing": {"start_time": "09:30"}},
                        }
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp_m1",
                                "state_key": "habits_state:morning_walk",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            },
                            "habits_state:budget_review": {
                                "item_id": "scp_b1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer budget review state.",
                            },
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        def ask_json(prompt: str):
            time.sleep(0.25)
            if "budget review" in prompt.lower():
                return {
                    "snapshot_state": {"habits_state:budget_review": {"timing": {"start_time": "09:30"}}},
                    "evidence": {"habits_state:budget_review": [{"app_log_id": "log_0001", "evidence_content": "budget review 09:30"}]},
                }
            return {
                "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                "evidence": {"habits_state:morning_walk": [{"app_log_id": "log_0001", "evidence_content": "morning walk 06:30"}]},
            }

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            start = time.perf_counter()
            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=ask_json,
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                baseline_name="rag",
                memory_prompt_mode="inline_memory",
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=False,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=False,
                checkpoint_workers=1,
                within_checkpoint_workers=2,
            )
            elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.45)
        self.assertEqual(result["predictions"][0]["metadata"]["effective_within_checkpoint_workers"], 2)

    def test_letta_policy_forces_serial_concurrency(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "item_id": "scp_m1",
                                "state_key": "habits_state:morning_walk",
                                "question_text": "Infer morning walk state.",
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}},
                                "retrieval_query": "Infer morning walk state.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=lambda prompt: {
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {"habits_state:morning_walk": [{"app_log_id": "log_0001", "evidence_content": "morning walk 06:30"}]},
                },
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                **_legacy_inline_pipeline_kwargs(_mock_retrieve_context),
                baseline_name="letta",
                memory_prompt_mode="agent_memory",
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                predict_per_key=True,
                enable_change_reasoning=False,
                enable_rq3_apply_service_qa=False,
                checkpoint_workers=4,
                within_checkpoint_workers=4,
            )

        md = result["predictions"][0]["metadata"]
        self.assertEqual(md["concurrency_policy"]["checkpoint_parallelism"], "forbidden")
        self.assertEqual(md["concurrency_policy"]["within_checkpoint_parallelism"], "forbidden")
        self.assertEqual(md["effective_checkpoint_workers"], 1)
        self.assertEqual(md["effective_within_checkpoint_workers"], 1)
        self.assertIn("[Agent memory]", result["predictions"][0]["metadata"]["prompt"][0])
        self.assertNotIn("[User memory]\n{'app_log_id':", result["predictions"][0]["metadata"]["prompt"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
