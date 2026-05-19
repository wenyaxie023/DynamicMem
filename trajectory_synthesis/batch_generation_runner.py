"""
Batch Generation Runner for Memory Benchmark

Orchestrates the full pipeline for user data generation:
1. Dynamic Profile Generation (per domain → domain conflict resolve → cross-domain align)
2. Events Chain Generation (w0 → w1 → w2 → w3 → w4)
3. App Logs Generation (merge all domains → convert to app logs)

Usage:
    # Single user run with debug mode
    python batch_generation_runner.py --debug
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dotenv
dotenv.load_dotenv()

from elite_persona_sampler import (
    DEFAULT_SAMPLE_PATH,
    DEFAULT_SEED,
    load_sampled_personas,
    sample_elite_personas,
)
from llm_client import GeminiJSONClient, OpenAICompatibleClient, create_client, LLMProvider
from logging_utils import setup_logger, get_logger
from stages import DynamicProfileStage, EventsChainStage, AppLogsStage
from stages.stage1_dynamic_profile import Domain, TimelineConfig
from domains import DOMAINS
DEFAULT_PERSONA_SAMPLE_SIZE = 10
DEFAULT_USER_COUNT = 10


def _slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict) -> None:
    """Write JSON to file with formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_text(path: Path, text: str) -> None:
    """Write text to file."""
    path.write_text(text)


def _load_or_sample_elite_personas(
    sample_path: str | Path,
    *,
    sample_size: int,
    seed: int,
) -> List[Dict]:
    """Ensure we have a local elite persona sample and return it."""
    sample_path = Path(sample_path)
    if not sample_path.exists():
        sample_elite_personas(
            output_path=sample_path,
            sample_size=sample_size,
            seed=seed,
        )
    return load_sampled_personas(sample_path)


def _extract_user_description(persona: Dict[str, Any]) -> Optional[str]:
    """Pull user description text from a PersonaHub record."""
    return persona.get("persona") or persona.get("user_description")


def _build_user_records(
    personas: Sequence[Dict[str, Any]],
    *,
    max_users: int,
) -> List[Dict[str, str]]:
    """Convert PersonaHub records into user_id/description payloads."""
    users: List[Dict[str, str]] = []
    for idx, persona in enumerate(personas):
        if len(users) >= max_users:
            break
        user_description = _extract_user_description(persona)
        if not user_description:
            continue
        user_id = (
            str(persona.get("user_id") or persona.get("id") or f"user_{idx + 1:03d}")
        )
        users.append(
            {
                "user_id": user_id,
                "user_description": user_description,
            }
        )
    return users


def _load_user_records_from_json(path: Path) -> List[Dict[str, str]]:
    """Load user records from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"User descriptions JSON not found: {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "users" in payload:
        users = payload["users"]
    elif isinstance(payload, list):
        users = payload
    else:
        raise ValueError(
            "User descriptions JSON must be a list or contain a 'users' field."
        )
    normalized: List[Dict[str, str]] = []
    for idx, entry in enumerate(users):
        if not isinstance(entry, dict):
            continue
        user_description = (
            entry.get("user_description")
            or entry.get("persona")
            or entry.get("description")
        )
        if not user_description:
            continue
        user_id = str(entry.get("user_id") or entry.get("id") or f"user_{idx + 1:03d}")
        normalized.append(
            {
                "user_id": user_id,
                "user_description": user_description,
            }
        )
    return normalized


def _prepare_persona_users(
    *,
    sample_path: str | Path,
    user_count: int,
    sample_size: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], Path]:
    """Stage 0: sample PersonaHub and return user records."""
    target_sample_size = max(user_count, sample_size)
    personas = _load_or_sample_elite_personas(
        sample_path=sample_path,
        sample_size=target_sample_size,
        seed=seed,
    )
    users = _build_user_records(personas, max_users=user_count)
    if len(users) < user_count:
        raise ValueError(
            f"Only found {len(users)} usable personas, expected {user_count}."
        )
    return users, Path(sample_path)


@dataclass
class RunConfig:
    """Configuration for a single run."""
    # Basic settings
    output_dir: Path = Path("./outputs")
    model_name: str = "gemini-3-flash-preview"

    # LLM provider settings
    provider: str = "google"  # "google", "openai", "aimlapi", "custom"
    base_url: Optional[str] = None  # Custom base URL (optional)
    api_key: Optional[str] = None  # API key (optional, falls back to env var)

    # Debug mode - saves all intermediate outputs
    debug_mode: bool = True

    # Stage control (for skipping stages if already done)
    skip_stage1: bool = False
    skip_stage2: bool = False
    skip_stage3: bool = False

    # Dry run mode - only save world background prompts, don't generate
    dry_run_world_bg: bool = False
    world_bg_mode: str = "require_existing"

    # Timeline configuration
    timeline: TimelineConfig = field(default_factory=TimelineConfig)

    # User configuration
    user_id: str = "user_001"

    # Cutoff date for app log generation
    cutoff_date: Optional[str] = None


@dataclass
class RunResult:
    """Result from a complete pipeline run."""
    user_basic_profile: Dict[str, Any]
    dynamic_profiles: Dict[str, Dict]
    events_chains_by_domain: Dict[str, List[Dict]]
    app_logs: List[Dict[str, Any]]
    usage_summary: Dict[str, Any]
    output_dir: Path


class BatchGenerationRunner:
    """
    Main orchestrator for user data generation.

    Pipeline:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Stage 1: Dynamic Profile Generation                                 │
    │   1.1 Generate raw dynamic profile per domain                       │
    │   1.2 In-domain conflict resolve (rule 1-5 cascade fix)             │
    │   1.3 Cross-domain key alignment                                    │
    │   1.4 Cross-domain conflict resolution (temporal + attribute)       │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Stage 2: Events Chain Generation                                    │
    │   For each domain:                                                  │
    │     For each window (w1 → w4):                                      │
    │       - Generate events chain from domain_window_state              │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Stage 3: App Logs Generation                                        │
    │   3.1 Merge all domain events chains                                │
    │   3.2 Sort events chronologically                                   │
    │   3.3 Convert each event to structured app log via LLM              │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: RunConfig,
        llm_client: Optional[GeminiJSONClient | OpenAICompatibleClient] = None,
    ):
        self.config = config
        self.llm_client = llm_client or create_client(
            provider=config.provider,
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.domains = self._load_domains()

        # Initialize output directory
        _ensure_dir(self.config.output_dir)

        # Setup logger with file output in the output directory
        self.logger = setup_logger(
            name=f"pipeline_{config.user_id}",
            log_dir=self.config.output_dir,
            log_filename="pipeline.log",
        )

        # Initialize stages with logger
        self.stage1 = DynamicProfileStage(
            llm_client=self.llm_client,
            output_dir=self.config.output_dir,
            timeline=self.config.timeline,
            debug_mode=self.config.debug_mode,
            dry_run_world_bg=self.config.dry_run_world_bg,
            world_bg_mode=self.config.world_bg_mode,
            logger=self.logger,
        )

        self.stage2 = EventsChainStage(
            llm_client=self.llm_client,
            output_dir=self.config.output_dir,
            debug_mode=self.config.debug_mode,
            logger=self.logger,
        )

        self.stage3 = AppLogsStage(
            llm_client=self.llm_client,
            output_dir=self.config.output_dir,
            debug_mode=self.config.debug_mode,
            logger=self.logger,
        )

    def _load_domains(self) -> List[Domain]:
        """Load domain definitions from domains.json."""
        domains_data = DOMAINS

        return [
            Domain(
                domain_name=d["domain_name"],
                domain_scope_definition=d["domain_scope_definition"],
            )
            for d in domains_data
        ]

    def run(
        self,
        user_description: str,
    ) -> RunResult:
        """
        Run the complete pipeline for a single user.

        Args:
            user_description: Raw user description text

        Returns:
            RunResult containing all generated data
        """
        self.logger.info("=" * 70)
        self.logger.info("BATCH GENERATION RUNNER")
        self.logger.info("=" * 70)
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Debug mode: {self.config.debug_mode}")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Domains: {[d.domain_name for d in self.domains]}")
        self.logger.info("=" * 70)

        aggregate_usage: Dict[str, Any] = {}

        # Save input for reference
        _write_text(
            self.config.output_dir / "input_user_description.txt",
            user_description,
        )

        # =====================================================================
        # Stage 1: Dynamic Profile Generation
        # =====================================================================
        if not self.config.skip_stage1:
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("STAGE 1: DYNAMIC PROFILE GENERATION")
            self.logger.info("=" * 70)
            stage1_result = self.stage1.run(
                user_description=user_description,
                domains=self.domains,
            )

            user_basic_profile = stage1_result.user_basic_profile
            dynamic_profiles = stage1_result.dynamic_profiles
            aggregate_usage["stage1"] = stage1_result.usage

            # Save stage 1 summary
            if self.config.debug_mode:
                _write_json(
                    self.config.output_dir / "stage1_summary.json",
                    {
                        "domains_processed": list(dynamic_profiles.keys()),
                        "conflicts_resolved": len(stage1_result.conflicts_summary),
                        "usage": stage1_result.usage,
                    },
                )

            # Early exit if dry_run_world_bg mode
            if self.config.dry_run_world_bg:
                self.logger.info("")
                self.logger.info("=" * 70)
                self.logger.info("DRY RUN MODE: World background prompts saved, pipeline stopped")
                self.logger.info("=" * 70)
                return RunResult(
                    user_basic_profile=user_basic_profile,
                    dynamic_profiles={},
                    events_chains_by_domain={},
                    app_logs=[],
                    usage_summary=aggregate_usage,
                    output_dir=self.config.output_dir,
                )
        else:
            self.logger.info("Skipping Stage 1 (using cached data)...")
            user_basic_profile = self._load_cached_basic_profile()
            dynamic_profiles = self._load_cached_dynamic_profiles()

        # =====================================================================
        # Stage 2: Events Chain Generation
        # =====================================================================
        if not self.config.skip_stage2:
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("STAGE 2: EVENTS CHAIN GENERATION")
            self.logger.info("=" * 70)

            # Convert domains to stage2 format
            stage2_domains = [
                self.stage2.__class__.__bases__[0].__subclasses__()[0]  # Get Domain from stage
                if hasattr(self.stage2, 'Domain') else
                type('Domain', (), {'domain_name': d.domain_name, 'domain_scope_definition': d.domain_scope_definition})()
                for d in self.domains
            ]

            # Use simple Domain objects
            from stages.stage2_events_chain import Domain as Stage2Domain
            stage2_domains = [
                Stage2Domain(
                    domain_name=d.domain_name,
                    domain_scope_definition=d.domain_scope_definition,
                )
                for d in self.domains
            ]

            stage2_result = self.stage2.run(
                domains=stage2_domains,
                user_basic_profile=user_basic_profile,
                dynamic_profiles=dynamic_profiles,
                world_background=None,
            )

            events_chains_by_domain = stage2_result.events_chains_by_domain
            aggregate_usage["stage2"] = stage2_result.usage

            # Save stage 2 summary
            if self.config.debug_mode:
                total_events = sum(
                    len(chain.get("events", []))
                    for windows in events_chains_by_domain.values()
                    for window in windows
                    for chain in window.get("event_chains", [])
                )
                _write_json(
                    self.config.output_dir / "stage2_summary.json",
                    {
                        "domains_processed": list(events_chains_by_domain.keys()),
                        "total_events_generated": total_events,
                        "usage": stage2_result.usage,
                    },
                )
        else:
            self.logger.info("Skipping Stage 2 (using cached data)...")
            events_chains_by_domain = self._load_cached_events_chains()

        # =====================================================================
        # Stage 3: App Logs Generation
        # =====================================================================
        if not self.config.skip_stage3:
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("STAGE 3: APP LOGS GENERATION")
            self.logger.info("=" * 70)

            stage3_result = self.stage3.run(
                events_chains_by_domain=events_chains_by_domain,
                user_basic_profile=user_basic_profile,
                user_id=self.config.user_id,
                cutoff_date=self.config.cutoff_date,
                resume_from_existing=True,
                # clear_checkpoint_on_start=True,
            )


            app_logs = stage3_result.app_logs
            aggregate_usage["stage3"] = stage3_result.usage

            # Save stage 3 summary
            if self.config.debug_mode:
                _write_json(
                    self.config.output_dir / "stage3_summary.json",
                    {
                        "total_events": stage3_result.total_events,
                        "total_logs_generated": stage3_result.total_logs_generated,
                        "usage": stage3_result.usage,
                    },
                )
        else:
            self.logger.info("Skipping Stage 3 (using cached data)...")
            app_logs = self._load_cached_app_logs()

        # =====================================================================
        # Final Summary
        # =====================================================================
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Domains processed: {len(self.domains)}")
        self.logger.info(f"App logs generated: {len(app_logs)}")
        self.logger.info(f"Output directory: {self.config.output_dir}")

        # Save final summary
        _write_json(
            self.config.output_dir / "pipeline_summary.json",
            {
                "user_id": self.config.user_id,
                "model": self.config.model_name,
                "domains": [d.domain_name for d in self.domains],
                "total_app_logs": len(app_logs),
                "usage_summary": aggregate_usage,
            },
        )

        return RunResult(
            user_basic_profile=user_basic_profile,
            dynamic_profiles=dynamic_profiles,
            events_chains_by_domain=events_chains_by_domain,
            app_logs=app_logs,
            usage_summary=aggregate_usage,
            output_dir=self.config.output_dir,
        )

    # =========================================================================
    # Cache Loading Methods
    # =========================================================================

    def _load_cached_basic_profile(self) -> Dict[str, Any]:
        """Load cached basic profile."""
        path = self.config.output_dir / "user_basic_profile.json"
        if path.exists():
            return json.loads(path.read_text())
        raise FileNotFoundError(f"Cached basic profile not found: {path}")

    def _load_cached_dynamic_profiles(self) -> Dict[str, Dict]:
        """Load cached dynamic profiles."""
        path = self.config.output_dir / "dynamic_profiles_final.json"
        if path.exists():
            return json.loads(path.read_text())
        raise FileNotFoundError(f"Cached dynamic profiles not found: {path}")

    def _load_cached_events_chains(self) -> Dict[str, List[Dict]]:
        """Load cached events chains."""
        path = self.config.output_dir / "all_events_chains.json"
        if path.exists():
            return json.loads(path.read_text())
        raise FileNotFoundError(f"Cached events chains not found: {path}")

    def _load_cached_app_logs(self) -> List[Dict[str, Any]]:
        """Load cached app logs."""
        path = self.config.output_dir / "app_logs_final.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("app_logs", [])
        raise FileNotFoundError(f"Cached app logs not found: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the batch generation pipeline for user data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="LLM model name to use",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "openai", "aimlapi", "custom"],
        help="LLM provider to use (google, openai, aimlapi, custom)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for the LLM API (optional)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM provider (optional, falls back to env var)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode (save all intermediate outputs)",
    )
    parser.add_argument(
        "--user-description",
        type=str,
        default=None,
        help="Path to user description file",
    )
    parser.add_argument(
        "--user-descriptions-json",
        type=str,
        default=None,
        help="Path to JSON list with user_id/user_description entries",
    )
    parser.add_argument(
        "--user-count",
        type=int,
        default=DEFAULT_USER_COUNT,
        help="Number of users to generate (e.g., 1/2/10)",
    )
    parser.add_argument(
        "--user-index",
        type=int,
        default=None,
        help="1-based index of the user to run (e.g., 2 for the second user)",
    )
    parser.add_argument(
        "--persona-sample-path",
        type=str,
        default=str(DEFAULT_SAMPLE_PATH),
        help="Path to PersonaHub sample JSONL file",
    )
    parser.add_argument(
        "--persona-sample-size",
        type=int,
        default=DEFAULT_PERSONA_SAMPLE_SIZE,
        help="Sample size to pull from PersonaHub for Stage 0",
    )
    parser.add_argument(
        "--persona-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for PersonaHub sampling",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 (use cached dynamic profiles)",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 (use cached events chains)",
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="Skip Stage 3 (use cached app logs)",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Cutoff date for app log generation (YYYY-MM-DD or MM.DD)",
    )
    parser.add_argument(
        "--dry-run-world-bg",
        action="store_true",
        help="Dry run mode: only generate basic profiles and save world background prompts (no LLM generation for world backgrounds)",
    )
    parser.add_argument(
        "--world-bg-mode",
        type=str,
        choices=["require_existing", "generate_if_missing"],
        default="require_existing",
        help=(
            "How Stage 1 resolves world backgrounds: "
            "'require_existing' loads pre-provided files and fails if missing; "
            "'generate_if_missing' falls back to LLM generation for missing domains."
        ),
    )

    args = parser.parse_args()

    # Setup paths (artifacts live at the repo-root outputs/ dir)
    base_dir = Path(__file__).resolve().parent.parent

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "outputs" / _slugify(args.model)

    _ensure_dir(output_dir)

    # Stage 0: Resolve user descriptions
    user_records: List[Dict[str, str]] = []
    if args.user_description:
        if args.user_index is not None:
            raise ValueError("--user-index cannot be used with --user-description")
        user_desc_path = Path(args.user_description)
        if not user_desc_path.exists():
            raise FileNotFoundError(
                f"User description file not found: {user_desc_path}"
            )
        user_records = [
            {
                "user_id": "user_001",
                "user_description": user_desc_path.read_text().strip(),
            }
        ]
    else:
        user_records, sample_path = _prepare_persona_users(
            sample_path=args.persona_sample_path,
            user_count=args.user_count,
            sample_size=args.persona_sample_size,
            seed=args.persona_seed,
        )
        _write_json(
            output_dir / "stage0_user_descriptions.json",
            {
                "source": "proj-persona/PersonaHub:elite_persona",
                "sample_path": str(sample_path),
                "user_count": args.user_count,
                "users": user_records,
            },
        )

    if args.user_count < 1:
        raise ValueError("--user-count must be >= 1")
    if len(user_records) < args.user_count:
        raise ValueError(
            f"Requested {args.user_count} users but only {len(user_records)} available."
        )
    user_records = user_records[: args.user_count]

    results: List[RunResult] = []
    user_entries = list(enumerate(user_records, start=1))
    if args.user_index is not None:
        if args.user_index < 1 or args.user_index > len(user_entries):
            raise ValueError(
                f"--user-index must be between 1 and {len(user_entries)}"
            )
        user_entries = [user_entries[args.user_index - 1]]
    for idx, record in user_entries:
        user_id = record["user_id"]
        user_description = record["user_description"]
        user_slug = _slugify(user_id) or f"user_{idx:03d}"
        user_output_dir = output_dir / f"{idx:03d}_{user_slug}"

        config = RunConfig(
            output_dir=user_output_dir,
            model_name=args.model,
            provider=args.provider,
            base_url=args.base_url,
            api_key=args.api_key,
            debug_mode=args.debug,
            skip_stage1=args.skip_stage1,
            skip_stage2=args.skip_stage2,
            skip_stage3=args.skip_stage3,
            dry_run_world_bg=args.dry_run_world_bg,
            world_bg_mode=args.world_bg_mode,
            cutoff_date=args.cutoff_date,
            user_id=user_id,
        )

        runner = BatchGenerationRunner(config)
        result = runner.run(
            user_description=user_description,
        )
        results.append(result)

        runner.logger.info(f"Pipeline completed successfully for {user_id}!")
        runner.logger.info(f"Output directory: {result.output_dir}")
        runner.logger.info(f"Total app logs: {len(result.app_logs)}")


if __name__ == "__main__":
    main()
