"""
Stage 1: Dynamic Profile Generation

Responsibilities:
1.1 Generate raw dynamic profile per domain
1.2 In-domain conflict resolution (rule 1-5 cascade)
1.3 Cross-domain key alignment
1.4 Cross-domain conflict resolution (temporal + attribute)

This stage produces the complete dynamic user profile across all domains,
with all conflicts resolved and ready for events chain generation.
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import GeminiJSONClient, LLMResult
from dynamic_profile_generator import (
    DynamicProfileRequest,
    generate_dynamic_profile,
    WorldBackgroundRequest,
    generate_world_background,
)
from prompt_templates import (
    ATTRIBUTE_CONFLICT_RESOLUTION_PROMPT,
    BASIC_PROFILE_PROMPT,
    DYNAMIC_PROFILE_TEMPLATE_EXCERPT,
    KEY_ALIGNMENT_PROMPT,
    TIME_CONFLICT_RESOLUTION_PROMPT,
    rule1_required_fields_prompt,
    rule2_prior_existence_prompt,
    rule3_essential_initialization_prompt,
    rule4_short_term_followup_prompt,
    rule5_time_conflict_prompt,
)
from stages.stage1_utils import (
    _apply_conflict_resolution_to_profiles,
    _apply_key_alignment,
    _apply_profile_revision,
    _collect_cross_domain_attribute_conflicts,
    _detect_rule1_required_field_issues,
    _detect_rule2_prior_existence_issues,
    _detect_rule3_first_add_issues,
    _detect_rule4_short_term_issues,
    _detect_rule5_time_conflict_issues,
    _normalize_window_id_label,
    detect_temporal_conflicts,
)


def _slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "domain"


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict) -> None:
    """Write JSON to file with formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_text(path: Path, text: str) -> None:
    """Write text to file."""
    path.write_text(text)


@dataclass
class Domain:
    """Domain specification for dynamic profile generation."""
    domain_name: str
    domain_scope_definition: str
    start_date: str | None = None
    end_date: str | None = None
    num_windows: int | None = None


@dataclass
class TimelineConfig:
    """Configuration for time windows in the simulation."""
    start_date: str = "2023-10-01"
    end_date: str = "2024-12-31"
    num_windows: int = 5
    time_windows: List[Dict[str, object]] | None = None

    def __post_init__(self):
        if self.time_windows is None:
            self.time_windows = [
                {"window_id": "w0", "time_range": ["2023-10-01", "2023-12-31"]},
                {"window_id": "w1", "time_range": ["2024-01-01", "2024-03-31"]},
                {"window_id": "w2", "time_range": ["2024-04-01", "2024-07-01"]},
                {"window_id": "w3", "time_range": ["2024-07-02", "2024-09-30"]},
                {"window_id": "w4", "time_range": ["2024-10-01", "2024-12-31"]},
            ]


@dataclass
class BasicProfileRequest:
    """Request for basic profile generation."""
    user_description: str


@dataclass
class Stage1Result:
    """Result from Stage 1: Dynamic Profile Generation."""
    user_basic_profile: Dict[str, Any]
    user_profile_text: str
    dynamic_profiles: Dict[str, Dict]
    world_backgrounds: Dict[str, Dict[str, Any]]  # domain_name -> world background data
    usage: Dict[str, Any]
    conflict_resolution_payload: Dict[str, Any]
    conflicts_summary: List[Dict[str, Any]]


class DynamicProfileStage:
    """
    Stage 1: Dynamic Profile Generation

    This stage handles:
    1. User basic profile generation from description
    2. Per-domain raw dynamic profile generation
    3. In-domain rule cascade fixes (rules 1-5)
    4. Cross-domain key alignment
    5. Cross-domain conflict resolution (attribute + temporal)
    """

    def __init__(
        self,
        llm_client: GeminiJSONClient,
        output_dir: Path,
        timeline: TimelineConfig | None = None,
        debug_mode: bool = True,
        dry_run_world_bg: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.timeline = timeline or TimelineConfig()
        self.debug_mode = debug_mode
        self.dry_run_world_bg = dry_run_world_bg
        self.logger = logger or logging.getLogger(__name__)
        _ensure_dir(self.output_dir)

        # Create debug subdirectory if needed
        if self.debug_mode:
            self.debug_dir = self.output_dir / "debug" / "stage1_dynamic_profile"
            _ensure_dir(self.debug_dir)

    def run(
        self,
        user_description: str,
        domains: Sequence[Domain],
    ) -> Stage1Result:
        """
        Run the complete Stage 1 pipeline.

        Args:
            user_description: Raw user description text
            domains: List of domains to generate profiles for

        Returns:
            Stage1Result containing all generated data and metadata
        """
        aggregate_usage: Dict[str, Any] = {}

        # Step 1: Generate basic profile
        self.logger.info("=== Step 1.1: Generating basic profile ===")
        user_basic_profile, basic_usage = self._generate_basic_profile(user_description)
        aggregate_usage["basic_profile"] = basic_usage
        user_profile_text = json.dumps(user_basic_profile, indent=2, ensure_ascii=False)

        # Step 1.5: Generate world backgrounds for each domain
        self.logger.info("=== Step 1.5: Generating world backgrounds per domain ===")
        world_backgrounds, world_bg_usage = self._generate_world_backgrounds(
            user_basic_profile=user_basic_profile,
            domains=domains,
        )
        aggregate_usage["world_backgrounds"] = world_bg_usage

        # Early exit if dry_run_world_bg mode - only save prompts, don't generate
        if self.dry_run_world_bg:
            self.logger.info("=== Dry run mode: World background prompts saved, stopping here ===")
            return Stage1Result(
                user_basic_profile=user_basic_profile,
                user_profile_text=user_profile_text,
                dynamic_profiles={},
                world_backgrounds=world_backgrounds,
                usage=aggregate_usage,
                conflict_resolution_payload={},
                conflicts_summary=[],
            )

        # Step 2: Generate raw dynamic profiles per domain
        self.logger.info("=== Step 1.2: Generating raw dynamic profiles per domain ===")
        raw_profiles, raw_usage = self._generate_raw_profiles(
            domains=domains,
            user_profile_text=user_profile_text,
            world_backgrounds=world_backgrounds,
        )
        aggregate_usage["raw_profiles"] = raw_usage

        # Step 3: Apply in-domain rule cascade fixes
        self.logger.info("=== Step 1.3: Applying in-domain rule cascade fixes ===")
        domain_fixed_profiles, fix_usage = self._apply_domain_level_fixes(
            raw_profiles=raw_profiles,
            domains=domains,
            user_profile_text=user_profile_text,
            world_backgrounds=world_backgrounds,
        )
        aggregate_usage["domain_fixes"] = fix_usage

        # Step 4: Cross-domain key alignment
        self.logger.info("=== Step 1.4: Cross-domain key alignment ===")
        aligned_profiles, alignment_usage = self._align_keys(
            dynamic_profiles=domain_fixed_profiles,
            user_basic_profile=user_basic_profile,
        )
        aggregate_usage["key_alignment"] = alignment_usage

        # Step 5: Cross-domain conflict resolution
        self.logger.info("=== Step 1.5: Cross-domain conflict resolution ===")
        (
            resolved_profiles,
            conflict_usage,
            conflict_resolution_payload,
            conflicts_summary,
        ) = self._resolve_cross_domain_conflicts(
            dynamic_profiles=aligned_profiles,
            domains=domains,
            user_basic_profile=user_basic_profile,
        )
        aggregate_usage["conflict_resolution"] = conflict_usage

        # Save final output
        _write_json(
            self.output_dir / "dynamic_profiles_final.json",
            resolved_profiles,
        )

        # Save world backgrounds
        _write_json(
            self.output_dir / "world_backgrounds.json",
            world_backgrounds,
        )

        return Stage1Result(
            user_basic_profile=user_basic_profile,
            user_profile_text=user_profile_text,
            dynamic_profiles=resolved_profiles,
            world_backgrounds=world_backgrounds,
            usage=aggregate_usage,
            conflict_resolution_payload=conflict_resolution_payload,
            conflicts_summary=conflicts_summary,
        )

    # =========================================================================
    # Step 1.1: Basic Profile Generation
    # =========================================================================

    def _generate_basic_profile(
        self,
        user_description: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate structured basic profile from user description."""

        # Check cache
        cache_path = self.output_dir / "user_basic_profile.json"
        if cache_path.exists():
            self.logger.info("Loading cached basic profile...")
            return json.loads(cache_path.read_text()), {}

        prompt = BASIC_PROFILE_PROMPT.render(user_description=user_description)
        result = self.llm_client.generate_json(prompt)

        # Save outputs
        _write_json(cache_path, result.data)

        if self.debug_mode:
            _write_text(
                self.debug_dir / "basic_profile_prompt.txt",
                result.prompt,
            )
            _write_text(
                self.debug_dir / "basic_profile_raw_response.txt",
                result.raw_text,
            )

        return result.data, result.usage

    # =========================================================================
    # Step 1.5: World Background Generation (per domain)
    # =========================================================================

    def _generate_world_backgrounds(
        self,
        user_basic_profile: Dict[str, Any],
        domains: Sequence[Domain],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Generate personalized world backgrounds for each domain.

        Each domain gets a tailored world background that considers:
        - The user's location and cultural context
        - Domain-specific relevant events and trends

        Args:
            user_basic_profile: The user's basic profile
            domains: List of domains to generate backgrounds for

        Returns:
            Tuple of (world_backgrounds dict, usage dict)
        """
        world_backgrounds: Dict[str, Dict[str, Any]] = {}
        aggregate_usage: Dict[str, Any] = {}

        # Import for rendering prompt in dry_run mode
        from dynamic_profile_generator import render_world_background_prompt

        for domain in domains:
            slug = _slugify(domain.domain_name)
            cache_path = self.output_dir / f"{slug}_world_background.json"

            if cache_path.exists():
                self.logger.info(f"Loading cached world background for {domain.domain_name}...")
                world_backgrounds[domain.domain_name] = json.loads(cache_path.read_text())
                continue

            # Build the request object
            request = WorldBackgroundRequest(
                user_basic_profile=user_basic_profile,
                domain_name=domain.domain_name,
                domain_scope_definition=domain.domain_scope_definition,
            )

            # Dry run mode: only save prompt, don't call LLM
            if self.dry_run_world_bg:
                self.logger.info(f"[Dry run] Saving world background prompt for {domain.domain_name}...")
                prompt = render_world_background_prompt(request)
                prompt_path = self.output_dir / f"{slug}_world_background_prompt.txt"
                _write_text(prompt_path, prompt)
                world_backgrounds[domain.domain_name] = {"dry_run": True, "prompt_saved": str(prompt_path)}
                continue

            self.logger.info(f"Generating world background for {domain.domain_name}...")

            result = generate_world_background(self.llm_client, request)

            world_backgrounds[domain.domain_name] = {
                "world_backgrounds": result.world_backgrounds,
                "combined_background": result.combined_background,
            }
            aggregate_usage[domain.domain_name] = result.usage

            # Save outputs
            _write_json(cache_path, world_backgrounds[domain.domain_name])

            if self.debug_mode:
                _write_text(
                    self.debug_dir / f"{slug}_world_background_prompt.txt",
                    result.prompt,
                )
                _write_text(
                    self.debug_dir / f"{slug}_world_background_raw_response.txt",
                    result.raw_text,
                )

        return world_backgrounds, aggregate_usage

    # =========================================================================
    # Step 1.2: Raw Dynamic Profile Generation (per domain)
    # =========================================================================

    def _generate_raw_profiles(
        self,
        domains: Sequence[Domain],
        user_profile_text: str,
        world_backgrounds: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """Generate raw dynamic profiles for each domain."""

        life_domain_list = [d.domain_name for d in domains]
        raw_profiles: Dict[str, Dict] = {}
        aggregate_usage: Dict[str, Any] = {}

        for domain in domains:
            slug = _slugify(domain.domain_name)
            cache_path = self.output_dir / f"{slug}_raw_dynamic_profile.json"

            if cache_path.exists():
                self.logger.info(f"Loading cached raw dynamic profile for {domain.domain_name}...")
                raw_profiles[domain.domain_name] = json.loads(cache_path.read_text())
                continue

            self.logger.info(f"Generating raw dynamic profile for {domain.domain_name}...")

            # Build time windows
            time_windows = self._build_time_windows_for_domain(domain)

            # Get the domain-specific world background
            domain_world_bg = world_backgrounds.get(domain.domain_name, {})
            world_background = domain_world_bg.get("combined_background", "")

            result = generate_dynamic_profile(
                self.llm_client,
                DynamicProfileRequest(
                    domain_name=domain.domain_name,
                    domain_scope_definition=domain.domain_scope_definition,
                    world_background=world_background,
                    user_profile=user_profile_text,
                    time_windows=time_windows,
                    life_domain_list=life_domain_list,
                ),
            )

            raw_profiles[domain.domain_name] = result.data
            aggregate_usage[domain.domain_name] = result.usage

            # Save outputs
            _write_json(cache_path, result.data)

            if self.debug_mode:
                _write_text(
                    self.debug_dir / f"{slug}_raw_profile_prompt.txt",
                    result.prompt,
                )

        # Save combined raw profiles
        _write_json(
            self.output_dir / "dynamic_profiles_raw.json",
            raw_profiles,
        )

        return raw_profiles, aggregate_usage

    def _build_time_windows_for_domain(self, domain: Domain) -> List[Dict]:
        """Build time windows for a specific domain."""
        time_windows = []

        if self.timeline.time_windows:
            for window in self.timeline.time_windows:
                window_id = str(window.get("window_id") or "").strip().lower()
                # Skip w0/initial for time_windows (it goes to initial_state)
                if window_id in {"w0", "initial", "initial_state", "init"}:
                    continue
                time_windows.append(window)

        return time_windows

    # =========================================================================
    # Step 1.3: In-Domain Rule Cascade Fixes
    # =========================================================================

    def _apply_domain_level_fixes(
        self,
        raw_profiles: Dict[str, Dict],
        domains: Sequence[Domain],
        user_profile_text: str,
        world_backgrounds: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """Apply rule cascade fixes to each domain profile."""

        # Check cache
        cache_path = self.output_dir / "dynamic_profiles_domain_level_fixes_applied.json"
        if cache_path.exists():
            self.logger.info("Loading cached domain-fixed profiles...")
            return json.loads(cache_path.read_text()), {}

        fixed_profiles: Dict[str, Dict] = {}
        aggregate_usage: Dict[str, Any] = {}
        domain_lookup = {d.domain_name: d for d in domains}

        for domain_name, profile in raw_profiles.items():
            domain = domain_lookup.get(domain_name)
            if domain is None:
                self.logger.warning(f"Domain {domain_name} not found, skipping fixes")
                fixed_profiles[domain_name] = profile
                continue

            slug = _slugify(domain_name)
            domain_cache_path = self.debug_dir / f"{slug}_domain_fixed_profile.json" if self.debug_mode else None
            if domain_cache_path and domain_cache_path.exists():
                self.logger.info(f"Loading cached domain-fixed profile for {domain_name}...")
                fixed_profiles[domain_name] = json.loads(domain_cache_path.read_text())
                continue

            self.logger.info(f"Applying rule fixes for {domain_name}...")

            domain_world_bg = world_backgrounds.get(domain.domain_name, {})
            world_background = domain_world_bg.get("combined_background", "")
            fixed_profile, domain_usage, domain_fixes = self._apply_rule_cascade(
                domain=domain,
                profile=profile,
                user_profile_text=user_profile_text,
                world_background=world_background,
            )

            fixed_profiles[domain_name] = fixed_profile
            aggregate_usage[domain_name] = domain_usage

            if domain_cache_path:
                _write_json(domain_cache_path, fixed_profile)

            if self.debug_mode and domain_fixes:
                _write_json(
                    self.debug_dir / f"{slug}_rule_fixes.json",
                    domain_fixes,
                )

        # Save combined fixed profiles
        _write_json(cache_path, fixed_profiles)

        return fixed_profiles, aggregate_usage

    def _apply_rule_cascade(
        self,
        domain: Domain,
        profile: Dict,
        user_profile_text: str,
        world_background: str,
    ) -> Tuple[Dict, Dict[str, Any], Dict[str, Any]]:
        """
        Apply rule cascade fixes (rules 1-5) to a single domain profile.

        Cascade order:
        - Rule 1: Required fields (structural integrity)
        - Rule 2: Prior existence (operation legality)
        - Rule 3: Essential initialization (semantic reasonableness)
        - Rule 4: Short-term followups (semantic consistency)
        - Rule 5: Time conflicts (final feasibility)
        """
        revised_profile = deepcopy(profile)
        all_fixes: Dict[str, Any] = {}
        aggregate_usage: Dict[str, Any] = {}
        slug = _slugify(domain.domain_name)

        # Rule 1: Required fields
        rule1_cached = self.debug_dir / f"{slug}_rule1_fixed_profile.json" if self.debug_mode else None
        if rule1_cached and rule1_cached.exists():
            self.logger.info(f"  Rule 1: Loading cached fixed profile for {domain.domain_name}")
            revised_profile = json.loads(rule1_cached.read_text())
        else:
            rule1_issues = self._detect_rule1_issues(revised_profile)
            if rule1_issues:
                self.logger.info(f"  Rule 1: Found {len(rule1_issues)} required field issues")
                fix_result, usage = self._fix_rule1(
                    domain, user_profile_text, revised_profile, rule1_issues
                )
                if fix_result:
                    all_fixes["rule1"] = fix_result
                    revised_profile = self._apply_fix(revised_profile, fix_result)
                    aggregate_usage["rule1"] = usage

                    if self.debug_mode:
                        _write_json(
                            self.debug_dir / f"{slug}_rule1_fix.json",
                            fix_result,
                        )
                        _write_json(
                            self.debug_dir / f"{slug}_rule1_fixed_profile.json",
                            revised_profile,
                        )

        # Rule 2: Prior existence
        rule2_cached = self.debug_dir / f"{slug}_rule2_fixed_profile.json" if self.debug_mode else None
        if rule2_cached and rule2_cached.exists():
            self.logger.info(f"  Rule 2: Loading cached fixed profile for {domain.domain_name}")
            revised_profile = json.loads(rule2_cached.read_text())
        else:
            rule2_issues = self._detect_rule2_issues(revised_profile)
            if rule2_issues:
                self.logger.info(f"  Rule 2: Found {len(rule2_issues)} prior existence issues")
                fix_result, usage = self._fix_rule2(
                    domain, user_profile_text, revised_profile, rule2_issues
                )
                if fix_result:
                    all_fixes["rule2"] = fix_result
                    revised_profile = self._apply_fix(revised_profile, fix_result)
                    aggregate_usage["rule2"] = usage

                    if self.debug_mode:
                        _write_json(
                            self.debug_dir / f"{slug}_rule2_fix.json",
                            fix_result,
                        )
                        _write_json(
                            self.debug_dir / f"{slug}_rule2_fixed_profile.json",
                            revised_profile,
                        )

        # Rule 3: Essential initialization
        rule3_cached = self.debug_dir / f"{slug}_rule3_fixed_profile.json" if self.debug_mode else None
        if rule3_cached and rule3_cached.exists():
            self.logger.info(f"  Rule 3: Loading cached fixed profile for {domain.domain_name}")
            revised_profile = json.loads(rule3_cached.read_text())
        else:
            rule3_issues = self._detect_rule3_issues(revised_profile)
            if rule3_issues:
                self.logger.info(f"  Rule 3: Found {len(rule3_issues)} essential initialization issues")
                fix_result, usage = self._fix_rule3(
                    domain, user_profile_text, revised_profile, rule3_issues
                )
                if fix_result:
                    all_fixes["rule3"] = fix_result
                    revised_profile = self._apply_fix(revised_profile, fix_result)
                    aggregate_usage["rule3"] = usage

                    if self.debug_mode:
                        _write_json(
                            self.debug_dir / f"{slug}_rule3_fix.json",
                            fix_result,
                        )
                        _write_json(
                            self.debug_dir / f"{slug}_rule3_fixed_profile.json",
                            revised_profile,
                        )

        # Rule 4: Short-term followups
        rule4_cached = self.debug_dir / f"{slug}_rule4_fixed_profile.json" if self.debug_mode else None
        if rule4_cached and rule4_cached.exists():
            self.logger.info(f"  Rule 4: Loading cached fixed profile for {domain.domain_name}")
            revised_profile = json.loads(rule4_cached.read_text())
        else:
            rule4_issues = self._detect_rule4_issues(revised_profile)
            if rule4_issues:
                self.logger.info(f"  Rule 4: Found {len(rule4_issues)} short-term followup issues")
                fix_result, usage = self._fix_rule4(
                    domain, user_profile_text, revised_profile, rule4_issues, world_background
                )
                if fix_result:
                    all_fixes["rule4"] = fix_result
                    revised_profile = self._apply_fix(revised_profile, fix_result)
                    aggregate_usage["rule4"] = usage

                    if self.debug_mode:
                        _write_json(
                            self.debug_dir / f"{slug}_rule4_fix.json",
                            fix_result,
                        )
                        _write_json(
                            self.debug_dir / f"{slug}_rule4_fixed_profile.json",
                            revised_profile,
                        )

        # Rule 5: Time conflicts
        rule5_cached = self.debug_dir / f"{slug}_rule5_fixed_profile.json" if self.debug_mode else None
        if rule5_cached and rule5_cached.exists():
            self.logger.info(f"  Rule 5: Loading cached fixed profile for {domain.domain_name}")
            revised_profile = json.loads(rule5_cached.read_text())
        else:
            rule5_issues = self._detect_rule5_issues(revised_profile)
            if rule5_issues and rule5_issues.get("conflicts"):
                self.logger.info(f"  Rule 5: Found {len(rule5_issues.get('conflicts', []))} time conflict issues")
                fix_result, usage = self._fix_rule5(
                    domain, user_profile_text, revised_profile, rule5_issues
                )
                if fix_result:
                    all_fixes["rule5"] = fix_result
                    revised_profile = self._apply_fix(revised_profile, fix_result)
                    aggregate_usage["rule5"] = usage

                    if self.debug_mode:
                        _write_json(
                            self.debug_dir / f"{slug}_rule5_fix.json",
                            fix_result,
                        )
                        _write_json(
                            self.debug_dir / f"{slug}_rule5_fixed_profile.json",
                            revised_profile,
                        )

        return revised_profile, aggregate_usage, all_fixes

    # -------------------------------------------------------------------------
    # Rule Detection Methods
    # -------------------------------------------------------------------------

    def _detect_rule1_issues(self, profile: Dict) -> List[Dict]:
        """Detect Rule 1: Required fields issues."""
        return _detect_rule1_required_field_issues(profile)

    def _detect_rule2_issues(self, profile: Dict) -> List[Dict]:
        """Detect Rule 2: Prior existence issues."""
        return _detect_rule2_prior_existence_issues(profile)

    def _detect_rule3_issues(self, profile: Dict) -> List[Dict]:
        """Detect Rule 3: Essential initialization issues."""
        return _detect_rule3_first_add_issues(profile)

    def _detect_rule4_issues(self, profile: Dict) -> List[Dict]:
        """Detect Rule 4: Short-term followup issues."""
        return _detect_rule4_short_term_issues(profile)

    def _detect_rule5_issues(self, profile: Dict) -> Dict:
        """Detect Rule 5: Time conflict issues."""
        return _detect_rule5_time_conflict_issues(profile)

    # -------------------------------------------------------------------------
    # Rule Fix Methods
    # -------------------------------------------------------------------------

    def _fix_rule1(
        self,
        domain: Domain,
        user_profile: str,
        profile: Dict,
        issues: List[Dict],
    ) -> Tuple[Dict | None, Dict]:
        """Fix Rule 1 violations."""
        if not issues:
            return None, {}

        prompt = rule1_required_fields_prompt.render(
            domain_name=domain.domain_name,
            domain_scope_definition=domain.domain_scope_definition,
            user_profile=user_profile,
            schema_excerpt=DYNAMIC_PROFILE_TEMPLATE_EXCERPT,
            detected_issues=json.dumps(issues, indent=2, ensure_ascii=False),
            dynamic_profile_json=json.dumps(profile, indent=2, ensure_ascii=False),
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            slug = _slugify(domain.domain_name)
            _write_text(
                self.debug_dir / f"{slug}_rule1_prompt.txt",
                prompt,
            )

        return result.data, result.usage

    def _fix_rule2(
        self,
        domain: Domain,
        user_profile: str,
        profile: Dict,
        issues: List[Dict],
    ) -> Tuple[Dict | None, Dict]:
        """Fix Rule 2 violations."""
        if not issues:
            return None, {}

        prompt = rule2_prior_existence_prompt.render(
            domain_name=domain.domain_name,
            domain_scope_definition=domain.domain_scope_definition,
            user_profile=user_profile,
            detected_issues=json.dumps(issues, indent=2, ensure_ascii=False),
            dynamic_profile_json=json.dumps(profile, indent=2, ensure_ascii=False),
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            slug = _slugify(domain.domain_name)
            _write_text(
                self.debug_dir / f"{slug}_rule2_prompt.txt",
                prompt,
            )

        return result.data, result.usage

    def _fix_rule3(
        self,
        domain: Domain,
        user_profile: str,
        profile: Dict,
        issues: List[Dict],
    ) -> Tuple[Dict | None, Dict]:
        """Fix Rule 3 violations."""
        if not issues:
            return None, {}

        prompt = rule3_essential_initialization_prompt.render(
            domain_name=domain.domain_name,
            domain_scope_definition=domain.domain_scope_definition,
            user_profile=user_profile,
            detected_issues=json.dumps(issues, indent=2, ensure_ascii=False),
            dynamic_profile_json=json.dumps(profile, indent=2, ensure_ascii=False),
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            slug = _slugify(domain.domain_name)
            _write_text(
                self.debug_dir / f"{slug}_rule3_prompt.txt",
                prompt,
            )

        return result.data, result.usage

    def _fix_rule4(
        self,
        domain: Domain,
        user_profile: str,
        profile: Dict,
        issues: List[Dict],
        world_background: str,
    ) -> Tuple[Dict | None, Dict]:
        """Fix Rule 4 violations."""
        if not issues:
            return None, {}

        prompt = rule4_short_term_followup_prompt.render(
            domain_name=domain.domain_name,
            domain_scope_definition=domain.domain_scope_definition,
            world_background=world_background,
            user_profile=user_profile,
            detected_issues=json.dumps(issues, indent=2, ensure_ascii=False),
            dynamic_profile_json=json.dumps(profile, indent=2, ensure_ascii=False),
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            slug = _slugify(domain.domain_name)
            _write_text(
                self.debug_dir / f"{slug}_rule4_prompt.txt",
                prompt,
            )

        return result.data, result.usage

    def _fix_rule5(
        self,
        domain: Domain,
        user_profile: str,
        profile: Dict,
        issues: Dict,
    ) -> Tuple[Dict | None, Dict]:
        """Fix Rule 5 violations."""
        conflicts_list = issues.get("conflicts") or []
        if not conflicts_list:
            return None, {}

        prompt = rule5_time_conflict_prompt.render(
            domain_name=domain.domain_name,
            domain_scope_definition=domain.domain_scope_definition,
            user_profile=user_profile,
            detected_issues=json.dumps(conflicts_list, indent=2, ensure_ascii=False),
            dynamic_profile_json=json.dumps(profile, indent=2, ensure_ascii=False),
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            slug = _slugify(domain.domain_name)
            _write_text(
                self.debug_dir / f"{slug}_rule5_prompt.txt",
                prompt,
            )

        return result.data, result.usage

    def _apply_fix(self, profile: Dict, fix_result: Dict) -> Dict:
        """Apply a fix result to a profile."""
        return _apply_profile_revision(profile, fix_result)

    # =========================================================================
    # Step 1.4: Cross-Domain Key Alignment
    # =========================================================================

    def _align_keys(
        self,
        dynamic_profiles: Dict[str, Dict],
        user_basic_profile: Dict,
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """Align attribute keys across domains."""

        # Check cache
        cache_path = self.output_dir / "dynamic_profiles_key_aligned.json"
        if cache_path.exists():
            self.logger.info("Loading cached key-aligned profiles...")
            return json.loads(cache_path.read_text()), {}

        # Generate key alignment
        prompt = KEY_ALIGNMENT_PROMPT.render(
            dynamic_profiles_json=json.dumps(dynamic_profiles, indent=2, ensure_ascii=False),
            user_basic_profile_json=json.dumps(user_basic_profile, indent=2, ensure_ascii=False),
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            _write_text(
                self.debug_dir / "key_alignment_prompt.txt",
                prompt,
            )
            _write_json(
                self.debug_dir / "key_alignment_result.json",
                result.data,
            )

        # Apply key alignment
        aligned_profiles, _, _ = _apply_key_alignment(dynamic_profiles, result.data)

        # Save aligned profiles
        _write_json(cache_path, aligned_profiles)

        return aligned_profiles, result.usage

    # =========================================================================
    # Step 1.5: Cross-Domain Conflict Resolution
    # =========================================================================

    def _resolve_cross_domain_conflicts(
        self,
        dynamic_profiles: Dict[str, Dict],
        domains: Sequence[Domain],
        user_basic_profile: Dict,
    ) -> Tuple[Dict[str, Dict], Dict[str, Any], Dict[str, Any], List[Dict]]:
        """
        Resolve cross-domain conflicts:
        - Attribute conflicts (same attribute, different values across domains)
        - Temporal conflicts (habit schedule overlaps)
        """

        # Check cache
        cache_path = self.output_dir / "dynamic_profiles_conflict_resolved.json"
        if cache_path.exists():
            self.logger.info("Loading cached conflict-resolved profiles...")
            return json.loads(cache_path.read_text()), {}, {}, []

        resolved_profiles = deepcopy(dynamic_profiles)
        aggregate_usage: Dict[str, Any] = {}
        resolution_payload: Dict[str, Any] = {}
        all_conflicts_summary: List[Dict] = []

        # Step 5a: Attribute conflict resolution
        self.logger.info("Resolving attribute conflicts...")
        (
            resolved_profiles,
            attr_usage,
            attr_payload,
            attr_conflicts,
        ) = self._resolve_attribute_conflicts(
            resolved_profiles,
            user_basic_profile,
        )
        aggregate_usage["attribute"] = attr_usage
        resolution_payload["attribute"] = attr_payload
        all_conflicts_summary.extend(attr_conflicts)

        # Step 5b: Temporal conflict resolution (iterative)
        self.logger.info("Resolving temporal conflicts...")
        (
            resolved_profiles,
            temporal_usage,
            temporal_payload,
            temporal_conflicts,
        ) = self._resolve_temporal_conflicts(
            resolved_profiles,
            domains,
            user_basic_profile,
        )
        aggregate_usage["temporal"] = temporal_usage
        resolution_payload["temporal"] = temporal_payload
        all_conflicts_summary.extend(temporal_conflicts)

        # Save resolved profiles
        _write_json(cache_path, resolved_profiles)

        return resolved_profiles, aggregate_usage, resolution_payload, all_conflicts_summary

    def _resolve_attribute_conflicts(
        self,
        dynamic_profiles: Dict[str, Dict],
        user_basic_profile: Dict,
    ) -> Tuple[Dict[str, Dict], Dict, Dict, List[Dict]]:
        """Resolve attribute conflicts across domains."""

        # Detect attribute conflicts
        attribute_conflicts = _collect_cross_domain_attribute_conflicts(dynamic_profiles)

        if self.debug_mode:
            _write_json(
                self.debug_dir / "detected_attribute_conflicts.json",
                {"conflicts": attribute_conflicts},
            )

        if not attribute_conflicts:
            return dynamic_profiles, {}, {}, []

        # Generate resolution
        prompt = ATTRIBUTE_CONFLICT_RESOLUTION_PROMPT.render(
            user_basic_profile_json=json.dumps(user_basic_profile, indent=2, ensure_ascii=False),
            dynamic_profiles_json=json.dumps(dynamic_profiles, indent=2, ensure_ascii=False),
            detected_attribute_conflicts_json=json.dumps(attribute_conflicts, indent=2, ensure_ascii=False),
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            _write_text(
                self.debug_dir / "attribute_conflict_resolution_prompt.txt",
                prompt,
            )
            _write_json(
                self.debug_dir / "attribute_conflict_resolution_result.json",
                result.data,
            )

        # Apply resolution
        resolved = _apply_conflict_resolution_to_profiles(dynamic_profiles, result.data)

        return resolved, result.usage, result.data, attribute_conflicts

    def _resolve_temporal_conflicts(
        self,
        dynamic_profiles: Dict[str, Dict],
        domains: Sequence[Domain],
        user_basic_profile: Dict,
    ) -> Tuple[Dict[str, Dict], Dict, Dict, List[Dict]]:
        """
        Resolve temporal conflicts across domains.

        This is an iterative process that resolves conflicts window by window.
        """

        # Try to load from the latest iteration cache
        (
            resolved_profiles,
            start_iteration,
            start_window_label,
        ) = self._load_latest_temporal_iteration_cache(dynamic_profiles)
        if start_iteration > 0:
            resume_window = _normalize_window_id_label(start_window_label) if start_window_label else None
            self.logger.info(
                f"Resuming temporal conflict resolution from iteration {start_iteration}"
                + (f" (window {resume_window})" if resume_window else "")
            )

        aggregate_usage: Dict[str, Any] = {}
        all_payloads: Dict[str, Any] = {}
        all_conflicts: List[Dict] = []

        # Detect initial conflicts
        temporal_conflicts = detect_temporal_conflicts(resolved_profiles)

        conflict_entries = temporal_conflicts.get("conflicts", [])
        all_conflicts.extend(conflict_entries)

        if self.debug_mode:
            _write_json(
                self.debug_dir / "detected_temporal_conflicts_initial.json",
                temporal_conflicts,
            )

        if not conflict_entries:
            return resolved_profiles, {}, {}, []

        # Get window order
        window_order = self._get_window_resolution_order(resolved_profiles)

        iteration_counter = start_iteration

        resume_started = start_iteration == 0 or not start_window_label
        for window_label in window_order:
            if not window_label or window_label == "unknown":
                continue

            if not resume_started:
                if _normalize_window_id_label(window_label) != _normalize_window_id_label(start_window_label):
                    continue
                resume_started = True

            self.logger.info(f"  Processing conflicts in {window_label}...")

            for _ in range(100):
                # Re-detect conflicts
                window_conflicts = detect_temporal_conflicts(resolved_profiles)
                window_conflict_entries = [
                    c for c in window_conflicts.get("conflicts", [])
                    if _normalize_window_id_label(c.get("window_id")) == _normalize_window_id_label(window_label)
                ]

                if not window_conflict_entries:
                    self.logger.info(f"    All conflicts resolved in {window_label}")
                    break

                iteration_counter += 1

                # Generate resolution for this iteration
                (
                    resolved_profiles,
                    iter_usage,
                    iter_payload,
                ) = self._resolve_single_temporal_iteration(
                    resolved_profiles,
                    user_basic_profile,
                    window_label,
                    window_conflict_entries,
                    iteration_counter,
                )

                iter_key = f"iter_{iteration_counter}_{window_label}"
                aggregate_usage[iter_key] = iter_usage
                all_payloads[iter_key] = iter_payload

                if self.debug_mode:
                    _write_json(
                        self.debug_dir / f"temporal_resolution_{iter_key}.json",
                        {
                            "conflicts": window_conflict_entries,
                            "resolution": iter_payload,
                        },
                    )
                    _write_json(
                        self.debug_dir / f"profiles_after_{iter_key}.json",
                        resolved_profiles,
                    )

        return resolved_profiles, aggregate_usage, all_payloads, all_conflicts

    def _resolve_single_temporal_iteration(
        self,
        dynamic_profiles: Dict[str, Dict],
        user_basic_profile: Dict,
        window_label: str,
        conflicts: List[Dict],
        iteration: int,
    ) -> Tuple[Dict[str, Dict], Dict, Dict]:
        """Resolve a single iteration of temporal conflicts."""

        # Build conflict hints (simplified)
        conflict_hints = {
            "conflicts": conflicts,
            "window_id": window_label,
        }

        prompt = TIME_CONFLICT_RESOLUTION_PROMPT.render(
            user_basic_profile_json=json.dumps(user_basic_profile, indent=2, ensure_ascii=False),
            dynamic_profiles_json=json.dumps(dynamic_profiles, indent=2, ensure_ascii=False),
            conflict_json=json.dumps(conflict_hints, indent=2, ensure_ascii=False),
            target_window_id=window_label,
        )

        result = self.llm_client.generate_json(prompt)

        if self.debug_mode:
            _write_text(
                self.debug_dir / f"temporal_resolution_iter{iteration}_{window_label}_prompt.txt",
                prompt,
            )

        # Apply resolution
        resolved = _apply_conflict_resolution_to_profiles(dynamic_profiles, result.data)

        return resolved, result.usage, result.data

    def _get_window_resolution_order(self, profiles: Dict[str, Dict]) -> List[str]:
        """Get the order of windows for conflict resolution."""
        ordered: List[str] = []
        seen = set()

        def _add(window_id):
            norm = _normalize_window_id_label(window_id)
            if norm not in seen:
                seen.add(norm)
                ordered.append(norm)

        _add("initial")
        for profile in profiles.values():
            for window in profile.get("time_windows", []):
                _add(window.get("window_id"))

        return ordered

    def _load_latest_temporal_iteration_cache(
        self,
        fallback_profiles: Dict[str, Dict],
    ) -> Tuple[Dict[str, Dict], int, Optional[str]]:
        """
        Load the latest temporal iteration cache if available.

        Scans debug directory for files matching 'profiles_after_iter_*.json'
        and returns the one with the highest iteration number.

        Args:
            fallback_profiles: Profiles to return if no cache is found

        Returns:
            Tuple of (profiles dict, iteration number). If no cache found,
            returns (fallback_profiles, 0, None).
        """
        if not self.debug_mode or not hasattr(self, 'debug_dir'):
            return deepcopy(fallback_profiles), 0, None

        # Find all iteration cache files
        pattern = "profiles_after_iter_*.json"
        cache_files = list(self.debug_dir.glob(pattern))

        if not cache_files:
            return deepcopy(fallback_profiles), 0, None

        # Parse iteration numbers from filenames
        # Format: profiles_after_iter_{iteration}_{window_label}.json
        max_iter = 0
        latest_file: Path | None = None
        latest_window_label: Optional[str] = None

        for cache_file in cache_files:
            # Extract iteration number from filename
            filename = cache_file.stem  # e.g., "profiles_after_iter_5_w1"
            parts = filename.split("_")
            # Find the part after "iter"
            try:
                iter_idx = parts.index("iter")
                if iter_idx + 1 < len(parts):
                    iter_num = int(parts[iter_idx + 1])
                    window_label = "_".join(parts[iter_idx + 2:]) if iter_idx + 2 < len(parts) else None
                    if iter_num > max_iter:
                        max_iter = iter_num
                        latest_file = cache_file
                        latest_window_label = window_label
            except (ValueError, IndexError):
                continue

        if latest_file is None or max_iter == 0:
            return deepcopy(fallback_profiles), 0, None

        try:
            self.logger.info(f"Loading temporal iteration cache from {latest_file.name}")
            cached_profiles = json.loads(latest_file.read_text())
            return cached_profiles, max_iter, latest_window_label
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to load cache {latest_file}: {e}")
            return deepcopy(fallback_profiles), 0, None
