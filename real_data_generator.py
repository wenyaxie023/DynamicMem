import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

from jinja2 import Template

from mem_bench.behavior_and_conversation.llm_client import (
    GeminiJSONClient,
    LLMResult,
)


data_realization_template = Template("""# Task: Generate Realistic Multi-Source Evidence Data

You are creating a synthetic digital footprint for a real person based on an evidence chain specification. This data will be used to train AI systems to understand how different data sources combine to reveal user states.

## Context Layers

### 1. User Identity
{{user_basic_profile}}

**Key attributes to maintain consistency:**
- Communication style (formal/casual)
- Technical literacy level
- Financial capacity
- Time availability

### 2. Spatiotemporal Constraints
{{life_context}}

### 3. Domain History (This provides behavioral continuity)
{{previous_window_summary_this_domain}}

**What this tells you:**
- Existing habits before this window
- Baseline preferences
- Prior devices/tools already in use

### 4. Current Window Motivation (Optional)
{{current_window_description}}

**Why this window is different:**
- Life changes triggering new behaviors
- Seasonal factors
- External events

---

## Evidence Chain Specification
```json
{{evidence_chain_json}}
```

**Evidence Story Summary:**
{{evidence_story}}

**Why Multi-Source?**
{{multi_source_dependency_rationale}}

---

## Generation Instructions

### Phase 1: Understand the State Change

First, identify what's changing:
- Review `proving_state_items` to see what states are being proven
- Note the `operation` type: stable/add/acquire/amplify/reduce/drop
- Understand the `change_reason` if provided

### Phase 2: Generate Data by Evidence Type

#### For App Logs (`data_source: "app_log"`):

**Structure Template:**
```json
{
  "log_id": "<evidence_id>_entry_<n>",
  "timestamp": "ISO 8601 format",
  "source_app": "<app name>",
  "event_type": "action_verb",
  "metadata": {
    "device_id": "realistic device ID",
    "os_version": "iOS 17.2 or Android 14",
    "app_version": "semantic version",
    "session_id": "UUID",
    "ip_address": "IP matching user location"
  },
  "data": {
    // Event-specific payload
  }
}
```

**Behavior Type Guidelines:**

- **transactional**: Include order ID, payment method, shipping address, product SKU, price
- **usage_tracking**: Include session duration, screens viewed, user actions, background sync status
- **self_reporting**: Include questionnaire responses, manual entries, user-submitted ratings

**Frequency Handling:**
- If `time_specification.type == "time_range"`:
  - `frequency: "daily"` → Generate one entry per day in range
  - `frequency: "weekly"` → Generate entries matching the frequency pattern (e.g., "3_times_per_week" = Mon/Wed/Fri)
  - Add realistic time jitter (±5-15 minutes from expected time)

#### For Dialogues (`data_source: "dialogue"`):

**Structure Template:**
```json
{
  "dialogue_id": "<evidence_id_placeholder>",
  "timestamp": "ISO 8601 start time",
  "conversation_type": "<behavior_type_placeholder>",
  "participants": ["user", "other_party"],
  "context": {
    "location": "where conversation happened",
    "medium": "in-person/text/voice/video",
    "platform": "iMessage/WhatsApp/face-to-face"
  },
  "messages": [
    {
      "speaker": "user or other",
      "content": "natural dialogue text",
      "timestamp": "ISO 8601",
      "metadata": {
        "read_status": true/false,
        "edited": false
      }
    }
  ]
}
```

**Behavior Type Guidelines:**

- **information_seeking**: User asks questions, explores options, compares products
- **social_interaction**: Casual conversation revealing preferences/habits
- **decision_making**: User explains choices, justifies decisions
- **feedback_expression**: User shares opinions, satisfaction, complaints
- **self_reporting**: User describes their own behavior or state

**Dialogue Realism Tips:**
- Use contractions (I'm, don't, can't)
- Include natural discourse markers (um, well, actually, honestly)
- Add typos occasionally if text-based
- Match vocabulary to user's education/profession
- Include emoji if user profile suggests casual communication style
- Multi-turn conversations should have realistic back-and-forth

### Phase 3: Cross-Validate Consistency

Before finalizing, check:

1. **Temporal Logic**: Events in `evidence_sequence` should occur in the specified order
2. **Detail Alignment**: 
   - Product name in dialogue = product name in transaction log
   - Quantities match
   - Prices are realistic for location/time
   - Device names are consistent across all mentions
3. **Behavioral Coherence**: 
   - Time-of-day patterns make sense (gym at 6pm for someone working 9-5)
   - Frequency claims in dialogue match actual log frequencies
   - Location data aligns with user's known addresses

---

## Output Format

Return a complete JSON object:
```json
{
  "evidence_chain_id": "string",
  "window_id": "string",
  "proving_states": [
    // Copy of proving_state_items for reference
  ],
  "generated_data": [
    {
      "evidence_id": "string",
      "behavior_type": "string",
      "data_source": "app_log or dialogue",
      "time_specification": {
        // Copy from source
      },
      "generated_count": "number of records generated",
      "content": {
        // Single record or array of records
      }
    }
  ],
  "generation_notes": {
    "consistency_checks_performed": ["list of validations"],
    "assumptions_made": ["any assumptions about missing details"]
  }
}
```

---

## Example Guidance (Based on Your Sample)

For `w1_health_whoop_philosophy`:

1. **Evidence w1_whoop_001** (dialogue, 2024-01-12):
   - Generate a conversation where user asks AI about CES 2024 wearables
   - Mention specific interest in "nervous system recovery" vs "steps"
   - Reference Whoop's AI Coach feature
   - Should feel like someone doing pre-purchase research

2. **Evidence w1_whoop_002** (app_log, 2024-01-15):
   - Generate Whoop.com purchase confirmation
   - Product: "Whoop 4.0 Onyx"
   - Plan: "12-month membership"
   - Price should match actual Whoop pricing (~$239 for annual)
   - Include realistic order number, payment method

3. **Evidence w1_whoop_003** (app_log, daily from 2024-01-18 to 2024-03-31):
   - Generate ~73 log entries (one per day)
   - Each should show:
     - Background sync (every 4-6 hours)
     - Active app open between 07:15-07:30 AM
     - User viewing "Recovery" and "Sleep" dashboards
   - Vary session durations (2-5 minutes realistic)

**The Story Arc**: Research → Purchase → Daily Use
The multi-source dependency: Without dialogue, we don't know WHY they bought it (algorithmic health optimization). Without logs, we don't know they actually USE it daily.

---

Generate the data now, maintaining high realism and internal consistency.
"""
)


@dataclass
class RealDataRequest:
    user_basic_profile: str | Dict[str, object]
    life_context: str | Dict[str, object]
    previous_window_summary_this_domain: str
    current_window_description: str
    evidence_chain: Dict[str, Any]
    window_id: str | None = None
    window_time_range: List[str] | None = None
    domain_name: str | None = None


def render_real_data_prompt(request: RealDataRequest) -> str:
    user_basic_profile = (
        request.user_basic_profile
        if isinstance(request.user_basic_profile, str)
        else json.dumps(request.user_basic_profile, indent=2, ensure_ascii=False)
    )
    life_context = (
        request.life_context
        if isinstance(request.life_context, str)
        else json.dumps(request.life_context, indent=2, ensure_ascii=False)
    )
    evidence_payload = deepcopy(request.evidence_chain or {})
    if request.window_id and "window_id" not in evidence_payload:
        evidence_payload["window_id"] = request.window_id
    if request.window_time_range and "time_range" not in evidence_payload:
        evidence_payload["time_range"] = request.window_time_range
    if request.domain_name and "domain_name" not in evidence_payload:
        evidence_payload["domain_name"] = request.domain_name

    evidence_story = evidence_payload.get("evidence_story") or ""
    multi_source_dependency_rationale = (
        evidence_payload.get("multi_source_dependency_rationale") or ""
    )

    return data_realization_template.render(
        user_basic_profile=user_basic_profile,
        life_context=life_context,
        previous_window_summary_this_domain=request.previous_window_summary_this_domain,
        current_window_description=request.current_window_description,
        evidence_chain_json=json.dumps(
            evidence_payload, indent=2, ensure_ascii=False
        ),
        evidence_story=evidence_story,
        multi_source_dependency_rationale=multi_source_dependency_rationale,
    )


def generate_real_data(
    llm_client: GeminiJSONClient, request: RealDataRequest
) -> LLMResult:
    prompt = render_real_data_prompt(request)
    return llm_client.generate_json(prompt)
