from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from collections import Counter
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
from jinja2 import Template
import dotenv
dotenv.load_dotenv()
import os

from mem_bench.behavior_and_conversation.atomic_events_generator import (
    AtomicEventsRequest,
    generate_atomic_events,
)
from mem_bench.behavior_and_conversation.dynamic_profile_generator import (
    DynamicProfileRequest,
    generate_dynamic_profile,
)
from mem_bench.behavior_and_conversation.llm_client import GeminiJSONClient, LLMResult
from mem_bench.behavior_and_conversation.real_data_generator import (
    RealDataRequest,
    generate_real_data,
)
from mem_bench.behavior_and_conversation.elite_persona_sampler import (
    DEFAULT_SAMPLE_PATH as DEFAULT_ELITE_SAMPLE_PATH,
    DEFAULT_SAMPLE_SIZE as DEFAULT_ELITE_SAMPLE_SIZE,
    DEFAULT_SEED as ELITE_SAMPLE_SEED,
    load_sampled_personas,
    sample_elite_personas,
)
from mem_bench.behavior_and_conversation.semantic_events_generator import (
    SemanticEventsRequest,
    generate_semantic_events,
)

DYNAMIC_PROFILE_TEMPLATE_EXCERPT = """
Expected JSON format:
{
  "life_domain": "...",
  "initial_state": {
    "user_attributes_state": {
      "singular": {
        "<attribute_name>": "<concrete value string>"
      },
      "collections": {
        "<collection_name>": [
          "<concrete description string 1>",
          "<concrete description string 2>"
        ]
      }
    },
    "habits_state": {
        "<habit_name>": {
          "schedule": {
            "frequency_type": "daily | weekly | biweekly | monthly_by_date | monthly_nth_weekday",
            "...": "required fields based on frequency_type, see schedule_format below"
          },
          "timing": {
            "start_time": "HH:MM",
            "end_time": "HH:MM"
          },
          "location": "<1-10 words>",
          "priority": "critical | high | medium | low",
        }
    },
    "preferences_state": {
        "<preference_name>": {
          "statement": "<10-30 word concrete preference statement>",
          "signals": [
            "<observable signal 1>",
            "<observable signal 2>"
          ]
        }
    },
    "summary": "..."
  },
  "time_windows": [
    {
      "window_id": "w1",
      "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "window_description": "...",
      "user_attributes_delta": {
        "operations": [
          {
            "op": "modify",
            "attribute_type": "singular",
            "attribute_name": "...",
            "delta": "<new value string>",
            "reason": "..."
          },
          {
            "op": "add|remove",
            "attribute_type": "collections",
            "collection_name": "...",
            "delta": [
              "<item 1>",
              "<item 2>"
            ],
            "reason": "..."
          }
        ]
      },
      "habits_delta": {
        "operations": [
          {
            "op": "acquire|adjust|drop",
            "habit_name": "...",
            "delta": {
              "schedule": {
                "frequency_type": "daily | weekly | biweekly | monthly_by_date | monthly_nth_weekday",
                "...": "required fields based on frequency_type, see schedule_format below"
              },
              "timing": {
                "start_time": "HH:MM",
                "end_time": "HH:MM"
              },
              "location": "<5-15 words>",
              "priority": "critical | high | medium | low",
            },
            "reason": "..."
          }
        ]
      },
      "preferences_delta": {
        "operations": [
          {
            "op": "shift | refine",
            "preference_name": "...",
            "delta": {
              "statement": "<10-30 word concrete preference statement>",
              "signals": [
                "<observable signal 1>",
                "<observable signal 2>"
              ]
            },
            "reason": "..."
          }
        ]
      },
      "summary": "..."
    }
  ]
}

schedule_format = {
    "daily": {
      "frequency_type": "daily"
    },
    "weekly": {
      "frequency_type": "weekly",
      "days_of_week": "[0-6 integers array, 0=Mon...6=Sun, e.g. [1,3,5] for Tue/Thu/Sat]"
    },
    "biweekly": {
      "frequency_type": "biweekly",
      "days_of_week": "[single integer 0-6]",
      "start_date": "YYYY-MM-DD (first occurrence)"
    },
    "monthly_by_date": {
      "frequency_type": "monthly_by_date",
      "days_of_month": "[1-28 integers array, e.g. [1,15] for 1st and 15th]"
    },
    "monthly_nth_weekday": {
      "frequency_type": "monthly_nth_weekday",
      "week_of_month": "1-4 or 'last'",
      "day_of_week": "0-6 integer (0=Mon...6=Sun)"
    }
}
"""
# Individual rule-specific prompts for targeted fixes
# Rule order (from structural to semantic to temporal):
# Rule 1: Required fields (structural integrity)
# Rule 2: Prior existence (operation legality)
# Rule 3: Essential initialization (semantic reasonableness)
# Rule 4: Short-term followups (semantic consistency)
# Rule 5: Time conflicts (final feasibility)

rule1_required_fields_prompt = Template("""You are a strict auditor for fixing VIOLATIONS in dynamic user profiles.

Your task: Add all missing required fields.

A complete dynamic user profile must follow the required schema. I will provide a reference format and a list of missing fields detected automatically. Review them and apply fixes according to the instructions below.

To keep your fixes reasonable, here is the background:

Life domain: {{ domain_name }}: {{ domain_scope_definition }}

Basic user profile: {{ user_profile }}

=================================================================================
DETECTED VIOLATIONS
=================================================================================

{{ detected_issues }}

=================================================================================
SCHEMA REFERENCE
=================================================================================

{{ schema_excerpt }}

=================================================================================
PATCH ACTION GUIDE
=================================================================================

**CRITICAL**: Use the correct action type:

- **"add_key"**: Add missing key to object (requires "key" + "value")
- **"append"**: Add to END of array (path = array, not array[index])
- **"replace"**: Replace existing value
- **"remove"**: Delete element

**NEVER use "add" - Always use "append" for arrays!**

=================================================================================
HOW TO FIX
=================================================================================

1. Add missing keys using "add_key" action
2. Generate appropriate content based on context
3. For habit objects, ensure all nested fields are complete
4. Replace "none" strings with JSON null for drop operations
5. Consider Cascade Effects: When fixing missing fields, check if your changes affect downstream operations or other fields(e.g., summary) and also fixed them in patches.

=================================================================================
OUTPUT FORMAT
=================================================================================

Return JSON with this EXACT structure (dict):
{
  "violations_and_fixes": [
    {
      "issue_id": "<string>",  // Must match detected issue ID (e.g., "rule1_000")
      "location": "<string>",  // Path to the violation location
      "violation_description": "<string>",  // Description of what field is missing
      "fix_applied": <boolean>,  // true if fix is needed, false if the automatically detected issue is not a real violation or doesn't need fixing
      "fix_description": "<string>",  // Explanation of the fix applied
      "cascade_note": "<string>",  // Make sure you consider the cascade effects and update the corresponding fields in patches
      "patches": [  // Array of patch objects (empty array if fix_applied is false)
        {
          "path": "<string>",  // JSON path to target location
          "action": "<string>",  // One of: "add_key", "append", "replace", "remove"
          "key": "<string>",  // Required for "add_key" action only
          "value": <any>  // Value to add/replace (not needed for "remove")
        }
      ]
    }
  ]
}

=================================================================================
CRITICAL CONSTRAINTS
=================================================================================

**1. ID-Based Response:**
- Each detected issue has a unique ID in the format [ID: ruleX_NNN]
- Make sure you fix all issues (if you think they are false positives, set fix_applied=false) and they match 1-to-1 by ID

**2. NO New Windows:**
- DO NOT create new time windows (w5, w6, etc.) and ONLY modify EXISTING windows in the profile

**Dynamic profile to fix:**
{{ dynamic_profile_json }}
""")

rule2_prior_existence_prompt = Template("""You are fixing INVALID OPERATION violations in a dynamic user profile.

Your task: Fix all invalid operations including:
1. Operations that modify/adjust/drop/shift/refine non-existent items
2. Invalid operation types for attribute types
3. Invalid habit adjust operations (only changing priority)

To keep your fixes reasonable, here is the background:

Life domain: {{ domain_name }}: {{ domain_scope_definition }}

Basic user profile: {{ user_profile }}

=================================================================================
DETECTED VIOLATIONS
=================================================================================

{{ detected_issues }}

=================================================================================
WHAT TO FIX: RULE 2 VIOLATIONS
=================================================================================

**RULE 2A: Valid Operation Types by Attribute Type**
- Singular attributes: ONLY "modify" is allowed
  Invalid: add, drop, remove, adjust on singular
  Valid: modify on singular

- Collection attributes: ONLY "add" or "remove" are allowed
  Invalid: modify, adjust, refine on collections
  Valid: add or drop on collections

**RULE 2B: Operations Require Prior Existence**
- Singular attributes: "modify" requires attribute exists in initial_state.singular or added earlier
- Collection attributes: "drop" requires collection exists in initial_state.collections or added earlier
- Habits: "adjust" and "drop" require habit exists in initial_state.habits_state or acquired earlier
- Preferences: "shift" and "refine" require preference exists in initial_state.preferences_state

**RULE 2C: Habit Adjust Must Have Structural Changes**
- Habit "adjust" operations MUST modify at least one of: schedule, timing, location, or context
- Adjusting ONLY priority is INVALID (priority changes should be minimal/implicit)
  Invalid: delta: {"priority": "high"} only
  Valid: delta: {"schedule": {...}, "priority": "high"}
  Valid: delta: {"timing": {...}}

**Examples of violations:**
w2: op="add", attribute_type="singular", attribute_name="primary_vehicle"
→ Invalid operation type: singular only supports "modify"

w2: op="modify", attribute_type="collections", collection_name="visited_countries"
→ Invalid operation type: collections only support "add" or "remove"

w2: op="modify", attribute_name="primary_vehicle"
→ "primary_vehicle" never defined in initial_state.singular

w2: op="adjust", habit_name="morning_run", delta: {"priority": "high"}
→ Adjust only changes priority, must modify schedule/timing/location

w2: op="drop", habit_name="dog_walking"
→ "dog_walking" never in initial_state.habits_state or acquired in previous windows

=================================================================================
PATCH ACTION GUIDE
=================================================================================

**CRITICAL**: Use the correct action type:

- **"add_key"**: Add missing key to object (requires "key" + "value")
- **"append"**: Add to END of array (path = array, not array[index])
- **"replace"**: Replace existing value
- **"remove"**: Delete element

**NEVER use "add" - Always use "append" for arrays!**

=================================================================================
HOW TO FIX
=================================================================================

**For RULE 2A violations (invalid operation type):**
- Change the operation to a valid type for that attribute:
  * Singular: change to "modify" (or remove if not needed)
  * Collections: change to "add" or "remove" (or remove if not needed)
- Update the operation structure to match the new operation type
- Adjust the "reason" field to reflect the corrected operation

**For RULE 2B violations (prior existence):**
- If a remove/modify/adjust/drop/shift/refine operation targets a non-existent item:
  * Add the missing item to initial_state with a baseline value
  * Update initial_state summary to mention it

**For RULE 2C violations (habit adjust without structural changes):**
- Add meaningful structural changes (timing/frequency/location) to the habit delta
- Keep priority if it makes sense, but ensure it's not the ONLY change
- If the adjustment is truly trivial, consider removing the operation entirely

=================================================================================
CRITICAL CONSTRAINTS
=================================================================================

**1. ID-Based Response:**
- Each detected issue has a unique ID in the format [ID: ruleX_NNN]
- Make sure you fix all issues (if you think they are false positives, set fix_applied=false) and they match 1-to-1 by ID

**2. NO New Windows:**
- DO NOT create new time windows (w5, w6, etc.) and ONLY modify EXISTING windows in the profile

=================================================================================
OUTPUT FORMAT
=================================================================================

Return JSON with this EXACT structure:

{
  "violations_and_fixes": [
    {
      "issue_id": "<string>",  // Must match detected issue ID (e.g., "rule2_000")
      "location": "<string>",  // Path to the violation location
      "violation_description": "<string>",  // Description of the prior existence violation
      "fix_applied": <boolean>,  // true if fix is needed, false if false positive
      "fix_description": "<string>",  // Explanation of the fix applied
      "cascade_note": "<string>",  // Explanation of downstream effects and cascades
      "patches": [  // Array of patch objects (empty array if fix_applied is false)
        {
          "path": "<string>",  // JSON path to target location
          "action": "<string>",  // One of: "add_key", "append", "replace", "remove"
          "key": "<string>",  // Required for "add_key" action only
          "value": <any>  // Value to add/replace (not needed for "remove")
        }
      ]
    }
  ]
}

**Dynamic profile to fix:**
{{ dynamic_profile_json }}
""")

rule3_essential_initialization_prompt = Template("""You are fixing ESSENTIAL COLLECTION INITIALIZATION violations in a dynamic user profile.

Your task: Fix collections that first appear via 'add' operations instead of being initialized in initial_state.

To keep your fixes reasonable, here is the background:

Life domain: {{ domain_name }}: {{ domain_scope_definition }}

Basic user profile: {{ user_profile }}

=================================================================================
DETECTED VIOLATIONS
=================================================================================

{{ detected_issues }}

=================================================================================
WHAT TO CHECK: COLLECTION ATTRIBUTES MUST BE INITIALIZED BEFORE FIRST 'ADD'
=================================================================================

**Principle:**
Collections should be initialized in initial_state before they are modified via 'add' operations.
This prevents unrealistic scenarios like a user getting their first smartphone in w3.

**Examples of violations:**
Scenario 1 (VIOLATION):
- initial_state.user_attributes_state.collections: {} (no "owned_devices" key)
- w3: add operation adds {"collection_name": "owned_devices", "delta": ["iPhone 12"]}
→ Violation: User's first device appears in w3, unrealistic for modern adult

Scenario 2 (CORRECT):
- initial_state.user_attributes_state.collections.owned_devices: ["Samsung Galaxy S10"]
- w3: add operation adds {"collection_name": "owned_devices", "delta": ["iPad Pro"]}
→ Correct: User already had a phone, now adding a tablet

=================================================================================
HOW TO FIX
=================================================================================

For each detected violation, evaluate if the collection is essential for this user:

**If essential (e.g., owned_devices for modern adult):**
1. Initialize the collection in initial_state.user_attributes_state.collections with baseline items
   Example: "owned_devices": ["Samsung Galaxy S10 (purchased 2020)"]
2. Update initial_state.summary to mention this baseline
3. **CASCADE:** Modify the first 'add' operation to reflect it's adding TO existing items
   - Keep the operation, just adjust the reason/context if needed

**If NOT essential or legitimately first-time (e.g., hobby equipment collection):**
1. Set fix_applied=false
2. Explain why this is acceptable in fix_description

=================================================================================
CRITICAL CONSTRAINTS
=================================================================================

**1. ID-Based Response:**
- Each detected issue has a unique ID in the format [ID: ruleX_NNN]
- Make sure you fix all issues (if you think they are false positives, set fix_applied=false) and they match 1-to-1 by ID

**2. NO New Windows:**
- DO NOT create new time windows (w5, w6, etc.) and ONLY modify EXISTING windows in the profile

=================================================================================
PATCH ACTION GUIDE
=================================================================================

**CRITICAL**: Use the correct action type:

- **"add_key"**: Add missing key to object (requires "key" + "value")
- **"append"**: Add to END of array (path = array, not array[index])
- **"replace"**: Replace existing value
- **"remove"**: Delete element

**NEVER use "add" - Always use "append" for arrays!**

**Common patch patterns:**
1. Add collection to initial_state:
   {"path": "initial_state.user_attributes_state.collections", "action": "add_key", "key": "owned_devices", "value": ["Samsung Galaxy S10"]}

2. Update initial_state summary:
   {"path": "initial_state.summary", "action": "replace", "value": <updated summary>}

3. Update first add operation's reason:
   {"path": "time_windows[2].user_attributes_delta.operations[0].reason", "action": "replace", "value": <updated reason>}

=================================================================================
OUTPUT FORMAT
=================================================================================

**IMPORTANT NOTES:**
- The detected violations come from automatic checks and may include false positives
- You must judge whether each detected collection is essential for this user
- Set fix_applied=false if the collection is legitimately first-time (not essential baseline)
- Include cascade_note to explain how initial_state addition affects later operations
- When adding collections to initial_state, consider updating the first 'add' operation's context

Return JSON with this EXACT structure:

{
  "violations_and_fixes": [
    {
      "issue_id": "<string>",  // Must match detected issue ID (e.g., "rule3_000")
      "location": "<string>",  // Path to the violation location
      "violation_description": "<string>",  // Description of the collection initialization violation
      "fix_applied": <boolean>,  // true if collection should be in initial_state, false if legitimately first-time
      "fix_description": "<string>",  // Explanation of the fix applied or why no fix needed
      "cascade_note": "<string>",  // Explanation of how initialization affects later operations
      "patches": [  // Array of patch objects (empty array if fix_applied is false)
        {
          "path": "<string>",  // JSON path to target location
          "action": "<string>",  // One of: "add_key", "append", "replace", "remove"
          "key": "<string>",  // Required for "add_key" action only
          "value": <any>  // Value to add/replace (not needed for "remove")
        }
      ]
    }
  ]
}

**Dynamic profile to fix:**
{{ dynamic_profile_json }}
""")

rule4_short_term_followup_prompt = Template("""You are fixing SHORT-TERM CHANGE FOLLOW-UP violations in a dynamic user profile.

Your task: Fix all short-term changes that lack proper follow-ups.

To keep your fixes reasonable, here is the background:

Life domain: {{ domain_name }}: {{ domain_scope_definition }}

Basic user profile: {{ user_profile }}

=================================================================================
DETECTED VIOLATIONS
=================================================================================

{{ detected_issues }}

=================================================================================
WHAT TO CHECK: SHORT-TERM CHANGES MUST HAVE FOLLOW-UPS
=================================================================================

**Principle:**
Changes motivated by SHORT-TERM external factors (seasonal, special events, temporary circumstances) must either:
1. Be rolled back when the factor ends (drop/adjust/refine/shift/modify/remove operations)
2. OR have explicit reasoning explaining why the change became permanent

**Examples of violations:**
w1: acquire "daily_hydration_habit" reason="combat dry winter air"
→ w3 (summer): no adjustment to this habit

w2: acquire "evening_olympic_viewing" reason="watch Olympics coverage"
→ w3 (after Olympics): habit still exists, not dropped


=================================================================================
PATCH ACTION GUIDE
=================================================================================

**CRITICAL**: Use the correct action type:

- **"add_key"**: Add missing key to object (requires "key" + "value")
- **"append"**: Add to END of array (path = array, not array[index])
- **"replace"**: Replace existing value
- **"remove"**: Delete element

**NEVER use "add" - Always use "append" for arrays!**

=================================================================================
HOW TO FIX
=================================================================================

For each detected violation:
1. Add a rollback operation in an appropriate later window (drop/adjust/shift/refine/modify/remove)
2. Update that window's summary to mention the rollback
3. Ensure timeline is realistic (Olympics ~2 weeks, winter ~3 months, seasonal ~3-6 months)
4. Include ALL cascade effects in your patches

**Patch format:** Use JSON patches with actions: "append", "replace", "add_key", "remove"

Note:
- The detected violations come from automatic checks and may include false positives
- Some short-term changes MAY be intentionally permanent (user adapted to the change)
- You must judge whether each detected issue truly needs fixing
- Set fix_applied=false if the change should remain permanent or detection is incorrect
- Include cascade_note to explain the rollback and its effects on later windows

=================================================================================
CRITICAL CONSTRAINTS
=================================================================================

**1. ID-Based Response:**
- Each detected issue has a unique ID in the format [ID: ruleX_NNN]
- Make sure you fix all issues (if you think they are false positives, set fix_applied=false) and they match 1-to-1 by ID

**2. NO New Windows:**
- DO NOT create new time windows (w5, w6, etc.) and ONLY modify EXISTING windows in the profile

=================================================================================
OUTPUT FORMAT
=================================================================================

Return JSON with this EXACT structure:

{
  "violations_and_fixes": [
    {
      "issue_id": "<string>",  // Must match detected issue ID (e.g., "rule4_000")
      "location": "<string>",  // Path to the violation location
      "violation_description": "<string>",  // Description of the short-term change without follow-up
      "fix_applied": <boolean>,  // true if rollback needed, false if intentionally permanent
      "fix_description": "<string>",  // Explanation of the fix applied
      "cascade_note": "<string>",  // Explanation of rollback effects on later windows
      "patches": [  // Array of patch objects (empty array if fix_applied is false)
        {
          "path": "<string>",  // JSON path to target location
          "action": "<string>",  // One of: "add_key", "append", "replace", "remove"
          "key": "<string>",  // Required for "add_key" action only
          "value": <any>  // Value to add/replace (not needed for "remove")
        }
      ]
    }
  ]
}

**Dynamic profile to fix:**
{{ dynamic_profile_json }}
""")

rule5_time_conflict_prompt = Template("""You are fixing TIME CONFLICT violations in a dynamic user profile.

Your task: Resolve ALL habit timing conflicts in one pass.

To keep your fixes reasonable, here is the background:

Life domain: {{ domain_name }}: {{ domain_scope_definition }}

Basic user profile: {{ user_profile }}

=================================================================================
DETECTED CONFLICTS
=================================================================================

{{ detected_issues }}

=================================================================================
RESOLUTION STRATEGY (CRITICAL)
=================================================================================

**Location-Based Conflict Rules:**
1. **Same location conflicts**: If two habits are at the SAME location, they conflict if their times overlap at all
2. **Different location conflicts**: If two habits are at DIFFERENT locations, they need at least 30 minutes gap between them for travel time
   - Gap is measured from the end of the first habit to the start of the second habit
   - Example: Habit A ends at 9:00 AM at Location X, Habit B starts at 9:20 AM at Location Y → CONFLICT (only 20 min gap)
   - Example: Habit A ends at 9:00 AM at Location X, Habit B starts at 9:30 AM at Location Y → OK (30 min gap)

**Resolution Approach:**
- Fix ALL conflicts in the domain at once (no need for iterative fixing since conflicts within a single domain are typically few)
- Adjust habit timings to eliminate all conflicts
- When adjusting times, consider:
  - Maintain realistic timing (e.g., don't move breakfast to 11 PM)
  - Preserve habit duration where possible
  - For same location: ensure no time overlap
  - For different locations: ensure at least 30 minute gap between activities

**IMPORTANT NOTES:**
- The detected violations come from automatic checks and may include false positives
- Some detected conflicts may be acceptable (e.g., overlapping background activities)
- You must judge whether each detected issue truly needs fixing
- Set fix_applied=false if the conflict is acceptable or doesn't need resolution
- When fixing timing in initial/time windows, check if later windows need updates too
- Include cascade_note to make sure you consider the cascade effects (e.g., modify summary or reason field or later window operations) and update the corresponding fields in patches

=================================================================================
PATCH ACTION GUIDE
=================================================================================

**CRITICAL**: Use the correct action type:

- **"add_key"**: Add missing key to object (requires "key" + "value")
- **"append"**: Add to END of array (path = array, not array[index])
- **"replace"**: Replace existing value
- **"remove"**: Delete element

**NEVER use "add" - Always use "append" for arrays!**

=================================================================================
CRITICAL CONSTRAINTS
=================================================================================

**1. ID-Based Response:**
- Each detected issue has a unique ID in the format [ID: ruleX_NNN]
- Make sure you fix all issues (if you think they are false positives, set fix_applied=false) and they match 1-to-1 by ID

**2. NO New Windows:**
- DO NOT create new time windows (w5, w6, etc.) and ONLY modify EXISTING windows in the profile

=================================================================================
OUTPUT FORMAT
=================================================================================

Return JSON with this EXACT structure:

{
  "violations_and_fixes": [
    {
      "issue_id": "<string>",  // Must match detected issue ID (e.g., "rule5_000")
      "location": "<string>",  // Path to the violation location
      "violation_description": "<string>",  // Description of the timing conflict
      "fix_applied": <boolean>,  // true if fix needed, false if acceptable overlap
      "fix_description": "<string>",  // Explanation of the timing adjustment
      "cascade_note": "<string>",  // Make sure you consider the cascade effects and update the corresponding fields in patches
      "patches": [  // Array of patch objects (empty array if fix_applied is false)
        {
          "path": "<string>",  // JSON path to target location
          "action": "<string>",  // One of: "add_key", "append", "replace", "remove"
          "key": "<string>",  // Required for "add_key" action only
          "value": <any>  // Value to add/replace (not needed for "remove")
        }
      ]
    }
  ]
}

**Dynamic profile to fix:**
{{ dynamic_profile_json }}
""")

in_domain_data_review_revise_prompt = Template("""You are a strict compliance auditor for dynamic user profiles.

Life domain: {{ domain_name }}: {{ domain_scope_definition }}
Basic user profile: {{ user_profile }}

Your task: Validate the candidate JSON against the generation rules from dynamic_profile_template and fix violations.

Auto-detected must-fix issues (from programmatic checks):
- Rule 1 (short-term changes require follow-ups): {{ auto_detected_rule1_issues }}
- Rule 2 (operation requires prior existence): {{ auto_detected_rule2_issues }}
- Rule 3 (required fields / schema completeness): {{ auto_detected_rule3_issues }}
- Rule 4 (essential items must be initialized if evolved): {{ auto_detected_rule4_issues }}
- Rule 5 (temporal feasibility / time conflicts): {{ auto_detected_rule5_issues }}
You must fix everything listed above. Still perform a full audit across ALL rules to catch any additional issues.

=================================================================================
OPERATION SEMANTICS REFERENCE
=================================================================================

Before auditing, understand the operation semantics:

**For user_attributes:**

USER ATTRIBUTES ARE DIVIDED INTO TWO TYPES:

1. SINGULAR ATTRIBUTES (single-value attributes):
   - Structure: Simple key-value where value is a string
   - Example: "primary_residence": "Rented studio apartment downtown"
   
   - "modify": Replace the current value with a new value
     * delta contains the NEW value string
     * Can be used multiple times on the same attribute across windows
     * The attribute MUST exist in initial_state or a prior window
     * Example: initial "Rented studio apartment" → w1 modify delta="Purchased two-bedroom condo" → w3 modify delta="Moved to suburban house"

2. COLLECTION ATTRIBUTES (multi-item collections):
   - Structure: Key-value where value is an ARRAY of description strings
   - Example: "owned_devices": ["Dell XPS 13 (laptop)", "Google Pixel 7 (phone)"]
   
   - "add": Add new items to the array
     * delta contains an array of NEW item strings to append
     * Can be used multiple times on the same collection across windows
     * Example: initial ["A", "B"] → w1 add delta=["C", "D"] → w3 add delta=["E"]
     * Result progression: ["A", "B"] → ["A", "B", "C", "D"] → ["A", "B", "C", "D", "E"]
   
   - "remove": Remove specific items from the array
     * delta contains an array of item strings to remove (must match exactly)
     * Example: initial ["A", "B", "C"] → w1 remove delta=["B"] → w3 remove delta=["A"]
     * Result progression: ["A", "B", "C"] → ["A", "C"] → ["C"]

**For habits:**
- "acquire": Create a new habit (habit must NOT exist before)
  * delta contains the complete habit object with all required fields
  
- "adjust": Modify an existing habit (habit MUST exist in initial_state or prior windows)
  * delta contains ONLY the fields being changed (partial habit object)
  
- "drop": Remove a habit entirely
  * delta must be JSON null (not string "none" or empty object)

**For preferences:**
- "shift": Change the preference direction (preference MUST exist before)
  * delta contains the full preference object (statement + 2-4 signals)
  
- "refine": Keep the same direction but adjust strength/specificity (preference MUST exist before)
  * delta contains the full preference object (statement + 2-4 signals)

=================================================================================
COMPLIANCE CHECKLIST (MUST ALL PASS)
=================================================================================

### RULE 1: SHORT-TERM CHANGES MUST HAVE FOLLOW-UPS
**What to check:**
- Identify all changes motivated by SHORT-TERM external factors:
  * Seasonal changes (e.g., "winter dry air" → hydration habit)
  * Special events (e.g., "Olympics" → daily viewing habit)
  * Temporary circumstances (e.g., "heatwave" → indoor exercise)

- For each short-term change, verify that a LATER window includes:
  * A corresponding rollback (drop/adjust/refine/shift) when the factor ends
  * OR explicit reasoning why the change became permanent

**How to detect:**
- Look for keywords (e.g. "winter", "summer", "seasonal", "holiday", etc.) in "reason" fields
- Check if subsequent windows mention the end of these factors
- Verify that habits/preferences acquired for short-term reasons are later adjusted

**Examples of violations:**
w1: acquire "daily_hydration_habit" reason="combat dry winter air" 
→ w3 (summer): no adjustment to this habit
   
w2: acquire "evening_olympic_viewing" reason="watch Olympics coverage"
→ w3 (after Olympics): habit still exists, not dropped

**How to fix:**
- Add a new operation in the appropriate later window to drop/adjust the temporary change
- Update the summary to mention the rollback
- Ensure the timeline is realistic (e.g., Olympics ~2 weeks, winter ~3 months, etc.)

---

### RULE 2: MODIFICATION OPERATIONS REQUIRE PRIOR EXISTENCE
**What to check:**
- For SINGULAR attributes: "modify" can ONLY be used if the attribute existed in a prior state
- For COLLECTION attributes: "add" can create or extend collections; "remove" requires the collection already exists
- For HABITS: "adjust" and "drop" can ONLY be used if the habit existed in a prior state
- For PREFERENCES: "shift" and "refine" can ONLY be used if the preference existed in a prior state

**How to detect:**
- For each modify/adjust/drop/shift/refine operation in window N:
  * Check if the target exists in initial_state OR was added/acquired in windows 1..N-1
  * If not found, this is a violation

**Examples of violations:**
w2: op="modify", attribute_name="primary_vehicle"
→ "primary_vehicle" never defined in initial_state.singular

w2: op="adjust", habit_name="morning_jog"
→ "morning_jog" never defined in initial_state.habits_state or acquired in w1

w3: op="shift", preference_name="exercise_style"
→ "exercise_style" never defined in initial_state.preferences_state

**How to fix:**
- If a modify/adjust/drop/shift/refine operation targets a non-existent item:
  * Add the missing item to initial_state with a baseline value
  * Update initial_state summary to mention it
- OR change the operation type:
  * For attributes: change "modify" to "add" (if creating new singular) or use "add" for collections
  * For habits: change "adjust"/"drop" to "acquire" if it never existed
  * For preferences: cannot fix this way - must add to initial_state

---

### RULE 3: REQUIRED FIELDS MUST BE PRESENT
**What to check:**
- Every window object MUST have: "window_description", "summary"
- initial_state object MUST have: "summary"
- user_attributes_state MUST have both "singular" and "collections" keys (can be empty objects)
- All habit objects (initial and acquire deltas) MUST include: action, schedule (with frequency_type + required fields), timing (start_time + end_time), context, priority, description
- Habit adjust deltas MUST include an updated "description" AND at least one substantive field change (schedule/timing/context/priority)
- Dropped habits MUST use JSON null (not string "none")
- Preferences (initial and deltas) MUST include: "statement" and "signals" (2-4 concrete signals)
- Every operation MUST include a "reason" explaining the change

**How to detect:**
- Check for missing keys in window objects
- Validate habit structure in all habits_state and habits_delta operations (including schedule/timing completeness)
- Check that drop operations set delta to null (not "none", not empty string)
- Validate preference objects have both statement and signals arrays (2-4 items)
- Verify user_attributes_state structure and reasons on operations

**Examples of violations:**
time_windows[1] missing "window_description"
initial_state missing "summary"
habit object: {"action": "walk_dog", "schedule": {"frequency_type": "daily"}} 
(Missing: timing, context, priority, description)
habit adjust: delta only updates "description" without schedule/timing/context/priority change
drop operation: "delta": "none" 
(Should be: "delta": null)
initial_state.user_attributes_state missing "collections" key
preference missing "signals" array
operation missing "reason"

**How to fix:**
- Add missing window_description based on window context
- Generate summary by synthesizing changes in that window
- Complete habit dicts with all required fields (including schedule + timing objects)
- Ensure habit adjust deltas change a substantive field and include updated description
- Replace "none" strings with JSON null
- Add missing structure keys with appropriate empty values
- Fill in preference statement/signals; add missing reasons

---

### RULE 4: ESSENTIAL ITEMS MUST BE INITIALIZED IF EVOLVED
**What to check:**
- Identify all items that are:
  * ESSENTIAL for this domain and this user profile (realistic baseline)
  * EVOLVED in later windows (modify/adjust/shift operations)

- Verify these items exist in initial_state with a baseline value

**Essential items by domain (examples):**
- Finances & Material Living: smartphone, primary bank account, payment methods
- Health & Self-care: basic toiletries, bed, clothing appropriate for climate
- Family & Close Relationships: communication devices if family is not co-located

**How to detect:**
- Scan all "modify", "adjust", "shift" operations in time_windows
- For each, check if that attribute/habit/preference exists in initial_state
- If missing, determine if it's essential (would a realistic person already have it?)

**Examples of violations:**
initial_state.user_attributes_state.collections.owned_devices: [] (empty)
→ w3: add operation adds first smartphone
(Violation: A modern adult should already own a phone in initial_state)

initial_state.preferences_state: no "exercise_style" preference
→ w2: shift operation changes "exercise_style"
(Violation: Must initialize the preference in initial_state before shifting it)

**How to fix:**
- Add the missing item to initial_state with a realistic baseline value:
  * For singular attributes: add to initial_state.user_attributes_state.singular
  * For collection attributes: add to initial_state.user_attributes_state.collections
  * For habits: add to initial_state.habits_state
  * For preferences: add to initial_state.preferences_state
- Update initial_state summary to mention it
- Ensure the baseline is appropriate for the user's profile (income, tech literacy, etc.)

---

### RULE 5: TEMPORAL FEASIBILITY
**What to check:**
- Within each window, habit timings must not overlap
- Habits must have realistic frequency (not "daily" and "5_times_per_day" for same person)
- Time windows must be chronologically ordered and non-overlapping

**How to detect:**
- Parse timing strings (e.g., "6:00-6:30 AM") and check for overlaps
- Check that time_range values are sequential across windows
- Verify frequencies are realistic given user's work schedule and commitments

**Examples of violations:**
- Same window: "morning_exercise" (6:00-7:00 AM) + "morning_commute" (6:30-7:30 AM)
- No spacing: Activity 'daily_ai_exploration' (10:00 PM) starts immediately after 'evening_streaming_routine' (ends 10:00 PM) without the required 15-30 minute spacing.
- Overload: The user's combined schedule is unrealistically packed

**How to fix:**
- Shift timing of one habit to avoid overlap (minimal adjustment)
- Reorder or merge windows if date ranges overlap
- Adjust frequency if unrealistic

=================================================================================
CRITICAL: CASCADE EFFECTS
=================================================================================

**IMPORTANT**: When you fix a violation, you MUST include ALL downstream changes that are affected by your fix.

**Examples of cascade effects:**

1. **Adding to initial_state affects later modifications:**
   - Fix: Add "primary_vehicle" to initial_state.user_attributes_state.singular
   - Cascade: Change w3's op from "add" to "modify" for the vehicle upgrade
   - Your patches MUST include BOTH changes

2. **Adding collection item to initial_state affects later adds:**
   - Fix: Add "Google Pixel 7" to initial_state.user_attributes_state.collections.owned_devices
   - Cascade: If w2 has an "add" operation that includes this phone, remove it from the delta
   - Your patches MUST include BOTH the initial_state addition and the w2 modification

3. **Adjusting a habit timing affects overlapping habits:**
   - Fix: Shift "morning_exercise" from 6:00-7:00 to 7:00-8:00
   - Cascade: If "morning_commute" was 7:30-8:30, it now overlaps and must also shift
   - Your patches MUST include BOTH timing changes

4. **Adding a rollback for short-term change affects summary:**
   - Fix: Add drop operation for "winter_hydration_habit" in w3
   - Cascade: Update w3's summary to mention the habit was dropped
   - Your patches MUST include BOTH the operation and summary update

5. **Changing operation type affects later references:**
   - Fix: Change w1's "adjust" to "acquire" for a habit
   - Cascade: If w2 references this habit with "adjust", verify it's still valid
   - Your patches MUST verify and fix any downstream references

**How to handle cascades:**
- After identifying a fix, scan ALL subsequent windows for items that reference or depend on the changed item
- Include patches for ALL affected locations in the same violation's patches array
- Patches will be applied in array order, so order them from earliest to latest window
- Always update summaries when you add/modify operations in a window

=================================================================================
PATCH FORMAT GUIDE
=================================================================================

**Path format:**
- Use dot notation with array indices: "time_windows[0].habits_delta.operations[1].delta.timing.start_time"
- Path should point to the MINIMAL unit that needs to change

**Action types:**
Choose the most appropriate action type for the operation.

1. **"remove"** - Delete an element from array or key from object
   Example: Remove an invalid operation
   {
     "path": "time_windows[1].habits_delta.operations[3]",
     "action": "remove"
   }

2. **"append"** - Add to the end of an array
   Example: Add a new operation to habits_delta
   {
     "path": "time_windows[2].habits_delta.operations",
     "action": "append",
     "value": {
       "op": "drop",
       "habit_name": "winter_hydration",
       "delta": null,
       "reason": "Summer humidity makes aggressive hydration unnecessary"
     }
   }
   
   Example: Add a new item to a collection in initial_state
   {
     "path": "initial_state.user_attributes_state.collections.owned_devices",
     "action": "append",
     "value": "Google Pixel 7 (Android smartphone for daily communication)"
   }

3. **"replace"** - Replace an existing value
   Example: Change a timing object
   {
     "path": "time_windows[0].habits_delta.operations[1].delta.timing",
     "action": "replace",
     "value": {
       "start_time": "08:45",
       "end_time": "09:15"
     }
   }
   
   Example: Change operation from adjust to acquire
   {
     "path": "time_windows[1].habits_delta.operations[0].op",
     "action": "replace",
     "value": "acquire"
   }

4. **"add_key"** - Add a new key-value pair to an object (for missing required fields or structural issues)
   Example: Add missing "summary" to initial_state
   {
     "path": "initial_state",
     "action": "add_key",
     "key": "summary",
     "value": "User is a tech-savvy professional with moderate fitness habits and a focus on long-term health."
   }
   
   Example: Add missing "window_description" to a window
   {
     "path": "time_windows[1]",
     "action": "add_key",
     "key": "window_description",
     "value": "Spring weather and increased outdoor activity opportunities motivate fitness improvements."
   }
   
   Example: Add missing singular attribute to initial_state
   {
     "path": "initial_state.user_attributes_state.singular",
     "action": "add_key",
     "key": "primary_vehicle",
     "value": "2018 Honda Civic (reliable sedan for daily commute)"
   }
   
   Example: Add entirely missing collections object to initial_state
   {
     "path": "initial_state.user_attributes_state",
     "action": "add_key",
     "key": "collections",
     "value": {
       "owned_devices": [
         "Dell XPS 13 (laptop for work)",
         "Google Pixel 7 (smartphone)"
       ]
     }
   }

=================================================================================
OUTPUT FORMAT
=================================================================================

Return a JSON object with this structure:

{
  "violations_and_fixes": [
    {
      "violation_type": "short_term_followups" | "operation_without_prior_existence" | "required_fields_violation" | "essential_items_violation" | "temporal_feasibility_violation",
      "location": "time_windows[2].habits_delta.operations[0]",
      "violation_description": "Short-term habit 'daily_hydration' acquired in winter (w1) but not adjusted in summer (w3)",
      "cascade_note": "Must also update w3 summary to reflect the habit adjustment",
      "patches": [
        {
          "path": "time_windows[2].habits_delta.operations",
          "action": "append",
          "value": {
            "op": "adjust",
            "habit_name": "daily_hydration",
            "delta": {
              "schedule": {"frequency_type": "weekly", "days_of_week": [0, 2, 4]},
              "description": "Scaled hydration focus to three structured check-ins per week after winter dryness ended"
            },
            "reason": "Summer humidity reduces need for aggressive hydration routine"
          }
        },
        {
          "path": "time_windows[2].summary",
          "action": "replace",
          "value": "Due to summer's increased humidity, the user scales hydration check-ins to three times per week and maintains other routines."
        }
      ]
    }
  ]
}

**CRITICAL REQUIREMENTS:**
- Each violation's patches array MUST include ALL cascading changes (operations + affected summaries)
- Patches are applied in array order, so order them chronologically (initial_state first, then w1, w2, etc.)
- Always include "cascade_note" field explaining what downstream effects your fix has
- If a fix has no cascade effects, set "cascade_note": "No downstream changes needed"

If no violations found, return:
{
  "violations_and_fixes": []
}

=================================================================================
SCHEMA REFERENCE
=================================================================================
{{ schema_excerpt }}

=================================================================================
CANDIDATE JSON TO AUDIT
=================================================================================
{{ dynamic_profile_json }}

Now perform the audit step by step:
1. Check RULE 1 (short-term followups)
2. Check RULE 2 (operation without prior existence)  
3. Check RULE 3 (required fields)
4. Check RULE 4 (essential items violation)
5. Check RULE 5 (temporal feasibility)

For each violation found, generate the minimal patches to fix it.
Output only the JSON result.
""")
# in_domain_data_review_revise_prompt = Template("""You are a strict auditor AND repairer for dynamic profiles produced with dynamic_profile_template.

# Life domain: {{ domain_name }}: {{ domain_scope_definition }}
# Basic user profile: {{ user_profile }}

# Your task: inspect the candidate JSON, detect issues, and think step by step to fix the issues.

# =================================================================================
# ISSUE TYPES YOU MUST CHECK
# =================================================================================

# 1) TIME_CONFLICT
#    - Time windows must be strictly chronological and non-overlapping
#    - Activities within and across windows need plausible spacing (>=15-30 mins buffer)
#    - If a change introduces overlap, adjust times minimally (shift/merge/drop the least-supported item)

# 2) CONSISTENCY_VIOLATION
#    - Every preference/habit change must reference an existing prior value
#    - If a change touches something missing from initial/prior window, add a plausible prior value first
#    - All referenced entities must exist and stay coherent across windows

# 3) FORMAT_VIOLATION
#    - Habits must be dicts with: action, frequency, timing, context, description
#    - Dropped habits use JSON null (not the string "none")
#    - Modified habits use complete habit dict in new_state
#    - Adhere to the schema excerpt below (types and shapes must stay intact)

# =================================================================================
# REPAIR PRINCIPLES
# =================================================================================
# - Make minimal, localized edits; keep unrelated content untouched
# - Preserve value types (list/dict/string/null)
# - Keep window ordering unless fixing a detected overlap/ordering error
# - If a fix in initial affects later windows, patch the downstream windows too

# =================================================================================
# OUTPUT FORMAT (STRICT JSON)
# =================================================================================

# Path format: 
# path 到最小可改的单元 
# 然后要注意cascade effect，比如调整了initial state的habit timing，那么后续的window的habit timing也要调整

# {
#   "conflicts_and_resolutions": [
#   // example for time conflict
#     {
#       "conflict": {
#         "location": "time_windows[0].habits_delta.operations[1].new_state",
#         "type": "time_conflict",
#         "conflict_description": "8:00 AM smoothie overlaps with 8:15-8:45 light therapy"
#       },
#       "resolution": {
#         "strategy": "adjust_timing",
#         "patches": [
#           {
#             "path": "time_windows[0].habits_delta.operations[1].new_state.timing",(这里到最小可改的单元）
#             "action": "replace",
#             "value": "8:45-9:15 AM"
#           },
#           ...
#         ]
#       }
#     },

#   // example for consistency violation
#     {
#       "conflict":{
#         "location": "time_windows[0].habits_delta.operations[1].new_state,
#         "type": "consistency_violation",
#         "description": "habit missing context/description"
#         },
#         "resolution": {
#         "strategy": "complete_habit_dict",
#             "patches": [
#                 {
#                 "path": "time_windows[0].habits_delta.operations[2].new_state",
#                 "action": "replace",
#                 "new_value": {<exact value in this location>, if is remove, the set to null},
#                 "reason": "..."
#                 }
#             ]
#             }
#     },
#     // example for format violation
#     {
#         "conflict":{
#         "location": "time_windows[0].habits_delta.operations[2].new_state",
#         "conflict_description": "habit missing context/description"
#         },
#         "resolution": {
#         "strategy": "complete_habit_dict",
#             "patches": [
#                 {
#                 "path": "time_windows[0].habits_delta.operations[2].new_state",
#                 "action": "replace",
#                 "new_value": {<exact value in this location>, if is remove, the set to null},
#                 "reason": "..."
#                 }
#         ]
#         }
# }

# If no issues: set each conflicts_and_resolutions array to [] and patch_ops to [].

# =================================================================================
# SCHEMA REFERENCE (follow strictly)
# =================================================================================
# {{ schema_excerpt }}

# Candidate JSON to audit and fix:
# {{ dynamic_profile_json }}
# """
# )

@dataclass
class TimelineConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    num_windows: int = 4


@dataclass
class SemanticEventsConfig:
    stable_state_reveal_probability: float = 0.35
    stable_state_reveal_seed: int | None = None


@dataclass
class AtomicEventsConfig:
    min_per_semantic: int = 1
    max_per_semantic: int = 3


@dataclass
class RealDataConfig:
    conversation_turns_range: tuple[int, int] = (2, 4)


@dataclass
class BasicProfileRequest:
    user_description: str


@dataclass
class SpatiotemporalConstraintsRequest:
    time_range: List[str] | tuple[str, str]
    user_basic_profile: Dict
    window_profile_summary: str
    window_description: str
    world_background: str


@dataclass
class LifeContextBaselineRequest:
    user_basic_profile: Dict
    dynamic_profiles_initial_state: Dict[str, Dict]


@dataclass
class LifeContextDeltaRequest:
    user_basic_profile: Dict
    life_context_baseline: Dict
    time_range: List[str] | tuple[str, str] | None
    window_id: str
    window_description: str
    window_profile_summary: str
    world_background: str
    previous_life_context: Dict | None = None
    window_state_by_domain: Dict[str, Dict] | None = None

# BASIC_PROFILE_PROMPT = """
# You convert a free-form user_description paragraph into a concise, structured basic profile.

# Input user_description:
# {user_description}

# Return JSON only (no code fences) using this structure and concrete values:
# {{
#   "summary": "2-3 sentences capturing the user overall",
#   "demographics": {{
#     "age": "mid-30s, late-20s, etc. if inferable",
#     "location": "city/region if present, otherwise null",
#     "education": "highest degree and field if mentioned, otherwise null"
#   }},
#   "work_and_income": {{
#     "occupation": "role and industry if given",
#     "work_style": "hybrid/remote/on-site if hinted",
#     "income_situation": "brief note if present, otherwise null"
#   }},
#   "household": {{
#     "living_situation": "who they live with and housing type if available",
#     "dependents_or_pets": "brief note or null"
#   }},
#   "interests_and_hobbies": [
#     "list of specific interests or hobbies pulled from the description"
#   ],
#   "values_and_personality": [
#     "short bullets on values/traits (e.g., efficiency-focused, community-minded)"
#   ],
#   "digital_life": [
#     "devices, platforms, or tech preferences mentioned; leave empty if unknown"
#   ],
#   "top_priorities": [
#     "1-3 concrete current priorities you can infer; leave empty if unknown"
#   ]
# }}
# """

# BASIC_PROFILE_PROMPT = Template("""You are an expert Social Scientist and Data Profiler. Your task is to convert a raw, unstructured description of a user into a structured, highly consistent Basic User Profile.

# # User Description
# {user_description}

# # Step-by-step Instructions
# 1.  **Analyze** the provided [User Description].
# 2.  **Extract** explicit facts.
# 3.  **Infer** missing attributes based on sociodemographic logic (e.g., if "University Student", infer "Low Income" and "Age 18-25"; if "Senior Surgeon", infer "High Income" and "High Conscientiousness").
# 4.  **Map** the data strictly into the JSON Schema provided below.
# 5.  **Output** ONLY the valid JSON object.

# # Constraints
# -   **Realism & Verisimilitude:** The profile must be **realistic** and **plausible** within the context of the modern world.
# -   **Internal Coherence:** Ensure logical consistency between attributes (e.g., Occupation and Income must match; Age and Education Level must be chronologically possible).
# -   **Enums:** You MUST select values from the provided [Enum Options]. Do not invent new strings.
# -   **No Nulls:** If a field is not mentioned in the text, infer the most **statistically probable** value based on the other known traits.

# # JSON Schema (Strictly Follow This)
# {
#   "basic_user_profile": {
#     "demographics": {
#       "age": "Integer (18-90)",
#       "gender": "Enum ['Male', 'Female', 'Non-binary']",
#       "location_type": "Enum ['Urban_Metropolis', 'Suburban', 'Rural']",
#       "cultural_background": "String (e.g., 'East Asian', 'North American', 'Western European')",
#       "primary_language": "String (ISO code, e.g., 'en', 'zh')"
#     },
#     "socio_economic": {
#       "education_level": "Enum ['High_School', 'Vocational', 'Bachelor', 'Master', 'Doctorate']",
#       "occupation": "String (e.g., 'Software Engineer', 'Marketing Manager', 'Unemployed')",
#       "economic_tier": "Enum ['Low_Income', 'Lower_Middle', 'Middle_Class', 'Upper_Middle', 'High_Net_Worth']",
#       "housing_status": "Enum ['Renting', 'Home_Owner', 'Living_with_Family', 'Dormitory']"
#     },
#     "household": {
#       "marital_status": "Enum ['Single', 'Partnered', 'Married', 'Divorced', 'Widowed']",
#       "parental_status": "Enum ['No_Children', 'Expecting', 'Young_Children', 'Teenage_Children', 'Adult_Children']",
#       "household_size": "Integer (1-10)"
#     },
#     "psychometrics": {
#       "big_five_traits": {
#         "openness": "Enum ['Low', 'Medium', 'High']",
#         "conscientiousness": "Enum ['Low', 'Medium', 'High']",
#         "extraversion": "Enum ['Low', 'Medium', 'High']",
#         "agreeableness": "Enum ['Low', 'Medium', 'High']",
#         "neuroticism": "Enum ['Low', 'Medium', 'High']"
#       },
#       "digital_literacy": "Enum ['Low', 'Basic', 'Advanced', 'Native']",
#       "risk_tolerance": "Enum ['Risk_Averse', 'Neutral', 'Risk_Seeking']"
#     }
#   }
# }
# """
# # )
# BASIC_PROFILE_PROMPT = Template("""You are an expert Social Scientist and Data Profiler. Your task is to convert a raw, unstructured description of a user into a structured, highly consistent Basic User Profile that can be used to synthesize realistic day-to-day behavioral trajectories.

# # User Description
# {user_description}

# # Core Objective
# Produce ONE concrete, internally consistent profile. Do not leave anything vague. Every field must be filled with a specific value from the schema.

# # Step-by-step Instructions
# 1. **Parse & Quote Evidence**
#    - Identify explicit facts in the description (e.g., age, job, family status, location hints).
#    - Treat explicit facts as high-priority constraints.

# 2. **Infer Missing Fields Deterministically**
#    - If a field is not mentioned, infer the most plausible value using the other fields.
#    - Prefer inferences that tighten behavioral constraints (time, money, health capacity, responsibilities) rather than soft “identity” guesses.
#    - Ensure chronological plausibility (age ↔ education ↔ employment).

# 3. **Resolve Conflicts**
#    - If the description contains conflicting signals, resolve to the most coherent interpretation.
#    - Use conservative assumptions (avoid extreme values unless clearly implied).

# 4. **Validate Cross-field Coherence (Hard Rules)**
#    - **Employment ↔ Work Hours**
#      - Student: weekly_work_hours typically 0–25 (unless explicitly working full-time).
#      - Full_time: typically 35–60.
#      - Part_time: typically 5–30.
#      - Retired: typically 0–15.
#      - Unemployed: typically 0–10.
#      - Caregiver: typically 0–20 unless explicitly employed.
#      - Self_employed: typically 20–70 depending on clues.
#    - **Age ↔ Education**
#      - Middle_School_or_Less: typically age 13+ (any)
#      - High_School: typically 16–20+
#      - Associate/Vocational: typically 18+
#      - Bachelor: typically 21+
#      - Master: typically 23+
#      - Doctorate: typically 26+
#      - If the description implies an uncommon path, keep it plausible and consistent.
#    - **Income ↔ Buffer ↔ Housing**
#      - Low income more likely: buffer '0'/'<1'/'1-3', housing Renting/Living_with_Family/Dormitory.
#      - High income more likely: buffer '3-6'/'6-12'/'12+', housing Home_Owner or higher-quality Renting.
#      - Do not assign 'High' income with '0' buffer unless strong evidence (e.g., debt crisis).
#    - **Household ↔ Caregiving**
#      - household_size >= 3 often implies at least Light caregiving if children/elders are implied.
#      - If parental/care duties are explicitly mentioned, increase caregiving_load accordingly.
#    - **Health Constraint ↔ Work**
#      - Severe health_constraint_level makes 50–70 weekly work hours unlikely unless explicitly stated.

# 5. **Use Inference Heuristics (Examples; apply consistently)**
#    - “University/college student” → employment_status Student; education_level Associate/Bachelor (choose based on wording); income_band Low or Lower_Mid; housing Dormitory or Renting; buffer '<1' or '1-3'; digital_literacy Advanced/Native.
#    - “Senior surgeon / partner / executive” → employment_status Full_time; education_level Doctorate/Master; income_band High; buffer '6-12' or '12+'; planning_orientation Future_oriented or Balanced; digital_literacy Advanced.
#    - “Single parent” or strong childcare cues → caregiving_load Moderate/High; household_size 2–5; weekly_work_hours adjusted downward unless explicit full-time.
#    - If no language clues, choose the statistically dominant language for the implied region ONLY if the region is mentioned; otherwise infer from the text (names, phrases).

# # Constraints
# - **Realism & Verisimilitude:** The profile must be realistic and plausible in the modern world.
# - **Internal Coherence:** Ensure logical consistency between attributes (e.g., work hours match employment status; age matches education).
# - **Enums:** You MUST select values from the provided Enum options. Do not invent new strings.
# - **No Nulls / No Unknown / No Prefer_not_to_say:** Every field must have a concrete value.
# - **No Overreach:** Do not infer sensitive details beyond what’s needed to constrain behavior. When multiple values are plausible, choose the one that best preserves coherence and typicality.

# # Output Format
# - Output ONLY one valid JSON object.
# - No commentary, no markdown, no extra keys.

# # JSON Schema (Strictly Follow This)
# {
#   "basic_user_profile": {
#     "context": {
#       "age": "Integer (13-100)",
#       "gender": "Enum ['Male','Female','Non-binary','Other']",
#       "location_type": "Enum ['Urban_Metropolis','Urban','Suburban','Town','Rural']",
#       "languages": "List[String] (ISO-639-1, e.g., ['en','zh'])"
#     },
#     "resources": {
#       "education_level": "Enum ['Middle_School_or_Less','High_School','Vocational','Associate','Bachelor','Master','Doctorate']",
#       "income_band": "Enum ['Low','Lower_Mid','Mid','Upper_Mid','High']",
#       "financial_buffer_months": "Enum ['0','<1','1-3','3-6','6-12','12+']",
#       "housing_status": "Enum ['Renting','Home_Owner','Living_with_Family','Dormitory','Other']"
#     },
#     "constraints": {
#       "employment_status": "Enum ['Full_time','Part_time','Self_employed','Unemployed','Student','Retired','Caregiver']",
#       "weekly_work_hours": "Integer (0-100)",
#       "caregiving_load": "Enum ['None','Light','Moderate','High']",
#       "household_size": "Integer (1-10)",
#       "health_constraint_level": "Enum ['None','Mild','Moderate','Severe']"
#     },
#     "capabilities": {
#       "digital_literacy": "Enum ['Low','Basic','Advanced','Native']",
#       "planning_orientation": "Enum ['Present_biased','Balanced','Future_oriented']",
#       "risk_tolerance": "Enum ['Risk_Averse','Neutral','Risk_Seeking']"
#     }
#   }
# }
# """)

BASIC_PROFILE_PROMPT = Template("""You are creating a concrete user profile for behavioral simulation. Generate ONE specific, realistic profile based on the description below.

# User Description
{{user_description}}

# Instructions
1. Extract explicit facts from the description
2. Infer missing details to create a complete, coherent profile
3. Ensure all fields are internally consistent (e.g., student income matches student occupation)
4. Choose specific values - no vague or placeholder text

# Consistency Rules
- **Students**: income Low/Lower_Mid, work_hours 0-25, education Associate/Bachelor
- **Full-time workers**: work_hours 35-50, income Mid or higher
- **Parents with young children**: caregiving_load Moderate/High, household_size 2+
- **Age & education**: Bachelor typically 22+, Master 24+, Doctorate 27+
- **Income & buffer**: High income → 6+ months buffer; Low income → 0-3 months

# Output Format (JSON only)
{
  "age": <integer 18-85>,
  "gender": "<Male|Female|Non_binary|Other>",
  "location_type": "<Urban_Metropolis|Urban|Suburban|Town|Rural>",
  "location_region": "<string, e.g., 'San Francisco Bay Area', 'Beijing', 'London'>",
  "languages": ["<ISO codes, e.g., en, zh, es>"],
  
  "occupation": "<specific job title, e.g., 'Software Engineer', 'Retail Cashier', 'Graduate Student', 'Retired Teacher'>",
  "employment_status": "<Full_time|Part_time|Self_employed|Unemployed|Student|Retired|Caregiver>",
  "industry": "<specific industry, e.g., 'Technology', 'Healthcare', 'Education', 'Retail'>",
  "weekly_work_hours": <integer 0-70>,
  
  "education_level": "<Middle_School|High_School|Vocational|Associate|Bachelor|Master|Doctorate>",
  "income_band": "<Low|Lower_Mid|Mid|Upper_Mid|High>",
  "financial_buffer_months": "<0|<1|1-3|3-6|6-12|12+>",
  "housing_status": "<Renting|Home_Owner|Living_with_Family|Dormitory|Other>",
  
  "household_size": <integer 1-8>,
  "household_composition": "<string, e.g., 'Lives alone', 'Married with 2 children', 'College roommates'>",
  "caregiving_load": "<None|Light|Moderate|High>",
  "health_constraint_level": "<None|Mild|Moderate|Severe>",
  
  "digital_literacy": "<Low|Basic|Advanced|Native>",
  "planning_orientation": "<Present_biased|Balanced|Future_oriented>",
  "risk_tolerance": "<Risk_Averse|Neutral|Risk_Seeking>"
}

""")

def render_basic_profile_prompt(request: BasicProfileRequest) -> str:
    return BASIC_PROFILE_PROMPT.render(user_description=request.user_description)


def generate_basic_profile(
    llm_client: GeminiJSONClient, request: BasicProfileRequest
) -> LLMResult:
    prompt = render_basic_profile_prompt(request)
    return llm_client.generate_json(prompt)


# SPATIOTEMPORAL_CONSTRAINTS_PROMPT = Template(
#     """You generate a precise spatiotemporal plan for ONE time window.
# you need generate a spatiotemporal plan in this time range for the user 
# Inputs:
# - time_range: {{ time_range }}  the time range
# - user_basic_profile (JSON): {{ user_basic_profile_json }}  this is the basic user profile that impact multiple life domains
# - window description: {{ window_description }} short description of the general conditions that motivate changes from previous time window to this time window
# - window_profile_summary (cross-domain, JSON text): {{ window_profile_summary }}  this is user fine-grained profile in multiple life domains in this time window, each is a summary
# - window_world_background: {{ world_background }} this is the world background in this time window

# Output ONLY JSON with this exact shape (no code fences):
# {
#   "whereabouts": [
#     {
#       "start": "YYYY-MM-DDTHH:MM",
#       "end":   "YYYY-MM-DDTHH:MM",
#       "loc":   "ISO-like location code or description (e.g., CN.SHANGHAI or US.SF)",
#       "kind":  "STAY"
#     },
#     {
#       "start": "YYYY-MM-DDTHH:MM",
#       "end":   "YYYY-MM-DDTHH:MM",
#       "loc":   "TRAVEL",
#       "kind":  "TRAVEL",
#       "from":  "origin code",
#       "to":    "destination code"
#     }
#   ],
#   "home_base": "default base city/area (e.g., CN.SHANGHAI)"
# }

# Rules:
# - Cover the entire time_range; combine into a minimal, coherent itinerary.
# - If no travel cues, use one STAY covering the whole range at the home_base.
# - Times can be approximate but must be within the time_range.
# - Use TRAVEL segments only when the summary or background implies travel; otherwise avoid them.
# - Keep texts concise; no extra fields.
# """
# )
Life_Context_Baseline_Prompt = Template("""
You are a constraint extraction agent. Generate a minimal life context that defines external constraints on when/where/with-whom events can occur.

# Framework
- **Capability**: Physical reach (travel time) + time budgets (when user is available)
- **Authority**: Location access rules (opening hours, memberships)
- **Coupling**: Coordination requirements (lead time, participants)

# Input Data
**Basic Profile**: {{ user_basic_profile_json }}
**Baseline Detailed Profile**: {{ dynamic_profiles_initial_state_json }}

---

## 1. CAPABILITY

### Time Budgets (4 blocks)
```json
{
  "name": "sleep|work_core|free_evening|free_weekend",
  "days": "all|weekday|weekend",
  "start": "HH:MM",
  "end": "HH:MM",
  "rigidity": "hard|soft"
}
```
- **hard**: Cannot violate (e.g., employer-mandated work hours)
- **soft**: Preferred but flexible (e.g., typical sleep window)

**Extract from**:
- Work habits → `work_core` (default 09:00-18:00 weekdays, rigidity=hard)
- Sleep habits → `sleep` (default 23:00-07:00, rigidity=soft)
- Remainder → `free_evening` (weekday 18:00-23:00), `free_weekend` (weekend 09:00-23:00)

### Mobility
```json
{
  "mode_defaults": {"local": "drive|walk|transit"},
  "edges": [
    {
      "from": "home|work|fitness|grocery|restaurant",
      "to": "home|work|fitness|grocery|restaurant",
      "mode": "drive|walk|transit",
      "travel_time": {"kind": "quantiles", "min": X, "p50": Y, "p90": Z, "max": W},
      "time_sensitivity": "rush_hour|weather|stable"
    }
  ]
}
```
- **Only generate edges for**: home↔work, home↔locations in 3+ weekly habits
- **Travel time estimation**: Urban car p50 ≈ distance_km × 3 minutes
- **Use generic location types**: "fitness" not "The Studio Climbing"

---

## 2. AUTHORITY

### Places (5-8 maximum)
```json
{
  "id": "home|work|fitness|grocery|restaurant|social_venue",
  "type": "residence|workplace|fitness|retail|social|healthcare",
  "access": {
    "kind": "private|restricted|public",
    "open_hours": [["weekday", "09:00", "18:00"]], // only if restricted
    "membership_required": true  // only if restricted
  }
}
```
- **Mandatory**: `home` (private), `work` (if employed)
- **Conditional**: Extract from habits (gym→fitness, grocery shopping→grocery, date night→restaurant)
- **Generic types only**: No real business names

---

## 3. COUPLING

### Commitments (recurring coordinated activities)
```json
{
  "id": "unique_id",
  "activity_tag": "date_night|board_game|gym_class",
  "rule": {
    "recurrence": "daily|weekly|monthly",
    "day": "monday|friday|last_saturday|15th",
    "time": "HH:MM",
    "duration_min": 120
  },
  "participants": ["partner", "group:board_game"],
  "location_hint": "restaurant|social_venue|fitness",
  "preemptible": false
}
```
- **Only include**: Habits with explicit timing + other people involved
- **Skip**: Solo habits, flexible activities

### Coordination Rules (by activity type)
```json
{
  "activity_tag": "date_night|group_event|casual_meetup",
  "participants": ["partner", "group:*", "friend:*"],
  "lead_time": {"min_h": 0, "preferred_h": 24},
  "flex_window_h": 2,
  "cancellation_penalty": "high|medium|low"
}
```
- **Standard patterns**:
  - Co-resident partner: min_h=0, preferred_h=24, penalty=high
  - Small group (4-8): min_h=48, preferred_h=168, penalty=high
  - Professional: min_h=24, preferred_h=72, penalty=low

---

# Output Format
```json
{
  "life_context": {
    "timezone": "America/Los_Angeles",
    "capability": {
      "time_budgets": [...],
      "mobility": {"mode_defaults": {...}, "edges": [...]}
    },
    "authority": {
      "places": [...]
    },
    "coupling": {
      "commitments": [...],
      "coordination_rules": [...]
    }
  }
}
```

# Rules
1. Use generic types, not specific names
2. Extract only what's in habits or blocks scheduling
3. Minimize entries: 4 time_budgets, 3-5 edges, 5-8 places
4. If unsure, omit (better sparse than speculative)

Generate the life context now.
""")

Life_Context_Delta_Prompt = Template("""
You are a constraint delta agent. Analyze a time window to identify TEMPORARY changes to the user's life context constraints.

# Input Data
**Basic Profile**: {{ user_basic_profile_json }}
**Current Life Context Baseline**: {{ life_context_baseline_json }}
**Time Window**:
- Description: {{ window_description }}
- Summary: {{ window_summary }}
- World Background: {{ window_world_background }}

---

# Task
Generate constraint deltas (additions, modifications, suspensions) that apply ONLY during this time window. Focus on:
1. **Travel/Relocation**: Does user leave primary region? (affects capability.mobility)
2. **Schedule Disruptions**: Do work hours change? Holidays? (affects capability.time_budgets)
3. **Access Changes**: New locations needed? Facilities closed? (affects authority.places)
4. **Coordination Shifts**: Are recurring commitments suspended? New group activities? (affects coupling)

---

## Delta Operations

### 1. CAPABILITY Deltas

#### Time Budget Modifications
```json
{
  "op": "suspend|modify|add",
  "target": "work_core|sleep|free_evening|free_weekend",
  "effective_dates": ["2024-07-15", "2024-07-20"], // specific date range, or null for entire window
  "new_state": {
    "name": "vacation_mode",
    "days": "all",
    "start": "08:00",
    "end": "23:00",
    "rigidity": "soft"
  },
  "reason": "User on vacation per window_summary"
}
```
**Triggers**:
- "vacation", "travel", "time off" → suspend work_core
- "holiday season", "extended break" → modify time_budgets
- "working from [other city]" → add temporary work hours

#### Mobility Changes
```json
{
  "op": "add_temporary_location|modify_edge",
  "details": {
    "location_id": "vacation_destination",
    "location_type": "temporary_residence",
    "from_baseline": "home", // which baseline location this replaces
    "effective_dates": ["2024-07-15", "2024-07-20"]
  },
  "reason": "Week-long vacation to [location] mentioned in window"
}
```
**Triggers**:
- "trip to", "vacation in", "visiting" → add temporary location
- "working remotely from" → add temporary work location
- Extreme weather mentioned → modify travel_time multipliers

---

### 2. AUTHORITY Deltas

#### Place Access Changes
```json
{
  "op": "add|suspend|modify",
  "place_id": "office|gym|new_venue",
  "effective_dates": ["2024-12-24", "2024-12-26"],
  "modification": {
    "access": {"kind": "restricted", "open_hours": []} // empty = closed
  },
  "reason": "Holiday closure per window_description"
}
```
**Triggers**:
- "office closed for holidays" → suspend work access
- "gym membership starts" → add fitness place
- "new climbing gym opens" → add alternative fitness venue

---

### 3. COUPLING Deltas

#### Commitment Suspensions
```json
{
  "op": "suspend|reschedule|add",
  "commitment_id": "date_night|board_game",
  "effective_dates": ["2024-08-01", "2024-08-14"],
  "modification": {
    "suspended": true,
    "reason": "Partner traveling for work per window_summary"
  }
}
```
**Triggers**:
- "partner away", "group on break" → suspend commitments
- "holiday gatherings" → add temporary family events
- Event like "Olympics viewing parties" → add temporary recurring commitment

#### Temporary Coordination Rules
```json
{
  "op": "add",
  "temporary_rule": {
    "activity_tag": "olympics_viewing",
    "participants": ["friend:*"],
    "lead_time": {"min_h": 4, "preferred_h": 24},
    "flex_window_h": 1,
    "cancellation_penalty": "low",
    "effective_window": "entire"
  },
  "reason": "Olympics viewing parties mentioned in habits_delta"
}
```

---

## Analysis Guidelines

### Look for these signals:

**In window_description**:
- Geographic terms: "traveling to", "vacation in", "visiting", "relocated temporarily"
- Schedule terms: "holiday", "break", "time off", "remote work period"
- Access terms: "gym closed", "new facility", "office shutdown"
- Social terms: "partner away", "hosting visitors", "group hiatus"

**In window_summary**:
- Habit changes: "drops outdoor_running" → may indicate location change or facility closure
- Attribute changes: "adds travel gear" → likely trip planned
- Relationship changes: "partner becomes fiancée" → may affect coordination patterns

**In window_world_background**:
- Major events: "Olympics", "holidays" → temporary commitments
- Natural events: "heatwave", "winter storm" → mobility constraints
- Market events: "Bitcoin halving" → usually no constraint impact (unless explicitly changes schedule)

### Default to NO DELTA if:
- Window only describes internal changes (skills learned, preferences shifted)
- Events are purely informational (reading about Olympics ≠ attending Olympics)
- Changes are covered by existing seasonal_modifiers in baseline

---

# Output Format
```json
{
  "window_id": "w1|w2|w3|w4",
  "window_dates": ["2024-01-01", "2024-03-31"],
  "has_constraint_changes": true,
  "deltas": {
    "capability": [
      // time_budget and mobility deltas
    ],
    "authority": [
      // place access deltas
    ],
    "coupling": [
      // commitment and coordination deltas
    ]
  },
  "rationale": "Brief explanation of why these deltas were generated"
}
```

**If no constraint changes**: 
```json
{
  "window_id": "w2",
  "window_dates": ["2024-04-01", "2024-07-01"],
  "has_constraint_changes": false,
  "rationale": "Window describes internal skill/preference changes with no physical/temporal/social constraint impacts"
}
```

---

# Critical Rules

1. **Only generate deltas with explicit evidence** in window text
2. **Specify effective_dates** when possible (vs. "entire window")
3. **Prefer suspend over delete**: Constraints usually return after window
4. **Don't infer lifestyle from events**: "Bitcoin Halving occurs" ≠ "user changes schedule"
5. **Check if seasonal_modifiers already cover it**: Don't duplicate baseline logic

Generate the constraint deltas now.
""")


# SPATIOTEMPORAL_CONSTRAINTS_PROMPT = Template("""You are an expert agent responsible for generating spatiotemporal trajectory plans.

# Your task: Generate a JSON-formatted schedule that covers the specified time window, based on user profiles and contextual information.

# # Input Data

# **Time Window**: {{ time_range }}

# **Window World Background**: {{ window_world_background }}

# **Basic Profile (Static/Long-term)**:
# {{ user_basic_profile_json }}

# **Motivation for This Window** (what drove the transition from previous state to current state):
# {{ window_description }}

# **Current State Summary (Dynamic/Window-specific)**:
# {{ window_profile_summary }}


# ---

# # Task Instructions

# 1. **User Modeling**: Analyze the `Basic Profile` to establish a baseline understanding of the user's characteristics, habits, and behavioral patterns.

# 2. **Window-specific Reasoning**: 
#    - Examine the `Motivation` to understand WHY the user's state changed
#    - Analyze the `Current State Summary` to understand WHAT the resulting state is
#    - Infer spatiotemporal changes based on the interplay between motivation and outcome

# 3. **Trajectory Generation Rules**:
#    - **Full Coverage**: The schedule MUST cover the entire `Time Window` with no gaps
#    - **Default Behavior**: If no explicit travel triggers exist, generate a single "STAY" segment at `home_base`
#    - **Travel Conditions**: Only add "TRAVEL" segments when `window_description` or `window_profile_summary` explicitly indicates movement/relocation
#    - **Consistency**: Ensure all transitions are logically justified by the input data

# ---

# # Output Format (JSON only)
# {
#   "whereabouts": [
#     {
#       "start": "YYYY-MM-DDTHH:MM",
#       "end": "YYYY-MM-DDTHH:MM",
#       "loc": "Location Code (e.g., CN.SHANGHAI) or 'TRAVEL'",
#       "kind": "STAY or TRAVEL",
#       "from": "Origin Code (required only if kind=TRAVEL)",
#       "to": "Destination Code (required only if kind=TRAVEL)"
#     }
#   ],
#   "home_base": "Default base city code (e.g., CN.SHANGHAI)"
# }
# """)

# CONFLICT_RESOLUTION_PROMPT = Template("""You are a cross-domain consistency auditor. Given full dynamic user profiles across domains, detect and resolve conflicts realistically, per window (initial + each window_id). Do NOT blanket-apply a single value to all windows.

# User basic profile (context, optional):
# {{ user_basic_profile_json }}

# Full dynamic profiles by domain (initial state + deltas):
# {{ dynamic_profiles_json }}

# What to look for (be exhaustive):
# - Inventory contradictions for the same entity class (e.g., one domain lists MacBook Pro M3, another lists MacBook Pro M2 14-inch).
# - Temporal collisions: overlapping recurring habits/events in the same day/slot; multiple "weekly" events all on Wed night—adjust to bi-weekly/alt days.
# - Preference/attribute drifts that cannot coexist (e.g., vegan in one domain, steakhouse lover in another).
# - Financial/resource feasibility anchored to the basic profile (time, money, energy).
# - Cross-window coherence across domains: if two windows overlap in time across domains and clash (e.g., two full-time jobs), flag and resolve.

# How to fix:
# - Prefer the most coherent, realistic single truth; keep shapes (list/dict/string) intact.
# - When schedule conflicts arise, resolve by spacing out (bi-weekly/monthly), shifting day/time, or dropping the least-supported item.
# - When similar attributes use different keys, pick a canonical key name and align all domains to the chosen value.
# - For set/list-type fields (subscriptions, devices, assets), resolve by union + dedupe unless items are mutually exclusive; then choose the consistent subset and note dropped items.
# - Resolve per window_id; only touch windows that need changes.
# - If nothing is conflicting, return empty sections.

# Return strict JSON (no code fences) with this structure:
# {
#   "canonical_attributes": {
#     "<canonical_key>": {
#       "resolved_value": "<final value>",
#       "reason": "short justification",
#       "windows": [
#         {
#           "window_id": "<initial|w1|...>",
#           "domains": [{"domain": "<domain_name>", "attribute_name": "<attr key as stored in that domain>"}]
#         }
#       ]
#     }
#   },
#   "per_domain_resolutions": {
#     "<domain_name>": [
#       {
#         "attribute_name": "<attr name exactly as it appears in that domain>",
#         "resolved_value": "<resolved value for that domain>",
#         "window_id": "<initial|w1|...>",
#         "note": "short note on the change"
#       }
#     ]
#   },
#   "detected_conflicts": [
#     {
#       "kind": "semantic|temporal|inventory|preference",
#       "window_id": "<initial|w1|...>",
#       "time_range": ["<start>", "<end>"],
#       "description": "brief description of the conflict and domains involved"
#     }
#   ]
# }

# Rules:
# - Outputs must be realistic and consistent with the basic profile.
# - Preserve attribute shapes (list/dict/string) when resolving.
# - Be explicit in per_domain_resolutions for every field that needs updating; specify window_id.
# """)

# CONFLICT_RESOLUTION_PROMPT = Template("""You are a cross-domain consistency auditor and profile reasonableness validator. 

# Your task is to:
# 1. Detect and resolve CONFLICTS across different life domains in the user's dynamic profile
# 2. Identify and fix UNREASONABLE patterns that emerge when domains are combined
# 3. Ensure the integrated profile reflects a realistic, livable human schedule and lifestyle

# IMPORTANT CONTEXT:
# - All attribute keys are already normalized across domains
# - Intra-domain conflicts have been resolved; focus ONLY on cross-domain issues
# - Your output will be directly applied as patches to the profile

# User basic profile:
# {{ user_basic_profile_json }}

# Full dynamic profiles by domain (initial state + deltas):
# {{ dynamic_profiles_json }}

# Auto-detected temporal conflicts (code-level hints):
# {{ detected_temporal_conflicts_json }}

# - Target window for this call: {{ target_window_id or "all_windows" }}. Only change that window unless a minimal cascade is unavoidable.
# - Deduped by window + habit pair with sample_dates and up to 3 overlap_examples. Includes per-window conflict graph (top nodes by degree). Resolve window-by-window (initial_state → w1 → w2 → w3...), tackling highest-degree habits first.

# =================================================================================
# DATA STRUCTURE REFERENCE
# =================================================================================

# Each domain profile has this structure:

# {
#   "life_domain": "<domain_name>",
#   "initial_state": {
#     "user_attributes_state": {
#       "singular": {
#         "<attribute_name>": "<value_string>",
#         ...
#       },
#       "collections": {
#         "<collection_name>": ["<item_1>", "<item_2>", ...],
#         ...
#       }
#     },
#     "habits_state": {
#       "initial": {
#         "<habit_name>": {
#           "action": "...",
#           "schedule": {
#             "frequency_type": "daily | weekly | biweekly | monthly_by_date | monthly_nth_weekday",
#             "...": "required schedule fields"
#           },
#           "timing": {"start_time": "HH:MM", "end_time": "HH:MM"},
#           "context": "...",
#           "priority": "critical | high | medium | low",
#         },
#         ...
#       }
#     },
#     "preferences_state": {
#       "initial": {
#         "<preference_name>": {
#           "statement": "...",
#           "signals": ["...", "..."]
#         },
#         ...
#       }
#     }
#   },
#   "time_windows": [
#     {
#       "window_id": "w1",
#       "user_attributes_delta": {
#         "operations": [
#           {
#             "op": "modify" | "add" | "remove",
#             "attribute_type": "singular" | "collections",
#             "attribute_name" | "collection_name": "...",
#             "delta": <value>,
#             "reason": "..."
#           }
#         ]
#       },
#       "habits_delta": { ... },
#       "preferences_delta": { ... }
#     }
#   ]
# }

# =================================================================================
# CONFLICT TYPE 1: ATTRIBUTE CONFLICT
# =================================================================================

# ## Definition & Detection

# **For Singular Attributes:**
# If the same singular attribute key exists in multiple domains with DIFFERENT values in the SAME window (including initial_state), it is a conflict.

# Path format: `<domain>.initial_state.user_attributes_state.singular.<attribute_name>`

# <Example>
# Job Title Conflict:
# // Domain: Work & Education @ initial_state.user_attributes_state.singular
# "primary_job": "Director of Product Management"

# // Domain: Finances & Material Living @ initial_state.user_attributes_state.singular
# "primary_job": "Senior Product Manager"

# → CONFLICT: Same singular attribute key, different values across domains
# </Example>

# **For Collection Attributes:**
# If the same collection key exists in multiple domains, the collections themselves are NOT automatically a conflict (they can coexist in different domains). However, if individual items WITHIN the collections are contradictory OR overlapping, it IS a conflict.

# Path format: `<domain>.initial_state.user_attributes_state.collections.<collection_name>`

# **Understanding Contradictory Items (Category Exclusivity):**
# Contradictory items belong to the same device/product category where a user typically owns only ONE primary item. Common exclusive categories include:
# - Smartphones: User has one primary phone (e.g., "iPhone 14" OR "Google Pixel 7", not both)
# - Laptops: User has one primary laptop (e.g., "MacBook Pro" OR "Lenovo ThinkPad", not both)
# - Tablets: User has one primary tablet (e.g., "iPad Pro" OR "Samsung Galaxy Tab", not both)
# - Fitness trackers: User wears one primary tracker (e.g., "Apple Watch" OR "Fitbit", not both)

# **Understanding Overlapping Items (Exact Duplicates):**
# Overlapping items are the EXACT SAME item (or highly similar description) listed in multiple domains.

# <Example>
# Device Inventory - Contradictory Items:
# // Domain A: Technology & Digital Life
# collections.owned_devices: [
#   "iPhone 14 (smartphone for daily communication)",
#   "MacBook Pro 2021 (laptop for development work)"
# ]

# // Domain B: Work & Education
# collections.work_devices: [
#   "Google Pixel 7 (Android phone for work)",
#   "Lenovo ThinkPad X1 (work laptop)"
# ]

# → CONFLICT: Collections contain contradictory items within the same device category
#    - "iPhone 14" (Domain A) vs "Google Pixel 7" (Domain B): Both are smartphones
#    - "MacBook Pro" (Domain A) vs "Lenovo ThinkPad" (Domain B): Both are laptops

# Resolution Approach:
#    - Evaluate which devices are more reasonable based on user's basic profile
#    - If user profile indicates Apple ecosystem preference:
#      * Keep "iPhone 14" and "MacBook Pro" in Domain A
#      * REMOVE "Google Pixel 7" from Domain B's collection
#      * REMOVE "Lenovo ThinkPad" from Domain B's collection
# </Example>

# <Example>
# Device Inventory - Overlapping Items:
# // Domain A: Technology & Digital Life
# collections.owned_devices: [
#   "iPhone 14 (smartphone for daily communication)",
#   "MacBook Pro 2021 (laptop)"
# ]

# // Domain B: Finances & Material Living
# collections.tracked_assets: [
#   "iPhone 14 (smartphone)",
#   "AirPods Pro (wireless earbuds)"
# ]

# → OVERLAP CONFLICT: "iPhone 14" appears in both domains
# Resolution: Keep in MOST AUTHORITATIVE domain (Technology) and remove from other domain (Finances)
# </Example>

# <Example>
# Subscription Coexistence (**NOT a conflict**):
# // Domain A: Entertainment & Leisure
# collections.streaming_subscriptions: [
#   "Netflix Standard (streaming service)",
#   "Spotify Premium (music streaming)"
# ]

# // Domain B: Learning & Personal Growth
# collections.learning_subscriptions: [
#   "O'Reilly Media (technical learning platform)"
# ]

# → NO CONFLICT: All items are unique across domains. User can have all simultaneously.
# </Example>

# **Key Distinction:**
# - Collection attributes can exist in multiple domains (no automatic merging)
# - **Contradictory items** = mutually exclusive items (competing devices, incompatible plans)
# - **Overlapping items** = exact duplicates appearing in multiple domains
# - When items are contradictory, identify and drop the less reasonable items
# - When items overlap, keep in the MOST AUTHORITATIVE domain and remove from others

# ---

# ## Resolution Strategies

# **For Singular Attributes:**
# 1. Analyze all conflicting values across domains
# 2. Select the MOST REASONABLE value based on:
#    - Consistency with user's basic profile
#    - Domain authority (e.g., "Work & Education" is authoritative for job_title)
# 3. Update ALL domains to use the selected canonical value
# 4. Document which values were dropped and why

# **For Collection Attributes:**
# 1. Identify specific ITEMS within collections that are contradictory or overlapping
# 2. For **contradictory items**: Evaluate each for reasonableness and drop the less reasonable ones
# 3. For **overlapping items**: Determine the most authoritative domain and remove duplicates from other domains
# 4. Keep collections separate per domain; do NOT merge across domains
# 5. Use "remove" operation targeting specific array indices

# <Example>
# Before:
#   Domain A "Technology & Digital Life" - collections.owned_devices:
#     ["iPhone 14 (smartphone)", "MacBook Pro (laptop)", "AirPods Pro (earbuds)"]
  
#   Domain B "Work & Education" - collections.work_devices:
#     ["Google Pixel 7 (work phone)", "iPad Pro (tablet)", "AirPods Pro (earbuds)"]

# Analysis: 
#   - "iPhone 14" vs "Google Pixel 7": Contradictory (mutually exclusive phones)
#   - "AirPods Pro": Overlapping (duplicate in both domains)

# Resolution (if user prefers Apple ecosystem):
#   Domain A: Keep as-is
#   Domain B: Remove contradictions and overlaps
  
# Patches:
#   [
#     {
#       "domain": "Work & Education",
#       "window_id": "initial",
#       "path": "user_attributes_state.collections.work_devices[0]",
#       "operation": "remove",
#       "reason": "Remove contradictory phone - user has iPhone 14 as primary device"
#     },
#     {
#       "domain": "Work & Education",
#       "window_id": "initial",
#       "path": "user_attributes_state.collections.work_devices[2]",
#       "operation": "remove",
#       "reason": "Remove duplicate AirPods - already tracked in Technology domain"
#     }
#   ]
# </Example>

# **Priority Rules for Determining Authoritative Domain (for overlapping items):**
# 1. If an item naturally belongs to a domain's core purpose (e.g., "work_laptop" in Work & Education), that domain is authoritative
# 2. For general items (devices, subscriptions), prioritize:
#    - Dedicated domain (e.g., "Technology & Digital Life" for devices)
#    - Financial tracking domain (e.g., "Finances & Material Living" for subscriptions)
#    - Context-specific domain (e.g., "Health & Wellness" for fitness devices)
# 3. When in doubt, keep the item in the domain with more contextual detail

# ---

# =================================================================================
# CONFLICT TYPE 2: TEMPORAL COLLISION
# =================================================================================

# ## Definition & Detection

# **Definition**: User cannot physically perform two activities at the same time

# Habits are stored in: `initial_state.habits_state.<habit_name>`
# Each habit has fields: `action`, `schedule` (frequency_type + required fields), `timing` (start_time + end_time), `context`, `priority`, `description`

# **Sub-types:**

# ### 2a. Direct Time Overlap

# <Example>
# Domain A - habit_weekly_class:
#   schedule: {"frequency_type": "weekly", "days_of_week": [2]}
#   timing: {"start_time": "19:00", "end_time": "20:00"}

# Domain B - habit_team_meeting:
#   schedule: {"frequency_type": "weekly", "days_of_week": [2]}
#   timing: {"start_time": "19:00", "end_time": "21:00"}

# → CONFLICT: Overlapping time blocks on the same day
# </Example>

# ### 2b. Frequency Saturation

# <Example>
# Domain A - habit_morning_gym:
#   schedule: {"frequency_type": "daily"}
#   timing: {"start_time": "08:30", "end_time": "09:00"}

# Domain B - habit_commute:
#   schedule: {"frequency_type": "daily"}
#   timing: {"start_time": "08:00", "end_time": "09:00"}

# Domain C - habit_breakfast_prep:
#   schedule: {"frequency_type": "daily"}
#   timing: {"start_time": "08:45", "end_time": "09:15"}

# → CONFLICT: Same day (daily), overlapping times
# </Example>

# ---

# ## Resolution Strategies

# **Option 1: Shift Timing**
# Modify the habit's `timing` field to a different time slot.

# <Example>
# Patch to shift timing:
# {
#   "domain": "Domain A",
#   "window_id": "initial",
#   "path": "habits_state.habit_weekly_class.timing",
#   "operation": "replace",
#   "new_value": {"start_time": "20:00", "end_time": "21:00"},
#   "reason": "Shifted later in the evening to avoid conflict with team meeting"
# }
# </Example>

# **Option 2: Reduce Frequency**
# Modify the habit's `schedule.frequency_type` (and days if needed) to create space.

# <Example>
# Patch to reduce frequency:
# {
#   "domain": "Domain B",
#   "window_id": "initial",
#   "path": "habits_state.habit_team_meeting.schedule",
#   "operation": "replace",
#   "new_value": {"frequency_type": "biweekly", "days_of_week": [2], "start_date": "2024-01-03"},
#   "reason": "Reduced from weekly to bi-weekly to accommodate other Wednesday commitments"
# }
# </Example>

# **Option 3: Remove Lower-Priority Habit**
# If timing adjustment is impractical, remove the entire habit.

# <Example>
# Patch to remove habit:
# {
#   "domain": "Domain C",
#   "window_id": "initial",
#   "path": "habits_state.habit_breakfast_prep",
#   "operation": "remove",
#   "reason": "Removed lower-priority breakfast habit due to morning schedule conflicts"
# }
# </Example>

# ---

# =================================================================================
# CONFLICT TYPE 3: SCHEDULE OVERLOAD/UNREASONABLE
# =================================================================================

# ## Definition & Detection

# **Definition**: While activities don't directly overlap, the overall schedule is unrealistically packed

# **Detection Criteria:**

# ### 3a. Single Time Slot Overcrowding
# Multiple habits scheduled for the same general time period (e.g., "Saturday morning", "weekday evenings")

# <Example>
# Saturday morning habits across domains:
# - Domain A: habit_grocery_shopping (9:00-10:30 AM)
# - Domain B: habit_family_breakfast (9:00-11:00 AM)
# - Domain C: habit_soccer_practice (9:00-10:00 AM)
# - Domain D: habit_home_cleaning (8:00-10:00 AM)
# - Domain E: habit_yoga_class (9:30-10:30 AM)

# → UNREASONABLE: Five activities in a 2-hour window
# </Example>

# ### 3b. Daily/Weekly Time Budget Exhaustion
# Total time commitment leaves no room for:
# - Work/sleep (assume ~8 hours each for working adults)
# - Meals and basic routines
# - Buffer time and flexibility
# - Unscheduled downtime

# <Example>
# Daily habits totaling 16+ hours plus 8 hours sleep = 24 hours with ZERO buffer
# → UNREASONABLE
# </Example>

# ### 3c. Frequency Overlap Within Same Time Slot
# Multiple "daily" or high-frequency habits scheduled for the same time-of-day

# <Example>
# "Every weekday evening after work" across domains:
# - Domain A: gym_session (daily, 6:00-7:30 PM)
# - Domain B: online_course (3x/week, 6:30-8:00 PM)
# - Domain C: family_dinner_prep (daily, 6:00-7:00 PM)

# → UNREASONABLE: Combined pattern is implausible
# </Example>

# ---

# ## Resolution Strategies

# **Guiding Principle**: Ensure the profile represents a REALISTIC, SUSTAINABLE human lifestyle

# **Step 1: Assess Priority**
# Rank activities using user's basic profile and domain context:
# - Core needs (work, sleep, meals) > social commitments > hobbies
# - Health-critical activities > optional recreation
# - Recurring commitments > flexible activities

# **Step 2: Apply Thinning Strategy**

# **Option A: Reduce Frequency**
# <Example>
# Patches:
# [
#   {
#     "domain": "Health & Wellness",
#     "window_id": "initial",
#     "path": "habits_state.morning_run.schedule",
#     "operation": "replace",
#     "new_value": {"frequency_type": "weekly", "days_of_week": [0, 2, 4, 5]},
#     "reason": "Reduced from daily to 4x/week to create schedule space while keeping fixed days"
#   },
#   {
#     "domain": "Fitness & Exercise",
#     "window_id": "initial",
#     "path": "habits_state.yoga_class.schedule",
#     "operation": "replace",
#     "new_value": {"frequency_type": "weekly", "days_of_week": [1, 4]},
#     "reason": "Reduced from 3x/week to 2x/week due to overall schedule density"
#   }
# ]
# </Example>

# **Option B: Shift to Different Time Slots**
# <Example>
# Patches:
# [
#   {
#     "domain": "Home & Living",
#     "window_id": "initial",
#     "path": "habits_state.home_cleaning.timing",
#     "operation": "replace",
#     "new_value": {"start_time": "18:00", "end_time": "20:00"},
#     "reason": "Shifted from Saturday morning to Friday evening to reduce weekend congestion"
#   }
# ]
# </Example>

# **Option C: Remove Lower-Priority Habits**
# <Example>
# Patch:
# {
#   "domain": "Entertainment & Leisure",
#   "window_id": "initial",
#   "path": "habits_state.podcast_listening",
#   "operation": "remove",
#   "reason": "Removed low-priority habit due to weekday evening overload; user can listen during commute instead"
# }
# </Example>

# **Step 3: Validate Reasonableness**
# After adjustments, verify:
# - No single time slot has more than 2-3 activities per week
# - Daily total time commitment leaves at least 2-3 hours unscheduled buffer
# - At least 1-2 free evenings per week
# - Weekend includes some unstructured time

# ---

# =================================================================================
# CRITICAL: CASCADE CHANGES ACROSS WINDOWS
# =================================================================================

# Since this is a dynamic profile with temporal evolution, resolving a conflict in initial_state may affect subsequent time_windows. When generating patches, carefully consider the ripple effects:

# - If you adjust a habit's timing in `initial_state`, check if any time_windows have operations that reference that habit
# - If a time_window delta modifies the adjusted habit, verify the modification is still coherent
# - Ensure consistency: if you change "Tuesday 7pm" to "Monday 7pm" in initial_state, and w2 says "adjust timing to 8pm", the w2 operation should reflect "Monday 8pm" not "Tuesday 8pm"

# <Example>
# Initial state: habit_yoga timing = {"start_time": "19:00", "end_time": "20:00"} with schedule {"frequency_type": "weekly", "days_of_week": [1]}  // Tuesday
# Window w2: adjust habit_yoga timing delta = {"timing": {"start_time": "20:00", "end_time": "21:00"}}

# If you resolve a conflict by changing initial to Monday (schedule.days_of_week = [0]):
# → You may need to patch w2 to clarify the adjusted timing still applies on Monday
# → OR add a note to the resolution explaining the inherited context
# </Example>

# However, in most cases, delta operations inherit the day context from the initial state, so only the time portion changes. Be judicious about whether cascade patches are truly needed.

# =================================================================================
# CRITICAL REQUIREMENTS
# =================================================================================

# 1. **Window-specific**: Each patch must specify exact window_id ("initial" for initial_state, "w1", "w2", etc. for time_windows)
# 2. **Minimal but sufficient changes**: Only patch what's necessary, but ensure profile is livable
# 3. **Preserve structure**: Don't change data types (string→string, array→array)
# 4. **Cascade awareness**: When modifying habits in initial_state, check and update references in subsequent time_windows if necessary
# 5. **One patch per change**: Don't combine multiple operations in one patch
# 6. **Clear reasoning**: Always include "reason" field explaining the resolution
# 7. **Holistic validation**: After resolving individual conflicts, validate that the overall schedule is reasonable
# 8. **Correct paths**: Use the new structure paths:
#    - Singular: `user_attributes_state.singular.<attr_name>`
#    - Collections: `user_attributes_state.collections.<collection_name>[<index>]`
#    - Habits: `habits_state.<habit_name>.<field>`

# =================================================================================
# PATCH FORMAT GUIDE
# =================================================================================

# **Path format:**
# Use dot notation with array indices where applicable:
# - For singular attributes: `"user_attributes_state.singular.primary_job"`
# - For collection items: `"user_attributes_state.collections.owned_devices[2]"` (to target specific item)
# - For habit fields: `"habits_state.morning_jog.timing"`
# - For entire habit: `"habits_state.morning_jog"` (to remove entire habit)

# **Action types:**

# 1. **"remove"** - Delete an element from array or remove a key from object
   
#    Remove item from collection by index:
#    {
#      "path": "user_attributes_state.collections.owned_devices[1]",
#      "operation": "remove",
#      "reason": "Remove contradictory device"
#    }
   
#    Remove entire habit:
#    {
#      "path": "habits_state.habit_name",
#      "operation": "remove",
#      "reason": "Remove lower-priority habit due to schedule conflict"
#    }

# 2. **"replace"** - Replace an existing value
   
#    Replace singular attribute value:
#    {
#      "path": "user_attributes_state.singular.primary_job",
#      "operation": "replace",
#      "new_value": "Director of Product Management",
#      "reason": "Align to canonical job title from Work domain"
#    }
   
#    Replace habit timing:
#    {
#      "path": "habits_state.morning_jog.timing",
#      "operation": "replace",
#      "new_value": "6:00-7:00 AM on weekdays",
#      "reason": "Shifted timing to avoid conflict with commute"
#    }

# 3. **"append"** - Add to the end of an array (rarely used in conflict resolution)
#    {
#      "path": "user_attributes_state.collections.owned_devices",
#      "operation": "append",
#      "new_value": "iPad Pro (tablet for work)",
#      "reason": "Add missing device for completeness"
#    }

# =================================================================================
# OUTPUT FORMAT (STRICT JSON)
# =================================================================================

# {
#   "conflicts_and_resolutions": [
#     {
#       "conflict": {
#         "type": "singular_conflict" | "collection_item_conflict" | "temporal_collision" | "schedule_overload",
#         "description": "<brief description of the conflict>",
#         "involved_data": [
#           {
#             "domain": "<domain_name>",
#             "window_id": "initial" | "w1" | "w2" | ...,
#             "path": "<full path to the conflicting element>",
#             "value": "<current value or description>"
#           },
#           ...
#         ]
#       },
#       "resolution": {
#         "strategy": "unify_singular_value" | "delete_contradictory_items" | "delete_overlapping_items" | "adjust_timing" | "reduce_frequency" | "drop_habit",
#         "explanation": "<detailed explanation of why this resolution was chosen>",
#         "patches": [
#           {
#             "domain": "<domain_name>",
#             "window_id": "initial" | "w1" | "w2" | ...,
#             "path": "<path to the element to modify>",
#             "operation": "remove" | "replace" | "append",
#             "new_value": <new value, if operation is replace or append>,
#             "reason": "<specific reason for this patch>"
#           },
#           ...
#         ]
#       }
#     },
#     ...
#   ]
# }

# **Example output:**

# {
#   "conflicts_and_resolutions": [
#     {
#       "conflict": {
#         "type": "collection_item_conflict",
#         "description": "Contradictory smartphones found in Technology and Work domains",
#         "involved_data": [
#           {
#             "domain": "Technology & Digital Life",
#             "window_id": "initial",
#             "path": "user_attributes_state.collections.owned_devices[0]",
#             "value": "iPhone 14 (smartphone for daily communication)"
#           },
#           {
#             "domain": "Work & Education",
#             "window_id": "initial",
#             "path": "user_attributes_state.collections.work_devices[0]",
#             "value": "Google Pixel 7 (Android phone for work)"
#           }
#         ]
#       },
#       "resolution": {
#         "strategy": "delete_contradictory_items",
#         "explanation": "User's basic profile indicates preference for Apple ecosystem. Keep iPhone 14 as primary phone and remove the contradictory Google Pixel 7 from work devices.",
#         "patches": [
#           {
#             "domain": "Work & Education",
#             "window_id": "initial",
#             "path": "user_attributes_state.collections.work_devices[0]",
#             "operation": "remove",
#             "reason": "Remove contradictory phone - user has iPhone 14 as primary smartphone in Technology domain"
#           }
#         ]
#       }
#     }
#   ]
# }

# Here are the temporal conflicts detected by code that you must resolve:
# {{ detected_temporal_conflicts_json }}
# """)

# {
#       "conflict": {
#         "type": "temporal_collision",
#         "description": "Tuesday 7pm time slot conflict between coding practice and cycling",
#         "involved_data": [
#           {
#             "domain": "Work & Education",
#             "window_id": "initial",
#             "path": "habits_state.weekly_technical_upskilling.timing",
#             "value": "Tuesday and Thursday evenings from 7:00-8:00 PM"
#           },
#           {
#             "domain": "Health & Self-care",
#             "window_id": "initial",
#             "path": "habits_state.indoor_cycling_sessions.timing",
#             "value": "after work around 7:00 PM"
#           }
#         ]
#       },
#       "resolution": {
#         "strategy": "adjust_timing",
#         "patches": [
#           {
#             "domain": "Health & Self-care",
#             "window_id": "initial",
#             "path": "habits_state.indoor_cycling_sessions.timing",
#             "operation": "update",
#             "old_value": "after work around 7:00 PM",
#             "new_value": "after work around 8:00 PM",
#             "reason": "Shifted to 8pm to avoid Tuesday collision with Work & Education's coding practice at 7pm"
#           }
#         ]
#       }
#     },
#     {
#       "conflict": {
#         "type": "temporal_collision",
#         "description": "Cross-window conflict: Domain A's w1 habit overlaps with Domain B's initial habit",
#         "involved_data": [
#           {
#             "domain": "Social & Community",
#             "window_id": "w1",
#             "path": "habits_state.w1.evening_networking_event.timing",
#             "value": "Every Wednesday 7:00-9:00 PM"
#           },
#           {
#             "domain": "Health & Self-care",
#             "window_id": "initial",
#             "path": "habits_state.yoga_class.timing",
#             "value": "Weekly Wednesday 7:30 PM"
#           }
#         ]
#       },
#       "resolution": {
#         "strategy": "adjust_timing",
#         "patches": [
#           {
#             "domain": "Social & Community",
#             "window_id": "w1",
#             "path": "habits_state.w1.evening_networking_event.timing",
#             "operation": "update",
#             "old_value": "Every Wednesday 7:00-9:00 PM",
#             "new_value": "Every Thursday 7:00-9:00 PM",
#             "reason": "Moved networking to Thursday to preserve Health & Self-care's established yoga routine"
#           }
#         ]
#       }
#     }
#   ]
# }

# **Field Specifications**:
# - `start`/`end`: ISO 8601 datetime format
# - `loc`: City/region code for STAY; use "TRAVEL" literal for TRAVEL segments
# - `kind`: Must be either "STAY" or "TRAVEL"
# - `from`/`to`: Required for TRAVEL, omit for STAY
# - `home_base`: The user's primary residence location

# **Example**:
# {
#   "whereabouts": [
#     {
#       "start": "2024-01-15T08:00",
#       "end": "2024-01-20T18:00",
#       "loc": "CN.SHANGHAI",
#       "kind": "STAY"
#     }
#   ],
#   "home_base": "CN.SHANGHAI"
# }

TIME_CONFLICT_RESOLUTION_PROMPT = Template("""You are a focused temporal conflict resolver for user dynamic profiles.

=================================================================================
TASK DESCRIPTION
=================================================================================

**Input:**
You will receive:
1. Conflict information - a list of conflicts with automatic ID, messages, and focus habit designation
2. User basic profile - user's background and context
3. Dynamic profiles - complete user dynamic profiles (all windows) that need patching

**Goal:**
Fix timing conflicts in the CURRENT window/state by generating patches to modify the **focus habit only**. You must:
- Each conflict entry has "focus_habit" and "conflicting_habit" fields
- The focus habit is automatically selected as the one with the MOST conflicts in this window
- You MUST ONLY modify the focus_habit - DO NOT edit the conflicting_habit
- The conflict "message" field provides a clear description including locations and conflict type
- Analyze all conflicts and generate patches to resolve them
- Ensure semantic consistency with existing narratives

**Important Context:**
- The focus habit is already selected for you (the habit with the most conflicts)
- Each conflict has an ID (e.g., "cross_domain_000") that you must reference in your fix
- You receive the FULL dynamic profiles (initial_state + all time_windows), but your task is to resolve conflicts ONLY in the current window
- When modifying habits, ensure your timing adjustments align with existing narrative context (operation reasons, window summaries)

=================================================================================
DEFINITIONS
=================================================================================

**Time Format:**
- All times use 24-hour format (HH:MM)
- timing field: {"start_time": "06:30", "end_time": "07:00"}
- Duration (end - start) MUST NOT exceed 3 hours

**Location-Based Conflict Rules:**
- **Same location**: If two habits are at the SAME location, they conflict if their times overlap at all
- **Different locations**: If two habits are at DIFFERENT locations, they need at least 30 minutes gap between them for travel time
  - Gap is measured from the end of the first habit to the start of the second habit
  - Example: Habit A ends at 9:00 AM at Location X, Habit B starts at 9:20 AM at Location Y → CONFLICT (only 20 min gap)
  - Example: Habit A ends at 9:00 AM at Location X, Habit B starts at 9:30 AM at Location Y → OK (30 min gap)
- **No location specified**: Treat as same location (time overlap = conflict)

**Schedule Format:**
Must use one of these standardized formats:

1. Daily: {"frequency_type": "daily"}

2. Weekly: {"frequency_type": "weekly", "days_of_week": [<integers 0-6>]}
   - 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
   - Must specify exact days (e.g., [1,3,5] for Tue/Thu/Sat, [5,6] for weekends)

3. Biweekly: {"frequency_type": "biweekly", "days_of_week": [<single integer 0-6>], "start_date": "YYYY-MM-DD"}
   - start_date: first occurrence, then repeats every 2 weeks

4. Monthly by date: {"frequency_type": "monthly_by_date", "days_of_month": [<integers 1-28>]}
   - Avoid 29-31 to prevent skipping months
   - Can specify multiple dates (e.g., [1,15] for 1st and 15th)

5. Monthly by nth weekday: {"frequency_type": "monthly_nth_weekday", "week_of_month": <1-4 or "last">, "day_of_week": <0-6>}
   - week_of_month: 1=first week, 2=second week, "last"=last week

**Patch Actions:**
- replace: Replace value at path (most common)
- remove: Delete the key/value or array element at path
- append: Add to an array (rarely needed)
- add_key: Add new key to object (rarely needed)

**Patch Fields:**
Each patch must include:
- domain: Domain name from focus_habit
- path: Dot-notation path to target location (see Path Construction below)
- action: One of the actions above
- value: New value to set (for replace, append, add_key)
- key: Key name (only for add_key action)
- fix_reason: Technical explanation of why THIS specific patch is needed

**Path Construction:**
1. For habits in initial_state:
   Format: `initial_state.habits_state.<habit_name>.<field>`
   Example: `initial_state.habits_state.morning_jog.timing.start_time`

2. For operations in time windows:
   Format: `time_windows[window_id=<id>].habits_delta.operations[<index>].<field>`
   Example: `time_windows[window_id=w3].habits_delta.operations[0].delta.timing.start_time`
   To remove entire operation: `time_windows[window_id=w3].habits_delta.operations[0]`

3. For window-level fields:
   Summary: `time_windows[window_id=<id>].summary`
   Description: `time_windows[window_id=<id>].window_description`

**Two Types of "reason" Fields:**
1. patch.fix_reason - Technical explanation of why this patch is needed
   Example: "Move dinner_prep start earlier by 45 minutes to avoid 19:30-20:00 overlap with evening_yoga"

2. operation.reason (the value being patched) - Narrative justification for the habit change in user's life
   Example: "Started meal prep earlier (18:30-19:15) to accommodate evening yoga schedule"
   When patching operation.reason, you're updating the story to match the new timing

=================================================================================
INPUT DATA
=================================================================================

**Conflict Information:**
{{ conflict_json }}

**User Basic Profile:**
{{ user_basic_profile_json }}

**Dynamic Profiles (Complete - all windows):**
{{ dynamic_profiles_json }}

=================================================================================
RESOLUTION STRATEGIES & CONSIDERATIONS
=================================================================================

**Core Resolution Principles:**
1. **Single Focus**: Only modify the focus_habit. Do NOT edit conflicting_habit
2. **Complete Resolution**: Clear ALL conflicts for the focus habit within the current window
3. **Minimal Changes**: Prefer the smallest adjustment that resolves all conflicts
4. **Current Window Scope**: Only patch the current window (initial_state or the specified time_window)
5. **Schema Integrity**: Keep data types and schemas intact
6. **Semantic Consistency**: Update operation.reason and window.summary when timing changes would contradict existing narratives
7. **Location Awareness**: When adjusting timing, respect location-based conflict rules:
   - For same location: ensure no time overlap
   - For different locations: ensure at least 30-minute gap for travel time
8. **Use Conflict IDs**: Reference the conflict ID in your fix_description to clearly identify which conflicts are being resolved

**Resolution Strategies:**

1. **adjust_schedule_and_timing** (STRONGLY PREFERRED):
   - Modify the habit's timing (start_time/end_time) to avoid conflicts
   - Adjust the habit's schedule (frequency, days_of_week) to avoid conflicts
   - Can adjust both timing and schedule together if needed
   - Examples:
     * Move to different time of day (earlier/later)
     * Change to different days of week
     * Reduce frequency (daily → 3x/week, weekly → monthly)

2. **drop_habit** (USE WITH EXTREME CAUTION):
   - Remove the habit entirely from the current window
   - **CRITICAL CASCADE WARNING**: Dropping a habit creates cascade effects in later windows
     * If you drop a habit in initial_state or early window, later windows may have operations (adjust/drop) that reference the now-deleted habit
     * These orphaned operations will become invalid and cause errors
   - **ONLY use when**:
     * The habit is truly incompatible and cannot be rescheduled
     * The habit represents a failed short-term experiment
     * A permanent life change makes the habit impossible
   - **IF you must drop**:
     * Clearly document in fix_description that this habit is permanently removed
     * Search the FULL dynamic profiles for ALL operations in later windows that reference this habit
     * Remove or update ALL such operations and their related summaries to prevent cascade errors

3. **other**:
   - Any other creative solution that resolves the conflict while maintaining data integrity

**Cascade Effect Handling:**
When using drop_habit strategy, you MUST:
- Scan ALL time_windows (not just current) in the dynamic profiles
- Identify ALL operations that reference the dropped habit
- Remove those operations (they become invalid once the habit is dropped)
- Update ALL affected window summaries to remove references to the dropped habit
- Document the cascade effect clearly in fix_description

**Important Notes:**
- When modifying operations in a time_window, also update that window's summary field
- If timing changes conflict with existing operation.reason, update the reason field to maintain coherence
- Prefer timing adjustments that naturally fit the existing narrative

=================================================================================
EXAMPLES
=================================================================================

**Example 1: Adjusting timing in initial_state**

Conflict scenario:
- morning_jog: 7:00-7:30 AM daily
- breakfast_routine: 7:15-7:45 AM daily
- Overlap: 7:15-7:30 AM every day
- Focus habit: morning_jog

Resolution:
{
    "strategy": "adjust_schedule_and_timing",
    "fix_description": "Shifted morning_jog from 7:00-7:30 to 6:30-7:00 AM to avoid overlap with breakfast_routine (7:15-7:45 AM)",
    "patches": [
        {
            "domain": "Health & Self-care",
            "path": "initial_state.habits_state.morning_jog.timing.start_time",
            "action": "replace",
            "value": "06:30",
            "fix_reason": "Move jog start time 30 minutes earlier to end before breakfast begins at 7:15"
        },
        {
            "domain": "Health & Self-care",
            "path": "initial_state.habits_state.morning_jog.timing.end_time",
            "action": "replace",
            "value": "07:00",
            "fix_reason": "Adjust end time to maintain 30-minute duration and eliminate overlap with breakfast"
        }
    ]
}

**Example 2: Adjusting time window operation with narrative updates**

Conflict scenario (window w2):
- Operation[0]: adjusts evening_coding_session to 19:00-21:00 (priority: medium, reason: "personal project sprint")
- Operation[1]: acquires family_dinner at 19:30-20:00 (priority: high)
- Overlap: 19:30-20:00
- Focus habit: evening_coding_session

Resolution:
{
    "strategy": "adjust_schedule_and_timing",
    "fix_description": "Moved evening_coding_session in w2 from 19:00-21:00 to 20:00-22:00 to clear overlap with family_dinner (19:30-20:00). Updated reason and summary to reflect the new timing.",
    "patches": [
        {
            "domain": "Work & Education",
            "path": "time_windows[window_id=w2].habits_delta.operations[0].delta.timing.start_time",
            "action": "replace",
            "value": "20:00",
            "fix_reason": "Push coding session later so dinner (19:30-20:00) finishes before it begins"
        },
        {
            "domain": "Work & Education",
            "path": "time_windows[window_id=w2].habits_delta.operations[0].delta.timing.end_time",
            "action": "replace",
            "value": "22:00",
            "fix_reason": "Preserve two-hour duration while starting after dinner"
        },
        {
            "domain": "Work & Education",
            "path": "time_windows[window_id=w2].habits_delta.operations[0].reason",
            "action": "replace",
            "value": "Shifted coding to 20:00-22:00 to avoid clashing with family dinner while keeping the nightly project block intact.",
            "fix_reason": "Align operation.reason with the new timing and conflict rationale"
        },
        {
            "domain": "Work & Education",
            "path": "time_windows[window_id=w2].summary",
            "action": "replace",
            "value": "Adjusted evening coding to 20:00-22:00 to keep family dinner uninterrupted while continuing the personal project sprint.",
            "fix_reason": "Keep the window summary consistent with the updated schedule"
        }
    ]
}

**Example 3: drop_habit with cascade effect handling**

Conflict scenario:
- Current window w1: morning_jog conflicts with new early_meeting (irreconcilable)
- Later window w2: has operation[1] to adjust morning_jog timing
- Later window w3: has operation[0] to drop morning_jog (seasonal change)
- Focus habit: morning_jog in w1

Resolution (with cascade handling):
{
    "strategy": "drop_habit",
    "fix_description": "Dropped morning_jog from w1 due to irreconcilable conflict with new early_meeting schedule. CASCADE HANDLING: Removed invalid operations in w2 (adjust morning_jog) and w3 (drop morning_jog) since the habit no longer exists after w1. Updated all affected window summaries.",
    "patches": [
        {
            "domain": "Health & Self-care",
            "path": "time_windows[window_id=w1].habits_delta.operations[2]",
            "action": "remove",
            "fix_reason": "Remove the acquire operation that introduced morning_jog, eliminating the habit from this window forward"
        },
        {
            "domain": "Health & Self-care",
            "path": "time_windows[window_id=w1].summary",
            "action": "replace",
            "value": "Removed morning_jog due to new early meeting schedule conflict. Could not find alternative time slot.",
            "fix_reason": "Update w1 summary to reflect the habit removal"
        },
        {
            "domain": "Health & Self-care",
            "path": "time_windows[window_id=w2].habits_delta.operations[1]",
            "action": "remove",
            "fix_reason": "CASCADE EFFECT: Remove invalid adjust operation for morning_jog since it was dropped in w1 and no longer exists"
        },
        {
            "domain": "Health & Self-care",
            "path": "time_windows[window_id=w2].summary",
            "action": "replace",
            "value": "Maintained regular fitness routine with afternoon workouts (morning_jog was discontinued in previous month).",
            "fix_reason": "Update w2 summary to remove reference to morning_jog timing adjustment"
        },
        {
            "domain": "Health & Self-care",
            "path": "time_windows[window_id=w3].habits_delta.operations[0]",
            "action": "remove",
            "fix_reason": "CASCADE EFFECT: Remove invalid drop operation for morning_jog since it was already dropped in w1"
        },
        {
            "domain": "Health & Self-care",
            "path": "time_windows[window_id=w3].summary",
            "action": "replace",
            "value": "Continued afternoon fitness schedule without changes this month.",
            "fix_reason": "Update w3 summary to remove reference to dropping morning_jog (already gone)"
        }
    ]
}

**KEY TAKEAWAY from Example 3**: When dropping a habit, scan the FULL dynamic profiles (all windows) and remove/update ALL operations that reference the dropped habit. This is why adjust_schedule_and_timing is STRONGLY PREFERRED.

=================================================================================
OUTPUT FORMAT
=================================================================================

Return JSON with this EXACT structure:

{
    "strategy": "adjust_schedule_and_timing | drop_habit | other",
    "fix_description": "<string: describe what you changed and why>",
    "patches": [
        {
            "domain": "<string: domain name from focus_habit>",
            "path": "<string: dot path to target location>",
            "action": "<string: replace | append | remove | add_key>",
            "key": "<string: only for add_key action>",
            "value": <any: new value>,
            "fix_reason": "<string: technical explanation of why THIS specific patch is needed>"
        }
    ]
}

""")



# def render_conflict_resolution_prompt(request: ConflictResolutionRequest) -> str:
#     user_basic_profile_json = json.dumps(
#         request.user_basic_profile or {}, indent=2, ensure_ascii=False
#     )
#     dynamic_profiles_json = json.dumps(
#         request.dynamic_profiles, indent=2, ensure_ascii=False
#     )
#     detected_attribute_conflicts_json = json.dumps(
#         request.detected_attribute_conflicts or [], indent=2, ensure_ascii=False
#     )
#     attribute_conflicts_note = (
#         "Attribute conflicts have already been resolved upstream; only touch attributes if timing fixes truly require it."
#         if not request.detected_attribute_conflicts
#         else "Resolve any remaining attribute conflicts if present, then handle temporal issues."
#     )
#     detected_temporal_conflicts_json = json.dumps(
#         request.detected_temporal_conflicts or [], indent=2, ensure_ascii=False
#     )
#     target_window_id = request.target_window_id
#     return CONFLICT_RESOLUTION_PROMPT.render(
#         user_basic_profile_json=user_basic_profile_json,
#         dynamic_profiles_json=dynamic_profiles_json,
#         target_window_id=target_window_id,
#     )


# def generate_conflict_resolution(
#     llm_client: GeminiJSONClient, request: ConflictResolutionRequest
# ) -> LLMResult:
#     prompt = render_conflict_resolution_prompt(request)
#     return llm_client.generate_json(prompt)


def _normalize_window_id_label(window_id: object) -> str:
    """
    Normalize window identifiers for consistent comparisons (e.g., initial_state -> initial).
    """
    if window_id is None:
        return "initial"
    normalized = str(window_id).strip()
    lower = normalized.lower()
    if lower in {"", "initial", "initial_state", "initialstate", "init"}:
        return "initial"
    return normalized


def _filter_dynamic_profiles_for_window(
    dynamic_profiles: Dict[str, Dict],
    target_window_id: str | None,
) -> Dict[str, Dict]:
    """
    Limit dynamic profile context to initial_state plus the target window to keep prompts focused.
    Falls back to the full profiles if filtering would result in an empty object.
    """
    normalized_target = _normalize_window_id_label(target_window_id)
    filtered: Dict[str, Dict] = {}
    for domain_name, profile in (dynamic_profiles or {}).items():
        if not isinstance(profile, dict):
            continue
        domain_entry: Dict[str, object] = {}
        if isinstance(profile.get("initial_state"), dict):
            domain_entry["initial_state"] = deepcopy(profile["initial_state"])
        if normalized_target != "initial":
            for window in profile.get("time_windows") or []:
                if _normalize_window_id_label(window.get("window_id")) == normalized_target:
                    domain_entry.setdefault("time_windows", []).append(deepcopy(window))
        if domain_entry:
            filtered[domain_name] = domain_entry
    return filtered or deepcopy(dynamic_profiles)


def render_time_conflict_resolution_prompt(
    request: ConflictResolutionRequest, *, iteration_index: int,
) -> str:
    user_basic_profile_json = json.dumps(
        request.user_basic_profile or {}, indent=2, ensure_ascii=False
    )

    # Use full dynamic profiles instead of filtering
    dynamic_profiles_json = json.dumps(
        request.dynamic_profiles, indent=2, ensure_ascii=False
    )

    # Extract conflict data from request
    conflict_data = request.detected_temporal_conflicts or {}
    all_conflicts = conflict_data.get("conflicts") or []
    graph_by_window = conflict_data.get("graph_by_window") or {}

    # Get target window
    window_id = _normalize_window_id_label(request.target_window_id or "initial")

    # Filter conflicts for this window only
    window_conflicts = [
        c for c in all_conflicts
        if _normalize_window_id_label(c.get("window_id") or "") == window_id
    ]

    if not window_conflicts:
        # No conflicts in this window, return empty structure
        conflict_json = json.dumps({"conflicts": []}, indent=2, ensure_ascii=False)
        return TIME_CONFLICT_RESOLUTION_PROMPT.render(
            user_basic_profile_json=user_basic_profile_json,
            dynamic_profiles_json=dynamic_profiles_json,
            conflict_json=conflict_json,
            target_window_id=window_id,
        )

    # Build conflict degree map: count how many conflicts each habit has
    conflict_degrees: Dict[Tuple[str, str], int] = {}  # (domain, habit) -> count
    for conflict in window_conflicts:
        habit_a = conflict.get("habit_a") or {}
        habit_b = conflict.get("habit_b") or {}

        key_a = (habit_a.get("domain", ""), habit_a.get("habit", ""))
        key_b = (habit_b.get("domain", ""), habit_b.get("habit", ""))

        occ = conflict.get("occurrences", 1)
        conflict_degrees[key_a] = conflict_degrees.get(key_a, 0) + occ
        conflict_degrees[key_b] = conflict_degrees.get(key_b, 0) + occ

    # Select the habit with the most conflicts as focus_habit
    if conflict_degrees:
        focus_key = max(conflict_degrees.items(), key=lambda x: x[1])[0]
        focus_domain, focus_habit_name = focus_key
    else:
        # Fallback: use first habit from first conflict
        first_conflict = window_conflicts[0]
        habit_a = first_conflict.get("habit_a") or {}
        focus_domain = habit_a.get("domain", "")
        focus_habit_name = habit_a.get("habit", "")
        focus_key = (focus_domain, focus_habit_name)

    # Restructure conflicts with focus_habit and conflicting_habit
    restructured_conflicts = []
    for idx, conflict in enumerate(window_conflicts):
        # Determine which habit is the focus
        habit_a = conflict.get("habit_a") or {}
        habit_b = conflict.get("habit_b") or {}

        key_a = (habit_a.get("domain", ""), habit_a.get("habit", ""))
        key_b = (habit_b.get("domain", ""), habit_b.get("habit", ""))

        if key_a == focus_key:
            focus_info = habit_a
            other_info = habit_b
        else:
            focus_info = habit_b
            other_info = habit_a

        # Generate automatic message
        focus_habit_name = focus_info.get("habit", "")
        other_habit_name = other_info.get("habit", "")
        focus_timing = focus_info.get("timing", "")
        other_timing = other_info.get("timing", "")
        focus_domain_str = focus_info.get("domain", "")
        other_domain = other_info.get("domain", "")

        loc_focus = focus_info.get("location", "")
        loc_other = other_info.get("location", "")
        conflict_type = conflict.get("conflict_type", "")
        occ_count = conflict.get("occurrences", 1)
        window_id_str = conflict.get("window_id", "")

        # Build location description
        loc_desc = ""
        if conflict_type == "same_location_time_overlap":
            loc_desc = f" (both at '{loc_focus}')"
        elif conflict_type == "different_location_insufficient_gap":
            loc_desc = f" ('{focus_habit_name}' at '{loc_focus}', '{other_habit_name}' at '{loc_other}' - insufficient travel time)"
        elif loc_focus or loc_other:
            loc_desc = f" ('{focus_habit_name}' at '{loc_focus}', '{other_habit_name}' at '{loc_other}')"

        conflict_id = f"cross_domain_{idx:03d}"
        message = (
            f"[ID: {conflict_id}] Time conflict in {window_id_str}: "
            f"[FOCUS] '{focus_habit_name}' ({focus_domain_str}, {focus_timing}) conflicts with "
            f"'{other_habit_name}' ({other_domain}, {other_timing}){loc_desc}. "
            f"Occurs {occ_count} time(s). You must adjust the FOCUS habit only."
        )

        # Create restructured conflict with focus_habit and conflicting_habit
        restructured_conflict = {
            "id": conflict_id,
            "window_id": window_id_str,
            "window_range": conflict.get("window_range"),
            "focus_habit": focus_info,
            "conflicting_habit": other_info,
            "conflict_type": conflict_type,
            "occurrences": occ_count,
            "overlap_examples": conflict.get("overlap_examples", []),
            "message": message,
        }
        restructured_conflicts.append(restructured_conflict)

    # Construct the conflict JSON structure (similar to rule5)
    conflict_structured = {
        "conflicts": restructured_conflicts,
        "focus_habit_info": {
            "domain": focus_domain,
            "habit_name": focus_habit_name,
            "total_conflicts": conflict_degrees.get(focus_key, 0),
            "note": "This is the habit with the most conflicts. You MUST adjust this habit to resolve all conflicts."
        }
    }

    conflict_json = json.dumps(conflict_structured, indent=2, ensure_ascii=False)

    return TIME_CONFLICT_RESOLUTION_PROMPT.render(
        user_basic_profile_json=user_basic_profile_json,
        dynamic_profiles_json=dynamic_profiles_json,
        conflict_json=conflict_json,
        target_window_id=window_id,
    )


def generate_time_conflict_resolution(
    llm_client: GeminiJSONClient,
    request: ConflictResolutionRequest,
    *,
    iteration_index: int,
) -> LLMResult:
    prompt = render_time_conflict_resolution_prompt(request, iteration_index=iteration_index)
    return llm_client.generate_json(prompt)


def render_attribute_conflict_resolution_prompt(
    request: ConflictResolutionRequest,
) -> str:
    user_basic_profile_json = json.dumps(
        request.user_basic_profile or {}, indent=2, ensure_ascii=False
    )
    dynamic_profiles_json = json.dumps(
        request.dynamic_profiles, indent=2, ensure_ascii=False
    )
    detected_attribute_conflicts_json = json.dumps(
        request.detected_attribute_conflicts or [], indent=2, ensure_ascii=False
    )
    return ATTRIBUTE_CONFLICT_RESOLUTION_PROMPT.render(
        user_basic_profile_json=user_basic_profile_json,
        dynamic_profiles_json=dynamic_profiles_json,
        detected_attribute_conflicts_json=detected_attribute_conflicts_json,
    )


def generate_attribute_conflict_resolution(
    llm_client: GeminiJSONClient, request: ConflictResolutionRequest
) -> LLMResult:
    prompt = render_attribute_conflict_resolution_prompt(request)
    return llm_client.generate_json(prompt)


def render_life_context_baseline_prompt(
    request: LifeContextBaselineRequest,
) -> str:
    user_basic_profile_json = json.dumps(
        request.user_basic_profile, indent=2, ensure_ascii=False
    )
    dynamic_profiles_initial_state_json = json.dumps(
        request.dynamic_profiles_initial_state, indent=2, ensure_ascii=False
    )
    return Life_Context_Baseline_Prompt.render(
        user_basic_profile_json=user_basic_profile_json,
        dynamic_profiles_initial_state_json=dynamic_profiles_initial_state_json,
    )


def generate_life_context_baseline(
    llm_client: GeminiJSONClient, request: LifeContextBaselineRequest
) -> LLMResult:
    prompt = render_life_context_baseline_prompt(request)
    return llm_client.generate_json(prompt)


def render_life_context_delta_prompt(request: LifeContextDeltaRequest) -> str:
    user_basic_profile_json = json.dumps(
        request.user_basic_profile, indent=2, ensure_ascii=False
    )
    life_context_baseline_json = json.dumps(
        request.life_context_baseline or {}, indent=2, ensure_ascii=False
    )
    window_summary = request.window_profile_summary or ""
    if request.time_range and isinstance(request.time_range, (list, tuple)):
        try:
            start, end = request.time_range
            prefix = f"[{request.window_id}] {start} to {end}"
            window_summary = f"{prefix} | {window_summary}" if window_summary else prefix
        except ValueError:
            # fallback if time_range isn't a 2-tuple/list
            window_summary = f"[{request.window_id}] {request.time_range} | {window_summary}"
    return Life_Context_Delta_Prompt.render(
        user_basic_profile_json=user_basic_profile_json,
        life_context_baseline_json=life_context_baseline_json,
        window_description=request.window_description,
        window_summary=window_summary,
    )


def generate_life_context_delta(
    llm_client: GeminiJSONClient, request: LifeContextDeltaRequest
) -> LLMResult:
    prompt = render_life_context_delta_prompt(request)
    return llm_client.generate_json(prompt)


def render_spatiotemporal_constraints_prompt(
    request: SpatiotemporalConstraintsRequest,
) -> str:
    user_basic_profile_json = json.dumps(
        request.user_basic_profile, indent=2, ensure_ascii=False
    )
    return SPATIOTEMPORAL_CONSTRAINTS_PROMPT.render(
        time_range=request.time_range,
        user_basic_profile_json=user_basic_profile_json,
        window_profile_summary=request.window_profile_summary,
        window_description=request.window_description,
        window_world_background=request.world_background,
    )


def generate_spatiotemporal_constraints(
    llm_client: GeminiJSONClient, request: SpatiotemporalConstraintsRequest
) -> LLMResult:
    prompt = render_spatiotemporal_constraints_prompt(request)
    return llm_client.generate_json(prompt)


@dataclass
class ConflictResolutionRequest:
    user_basic_profile: Dict | None
    dynamic_profiles: Dict[str, Dict]
    detected_attribute_conflicts: Dict[str, object] | List[Dict[str, object]] | None = None
    detected_temporal_conflicts: Dict[str, object] | List[Dict[str, object]] | None = None
    target_window_id: str | None = None


@dataclass
class KeyAlignmentRequest:
    dynamic_profiles: Dict[str, Dict]
    user_basic_profile: Dict | None = None

KEY_ALIGNMENT_PROMPT = Template("""Normalize attribute key names across domains by identifying semantically identical keys.

Input dynamic profiles (initial + deltas):
{{ dynamic_profiles_json }}

Basic user profile:
{{ user_basic_profile_json }}

**IMPORTANT**: User attributes are structured as:
- user_attributes_state.singular: single-value attributes (e.g., primary_residence, primary_job)
- user_attributes_state.collections: multi-item collections (e.g., owned_devices, active_subscriptions)

Task:
Identify keys that represent the SAME concept across different domains and map them to ONE canonical key name.

Alignment Rules:
1. Keys are "semantically identical" if they track the exact same attribute (e.g., "primary_job" in one domain and "main_occupation" in another)
2. Prefer the shortest, most common key name as canonical
3. DO NOT merge keys that are similar but distinct (e.g., "primary_residence" vs "secondary_residence")
4. If domains have different attribute types (singular vs collection) for same concept, note this in the mapping
5. When specifying original_key, use the full path including attribute_type (e.g., "singular.primary_job" or "collections.owned_devices")

Examples of what TO align:
- Domain A has "singular.primary_vehicle", Domain B has "singular.main_car" → canonical: "primary_vehicle"
- Domain A has "collections.electronic_devices", Domain B has "collections.owned_devices" → canonical: "owned_devices"

Examples of what NOT to align:
- "singular.primary_residence" vs "collections.owned_properties" (different semantics: one primary vs multiple items)
- "collections.close_friends" vs "collections.professional_contacts" (different relationship types)

Output format (JSON only):
{
  "canonical_key_mappings": {
    "<canonical_key>": {
      "description": "<brief description of what this represents>",
      "attribute_type": "singular" | "collections",
      "domains": [
        {
          "domain": "<domain_name>",
          "original_key": "<attribute_type>.<key_name>",
          "note": "<optional: note if data type differs or other special cases>"
        }
      ]
    }
  }
}

Example output:
{
  "canonical_key_mappings": {
    "primary_vehicle": {
      "description": "User's main vehicle for daily transportation",
      "attribute_type": "singular",
      "domains": [
        {"domain": "Transportation & Mobility", "original_key": "singular.primary_vehicle"},
        {"domain": "Finances & Material Living", "original_key": "singular.main_car"}
      ]
    },
    "owned_devices": {
      "description": "Electronic devices owned by the user",
      "attribute_type": "collections",
      "domains": [
        {"domain": "Technology & Digital Life", "original_key": "collections.owned_devices"},
        {"domain": "Finances & Material Living", "original_key": "collections.electronic_devices"}
      ]
    }
  }
}
""")




def generate_key_alignment(
    llm_client: GeminiJSONClient, request: KeyAlignmentRequest
) -> LLMResult:
    user_basic_profile_json = json.dumps(
        request.user_basic_profile or {}, indent=2, ensure_ascii=False
    )
    dynamic_profiles_json = json.dumps(
        request.dynamic_profiles, indent=2, ensure_ascii=False
    )
    prompt = KEY_ALIGNMENT_PROMPT.render(
        dynamic_profiles_json=dynamic_profiles_json,
        user_basic_profile_json=user_basic_profile_json,
    )
    return llm_client.generate_json(prompt)


ATTRIBUTE_CONFLICT_RESOLUTION_PROMPT = Template("""You are a cross-domain ATTRIBUTE conflict resolver.

Your task: Detect and resolve conflicts in user_attributes_state (singular + collections) across domains. Do not change habits or preferences unless absolutely required by a cascade.

To keep your fixes reasonable, here is the background:

Basic user profile:
{{ user_basic_profile_json }}

=================================================================================
SHARED ATTRIBUTES ACROSS DOMAINS
=================================================================================

{{ detected_attribute_conflicts_json }}

**Data format explanation:**
- After key alignment, attributes with the same name across domains are listed here
- Each entry shows a "shared_key" (the aligned attribute name after normalization)
- "windows" shows each window where this attribute appears in multiple domains
- For each window, "domains" lists the value in each domain
- Your job: check if the values across domains are consistent or conflicting

=================================================================================
DYNAMIC PROFILES (KEY-ALIGNED)
=================================================================================

{{ dynamic_profiles_json }}

=================================================================================
CONFLICT DETECTION AND RESOLUTION RULES (CRITICAL)
=================================================================================

**Core principles:**
- Review each shared attribute to determine if there is an actual conflict
- Only create resolutions for attributes that have genuine conflicts
- Make minimal edits in the specific window (initial or wX)
- Avoid touching other windows unless strictly needed for consistency
- Keep value shapes intact (string vs list)
- Leave habits_state and preferences_state untouched unless a cascade is unavoidable

**Conflict detection criteria:**

1. **Singular attributes**: A conflict exists if the same attribute has different values across domains in the same window
   - Example conflict: domain A has "occupation: engineer", domain B has "occupation: teacher" in the same window
   - Example NOT a conflict: domain A has "occupation: engineer", domain B has "occupation: engineer" (same value)

2. **Collection attributes**: A conflict exists if there are duplicate items across domains
   - Example conflict: domain A has ["project_x", "project_y"], domain B has ["project_y", "project_z"] (project_y is duplicated)
   - Example NOT a conflict: domain A has ["project_x"], domain B has ["project_y"] (no overlap, each domain can have its own items)

**Resolution strategies:**

1. **For singular conflicts**:
   - Pick the most coherent value (based on basic profile + domain authority)
   - Align all domains for that window to use this value

2. **For collection conflicts** (duplicates):
   - Dedupe: keep one authoritative instance of the duplicate item
   - Remove the duplicate from other domains
   - IMPORTANT: Keep collections separate per domain; DO NOT merge all items into one domain
   - For removals, specify the target by index or by matching the item value

3. **Cascade changes**: if you must change habits or preferences
   - Explain why in the reason field

=================================================================================
OUTPUT FORMAT
=================================================================================

Output strict JSON:
{
  "conflicts_and_resolutions": [
    {
      "conflict": {
        "description": "...",
        "shared_key": "...",
        "window_id": "<initial|w1|...>"
      },
      "resolution": {
        "strategy": "align_singular | dedupe_collection | drop_duplicate | other",
        "patches": [
          {"domain": "...", "window_id": "...", "path": "<dot path>", "action": "replace|append|remove|add_key|update", "value": <new_value_optional>, "reason": "..."}
        ]
      }
    }
  ]
}

**Important**: Only include conflicts that genuinely need resolution. If a shared attribute has the same values across domains, or collections have no overlaps, do NOT treat it as a conflict.
""")


@dataclass
class Domain:
    domain_name: str
    domain_scope_definition: str
    start_date: str | None = None
    end_date: str | None = None
    num_windows: int | None = None

    def resolve_timeline(self, default: TimelineConfig) -> TimelineConfig:
        return TimelineConfig(
            start_date=self.start_date or default.start_date,
            end_date=self.end_date or default.end_date,
            num_windows=self.num_windows or default.num_windows,
        )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "domain"


def _format_window_range(window_state: Dict) -> str:
    time_range = window_state.get("time_range")
    if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
        return f"{time_range[0]} to {time_range[1]}"
    return str(time_range)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

def _write_text(path: Path, text: str) -> None:
    path.write_text(text)


# ==== Temporal conflict detection utilities ====
PRIORITY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _safe_parse_date(value: object) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _parse_window_date_range(time_range: object) -> Tuple[Optional[date], Optional[date]]:
    if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
        start = _safe_parse_date(time_range[0])
        end = _safe_parse_date(time_range[1])
        if start and end:
            return start, end
    return None, None


def _iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _coerce_days_of_week(raw: object) -> List[int]:
    days: List[int] = []
    for item in raw or []:
        try:
            val = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= val <= 6:
            days.append(val)
    return days


def _iter_month_starts(start_date: date, end_date: date):
    current = date(start_date.year, start_date.month, 1)
    last_month = date(end_date.year, end_date.month, 1)
    while current <= last_month:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _dates_from_schedule(schedule: Dict[str, Any], start_date: date, end_date: date) -> List[date]:
    freq = (schedule.get("frequency_type") or "").lower()
    if not freq:
        return []
    if freq == "daily":
        return list(_iter_dates(start_date, end_date))
    if freq == "weekly":
        days = _coerce_days_of_week(schedule.get("days_of_week"))
        days = days or list(range(7))
        return [dt for dt in _iter_dates(start_date, end_date) if dt.weekday() in days]
    if freq == "biweekly":
        days = _coerce_days_of_week(schedule.get("days_of_week"))
        days = days or list(range(7))
        anchor = _safe_parse_date(schedule.get("start_date")) or start_date
        matches: List[date] = []
        for dt in _iter_dates(start_date, end_date):
            if dt.weekday() not in days:
                continue
            if anchor and dt >= anchor and (dt - anchor).days % 14 == 0:
                matches.append(dt)
        return matches
    if freq == "monthly_by_date":
        days_of_month = []
        for dom in schedule.get("days_of_month") or []:
            try:
                dom_int = int(dom)
            except (TypeError, ValueError):
                continue
            if 1 <= dom_int <= 31:
                days_of_month.append(dom_int)
        return [
            dt
            for dt in _iter_dates(start_date, end_date)
            if days_of_month and dt.day in days_of_month
        ]
    if freq == "monthly_nth_weekday":
        week_of_month = schedule.get("week_of_month")
        day_of_week = schedule.get("day_of_week")
        try:
            target_week = int(week_of_month)
        except (TypeError, ValueError):
            target_week = None
        if isinstance(week_of_month, str) and week_of_month.lower() == "last":
            target_week = -1
        try:
            target_dow = int(day_of_week)
        except (TypeError, ValueError):
            target_dow = None
        if target_dow is None:
            return []

        matches: List[date] = []
        for month_start in _iter_month_starts(start_date, end_date):
            days_in_month: List[date] = []
            current = month_start
            while current.month == month_start.month and current <= end_date:
                if current >= start_date and current.weekday() == target_dow:
                    days_in_month.append(current)
                current += timedelta(days=1)
            if not days_in_month:
                continue
            if target_week == -1:
                candidate = days_in_month[-1]
            elif target_week and 1 <= target_week <= len(days_in_month):
                candidate = days_in_month[target_week - 1]
            else:
                candidate = days_in_month[0]
            if start_date <= candidate <= end_date:
                matches.append(candidate)
        return matches
    return []


def _time_to_minutes(value: str) -> Optional[int]:
    value = value.strip().lower()
    match = re.match(r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>am|pm)?", value)
    if not match:
        return None
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    ampm = match.group("ampm")
    if ampm:
        if hours == 12:
            hours = 0
        if ampm == "pm":
            hours += 12
    return hours * 60 + minutes


def _parse_time_range_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(
        r"(\d{1,2}:\d{2}\s*(?:am|pm)?)\s*[-–]\s*(\d{1,2}:\d{2}\s*(?:am|pm)?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None, None
    start = _time_to_minutes(match.group(1))
    end = _time_to_minutes(match.group(2))
    if start is None or end is None:
        return None, None
    if end <= start:
        end += 24 * 60
    return start, end


def _parse_structured_timing(timing: object) -> Tuple[Optional[int], Optional[int], str]:
    if isinstance(timing, dict):
        start_text = str(timing.get("start_time", "")).strip()
        end_text = str(timing.get("end_time", "")).strip()
        label = f"{start_text}-{end_text}".strip("-") if start_text or end_text else ""
        start = _time_to_minutes(start_text) if start_text else None
        end = _time_to_minutes(end_text) if end_text else None
        if start is None or end is None:
            return None, None, label
        if end <= start:
            end += 24 * 60
        return start, end, label
    if isinstance(timing, str):
        start, end = _parse_time_range_text(timing)
        return start, end, timing.strip()
    return None, None, ""


def _materialize_habit_snapshots_for_conflicts(domain: Dict[str, Any]) -> List[Dict[str, Any]]:
    initial = (domain.get("initial_state", {}) or {})
    base_habits = deepcopy(initial.get("habits_state") or {})

    # Track where each habit is defined (for path construction in Rule 5)
    habit_sources: Dict[str, str] = {}
    for habit_name in base_habits.keys():
        habit_sources[habit_name] = "initial_state.habits_state"

    snapshots = [
        {
            "window_id": "initial_state",
            "time_range": initial.get("time_range"),
            "habits": deepcopy(base_habits),
            "habit_sources": deepcopy(habit_sources),
        }
    ]
    current = deepcopy(base_habits)

    for w_idx, window in enumerate(domain.get("time_windows") or []):
        window_id = window.get("window_id") or f"w{w_idx}"
        for op in (window.get("habits_delta") or {}).get("operations") or []:
            name = op.get("habit_name") or "unnamed_habit"
            op_type = (op.get("op") or "").lower()
            delta = op.get("delta")
            if op_type == "acquire" and isinstance(delta, dict):
                current[name] = deepcopy(delta)
                habit_sources[name] = f"time_windows[{w_idx}].habits_delta.operations[?habit_name='{name}']"
            elif op_type == "adjust" and isinstance(delta, dict):
                existing = current.get(name, {})
                if not isinstance(existing, dict):
                    existing = {}
                merged = deepcopy(existing)
                merged.update(delta)
                current[name] = merged
                # For adjust, the habit was defined earlier, but we track the latest modification
                if name not in habit_sources:
                    habit_sources[name] = f"time_windows[{w_idx}].habits_delta.operations[?habit_name='{name}']"
            elif op_type == "drop":
                current.pop(name, None)
                habit_sources.pop(name, None)
        snapshots.append(
            {
                "window_id": window_id,
                "time_range": window.get("time_range"),
                "habits": deepcopy(current),
                "habit_sources": deepcopy(habit_sources),
            }
        )
    return snapshots


def _collect_temporal_events(dynamic_profiles: Dict[str, Dict]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for domain_name, profile in dynamic_profiles.items():
        for snapshot in _materialize_habit_snapshots_for_conflicts(profile or {}):
            window_id = snapshot.get("window_id") or "unknown_window"
            normalized_window_id = _normalize_window_id_label(window_id)

            start_date, end_date = _parse_window_date_range(snapshot.get("time_range"))

            # For initial_state without time_range, use a default sampling period
            if not start_date or not end_date:
                if normalized_window_id == "initial":
                    # Sample 1 year for initial_state conflict detection
                    start_date = date(2024, 1, 1)
                    end_date = date(2024, 12, 31)
                else:
                    continue

            for habit_name, habit in (snapshot.get("habits") or {}).items():
                start_min, end_min, timing_label = _parse_structured_timing(
                    habit.get("timing")
                )
                if start_min is None or end_min is None:
                    continue
                schedule = habit.get("schedule") or {}
                occurrences = _dates_from_schedule(schedule, start_date, end_date)
                if not occurrences:
                    occurrences = list(_iter_dates(start_date, end_date))
                location = habit.get("location", "")
                for dt in occurrences:
                    events.append(
                        {
                            "domain": domain_name,
                            "window_id": normalized_window_id,  # Use normalized window_id
                            "window_range": snapshot.get("time_range"),
                            "habit": habit_name,
                            "priority": (habit.get("priority") or "").lower(),
                            "timing": timing_label,
                            "start_min": start_min,
                            "end_min": end_min,
                            "location": location,
                            "date": dt,
                        }
                    )
    return events


def _format_minutes(value: int) -> str:
    hours = (value // 60) % 24
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"


def _conflict_habit_summary(event: Dict[str, object]) -> Dict[str, object]:
    return {
        "domain": event.get("domain"),
        "habit": event.get("habit"),
        "priority": event.get("priority"),
        "timing": event.get("timing"),
        "location": event.get("location", ""),
        "window_id": event.get("window_id"),
    }


def _priority_resolution_suggestion(event_a: Dict[str, object], event_b: Dict[str, object]) -> Optional[Dict[str, object]]:
    pa = PRIORITY_ORDER.get(str(event_a.get("priority") or "").lower())
    pb = PRIORITY_ORDER.get(str(event_b.get("priority") or "").lower())
    if pa is None or pb is None or pa == pb:
        return None
    winner, loser = (event_a, event_b) if pa > pb else (event_b, event_a)
    return {
        "strategy": "keep_highest_priority",
        "winner": _conflict_habit_summary(winner),
        "loser": _conflict_habit_summary(loser),
        "note": "Keep the higher-priority habit; shift or drop the lower-priority one to remove the overlap.",
    }


def _canonical_pair(
    a: Dict[str, object], b: Dict[str, object], window_id: str | None
) -> Tuple[Dict[str, object], Dict[str, object], Tuple[str | None, str, str, str, str]]:
    """
    Order pair deterministically so we don't duplicate conflicts for swapped pairs.
    """
    key_a = (str(a.get("domain") or ""), str(a.get("habit") or ""))
    key_b = (str(b.get("domain") or ""), str(b.get("habit") or ""))
    if key_a <= key_b:
        return a, b, (window_id, key_a[0], key_a[1], key_b[0], key_b[1])
    return b, a, (window_id, key_b[0], key_b[1], key_a[0], key_a[1])


def detect_temporal_conflicts(dynamic_profiles: Dict[str, Dict]) -> Dict[str, object]:
    events = _collect_temporal_events(dynamic_profiles)
    aggregated: Dict[
        Tuple[str | None, str, str, str, str], Dict[str, object]
    ] = {}
    grouped: Dict[Tuple[str, date], List[Dict[str, object]]] = {}
    for ev in events:
        dt = ev.get("date")
        if not isinstance(dt, date):
            continue
        key = (ev.get("window_id") or "unknown_window", dt)
        grouped.setdefault(key, []).append(ev)

    for (window_id, dt), bucket in grouped.items():
        bucket = sorted(bucket, key=lambda e: e["start_min"])
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]

                # Get locations
                loc_a = str(a.get("location") or "").strip()
                loc_b = str(b.get("location") or "").strip()

                # Determine if there's a conflict based on location
                has_conflict = False
                conflict_type = ""

                if loc_a and loc_b and loc_a == loc_b:
                    # Same location: conflict if time overlaps
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "same_location_time_overlap"
                elif loc_a and loc_b and loc_a != loc_b:
                    # Different locations: conflict if less than 30 min gap
                    gap_min = b["start_min"] - a["end_min"]
                    if gap_min < 30:
                        has_conflict = True
                        conflict_type = "different_location_insufficient_gap"
                else:
                    # One or both locations are empty, treat as same location
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "time_overlap_no_location"

                if has_conflict:
                    left, right, key = _canonical_pair(a, b, window_id)
                    overlap_range = f"{_format_minutes(max(a['start_min'], b['start_min']))}-{_format_minutes(min(a['end_min'], b['end_min']))}"

                    entry = aggregated.setdefault(
                        key,
                        {
                            "window_id": window_id,
                            "window_range": bucket[0].get("window_range"),
                            "habit_a": _conflict_habit_summary(left),
                            "habit_b": _conflict_habit_summary(right),
                            "conflict_type": conflict_type,
                            "occurrences": 0,
                            "sample_dates": [],
                            "overlap_examples": [],
                        },
                    )
                    entry["occurrences"] = entry.get("occurrences", 0) + 1
                    if len(entry["sample_dates"]) < 3:
                        entry["sample_dates"].append(dt.isoformat())
                    if len(entry["overlap_examples"]) < 3:
                        gap_info = ""
                        if conflict_type == "different_location_insufficient_gap":
                            gap_min = b["start_min"] - a["end_min"]
                            gap_info = f", gap: {gap_min} min"
                        entry["overlap_examples"].append(
                            {"date": dt.isoformat(), "overlap": overlap_range + gap_info}
                        )
    conflicts = list(aggregated.values())

    # Build per-window conflict graph summary (node degree by habit).
    graph_by_window: Dict[str, Dict[str, object]] = {}
    TOP_N = 8
    for entry in conflicts:
        window_id = entry.get("window_id") or "unknown_window"
        occ = int(entry.get("occurrences") or 1)
        window_graph = graph_by_window.setdefault(
            window_id, {"total_conflicts": 0, "node_degrees": {}}
        )
        window_graph["total_conflicts"] += occ
        for habit_key in ("habit_a", "habit_b"):
            habit_info = entry.get(habit_key) or {}
            node_key = (habit_info.get("domain") or "", habit_info.get("habit") or "")
            node_degrees: Dict[Tuple[str, str], int] = window_graph["node_degrees"]  # type: ignore
            node_degrees[node_key] = node_degrees.get(node_key, 0) + occ

    # Convert node_degrees to sorted top list.
    for window_id, data in graph_by_window.items():
        node_degrees = data.get("node_degrees", {}) or {}
        top_nodes = sorted(
            (
                {
                    "domain": domain,
                    "habit": habit,
                    "degree": degree,
                }
                for (domain, habit), degree in node_degrees.items()
            ),
            key=lambda x: (-x["degree"], x["domain"], x["habit"]),
        )[:TOP_N]
        data["top_nodes"] = top_nodes
        data.pop("node_degrees", None)

    return {"conflicts": conflicts, "graph_by_window": graph_by_window}


def _path_key(key: object) -> object:
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.isdigit():
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _location_to_path(location: object) -> List[object]:
    """
    Convert dotted + bracket notation paths (e.g., time_windows[0].summary) into
    a list/tuple path usable by the patch applier. If the input is already a
    list/tuple, it is returned as-is.
    """
    if isinstance(location, (list, tuple)):
        return list(location)
    parts: List[object] = []
    if location is None:
        return parts
    for segment in str(location).split("."):
        if not segment:
            continue
        remainder = segment
        while remainder:
            if "[" in remainder:
                before, after = remainder.split("[", 1)
                if before:
                    parts.append(before)
                idx_str, remainder = after.split("]", 1)
                parts.append(_path_key(idx_str))
            else:
                parts.append(remainder)
                remainder = ""
    return parts


def _traverse_path(
    root: Any, path: Sequence[object], *, create_missing: bool = False
) -> Any:
    """
    Walk the object graph following a path. Optionally create missing containers
    (dict or list) when create_missing is True.
    """
    target: Any = root
    for idx, raw_key in enumerate(path):
        key = _path_key(raw_key)
        if isinstance(target, list):
            if not isinstance(key, int):
                return None
            if 0 <= key < len(target):
                target = target[key]
            elif create_missing and key == len(target):
                next_key = path[idx + 1] if idx + 1 < len(path) else None
                target.append({} if isinstance(next_key, str) else [])
                target = target[key]
            else:
                return None
        elif isinstance(target, dict):
            if key not in target:
                if create_missing:
                    next_key = path[idx + 1] if idx + 1 < len(path) else None
                    target[key] = {} if isinstance(next_key, str) else []
                else:
                    return None
            target = target[key]
        else:
            return None
    return target


def _apply_patch_ops(base: Dict, patch_ops: List[Dict]) -> Dict:
    patched = deepcopy(base)
    for op in patch_ops or []:
        path = _location_to_path(op.get("path"))
        action = op.get("action")
        value = op.get("value")
        key_name = op.get("key")
        if action not in {"add", "replace", "remove", "append", "add_key"}:
            continue
        if not isinstance(path, list) or len(path) == 0:
            continue

        if action == "append":
            target_list = _traverse_path(patched, path, create_missing=True)
            if isinstance(target_list, list):
                target_list.append(value)
            continue

        # Allow nested field updates without replacing the whole object when a key is provided
        if key_name is not None and action in {"add", "replace", "remove", "add_key"}:
            target_dict = _traverse_path(
                patched, path, create_missing=action in {"add", "replace", "add_key"}
            )
            if isinstance(target_dict, dict):
                if action == "remove":
                    target_dict.pop(key_name, None)
                else:
                    target_dict[key_name] = value
                continue

        if action == "add_key":
            target_dict = _traverse_path(patched, path, create_missing=True)
            if isinstance(target_dict, dict) and key_name is not None:
                target_dict[key_name] = value
            continue

        parent = _traverse_path(
            patched, path[:-1], create_missing=action in {"add", "replace"}
        )
        if parent is None:
            continue

        last_key = _path_key(path[-1])
        if isinstance(parent, list):
            if not isinstance(last_key, int):
                continue
            if action == "add":
                if 0 <= last_key <= len(parent):
                    parent.insert(last_key, value)
                else:
                    parent.append(value)
            elif action == "replace":
                if 0 <= last_key < len(parent):
                    parent[last_key] = value
                elif last_key == len(parent):
                    parent.append(value)
            elif action == "remove" and 0 <= last_key < len(parent):
                parent.pop(last_key)
        elif isinstance(parent, dict):
            if action in {"add", "replace"}:
                parent[last_key] = value
            elif action == "remove":
                parent.pop(last_key, None)
    return patched


def _apply_profile_revision(original_profile: Dict, review_payload: Dict) -> Dict:
    if not isinstance(review_payload, dict):
        return deepcopy(original_profile)
    patch_ops = review_payload.get("patch") or review_payload.get("patch_ops") or []
    fixes = review_payload.get("fixes") or []
    revised = review_payload.get("revised_profile")
    base_from_singleton_list = (
        isinstance(original_profile, list)
        and len(original_profile) == 1
        and isinstance(original_profile[0], dict)
    )
    base_obj = (
        deepcopy(original_profile[0])
        if base_from_singleton_list
        else deepcopy(original_profile)
    )

    violations = review_payload.get("violations_and_fixes") or []
    if violations and not patch_ops:
        translated_ops: List[Dict] = []
        for violation in violations:
            for patch in violation.get("patches") or []:
                loc = patch.get("path") or patch.get("location")
                if loc is None:
                    continue
                translated_ops.append(
                    {
                        "path": _location_to_path(loc),
                        "action": patch.get("action") or "replace",
                        "value": patch.get("value"),
                        "key": patch.get("key"),
                    }
                )
        patch_ops = translated_ops

    if fixes and not patch_ops:
        translated_ops: List[Dict] = []
        for fix in fixes:
            loc = fix.get("location") or fix.get("path")
            if not loc:
                continue
            path = _location_to_path(loc)
            action = fix.get("action") or "replace"
            value = fix.get("fix")
            if "value" in fix and value is None:
                value = fix.get("value")
            translated_ops.append({"path": path, "action": action, "value": value})
        patch_ops = translated_ops

    patched = deepcopy(base_obj)
    if patch_ops:
        try:
            patched = _apply_patch_ops(patched, patch_ops)
        except Exception:
            if revised is not None:
                patched = deepcopy(revised)
            else:
                patched = deepcopy(base_obj)
    elif revised is not None:
        patched = deepcopy(revised)
    if base_from_singleton_list and isinstance(patched, dict):
        return [patched]
    return patched


def _append_issue(
    issues: List[Dict[str, str]], path: str, message: str, window_id: str | None = None, issue_id: str | None = None
) -> None:
    entry: Dict[str, str] = {"path": path, "message": message}
    if window_id:
        entry["window_id"] = window_id
    if issue_id:
        entry["id"] = issue_id
    issues.append(entry)


def _validate_schedule_structure(schedule: Any) -> List[str]:
    if not isinstance(schedule, dict):
        return ["schedule must be an object with frequency_type"]
    missing: List[str] = []
    freq = schedule.get("frequency_type")
    allowed = {"daily", "weekly", "biweekly", "monthly_by_date", "monthly_nth_weekday"}
    if not freq:
        missing.append("schedule.frequency_type missing")
    elif freq not in allowed:
        missing.append(f"schedule.frequency_type '{freq}' is invalid")
    else:
        if freq == "weekly" and "days_of_week" not in schedule:
            missing.append("schedule.days_of_week missing for weekly")
        if freq == "biweekly":
            if "days_of_week" not in schedule:
                missing.append("schedule.days_of_week missing for biweekly")
            if "start_date" not in schedule:
                missing.append("schedule.start_date missing for biweekly")
        if freq == "monthly_by_date" and "days_of_month" not in schedule:
            missing.append("schedule.days_of_month missing for monthly_by_date")
        if freq == "monthly_nth_weekday":
            if "week_of_month" not in schedule:
                missing.append("schedule.week_of_month missing for monthly_nth_weekday")
            if "day_of_week" not in schedule:
                missing.append("schedule.day_of_week missing for monthly_nth_weekday")
    return missing


def _validate_timing_structure(timing: Any) -> List[str]:
    if not isinstance(timing, dict):
        return ["timing must be an object with start_time and end_time"]
    missing: List[str] = []
    if "start_time" not in timing:
        missing.append("timing.start_time missing")
    if "end_time" not in timing:
        missing.append("timing.end_time missing")
    return missing


def _validate_habit_object(habit_obj: Any) -> List[str]:
    if not isinstance(habit_obj, dict):
        return ["habit must be an object with required fields"]
    required_fields = ["schedule", "timing", "location", "priority"]
    missing = [f for f in required_fields if f not in habit_obj]
    missing += _validate_schedule_structure(habit_obj.get("schedule"))
    missing += _validate_timing_structure(habit_obj.get("timing"))
    return missing


def _validate_preference_object(pref_obj: Any) -> List[str]:
    if not isinstance(pref_obj, dict):
        return ["preference must be an object with statement and signals"]
    missing: List[str] = []
    if not pref_obj.get("statement"):
        missing.append("statement missing")
    signals = pref_obj.get("signals")
    if not isinstance(signals, list) or len(signals) < 2:
        missing.append("signals missing or too few (need 2-4)")
    return missing


def _detect_rule2_prior_existence_issues(profile: Dict) -> List[Dict[str, str]]:
    # Some generators output a singleton list; normalize to dict
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}
    issues: List[Dict[str, str]] = []
    initial_state = profile.get("initial_state") or {}
    user_attributes_state = initial_state.get("user_attributes_state") or {}
    known_singular = set((user_attributes_state.get("singular") or {}).keys())
    known_collections = set((user_attributes_state.get("collections") or {}).keys())
    habits_state = initial_state.get("habits_state") or {}
    known_habits = set(habits_state.keys())
    preferences_state = initial_state.get("preferences_state") or {}
    known_preferences = set(preferences_state.keys())

    time_windows = profile.get("time_windows") or []
    for w_idx, window in enumerate(time_windows):
        window_id = window.get("window_id") or f"w{w_idx + 1}"

        attr_ops = (window.get("user_attributes_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(attr_ops):
            op_type = op.get("op")
            attr_type = op.get("attribute_type")

            # Check 1: singular attributes can ONLY use "modify"
            if attr_type == "singular" and op_type != "modify":
                name = op.get("attribute_name")
                _append_issue(
                    issues,
                    f"time_windows[{w_idx}].user_attributes_delta.operations[{op_idx}]",
                    f"invalid operation '{op_type}' on singular attribute '{name}' (singular only supports 'modify')",
                    window_id,
                )

            # Check 2: collections can ONLY use "add" or "remove"
            if attr_type == "collections" and op_type not in {"add", "remove"}:
                name = op.get("collection_name")
                _append_issue(
                    issues,
                    f"time_windows[{w_idx}].user_attributes_delta.operations[{op_idx}]",
                    f"invalid operation '{op_type}' on collection '{name}' (collections only support 'add' or 'drop')",
                    window_id,
                )

            # Check 3: modify on singular requires prior existence
            if op_type == "modify" and attr_type == "singular":
                name = op.get("attribute_name")
                if name and name not in known_singular:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].user_attributes_delta.operations[{op_idx}]",
                        f"modify '{name}' before it exists in prior state",
                        window_id,
                    )
                    known_singular.add(name)

            # Track collections
            elif op_type == "add" and attr_type == "collections":
                name = op.get("collection_name")
                if name:
                    known_collections.add(name)

            # Check 4: drop on collection requires prior existence
            elif op_type == "remove" and attr_type == "collections":
                name = op.get("collection_name")
                if name and name not in known_collections:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].user_attributes_delta.operations[{op_idx}]",
                        f"drop collection '{name}' before it exists",
                        window_id,
                    )

        habit_ops = (window.get("habits_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(habit_ops):
            op_type = op.get("op")
            habit_name = op.get("habit_name")
            if op_type == "acquire":
                if habit_name:
                    known_habits.add(habit_name)
            elif op_type in {"adjust", "drop"}:
                if habit_name and habit_name not in known_habits:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].habits_delta.operations[{op_idx}]",
                        f"{op_type} '{habit_name}' before it exists in prior state",
                        window_id,
                    )
                    known_habits.add(habit_name)

                # Check 5: adjust on habit must modify schedule/timing/location, not just priority
                if op_type == "adjust":
                    delta = op.get("delta") or {}
                    if isinstance(delta, dict):
                        # Check if only priority is being modified
                        has_structural_change = any(
                            key in delta for key in ["schedule", "timing", "location"]
                        )
                        if not has_structural_change and "priority" in delta:
                            _append_issue(
                                issues,
                                f"time_windows[{w_idx}].habits_delta.operations[{op_idx}]",
                                f"adjust '{habit_name}' only modifies priority (must modify at least one of: schedule, timing, location, context)",
                                window_id,
                            )

                if op_type == "drop" and habit_name in known_habits:
                    known_habits.remove(habit_name)

        pref_ops = (window.get("preferences_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(pref_ops):
            op_type = op.get("op")
            pref_name = op.get("preference_name")
            if op_type in {"shift", "refine"} and pref_name:
                if pref_name not in known_preferences:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].preferences_delta.operations[{op_idx}]",
                        f"{op_type} '{pref_name}' before it exists in prior state",
                        window_id,
                    )
                    known_preferences.add(pref_name)

    
    # Add unique IDs to all issues
    for idx, issue in enumerate(issues):
        if "id" not in issue:
            issue["id"] = f"rule2_{idx:03d}"
            issue["message"] = f"[ID: {issue['id']}] {issue['message']}"
    
    return issues


def _detect_rule1_required_field_issues(profile: Dict) -> List[Dict[str, str]]:
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}
    issues: List[Dict[str, str]] = []
    initial_state = profile.get("initial_state") or {}
    if "summary" not in initial_state:
        _append_issue(issues, "initial_state.summary", "initial_state missing summary")

    user_attributes_state = initial_state.get("user_attributes_state")
    if not isinstance(user_attributes_state, dict):
        _append_issue(
            issues,
            "initial_state.user_attributes_state",
            "user_attributes_state missing or not an object",
        )
    else:
        if "singular" not in user_attributes_state:
            _append_issue(
                issues,
                "initial_state.user_attributes_state.singular",
                "singular attributes missing (can be empty object)",
            )
        if "collections" not in user_attributes_state:
            _append_issue(
                issues,
                "initial_state.user_attributes_state.collections",
                "collections missing (can be empty object)",
            )

    habits_state = initial_state.get("habits_state") or {}
    for habit_name, habit_obj in habits_state.items():
        missing = _validate_habit_object(habit_obj)
        if missing:
            _append_issue(
                issues,
                f"initial_state.habits_state.{habit_name}",
                "; ".join(missing),
                "initial",
            )

    pref_state = initial_state.get("preferences_state") or {}
    for pref_name, pref_obj in pref_state.items():
        missing = _validate_preference_object(pref_obj)
        if missing:
            _append_issue(
                issues,
                f"initial_state.preferences_state.{pref_name}",
                "; ".join(missing),
                "initial",
            )

    time_windows = profile.get("time_windows") or []
    for w_idx, window in enumerate(time_windows):
        window_id = window.get("window_id") or f"w{w_idx + 1}"
        if "window_description" not in window:
            _append_issue(
                issues,
                f"time_windows[{w_idx}].window_description",
                "window_description missing",
                window_id,
            )
        if "summary" not in window:
            _append_issue(
                issues,
                f"time_windows[{w_idx}].summary",
                "summary missing",
                window_id,
            )

        attr_ops = (window.get("user_attributes_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(attr_ops):
            op_path = f"time_windows[{w_idx}].user_attributes_delta.operations[{op_idx}]"
            op_type = op.get("op")
            if not op_type:
                _append_issue(issues, op_path, "operation missing op", window_id)
            if not op.get("reason"):
                _append_issue(issues, f"{op_path}.reason", "reason missing", window_id)
            attr_type = op.get("attribute_type")
            if not attr_type:
                _append_issue(
                    issues, f"{op_path}.attribute_type", "attribute_type missing", window_id
                )
            if op_type == "modify":
                if not op.get("attribute_name"):
                    _append_issue(
                        issues, f"{op_path}.attribute_name", "attribute_name missing", window_id
                    )
                if op.get("delta") in [None, ""]:
                    _append_issue(
                        issues, f"{op_path}.delta", "delta missing for modify", window_id
                    )
            elif op_type in {"add", "remove"}:
                if not op.get("attribute_name"):
                    _append_issue(
                        issues,
                        f"{op_path}.attribute_name",
                        "attribute_name missing",
                        window_id,
                    )
                delta = op.get("delta")
                if not isinstance(delta, list) or len(delta) == 0:
                    _append_issue(
                        issues,
                        f"{op_path}.delta",
                        f"delta list missing/empty for {op_type}",
                        window_id,
                    )

        habit_ops = (window.get("habits_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(habit_ops):
            op_path = f"time_windows[{w_idx}].habits_delta.operations[{op_idx}]"
            op_type = op.get("op")
            if not op_type:
                _append_issue(issues, op_path, "operation missing op", window_id)
            if not op.get("reason"):
                _append_issue(issues, f"{op_path}.reason", "reason missing", window_id)

            habit_name = op.get("habit_name")
            if not habit_name:
                _append_issue(
                    issues, f"{op_path}.habit_name", "habit_name missing", window_id
                )

            if op_type == "acquire":
                delta = op.get("delta")
                missing = _validate_habit_object(delta)
                if missing:
                    _append_issue(
                        issues, f"{op_path}.delta", "; ".join(missing), window_id
                    )
            elif op_type == "adjust":
                delta = op.get("delta")
                if not isinstance(delta, dict):
                    _append_issue(
                        issues,
                        f"{op_path}.delta",
                        "delta must be an object for adjust",
                        window_id,
                    )
                else:
                    substantive_change = any(
                        key in delta for key in ("schedule", "timing", "location", "priority")
                    )
                    if not substantive_change:
                        _append_issue(
                            issues,
                            f"{op_path}.delta",
                            "adjust must change schedule/timing/location",
                            window_id,
                        )
                    if "schedule" in delta:
                        missing = _validate_schedule_structure(delta.get("schedule"))
                        if missing:
                            _append_issue(
                                issues,
                                f"{op_path}.delta.schedule",
                                "; ".join(missing),
                                window_id,
                            )
                    if "timing" in delta:
                        missing = _validate_timing_structure(delta.get("timing"))
                        if missing:
                            _append_issue(
                                issues,
                                f"{op_path}.delta.timing",
                                "; ".join(missing),
                                window_id,
                            )
            elif op_type == "drop":
                if op.get("delta") is not None:
                    _append_issue(
                        issues,
                        f"{op_path}.delta",
                        "drop operation must use JSON null",
                        window_id,
                    )

        pref_ops = (window.get("preferences_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(pref_ops):
            op_path = f"time_windows[{w_idx}].preferences_delta.operations[{op_idx}]"
            op_type = op.get("op")
            if not op_type:
                _append_issue(issues, op_path, "operation missing op", window_id)
            if not op.get("reason"):
                _append_issue(issues, f"{op_path}.reason", "reason missing", window_id)
            pref_name = op.get("preference_name")
            if not pref_name:
                _append_issue(
                    issues, f"{op_path}.preference_name", "preference_name missing", window_id
                )

            delta = op.get("delta")
            if op_type in {"shift", "refine"}:
                missing = _validate_preference_object(delta)
                if missing:
                    _append_issue(
                        issues, f"{op_path}.delta", "; ".join(missing), window_id
                    )

    
    # Add unique IDs to all issues
    for idx, issue in enumerate(issues):
        if "id" not in issue:
            issue["id"] = f"rule1_{idx:03d}"
            issue["message"] = f"[ID: {issue['id']}] {issue['message']}"
    
    return issues


def _detect_rule4_short_term_issues(profile: Dict) -> List[Dict[str, str]]:
    """
    Detect short-term changes (seasonal, special events, temporary circumstances)
    that lack appropriate follow-up operations (rollback or permanence reasoning).

    Only checks habits and attributes (NOT preferences).
    Ignores short-term changes in the LAST window (no follow-up window available).
    """
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}

    issues: List[Dict[str, str]] = []

    # Keywords indicating short-term reasons
    short_term_keywords = [
        "winter", "summer", "spring", "fall", "autumn", "seasonal",
        "holiday", "vacation", "olympics", "event", "festival",
        "heatwave", "cold snap", "temporary", "temporarily",
        "this month", "this week", "during", "special occasion"
    ]

    # Track short-term changes: (window_idx, type, name, reason, keywords_found)
    short_term_changes: List[tuple] = []

    time_windows = profile.get("time_windows") or []
    last_window_idx = len(time_windows) - 1

    # First pass: identify short-term changes (but NOT in the last window)
    for w_idx, window in enumerate(time_windows):
        # Skip the last window - no follow-up window available
        if w_idx == last_window_idx:
            continue

        window_id = window.get("window_id") or f"w{w_idx + 1}"

        # Check habits_delta operations
        habit_ops = (window.get("habits_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(habit_ops):
            op_type = op.get("op")
            habit_name = op.get("habit_name")
            reason = (op.get("reason") or "").lower()

            # Check for short-term keywords in acquire or adjust operations
            if op_type in {"acquire", "adjust"} and reason:
                found_keywords = [kw for kw in short_term_keywords if kw in reason]
                if found_keywords:
                    short_term_changes.append((
                        w_idx, "habit", habit_name, reason, found_keywords
                    ))

        # Check user_attributes_delta operations
        attr_ops = (window.get("user_attributes_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(attr_ops):
            op_type = op.get("op")
            reason = (op.get("reason") or "").lower()

            if op_type in {"modify", "add"} and reason:
                found_keywords = [kw for kw in short_term_keywords if kw in reason]
                if found_keywords:
                    attr_name = op.get("attribute_name") or op.get("collection_name")
                    short_term_changes.append((
                        w_idx, "attribute", attr_name, reason, found_keywords
                    ))

    # Second pass: check for follow-ups
    issue_id_counter = 0
    for orig_w_idx, change_type, change_name, change_reason, keywords in short_term_changes:
        has_followup = False

        # Check subsequent windows for rollback or permanence mention
        for check_w_idx in range(orig_w_idx + 1, len(time_windows)):
            window = time_windows[check_w_idx]

            if change_type == "habit":
                habit_ops = (window.get("habits_delta") or {}).get("operations") or []
                for op in habit_ops:
                    if op.get("habit_name") == change_name:
                        # Found a follow-up operation (adjust or drop)
                        if op.get("op") in {"adjust", "drop"}:
                            has_followup = True
                            break
                        # Or explicit reasoning about permanence
                        reason = (op.get("reason") or "").lower()
                        if any(kw in reason for kw in ["permanent", "decided to keep", "became habit"]):
                            has_followup = True
                            break

            elif change_type == "attribute":
                attr_ops = (window.get("user_attributes_delta") or {}).get("operations") or []
                for op in attr_ops:
                    attr_name = op.get("attribute_name") or op.get("collection_name")
                    if attr_name == change_name:
                        # Found a follow-up operation (modify or remove)
                        if op.get("op") in {"modify", "remove"}:
                            has_followup = True
                            break
                        reason = (op.get("reason") or "").lower()
                        if any(kw in reason for kw in ["permanent", "decided to keep"]):
                            has_followup = True
                            break

            if has_followup:
                break

        if not has_followup:
            window_id = time_windows[orig_w_idx].get("window_id") or f"w{orig_w_idx + 1}"
            issue_id = f"rule4_{issue_id_counter:03d}"
            issue_id_counter += 1
            _append_issue(
                issues,
                f"time_windows[{orig_w_idx}]",
                f"[ID: {issue_id}] Short-term {change_type} '{change_name}' (keywords: {', '.join(keywords)}) maybe lacks follow-up in later windows. "
                f"Maybe need rollback operation (drop/adjust/modify/remove) or explicit reasoning about permanence.",
                window_id,
            )
            # Add ID to the issue dict
            if issues:
                issues[-1]["id"] = issue_id

    return issues


def _detect_rule3_first_add_issues(profile: Dict) -> List[Dict[str, str]]:
    """
    Detect collection-type attributes that first appear via 'add' operation
    instead of being initialized in initial_state.

    This catches unrealistic scenarios like getting a first smartphone in w3,
    when it should already exist in initial_state.
    """
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}

    issues: List[Dict[str, str]] = []

    # Track what collection attributes exist in initial_state
    initial_state = profile.get("initial_state") or {}
    user_attributes_state = initial_state.get("user_attributes_state") or {}
    known_collections = set((user_attributes_state.get("collections") or {}).keys())

    time_windows = profile.get("time_windows") or []

    # Track first 'add' operations for collection attributes
    for w_idx, window in enumerate(time_windows):
        window_id = window.get("window_id") or f"w{w_idx + 1}"

        # Check user_attributes_delta for 'add' operations
        attr_ops = (window.get("user_attributes_delta") or {}).get("operations") or []
        for op_idx, op in enumerate(attr_ops):
            op_type = op.get("op")

            # Check for first 'add' to a collection
            if op_type == "add":
                collection_name = op.get("collection_name") or ""

                # If this collection doesn't exist in initial_state, it's a first-time add
                if collection_name and collection_name not in known_collections:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].user_attributes_delta.operations[{op_idx}]",
                        f"Collection '{collection_name}' first appears via 'add' operation in {window_id}. "
                        f"If this is an essential collection for the user, it should be initialized in initial_state "
                        f"with realistic baseline items.",
                        window_id,
                    )
                    # Mark as known to avoid duplicate reports
                    known_collections.add(collection_name)

    # Add unique IDs to all issues
    for idx, issue in enumerate(issues):
        if "id" not in issue:
            issue["id"] = f"rule3_{idx:03d}"
            issue["message"] = f"[ID: {issue['id']}] {issue['message']}"

    return issues


def _parse_time(time_str: str) -> tuple[int, int] | None:
    """Parse time string like '6:00 AM' or '18:30' to (hour, minute) in 24h format."""
    if not time_str:
        return None

    time_str = time_str.strip().upper()

    # Handle AM/PM format
    if 'AM' in time_str or 'PM' in time_str:
        is_pm = 'PM' in time_str
        time_str = time_str.replace('AM', '').replace('PM', '').strip()

        if ':' in time_str:
            parts = time_str.split(':')
            try:
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0

                # Convert to 24h
                if is_pm and hour != 12:
                    hour += 12
                elif not is_pm and hour == 12:
                    hour = 0

                return (hour, minute)
            except (ValueError, IndexError):
                return None

    # Handle 24h format
    if ':' in time_str:
        parts = time_str.split(':')
        try:
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            return (hour, minute)
        except (ValueError, IndexError):
            return None

    return None


def _detect_rule5_time_conflict_issues(profile: Dict) -> Dict[str, object]:
    """
    Detect time conflicts using date-aware temporal event expansion (same logic as cross-domain).

    Key improvements over old implementation:
    1. Considers actual dates - habits on different dates don't conflict
    2. Respects schedule frequency (daily/weekly/monthly)
    3. Counts actual occurrence conflicts, not just habit pair conflicts
    4. Groups events by (window_id, date) before checking overlaps

    Returns:
        {
            "conflicts": [...],  # List of conflict entries with occurrence counts
            "graph_by_window": {  # Conflict graph analysis
                "window_id": {
                    "total_conflicts": int (total occurrences),
                    "top_nodes": [{"habit": str, "degree": int}, ...]
                }
            }
        }
    """
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}

    # Collect all temporal events (expanded by date)
    events: List[Dict[str, object]] = []
    for snapshot in _materialize_habit_snapshots_for_conflicts(profile or {}):
        window_id = snapshot.get("window_id") or "unknown_window"
        habit_sources = snapshot.get("habit_sources") or {}
        start_date, end_date = _parse_window_date_range(snapshot.get("time_range"))
        if not start_date or not end_date:
            continue

        for habit_name, habit in (snapshot.get("habits") or {}).items():
            start_min, end_min, timing_label = _parse_structured_timing(habit.get("timing"))
            if start_min is None or end_min is None:
                continue

            schedule = habit.get("schedule") or {}
            occurrences = _dates_from_schedule(schedule, start_date, end_date)
            if not occurrences:
                # No schedule specified, assume daily
                occurrences = list(_iter_dates(start_date, end_date))

            location = habit.get("location", "")
            habit_source_path = habit_sources.get(habit_name, "")

            for dt in occurrences:
                events.append({
                    "window_id": window_id,
                    "window_range": snapshot.get("time_range"),
                    "habit": habit_name,
                    "timing": timing_label,
                    "start_min": start_min,
                    "end_min": end_min,
                    "location": location,
                    "date": dt,
                    "habit_source_path": habit_source_path,
                })

    # Group events by (window_id, date) and detect conflicts
    aggregated: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    grouped: Dict[Tuple[str, date], List[Dict[str, object]]] = {}

    for ev in events:
        dt = ev.get("date")
        if not isinstance(dt, date):
            continue
        key = (ev.get("window_id") or "unknown_window", dt)
        grouped.setdefault(key, []).append(ev)

    # Check for overlaps within each (window, date) bucket
    for (window_id, dt), bucket in grouped.items():
        bucket = sorted(bucket, key=lambda e: e["start_min"])
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]

                # Get locations
                loc_a = str(a.get("location") or "").strip()
                loc_b = str(b.get("location") or "").strip()

                # Determine if there's a conflict based on location
                has_conflict = False
                conflict_type = ""

                if loc_a and loc_b and loc_a == loc_b:
                    # Same location: conflict if time overlaps
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "same_location_time_overlap"
                elif loc_a and loc_b and loc_a != loc_b:
                    # Different locations: conflict if less than 30 min gap
                    # Gap is the time between end of earlier event and start of later event
                    gap_min = b["start_min"] - a["end_min"]
                    if gap_min < 30:
                        has_conflict = True
                        conflict_type = "different_location_insufficient_gap"
                else:
                    # One or both locations are empty, treat as same location
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "time_overlap_no_location"

                if has_conflict:
                    # Canonical pair key to aggregate same habit pairs
                    habit_a = str(a.get("habit") or "")
                    habit_b = str(b.get("habit") or "")
                    if habit_a <= habit_b:
                        pair_key = (window_id, habit_a, habit_b)
                        left, right = a, b
                    else:
                        pair_key = (window_id, habit_b, habit_a)
                        left, right = b, a

                    overlap_range = f"{_format_minutes(max(a['start_min'], b['start_min']))}-{_format_minutes(min(a['end_min'], b['end_min']))}"

                    # Get the source paths for both habits
                    path_a = str(a.get("habit_source_path") or "")
                    path_b = str(b.get("habit_source_path") or "")

                    # If source paths are not available, construct default paths
                    if not path_a:
                        if window_id == "initial_state":
                            path_a = f"initial_state.habits_state.{habit_a}"
                        else:
                            path_a = f"{window_id}.habits.{habit_a}"

                    if not path_b:
                        if window_id == "initial_state":
                            path_b = f"initial_state.habits_state.{habit_b}"
                        else:
                            path_b = f"{window_id}.habits.{habit_b}"

                    # Use a combined path for the conflict
                    path = f"{path_a} <-> {path_b}"

                    # Aggregate by pair key
                    entry = aggregated.setdefault(pair_key, {
                        "window_id": window_id,
                        "window_range": bucket[0].get("window_range"),
                        "habit_a": habit_a,
                        "habit_b": habit_b,
                        "habit_a_timing": left.get("timing"),
                        "habit_b_timing": right.get("timing"),
                        "habit_a_location": left.get("location", ""),
                        "habit_b_location": right.get("location", ""),
                        "conflict_type": conflict_type,
                        "path": path,  # Add path field with actual source locations
                        "occurrences": 0,
                        "overlap_examples": [],
                    })
                    entry["occurrences"] = entry.get("occurrences", 0) + 1
                    if len(entry["overlap_examples"]) < 3:
                        gap_info = ""
                        if conflict_type == "different_location_insufficient_gap":
                            gap_min = b["start_min"] - a["end_min"]
                            gap_info = f", gap: {gap_min} min"
                        entry["overlap_examples"].append({
                            "date": dt.isoformat(),
                            "overlap": overlap_range + gap_info,
                        })

    conflicts = list(aggregated.values())

    # # Build conflict graph by window
    # graph_by_window: Dict[str, Dict[str, object]] = {}
    # TOP_N = 8

    # for entry in conflicts:
    #     window_id = entry.get("window_id") or "unknown_window"
    #     occ = int(entry.get("occurrences") or 1)
    #     window_graph = graph_by_window.setdefault(
    #         window_id, {"total_conflicts": 0, "node_degrees": {}}
    #     )
    #     window_graph["total_conflicts"] += occ

    #     # Count degree for each habit (number of conflict occurrences)
    #     for habit_key in ("habit_a", "habit_b"):
    #         habit_name = entry.get(habit_key) or ""
    #         node_degrees: Dict[str, int] = window_graph["node_degrees"]  # type: ignore
    #         node_degrees[habit_name] = node_degrees.get(habit_name, 0) + occ

    # # Convert node_degrees to sorted top list
    # for window_id, data in graph_by_window.items():
    #     node_degrees = data.get("node_degrees", {}) or {}
    #     top_nodes = sorted(
    #         [{"habit": habit, "degree": degree} for habit, degree in node_degrees.items()],
    #         key=lambda x: (-x["degree"], x["habit"])
    #     )[:TOP_N]
    #     data["top_nodes"] = top_nodes
    #     data.pop("node_degrees", None)

    # Add unique IDs to conflicts
    for idx, conflict in enumerate(conflicts):
        if "id" not in conflict:
            conflict["id"] = f"rule5_{idx:03d}"
            # Build message with occurrence info
            habit_a = conflict.get("habit_a")
            habit_b = conflict.get("habit_b")
            timing_a = conflict.get("habit_a_timing")
            timing_b = conflict.get("habit_b_timing")
            loc_a = conflict.get("habit_a_location", "")
            loc_b = conflict.get("habit_b_location", "")
            conflict_type = conflict.get("conflict_type", "")
            occ_count = conflict.get("occurrences", 1)
            window_id = conflict.get("window_id")

            # Build location description
            loc_desc = ""
            if conflict_type == "same_location_time_overlap":
                loc_desc = f" (both at '{loc_a}')"
            elif conflict_type == "different_location_insufficient_gap":
                loc_desc = f" ('{habit_a}' at '{loc_a}', '{habit_b}' at '{loc_b}' - insufficient travel time)"
            elif loc_a or loc_b:
                loc_desc = f" ('{habit_a}' at '{loc_a}', '{habit_b}' at '{loc_b}')"

            conflict["message"] = (
                f"[ID: {conflict['id']}] Time conflict in {window_id}: "
                f"'{habit_a}' ({timing_a}) overlaps with '{habit_b}' ({timing_b}){loc_desc}. "
                f"Occurs {occ_count} time(s) in this window."
            )

    return {
        "conflicts": conflicts,
        # "graph_by_window": graph_by_window,
    }


def _format_minutes(minutes: int) -> str:
    """Convert minutes since midnight to time string like '6:30 AM'."""
    hours = minutes // 60
    mins = minutes % 60

    if hours >= 12:
        period = "PM"
        display_hour = hours if hours == 12 else hours - 12
    else:
        period = "AM"
        display_hour = hours if hours > 0 else 12

    return f"{display_hour}:{mins:02d} {period}"


# Individual rule fix functions
def _fix_rule4_violations(
    llm_client,
    domain,
    user_profile: str,
    dynamic_profile: Dict,
    detected_issues: List[Dict[str, str]],
) -> tuple[Dict | None, Dict]:
    """Fix Rule 4 violations using LLM."""
    if not detected_issues:
        return None, {}

    prompt = rule4_short_term_followup_prompt.render(
        domain_name=domain.domain_name,
        domain_scope_definition=domain.domain_scope_definition,
        user_profile=user_profile,
        detected_issues=json.dumps(detected_issues, indent=2, ensure_ascii=False),
        dynamic_profile_json=json.dumps(dynamic_profile, indent=2, ensure_ascii=False),
    )

    result = llm_client.generate_json(prompt)
    return result.data, {"rule4_fix": result.usage}


def _fix_rule2_violations(
    llm_client,
    domain,
    user_profile: str,
    dynamic_profile: Dict,
    detected_issues: List[Dict[str, str]],
) -> tuple[Dict | None, Dict]:
    """Fix Rule 2 violations using LLM."""
    if not detected_issues:
        return None, {}

    prompt = rule2_prior_existence_prompt.render(
        domain_name=domain.domain_name,
        domain_scope_definition=domain.domain_scope_definition,
        user_profile=user_profile,
        detected_issues=json.dumps(detected_issues, indent=2, ensure_ascii=False),
        dynamic_profile_json=json.dumps(dynamic_profile, indent=2, ensure_ascii=False),
    )

    result = llm_client.generate_json(prompt)
    return result.data, {"rule2_fix": result.usage}


def _fix_rule1_violations(
    llm_client,
    domain,
    user_profile: str,
    dynamic_profile: Dict,
    detected_issues: List[Dict[str, str]],
) -> tuple[Dict | None, Dict]:
    """Fix Rule 1 violations using LLM."""
    if not detected_issues:
        return None, {}

    prompt = rule1_required_fields_prompt.render(
        domain_name=domain.domain_name,
        domain_scope_definition=domain.domain_scope_definition,
        user_profile=user_profile,
        schema_excerpt=DYNAMIC_PROFILE_TEMPLATE_EXCERPT,
        detected_issues=json.dumps(detected_issues, indent=2, ensure_ascii=False),
        dynamic_profile_json=json.dumps(dynamic_profile, indent=2, ensure_ascii=False),
    )

    result = llm_client.generate_json(prompt)
    return result.data, {"rule1_fix": result.usage}


def _fix_rule3_violations(
    llm_client,
    domain,
    user_profile: str,
    dynamic_profile: Dict,
    detected_issues: List[Dict[str, str]],
) -> tuple[Dict | None, Dict]:
    """Fix Rule 3 violations using LLM."""
    if not detected_issues:
        return None, {}

    prompt = rule3_essential_initialization_prompt.render(
        domain_name=domain.domain_name,
        domain_scope_definition=domain.domain_scope_definition,
        user_profile=user_profile,
        detected_issues=json.dumps(detected_issues, indent=2, ensure_ascii=False),
        dynamic_profile_json=json.dumps(dynamic_profile, indent=2, ensure_ascii=False),
    )

    result = llm_client.generate_json(prompt)
    return result.data, {"rule3_fix": result.usage}


def _fix_rule5_violations(
    llm_client,
    domain,
    user_profile: str,
    dynamic_profile: Dict,
    detected_conflicts: Dict[str, object],
) -> tuple[Dict | None, Dict]:
    """Fix Rule 5 violations using LLM."""
    conflicts_list = detected_conflicts.get("conflicts") or []
    if not conflicts_list:
        return None, {}

    prompt = rule5_time_conflict_prompt.render(
        domain_name=domain.domain_name,
        domain_scope_definition=domain.domain_scope_definition,
        user_profile=user_profile,
        detected_issues=json.dumps(conflicts_list, indent=2, ensure_ascii=False),
        dynamic_profile_json=json.dumps(dynamic_profile, indent=2, ensure_ascii=False),
    )

    result = llm_client.generate_json(prompt)
    ## record prompt and result to file

    return result.data, {"rule5_fix": result.usage}, prompt


PRICING_TABLE: Dict[str, Dict[str, Any]] = {
    "gemini_2_5_flash_lite": {
        "tier": "flash_lite",
        "pricing": {
            "input_per_1M_tokens_usd": 0.10,
            "output_per_1M_tokens_usd": 0.40,
        },
    },
    "gemini_2_5_flash": {
        "tier": "flash",
        "pricing": {
            "input_per_1M_tokens_usd": 0.50,
            "output_per_1M_tokens_usd": 3.00,
        },
    },
    "gemini_3_flash_preview": {
        "tier": "flash",
        "pricing": {
            "input_per_1M_tokens_usd": 0.50,
            "output_per_1M_tokens_usd": 3.00,
        },
    },
    "gemini_3_pro": {
        "tier": "pro",
        "pricing_tiers": {
            "up_to_200k_context_tokens": {
                "input_per_1M_tokens_usd": 2.00,
                "output_per_1M_tokens_usd": 12.00,
            },
            "over_200k_context_tokens": {
                "input_per_1M_tokens_usd": 4.00,
                "output_per_1M_tokens_usd": 18.00,
            },
        },
    },
}


def _summarize_usage_counts(node: Any) -> Dict[str, int]:
    """
    Walk a nested usage tree and accumulate token counts.
    A usage leaf is any dict containing prompt/completion/total keys.
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            if {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            }.issubset(value.keys()):
                totals["prompt_tokens"] += int(value.get("prompt_tokens", 0) or 0)
                totals["completion_tokens"] += int(
                    value.get("completion_tokens", 0) or 0
                )
                totals["total_tokens"] += int(value.get("total_tokens", 0) or 0)
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(node)
    return totals


def _summarize_usage(aggregate_usage: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Summarize token usage per top-level section plus a grand total.
    """
    summary = {
        key: _summarize_usage_counts(value) for key, value in aggregate_usage.items()
    }
    summary["grand_total"] = _summarize_usage_counts(aggregate_usage)
    return summary


def _normalize_model_key(model_name: str) -> str:
    return _slugify(model_name.replace(".", "_"))


def _lookup_pricing_for_model(model_name: str) -> Dict[str, float] | None:
    pricing_key = _normalize_model_key(model_name)
    entry = PRICING_TABLE.get(pricing_key)
    if not entry:
        return None
    if "pricing" in entry:
        return entry["pricing"]
    if "pricing_tiers" in entry:
        tiers = entry["pricing_tiers"]
        # Default to the lower tier unless specified otherwise.
        return tiers.get("up_to_200k_context_tokens") or next(iter(tiers.values()), None)
    return None


def _compute_usage_pricing(
    usage_summary: Dict[str, Dict[str, int]], model_name: str
) -> Dict[str, object] | None:
    pricing = _lookup_pricing_for_model(model_name)
    if not pricing:
        return None

    grand = usage_summary.get("grand_total") or {}
    prompt_tokens = int(grand.get("prompt_tokens", 0) or 0)
    completion_tokens = int(grand.get("completion_tokens", 0) or 0)

    input_rate = pricing.get("input_per_1M_tokens_usd")
    output_rate = pricing.get("output_per_1M_tokens_usd")
    if input_rate is None or output_rate is None:
        return None

    input_cost = prompt_tokens / 1_000_000 * input_rate
    output_cost = completion_tokens / 1_000_000 * output_rate

    return {
        "model_name": model_name,
        "pricing_key": _normalize_model_key(model_name),
        "input_rate_per_million": input_rate,
        "output_rate_per_million": output_rate,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def _persist_usage_artifacts(
    aggregate_usage: Dict[str, Any],
    output_dir: Path,
    *,
    basename: str = "token_usage",
    model_name: str | None = None,
) -> tuple[Dict[str, Dict[str, int]], Dict[str, object] | None]:
    """
    Persist raw and summarized token usage to disk.
    """
    summary = _summarize_usage(aggregate_usage)
    pricing = _compute_usage_pricing(summary, model_name) if model_name else None
    _write_json(output_dir / f"{basename}.json", aggregate_usage)
    _write_json(output_dir / f"{basename}_summary.json", summary)
    if pricing:
        _write_json(output_dir / f"{basename}_pricing.json", pricing)
    return summary, pricing


def _segment_world_background_text(world_background: str) -> List[str]:
    """
    Split a world background text into segments using bold headers as boundaries.
    If no headers are found, return the whole text as a single segment.
    """
    if not isinstance(world_background, str):
        return []

    segments: List[str] = []
    current_lines: List[str] = []
    header_pattern = re.compile(r"^\s*\*\*.+\*\*\s*$")

    for line in world_background.splitlines():
        if header_pattern.match(line):
            segment = "\n".join(current_lines).strip()
            if segment:
                segments.append(segment)
            current_lines = []
            continue
        current_lines.append(line)

    final_segment = "\n".join(current_lines).strip()
    if final_segment:
        segments.append(final_segment)

    if not segments and world_background.strip():
        return [world_background.strip()]
    return segments


def _parse_world_background_by_window(text: str) -> Dict[str, str]:
    """
    Parse a world background text that labels sections with window ids (e.g., 'w1:', 'w2:').
    We scan the whole text for 'w<number>:' markers and slice between them, so labels do not
    need to be at line starts.
    """
    sections: Dict[str, str] = {}
    pattern = re.compile(r"(w\d+)\s*:", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    for idx, match in enumerate(matches):
        window_id = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[window_id] = content
    return sections


def _map_world_background_to_windows(
    world_background: str | Dict[str, str] | None, window_ids: List[str]
) -> Dict[str, str]:
    """
    Map world background to each window. Supports:
      - dict mapping window_id -> text (with optional "default")
      - JSON string that can be parsed into such a dict (common when loading from disk)
      - plain text split into sequential segments (aligned by index to window_ids)
      - fallback: same text for all windows
    """
    mapped: Dict[str, str] = {}
    if isinstance(world_background, str):
        try:
            parsed = json.loads(world_background)
            if isinstance(parsed, dict):
                world_background = parsed
        except json.JSONDecodeError:
            # Not a JSON payload; treat as plain text.
            parsed_sections = _parse_world_background_by_window(world_background)
            if parsed_sections:
                default_text = parsed_sections.get("default", "")
                for window_id in window_ids:
                    mapped[window_id] = parsed_sections.get(window_id, default_text)
                return mapped

    if isinstance(world_background, dict):
        default_text = world_background.get("default", "")
        for window_id in window_ids:
            mapped[window_id] = world_background.get(window_id, default_text)
        return mapped

    text = world_background or ""
    segments = _segment_world_background_text(text)
    if not segments:
        return {window_id: "" for window_id in window_ids}

    for idx, window_id in enumerate(window_ids):
        mapped[window_id] = segments[min(idx, len(segments) - 1)]
    return mapped


ATTRIBUTE_CANONICAL_KEYS: Dict[str, str] = {
    # Device/possession clusters
    "user_material_possessions": "user_devices_and_possessions",
    "user_tech_gadgets": "user_devices_and_possessions",
    "user_owned_devices": "user_devices_and_possessions",
    "user_smart_home_devices": "user_devices_and_possessions",
}


def _canonical_attribute_key(attribute_name: str) -> str:
    return ATTRIBUTE_CANONICAL_KEYS.get(attribute_name, attribute_name)


def _normalize_attribute_value(value: object) -> object:
    """Recursively normalize values for comparison across domains."""
    if isinstance(value, list):
        return tuple(_normalize_attribute_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _normalize_attribute_value(v)) for k, v in value.items()))
    return str(value)


def _normalize_collection_item(value: object) -> str:
    """Lightweight normalization for collection items to detect overlaps."""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
        except TypeError:
            return str(value)
    return str(value).strip().lower()


def _collect_cross_domain_attribute_conflicts(
    dynamic_profiles: Dict[str, Dict],
) -> List[Dict[str, object]]:
    """
    Collect all shared attributes across domains after key alignment.

    Simply lists all attributes that appear in multiple domains, showing their values
    in each domain. Does NOT judge whether there are conflicts - that's for the LLM.

    Returns: List of shared attributes, each containing:
    - shared_key: the aligned attribute name
    - attribute_type: "singular" or "collections"
    - windows: list of windows where this attribute appears, with domain values
    """
    # Group by (attr_type, attr_name) across domains
    # After key alignment, attribute names should already be normalized
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict[str, object]]]] = {}

    def _record_entry(
        *,
        window_id: str,
        time_range: object,
        attr_type: str,
        attr_name: str,
        value: object,
        domain: str,
    ) -> None:
        key = (("collections" if attr_type == "collections" else "singular"), attr_name)
        entry = {
            "domain": domain,
            "value": deepcopy(value),
            "time_range": time_range,
        }
        grouped.setdefault(key, {}).setdefault(window_id or "initial", []).append(entry)

    for domain_name, profile in dynamic_profiles.items():
        if not isinstance(profile, dict):
            continue

        user_attrs_state = (profile.get("initial_state") or {}).get("user_attributes_state") or {}
        for attr_type in ("singular", "collections"):
            entries = user_attrs_state.get(attr_type)
            if isinstance(entries, dict):
                for attr_name, value in entries.items():
                    _record_entry(
                        window_id="initial",
                        time_range=None,
                        attr_type=attr_type,
                        attr_name=attr_name,
                        value=value,
                        domain=domain_name,
                    )

        # Legacy fallback: user_attributes_state.initial
        legacy_entries = _extract_initial_state_entries(user_attrs_state.get("initial"))
        for attr_name, entry in legacy_entries.items():
            _record_entry(
                window_id="initial",
                time_range=None,
                attr_type="singular",
                attr_name=attr_name,
                value=entry.get("current_value"),
                domain=domain_name,
            )

        for window in _resolve_window_states(profile):
            window_id = window.get("window_id") or "unknown_window"
            time_range = window.get("time_range")
            for item in window.get("user_attributes_state", []) or []:
                attr_name = item.get("name")
                if not attr_name:
                    continue
                attr_type = (item.get("attribute_type") or "singular").lower()
                _record_entry(
                    window_id=window_id,
                    time_range=time_range,
                    attr_type=attr_type,
                    attr_name=attr_name,
                    value=item.get("current_value"),
                    domain=domain_name,
                )

    # Collect all shared attributes (appearing in multiple domains)
    shared_attributes: List[Dict[str, object]] = []
    for (attr_type, attr_name), windows_map in grouped.items():
        windows_payload: List[Dict[str, object]] = []

        for window_id, entries in windows_map.items():
            # Only include windows where this attribute appears in multiple domains
            if len(entries) <= 1:
                continue

            time_range = next(
                (entry.get("time_range") for entry in entries if entry.get("time_range") is not None),
                None,
            )

            # Simply list all domain values
            domains_list = [
                {
                    "domain": entry["domain"],
                    "value": entry["value"],
                }
                for entry in entries
            ]

            windows_payload.append(
                {
                    "window_id": window_id,
                    "time_range": time_range,
                    "domains": domains_list,
                }
            )

        # Only include attributes that appear in multiple domains in at least one window
        if windows_payload:
            shared_attributes.append(
                {
                    "shared_key": attr_name,
                    "attribute_type": attr_type,
                    "windows": windows_payload,
                }
            )

    return shared_attributes


def _replace_or_add_initial_attribute(
    profile: Dict,
    attribute_name: str,
    new_value: object,
    *,
    attribute_type: str = "singular",
) -> None:
    """
    Upsert an attribute into the initial_state using the new schema
    (singular/collections). Falls back to legacy "initial" if present.
    """
    user_attrs_state = profile.setdefault("initial_state", {}).setdefault(
        "user_attributes_state", {}
    )
    attr_type = (
        "collections" if (attribute_type or "").lower() == "collections" else "singular"
    )
    if attr_type == "collections":
        collections = user_attrs_state.setdefault("collections", {})
        if not isinstance(collections, dict):
            collections = {}
            user_attrs_state["collections"] = collections
        collections[attribute_name] = (
            new_value if isinstance(new_value, list) else ([] if new_value is None else [new_value])
        )
        return

    singular = user_attrs_state.setdefault("singular", {})
    if not isinstance(singular, dict):
        singular = {}
        user_attrs_state["singular"] = singular
    singular[attribute_name] = new_value

    # Legacy fallback: mirror into user_attributes_state.initial if present
    legacy_initial = user_attrs_state.get("initial")
    if isinstance(legacy_initial, dict):
        legacy_initial[attribute_name] = new_value
    elif isinstance(legacy_initial, list):
        replaced = False
        for entry in legacy_initial:
            if isinstance(entry, dict) and attribute_name in entry:
                entry[attribute_name] = new_value
                replaced = True
        if not replaced:
            legacy_initial.append({attribute_name: new_value})


def _merge_values(
    existing: object, new_value: object, *, value_kind: str | None
) -> object:
    """
    Merge values when appropriate. For set-like fields, perform union.
    """
    if value_kind == "set":
        if isinstance(existing, list) and isinstance(new_value, list):
            seen = set()
            merged: List[object] = []
            for item in list(existing) + list(new_value):
                marker = _normalize_attribute_value(item)
                if marker in seen:
                    continue
                seen.add(marker)
                merged.append(item)
            return merged
    return new_value


def _set_attribute_in_window(
    profile: Dict,
    window_id: str | None,
    attribute_name: str,
    new_value: object,
    *,
    attribute_type: str = "singular",
    value_kind: str | None = None,
) -> None:
    """
    Apply a resolved attribute to either the initial state or a specific window delta.
    """
    if not window_id or window_id == "initial":
        initial_state = profile.setdefault("initial_state", {}).setdefault(
            "user_attributes_state", {}
        )
        attr_type = (
            "collections"
            if (attribute_type or "").lower() == "collections"
            else "singular"
        )
        container = initial_state.get(attr_type)
        current_val = None
        if isinstance(container, dict):
            current_val = container.get(attribute_name)
        merged_value = _merge_values(current_val, new_value, value_kind=value_kind)
        _replace_or_add_initial_attribute(
            profile,
            attribute_name,
            merged_value,
            attribute_type=attr_type,
        )
        return

    for window in profile.get("time_windows", []):
        if window.get("window_id") != window_id:
            continue
        delta = window.setdefault("user_attributes_delta", {}).setdefault(
            "operations", []
        )
        attr_type = (
            "collections"
            if (attribute_type or "").lower() == "collections"
            else "singular"
        )
        name_field = "collection_name" if attr_type == "collections" else "attribute_name"
        # Update last matching operation if present.
        updated = False
        for op in reversed(delta):
            if (
                isinstance(op, dict)
                and op.get(name_field) == attribute_name
                and (op.get("attribute_type") in {None, attr_type} or attr_type == "singular")
            ):
                current_val = op.get("delta") or op.get("new_state")
                merged_value = _merge_values(
                    current_val, new_value, value_kind=value_kind
                )
                op["delta"] = merged_value
                op["attribute_type"] = attr_type
                op[name_field] = attribute_name
                updated = True
                break
        if not updated:
            delta.append(
                {
                    "op": "modify" if attr_type == "singular" else "add",
                    "attribute_type": attr_type,
                    name_field: attribute_name,
                    "delta": new_value,
                    "reason": "conflict_resolution",
                }
            )
        break


def _update_attribute_deltas(profile: Dict, attribute_name: str, new_value: object) -> None:
    for window in profile.get("time_windows", []):
        operations = (
            (window.get("user_attributes_delta") or {}).get("operations") or []
        )
        for op in operations:
            target_name = op.get("attribute_name") or op.get("collection_name")
            if target_name == attribute_name and op.get("op") in {"add", "modify"}:
                op["delta"] = new_value
                if "new_state" in op:
                    op["new_state"] = new_value


def _apply_key_alignment(
    dynamic_profiles: Dict[str, Dict],
    alignment_payload: Dict[str, object],
) -> tuple[Dict[str, Dict], Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Apply LLM-suggested key alignment mappings and deterministic renames (no value merging).
    """
    resolved_profiles = deepcopy(dynamic_profiles)
    if not isinstance(alignment_payload, dict):
        return resolved_profiles, {}, {}

    # Normalize the LLM output into a per-domain mapping:
    # domain -> original_key -> canonical_key
    domain_key_mapping: Dict[str, Dict[str, str]] = {}
    canonical_descriptions: Dict[str, str] = {}

    canonical_entries = alignment_payload.get("canonical_key_mappings") or alignment_payload.get("canonical_keys") or {}
    if isinstance(canonical_entries, dict):
        for canonical_key, entry in canonical_entries.items():
            if not isinstance(entry, dict):
                continue
            description = entry.get("description") or entry.get("note")
            if description:
                canonical_descriptions[canonical_key] = description

            for domain_entry in entry.get("domains") or []:
                if not isinstance(domain_entry, dict):
                    continue
                domain_name = domain_entry.get("domain")
                original_key = (
                    domain_entry.get("original_key")
                    or domain_entry.get("attribute_name")
                    or domain_entry.get("key")
                )
                if not domain_name or not original_key:
                    continue
                domain_mapping = domain_key_mapping.setdefault(domain_name, {})
                domain_mapping[original_key] = canonical_key
                if "." in original_key:
                    base_key = original_key.rsplit(".", 1)[-1]
                    if base_key:
                        domain_mapping.setdefault(base_key, canonical_key)

    def _rename_initial_entries(entries: object, mapping: Dict[str, str]) -> object:
        """
        Rename keys in initial_state.user_attributes_state.initial while preserving shape (dict vs list).
        """
        if entries is None:
            return entries

        aggregated: Dict[str, object] = {}

        # Support both dict and list-of-dicts forms.
        items = []
        if isinstance(entries, dict):
            items = list(entries.items())
        elif isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    items.extend(entry.items())
        else:
            return entries

        for key, value in items:
            target_key = mapping.get(key, key)
            aggregated[target_key] = deepcopy(value)

        if isinstance(entries, dict):
            return aggregated
        return [{k: v} for k, v in aggregated.items()]

    def _drop_prefixed_duplicates(entries: object, mapping: Dict[str, str]) -> object:
        """
        Remove prefixed duplicates such as specific_<key>/user_specific_<key> when the base key is present.
        """
        prefixes = ("specific_", "user_specific_")

        def should_drop(key: str, container: Dict[str, object]) -> bool:
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                base = key[len(prefix) :]
                canonical_base = mapping.get(base, base)
                canonical_key = mapping.get(key, key)
                # Drop if the base (or canonical base) is already present.
                if base in container or canonical_base in container or canonical_key == canonical_base:
                    return True
            return False

        if isinstance(entries, dict):
            return {k: v for k, v in entries.items() if not should_drop(k, entries)}

        if isinstance(entries, list):
            cleaned: List[object] = []
            for entry in entries:
                if isinstance(entry, dict):
                    cleaned.append({k: v for k, v in entry.items() if not should_drop(k, entry)})
                else:
                    cleaned.append(entry)
            return cleaned
        return entries

    for domain_name, profile in resolved_profiles.items():
        # Some LLM responses wrap the profile object in a singleton list; unwrap to keep a consistent shape.
        if isinstance(profile, list):
            if len(profile) == 1 and isinstance(profile[0], dict):
                profile = profile[0]
                resolved_profiles[domain_name] = profile
            else:
                raise ValueError(
                    f"Dynamic profile for {domain_name} should be an object, got list (len={len(profile)})"
                )
        if not isinstance(profile, dict):
            raise ValueError(
                f"Dynamic profile for {domain_name} should be a dict, got {type(profile).__name__}"
            )
        mapping = dict(ATTRIBUTE_CANONICAL_KEYS)
        mapping.update(domain_key_mapping.get(domain_name, {}))

        # Rename in initial_state.user_attributes_state (singular/collections)
        initial_state = (profile.get("initial_state") or {}).get(
            "user_attributes_state"
        ) or {}
        for section in ("singular", "collections"):
            entries = initial_state.get(section)
            if isinstance(entries, dict):
                renamed_section = {
                    mapping.get(key, key): deepcopy(value)
                    for key, value in entries.items()
                }
                initial_state[section] = _drop_prefixed_duplicates(
                    renamed_section, mapping
                )

        # Legacy fallback: user_attributes_state.initial
        if "initial" in initial_state:
            renamed = _rename_initial_entries(initial_state.get("initial"), mapping)
            initial_state["initial"] = _drop_prefixed_duplicates(renamed, mapping)

        # Rename in each window's user_attributes_delta.operations
        for window in profile.get("time_windows", []):
            delta_ops = _get_delta_operations(
                (window.get("user_attributes_delta") or {})
            )
            for op in delta_ops:
                if not isinstance(op, dict):
                    continue
                attr_type_raw = (op.get("attribute_type") or "").lower()
                inferred_type = (
                    "collections" if op.get("collection_name") else "singular"
                )
                attr_type = (
                    "collections"
                    if attr_type_raw in {"collection", "collections"}
                    else attr_type_raw
                    or inferred_type
                )
                name_field = "collection_name" if attr_type == "collections" else "attribute_name"
                attr_name = (
                    op.get(name_field)
                    or op.get("attribute_name")
                    or op.get("collection_name")
                )
                if not attr_name:
                    continue
                mapped = mapping.get(attr_name)
                if mapped:
                    op[name_field] = mapped
                op["attribute_type"] = attr_type

    return resolved_profiles, domain_key_mapping, canonical_descriptions


def _apply_conflict_resolution_to_profiles(
    dynamic_profiles: Dict[str, Dict],
    resolution_payload: Dict[str, object] | None,
) -> Dict[str, Dict]:
    resolved_profiles = deepcopy(dynamic_profiles)
    if not isinstance(resolution_payload, dict):
        return resolved_profiles

    def _parse_path_tokens(path: str | None) -> List[object]:
        """
        Split a dotted path like "habits_delta.operations[0].new_state.timing"
        into ["habits_delta", "operations", 0, "new_state", "timing"].
        Supports selectors: [window_id=w3] or [habit_name=foo] → {"_selector_key": "...", "_selector_value": "..."}.
        """
        if not isinstance(path, str):
            return []

        def _make_selector(raw: str) -> Dict[str, str] | str | int:
            raw = raw.strip()
            if raw.isdigit():
                return int(raw)
            if "=" in raw:
                key, value = raw.split("=", 1)
                return {"_selector_key": key.strip(), "_selector_value": value.strip()}
            return raw

        tokens: List[object] = []
        for segment in path.split("."):
            if not segment:
                continue
            for match in re.finditer(r"([^\[\]]+)|\[(.*?)\]", segment):
                plain, bracket = match.groups()
                if plain:
                    tokens.append(plain)
                elif bracket:
                    selector = _make_selector(bracket)
                    tokens.append(selector)
        return tokens

    def _is_selector_token(token: object) -> bool:
        return isinstance(token, dict) and "_selector_key" in token and "_selector_value" in token

    def _normalize_window_id(window_id: str | None) -> str:
        """
        Normalize window identifiers to match stored window_ids.
        Treats None/""/initial_state as "initial".
        """
        if window_id is None:
            return "initial"
        normalized = str(window_id).strip()
        lower = normalized.lower()
        if lower in {"", "initial", "initial_state", "initialstate", "init"}:
            return "initial"
        return normalized

    def _get_window_root(profile: Dict, window_id: str | None) -> Dict | None:
        normalized = _normalize_window_id(window_id)
        if normalized == "initial":
            return profile.setdefault("initial_state", {})
        for window in profile.get("time_windows", []):
            wid = window.get("window_id")
            if isinstance(wid, str) and wid.lower() == normalized.lower():
                return window
        return None

    def _strip_redundant_window_tokens(
        tokens: List[object], window_id: str | None, root: Dict | None
    ) -> List[object]:
        """
        Some patch formats repeat the window_id inside the path (e.g., habits_state.*).
        Strip that marker since we already route to the correct window root.
        """
        if not tokens or not isinstance(window_id, str):
            return tokens
        normalized = _normalize_window_id(window_id).lower()

        def _matches_window(token_str: str) -> bool:
            base = token_str.lower()
            if base.endswith("_state"):
                base = base[: -len("_state")]
            return base == normalized

        # Drop a leading window_id marker, e.g., "initial.habits_state..."
        if isinstance(tokens[0], str) and _matches_window(tokens[0]):
            tokens = tokens[1:]

        # Drop a window_id marker immediately after a state section.
        if len(tokens) >= 2 and isinstance(tokens[0], str) and isinstance(tokens[1], str):
            parent_container = None
            if isinstance(root, dict):
                parent_container = root.get(tokens[0])
            has_window_key = False
            if isinstance(parent_container, dict):
                for key in parent_container.keys():
                    if isinstance(key, str) and _matches_window(key):
                        has_window_key = True
                        break
            if _matches_window(tokens[1]) and tokens[0] in {
                "user_attributes_state",
                "habits_state",
                "preferences_state",
            } and not has_window_key:
                tokens = [tokens[0]] + tokens[2:]

        # Drop leading explicit container references like time_windows[2] or initial_state.*
        if len(tokens) >= 2 and tokens[0] == "time_windows" and (
            isinstance(tokens[1], int) or _is_selector_token(tokens[1])
        ):
            tokens = tokens[2:]
        if tokens:
            first = tokens[0]
            if isinstance(first, str) and first.lower() in {"initial_state", "time_windows"}:
                tokens = tokens[1:]

        return tokens

    def _strip_context_prefix(tokens: List[object]) -> List[object]:
        """
        Remove leading context markers (e.g., 'Dynamic profiles') that aren't part of the domain path.
        """
        prefixes = {
            "dynamic profiles",
            "dynamic profile",
            "dynamic_profiles",
            "dynamic_profile",
            "dynamicprofiles",
        }
        while tokens and isinstance(tokens[0], str) and tokens[0].strip().lower() in prefixes:
            tokens = tokens[1:]
        return tokens

    def _infer_window_id_from_tokens(tokens: List[object], profile: Dict | None) -> str | None:
        """
        Deduce the window_id using explicit markers (w3, initial) or time_windows indices/selectors.
        """
        for token in tokens:
            if isinstance(token, str):
                token_lower = token.lower()
                if token_lower in {"initial", "initial_state", "init"}:
                    return "initial"
                if re.fullmatch(r"w\d+", token_lower):
                    return token
            elif _is_selector_token(token):
                key = str(token.get("_selector_key", "")).lower()
                if key == "window_id":
                    return token.get("_selector_value")

        if not isinstance(profile, dict):
            return None
        windows = profile.get("time_windows")
        for idx, token in enumerate(tokens):
            if token == "time_windows" and idx + 1 < len(tokens):
                win_token = tokens[idx + 1]
                if isinstance(win_token, int):
                    if isinstance(windows, list) and 0 <= win_token < len(windows):
                        window_entry = windows[win_token]
                        if isinstance(window_entry, dict):
                            return window_entry.get("window_id") or f"w{win_token + 1}"
                        return f"w{win_token + 1}"
                elif _is_selector_token(win_token):
                    selector_key = str(win_token.get("_selector_key", "")).lower()
                    selector_val = win_token.get("_selector_value")
                    if selector_key == "window_id":
                        return selector_val
                    if isinstance(windows, list):
                        for window in windows:
                            if isinstance(window, dict) and str(window.get(selector_key)) == str(selector_val):
                                return window.get("window_id") or selector_val
                break
        return None

    def _apply_path_update(
        root: Dict, tokens: List[object], op: str, value: object, *, value_provided: bool
    ) -> None:
        """
        Generic setter/deleter/append that can walk dicts/lists, creating containers as needed for updates.
        """
        if not tokens:
            return
        op_lower = (op or "update").lower()
        is_delete = op_lower in {"delete", "remove"}
        is_append = op_lower in {"append", "add", "extend", "push"}

        parent = root
        for idx, token in enumerate(tokens[:-1]):
            next_token = tokens[idx + 1] if idx + 1 < len(tokens) else None

            if _is_selector_token(token):
                key = token["_selector_key"]  # type: ignore
                val = token["_selector_value"]  # type: ignore
                if not isinstance(parent, list):
                    return
                match_idx = None
                for i, item in enumerate(parent):
                    if isinstance(item, dict) and str(item.get(key)) == str(val):
                        match_idx = i
                        break
                if match_idx is None:
                    if is_delete:
                        return
                    new_item: Dict[str, object] = {key: val}
                    if isinstance(next_token, int):
                        new_item[key] = []
                    parent.append(new_item)
                    match_idx = len(parent) - 1
                if not isinstance(parent[match_idx], (dict, list)):
                    if is_delete:
                        return
                    parent[match_idx] = {} if isinstance(next_token, (str, dict)) else []  # type: ignore
                parent = parent[match_idx]  # type: ignore
            elif isinstance(token, str):
                if not isinstance(parent, dict):
                    if is_delete:
                        return
                    return
                if token not in parent or not isinstance(parent[token], (dict, list)):
                    if is_delete:
                        return
                    parent[token] = [] if isinstance(next_token, int) else {}
                parent = parent[token]
            else:  # token is int
                if not isinstance(parent, list):
                    return
                while len(parent) <= token:  # type: ignore
                    parent.append({} if isinstance(next_token, str) else None)  # type: ignore
                if not isinstance(parent[token], (dict, list)) and isinstance(next_token, (str, int, dict)):
                    if not is_delete:
                        parent[token] = {} if isinstance(next_token, (str, dict)) else []  # type: ignore
                parent = parent[token]  # type: ignore

        last = tokens[-1]
        if _is_selector_token(last):
            return  # selectors should not be terminal
        if isinstance(last, str):
            if not isinstance(parent, dict):
                return
            if is_delete:
                parent.pop(last, None)
                return
            if is_append:
                existing = parent.get(last)
                if isinstance(existing, list):
                    existing.append(value)
                elif existing is None:
                    parent[last] = [value]
                else:
                    parent[last] = [existing, value] if value_provided else existing
                return
            if op_lower in {"replace", "update", "modify", "set"} and value_provided and (
                value == [] or value == {}
            ):
                # Treat explicit empty replacements as removal to avoid leaving empty containers.
                parent.pop(last, None)
                return
            parent[last] = value
        else:
            if not isinstance(parent, list):
                return
            while len(parent) <= last:
                parent.append(None)
            if is_delete:
                if 0 <= last < len(parent):
                    parent.pop(last)
                return
            if is_append:
                if last < len(parent) and isinstance(parent[last], list):
                    parent[last].append(value)
                elif last == len(parent):
                    parent.append(value)
                else:
                    parent[last] = value
                return
            parent[last] = value

    if not isinstance(resolution_payload, dict):
        return resolved_profiles

    def _iter_patch_lists(payload: Dict[str, object]) -> List[List[Dict[str, object]]]:
        patch_lists: List[List[Dict[str, object]]] = []
        direct_patches = payload.get("patches")
        if isinstance(direct_patches, list):
            patch_lists.append(direct_patches)
        resolution_block = payload.get("resolution")
        if isinstance(resolution_block, dict):
            nested = resolution_block.get("patches")
            if isinstance(nested, list):
                patch_lists.append(nested)
        conflicts_and_resolutions = payload.get("conflicts_and_resolutions")
        if isinstance(conflicts_and_resolutions, list):
            for conflict_entry in conflicts_and_resolutions:
                if not isinstance(conflict_entry, dict):
                    continue
                entry_patches = None
                resolution_section = conflict_entry.get("resolution")
                if isinstance(resolution_section, dict):
                    entry_patches = resolution_section.get("patches")
                if entry_patches is None:
                    entry_patches = conflict_entry.get("patches")
                if isinstance(entry_patches, list):
                    patch_lists.append(entry_patches)
        return patch_lists

    patch_batches = _iter_patch_lists(resolution_payload)
    if not patch_batches:
        return resolved_profiles

    for patches in patch_batches:
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            domain_name = patch.get("domain") or patch.get("domain_name")
            profile = resolved_profiles.get(domain_name) if domain_name else None
            tokens = _strip_context_prefix(_parse_path_tokens(patch.get("path")))
            if profile is None:
                # Attempt to infer domain from the path prefix (e.g., "Work & Education.initial_state...")
                if tokens and isinstance(tokens[0], str):
                    token_domain = tokens[0]
                    # exact match first
                    if token_domain in resolved_profiles:
                        domain_name = token_domain
                        tokens = tokens[1:]
                        profile = resolved_profiles.get(domain_name)
                    else:
                        # case-insensitive match
                        for candidate in resolved_profiles.keys():
                            if candidate.lower() == token_domain.lower():
                                domain_name = candidate
                                tokens = tokens[1:]
                                profile = resolved_profiles.get(domain_name)
                                break
                if profile is None:
                    continue
            else:
                if len(tokens) < 1:
                    continue

            if tokens and isinstance(tokens[0], str) and domain_name and tokens[0].lower() == str(domain_name).lower():
                tokens = tokens[1:]

            op = str(patch.get("operation") or patch.get("action") or "update").lower()
            value = None
            value_provided = False
            for key in ("new_value", "value", "resolved_value", "new_state", "delta", "replacement"):
                if key in patch:
                    value = patch.get(key)
                    value_provided = True
                    break

            window_id = patch.get("window_id") or patch.get("window")
            if not window_id:
                window_id = _infer_window_id_from_tokens(tokens, profile)

            normalized_window_id = _normalize_window_id(window_id)
            root = _get_window_root(profile, normalized_window_id)
            if root is None:
                continue

            if op in {"delete", "remove"}:
                value_provided = True  # allow deletion without explicit value
            if not value_provided and op in {
                "append",
                "add",
                "extend",
                "push",
                "replace",
                "update",
                "modify",
                "add_key",
                "set",
            }:
                # Skip patches that would overwrite with None when no explicit value is provided.
                continue

            tokens = _strip_redundant_window_tokens(tokens, normalized_window_id, root)
            _apply_path_update(root, tokens, op, value, value_provided=value_provided)

    return resolved_profiles


def _extract_initial_state_entries(
    entries: Dict[str, object] | List[Dict[str, object]] | None,
) -> Dict[str, Dict[str, object]]:
    """
    Extract initial state entries from the new format.
    Each entry is a dict like {"attribute_name": "attribute_value"} or a map of
    names to values.
    """
    state: Dict[str, Dict[str, object]] = {}
    # Some generators produce an object map instead of a list of singleton dicts.
    if isinstance(entries, dict):
        for key, value in entries.items():
            state[key] = {"current_value": value}
        return state

    for entry in entries or []:
        if not isinstance(entry, dict):
            # Treat bare strings as keys with unknown value to avoid crashing.
            if isinstance(entry, str):
                state[entry] = {"current_value": None}
            continue
        for key, value in entry.items():
            state[key] = {"current_value": value}
    return state


def _get_delta_operations(delta_section: Dict | None) -> List[Dict[str, object]]:
    if isinstance(delta_section, dict):
        operations = delta_section.get("operations")
        if isinstance(operations, list):
            return operations
    return []


def _apply_delta_operations(
    state: Dict[str, Dict[str, object]],
    operations: List[Dict[str, object]] | None,
    *,
    name_field: str,
) -> Dict[str, Dict[str, object]]:
    """
    Apply delta operations to state and return a dict of changes.
    
    Returns:
        Dict mapping key to change info: {"previous_value": ..., "change_reason": ..., "op": ...}
    """
    changes: Dict[str, Dict[str, object]] = {}
    if not operations:
        return changes

    for operation in operations:
        key = operation.get(name_field)
        if not key:
            continue

        op_type = (operation.get("op") or "").lower()
        # Support both the old schema (before/after) and the new schema (new_state)
        # Previous value is always taken from the current state snapshot
        prev_entry = state.get(key) or {}
        prev_value = prev_entry.get("current_value")

        new_value = None
        if "delta" in operation:
            new_value = operation.get("delta")
        if new_value is None and "new_state" in operation:
            new_value = operation.get("new_state")
        if new_value is None:
            before = operation.get("before") or {}
            after = operation.get("after") or {}
            # Old schema stores the new value under after["value"]
            new_value = after.get("value")

            # Some generators emit flat fields (e.g., action/schedule/timing) instead
            # of wrapping them under new_state; treat those as the new state payload.
            if new_value is None:
                candidate = {
                    k: v
                    for k, v in operation.items()
                    if k
                    not in {
                        name_field,
                        "delta",
                        "op",
                        "reason",
                        "before",
                        "after",
                        "change_reason",
                    }
                }
                if candidate:
                    new_value = candidate

        reason = operation.get("reason", "")
        
        # Record the change information
        changes[key] = {
            "previous_value": prev_value,
            "change_reason": reason,
            "op": op_type,
        }

        # We keep a tombstone entry for remove/drop so that semantic-event
        # generation can still see the op on this field.
        if op_type in {"remove", "drop"} and "new_state" not in operation and "delta" not in operation:
            # Old schema remove/drop: no explicit new_state, treat as cleared.
            state[key] = {"current_value": None}
        else:
            state[key] = {
                "current_value": new_value,
            }
    
    return changes


def _init_user_attributes_state(
    initial_state: Dict[str, object] | None,
) -> Dict[str, Dict[str, object]]:
    """
    Build a {singular, collections} map from the new schema, with a legacy
    fallback to user_attributes_state.initial when present.
    """
    attrs = {"singular": {}, "collections": {}}
    user_attrs_state = (initial_state or {}).get("user_attributes_state") or {}
    if isinstance(user_attrs_state, dict):
        singular = user_attrs_state.get("singular")
        collections = user_attrs_state.get("collections")
        if isinstance(singular, dict):
            attrs["singular"] = deepcopy(singular)
        if isinstance(collections, dict):
            attrs["collections"] = deepcopy(collections)

        # Legacy fallback: user_attributes_state.initial (dict or list of dicts)
        if not attrs["singular"] and not attrs["collections"]:
            legacy_entries = _extract_initial_state_entries(
                user_attrs_state.get("initial")
            )
            if legacy_entries:
                attrs["singular"] = {
                    name: entry.get("current_value")
                    for name, entry in legacy_entries.items()
                }
    return attrs


def _apply_user_attribute_operations(
    state: Dict[str, Dict[str, object]],
    operations: List[Dict[str, object]] | None,
) -> tuple[Dict[str, Dict[str, object]], Dict[tuple[str, str], Dict[str, object]]]:
    """
    Apply attribute deltas using the new singular/collections semantics.
    """
    updated = {
        "singular": deepcopy(state.get("singular") or {}),
        "collections": deepcopy(state.get("collections") or {}),
    }
    changes: Dict[tuple[str, str], Dict[str, object]] = {}
    if not operations:
        return updated, changes

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        raw_attr_type = (operation.get("attribute_type") or "").lower()
        inferred_type = "collections" if operation.get("collection_name") else "singular"
        attr_type = "collections" if raw_attr_type in {"collection", "collections"} else raw_attr_type or inferred_type
        name = operation.get("attribute_name") or operation.get("collection_name")
        if not name:
            continue

        op_type = (operation.get("op") or "").lower()
        reason = operation.get("reason", "")
        prev_value = deepcopy(updated.get(attr_type, {}).get(name))
        delta_payload = operation.get("delta")
        if delta_payload is None:
            delta_payload = operation.get("new_state")

        if attr_type == "collections":
            prev_list = (
                prev_value
                if isinstance(prev_value, list)
                else ([] if prev_value is None else [prev_value])
            )
            if op_type == "remove":
                removals = set()
                if isinstance(delta_payload, list):
                    removals = set(delta_payload)
                elif delta_payload is not None:
                    removals = {delta_payload}
                new_list = [item for item in prev_list if item not in removals]
            elif op_type == "add":
                additions = (
                    delta_payload
                    if isinstance(delta_payload, list)
                    else ([] if delta_payload is None else [delta_payload])
                )
                new_list = list(prev_list)
                for item in additions:
                    if item not in new_list:
                        new_list.append(item)
            elif op_type in {"modify", "replace", "set"}:
                new_list = (
                    delta_payload if isinstance(delta_payload, list) else prev_list
                )
            else:
                new_list = prev_list
            new_value = new_list
        else:
            # singular attribute
            new_value = delta_payload if delta_payload is not None else prev_value

        updated.setdefault(attr_type, {})[name] = new_value
        changes[(attr_type, name)] = {
            "previous_value": prev_value,
            "change_reason": reason,
            "op": op_type,
            "attribute_type": attr_type,
        }

    return updated, changes


def _user_attributes_state_to_list(
    state: Dict[str, Dict[str, object]],
    changes: Dict[tuple[str, str], Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """
    Convert user_attributes state map into a list with change metadata.
    """
    changes = changes or {}
    result: List[Dict[str, object]] = []
    for attr_type in ("singular", "collections"):
        entries = state.get(attr_type) or {}
        for name, value in sorted(entries.items()):
            item: Dict[str, object] = {
                "name": name,
                "attribute_type": attr_type,
                "current_value": value,
            }
            change = changes.get((attr_type, name))
            if change:
                item["op"] = change.get("op")
                if "previous_value" in change:
                    item["previous_value"] = change.get("previous_value")
                if change.get("change_reason"):
                    item["change_reason"] = change.get("change_reason")
            result.append(item)
    return result


def _init_generic_state(entries: Dict[str, object] | None) -> Dict[str, Dict[str, object]]:
    """
    Convert an initial {name: value} mapping into {name: {current_value: value}}.
    """
    state: Dict[str, Dict[str, object]] = {}
    if isinstance(entries, dict):
        for key, value in entries.items():
            state[key] = {"current_value": deepcopy(value)}
    return state


def _apply_habit_operations(
    state: Dict[str, Dict[str, object]],
    operations: List[Dict[str, object]] | None,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """
    Apply habit deltas; adjust merges partial fields into the previous habit.
    """
    updated = deepcopy(state)
    changes: Dict[str, Dict[str, object]] = {}
    if not operations:
        return updated, changes

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        name = operation.get("habit_name")
        if not name:
            continue

        op_type = (operation.get("op") or "").lower()
        reason = operation.get("reason", "")
        prev_value = deepcopy((updated.get(name) or {}).get("current_value"))
        delta_payload = operation.get("delta")
        if delta_payload is None:
            delta_payload = operation.get("new_state")

        if delta_payload is None:
            candidate = {
                k: v
                for k, v in operation.items()
                if k
                not in {
                    "habit_name",
                    "op",
                    "reason",
                    "before",
                    "after",
                    "change_reason",
                    "delta",
                    "new_state",
                    "attribute_type",
                }
            }
            if candidate:
                delta_payload = candidate

        if op_type == "drop":
            new_value = None
        elif op_type == "adjust":
            base = deepcopy(prev_value) if isinstance(prev_value, dict) else {}
            if isinstance(delta_payload, dict):
                base.update(delta_payload)
            elif delta_payload is not None:
                base = delta_payload
            new_value = base
        else:
            # acquire or fallback
            new_value = delta_payload if delta_payload is not None else prev_value

        updated[name] = {"current_value": new_value}
        changes[name] = {
            "previous_value": prev_value,
            "change_reason": reason,
            "op": op_type,
        }

    return updated, changes


def _apply_preference_operations(
    state: Dict[str, Dict[str, object]],
    operations: List[Dict[str, object]] | None,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """
    Apply preference deltas with the new delta field.
    """
    updated = deepcopy(state)
    changes: Dict[str, Dict[str, object]] = {}
    if not operations:
        return updated, changes

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        name = operation.get("preference_name")
        if not name:
            continue
        op_type = (operation.get("op") or "").lower()
        reason = operation.get("reason", "")
        prev_value = deepcopy((updated.get(name) or {}).get("current_value"))
        delta_payload = operation.get("delta")
        if delta_payload is None:
            delta_payload = operation.get("new_state")

        if op_type in {"drop", "remove"}:
            new_value = None
        else:
            new_value = delta_payload if delta_payload is not None else prev_value

        updated[name] = {"current_value": new_value}
        changes[name] = {
            "previous_value": prev_value,
            "change_reason": reason,
            "op": op_type,
        }

    return updated, changes


def _state_dict_to_list(
    state: Dict[str, Dict[str, object]],
    changes: Dict[str, Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """
    Convert state dict to list format, including change information if available.
    
    Args:
        state: Current state dict
        changes: Optional dict of changes (from _apply_delta_operations)
    """
    if changes is None:
        changes = {}
    
    result = []
    for key, entry in sorted(state.items()):
        current_value = (
            entry.get("current_value") if isinstance(entry, dict) else entry
        )
        item: Dict[str, object] = {
            "name": key,
            "current_value": current_value,
        }
        if isinstance(entry, dict) and "attribute_type" in entry:
            item["attribute_type"] = entry.get("attribute_type")
        
        # Add change information if this item changed
        if key in changes:
            change_info = changes[key]
            item["previous_value"] = change_info.get("previous_value")
            item["change_reason"] = change_info.get("change_reason")
            item["op"] = change_info.get("op")
        
        result.append(item)
    
    return result


def _resolve_window_states(dynamic_profile_data: Dict) -> List[Dict]:
    """
    Resolve initial state + deltas into full window states for downstream prompts.
    Works with the newer dynamic_profile format that has:
      - top-level "initial_state"
      - per-window *_delta sections with {op, *_name, delta, reason}
    Preserves change information (previous_value, change_reason, op) for each state item.
    """
    resolved: List[Dict] = []
    # Build initial snapshots from the top-level initial_state
    initial_state = dynamic_profile_data.get("initial_state") or {}
    current_attributes = _init_user_attributes_state(initial_state)
    current_habits = _init_generic_state(
        initial_state.get("habits_state")
    )
    current_preferences = _init_generic_state(
        initial_state.get("preferences_state")
    )

    for window in dynamic_profile_data.get("time_windows", []):
        window_id = window.get("window_id")
        if not window_id:
            continue

        # Track changes for each state type
        attributes_snapshot, attributes_changes = _apply_user_attribute_operations(
            current_attributes,
            _get_delta_operations(window.get("user_attributes_delta")),
        )

        habits_snapshot, habits_changes = _apply_habit_operations(
            current_habits,
            _get_delta_operations(window.get("habits_delta")),
        )

        preferences_snapshot, preferences_changes = _apply_preference_operations(
            current_preferences,
            _get_delta_operations(window.get("preferences_delta")),
        )

        resolved.append(
            {
                "window_id": window_id,
                "time_range": window.get("time_range"),
                "window_description": window.get("window_description"),
                "summary": window.get("summary", ""),
                "user_attributes_state": _user_attributes_state_to_list(
                    attributes_snapshot, changes=attributes_changes
                ),
                "habits_state": _state_dict_to_list(
                    habits_snapshot, changes=habits_changes
                ),
                "preferences_state": _state_dict_to_list(
                    preferences_snapshot, changes=preferences_changes
                ),
            }
        )

        current_attributes = attributes_snapshot
        current_habits = habits_snapshot
        current_preferences = preferences_snapshot

    return resolved


def _reorganize_by_key(resolved_windows: List[Dict]) -> Dict:
    """
    Reorganize window-based data into key-based timelines.
    Each key maintains a timeline of its values across all windows.
    Merges consecutive windows with no changes (no op field) into a single time range.
    
    Args:
        resolved_windows: List of window states from _resolve_window_states
        
    Returns:
        Dict with structure:
        {
            "user_attributes_state": {
                "key_name": {
                    "timeline": [
                        {
                            "time_range": ["2024-01-01", "2024-12-31"],  # merged if no changes
                            "current_value": ...,
                            # If there's a change:
                            "op": "modify",  # or "add", "drop", etc.
                            "previous_value": ...,
                            "change_reason": ...,
                        },
                        ...
                    ]
                },
                ...
            },
            "habits_state": {...},
            "preferences_state": {...}
        }
    """
    reorganized: Dict[str, Dict[str, Dict]] = {
        "user_attributes_state": {},
        "habits_state": {},
        "preferences_state": {},
    }
    
    # First pass: collect all timeline entries
    for window in resolved_windows:
        window_id = window.get("window_id")
        time_range = window.get("time_range")
        if not window_id:
            continue
        
        # Process each state type
        for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
            state_list = window.get(state_type, [])
            for item in state_list:
                key_name = item.get("name")
                attr_type = item.get("attribute_type")
                if not key_name:
                    continue

                composite_key = (
                    f"{attr_type}.{key_name}"
                    if state_type == "user_attributes_state" and attr_type
                    else key_name
                )

                # Initialize timeline for this key if not exists
                if composite_key not in reorganized[state_type]:
                    reorganized[state_type][composite_key] = {"timeline": []}
                    if state_type == "user_attributes_state" and attr_type:
                        reorganized[state_type][composite_key]["attribute_type"] = attr_type
                
                # Create timeline entry (with window_id for now, will be removed after merging)
                timeline_entry: Dict[str, object] = {
                    "window_id": window_id,
                    "time_range": time_range,
                    "current_value": item.get("current_value"),
                }
                if state_type == "user_attributes_state" and attr_type:
                    timeline_entry["attribute_type"] = attr_type
                
                # Add change information if present
                if "op" in item:
                    timeline_entry["op"] = item.get("op")
                    if "previous_value" in item:
                        timeline_entry["previous_value"] = item.get("previous_value")
                    if "change_reason" in item:
                        timeline_entry["change_reason"] = item.get("change_reason")
                
                reorganized[state_type][composite_key]["timeline"].append(timeline_entry)
    
    # Second pass: merge consecutive entries with no changes
    for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
        for key_name, key_data in reorganized[state_type].items():
            timeline = key_data["timeline"]
            if not timeline:
                continue
            
            merged_timeline: List[Dict[str, object]] = []
            i = 0
            
            while i < len(timeline):
                current_entry = timeline[i]
                
                # If this entry has an op (change), add it as-is (without window_id)
                if "op" in current_entry:
                    merged_entry: Dict[str, object] = {
                        "time_range": list(current_entry["time_range"]),  # copy
                        "current_value": current_entry["current_value"],
                        "op": current_entry["op"],
                    }
                    if "attribute_type" in current_entry:
                        merged_entry["attribute_type"] = current_entry["attribute_type"]
                    if "previous_value" in current_entry:
                        merged_entry["previous_value"] = current_entry["previous_value"]
                    if "change_reason" in current_entry:
                        merged_entry["change_reason"] = current_entry["change_reason"]
                    merged_timeline.append(merged_entry)
                    i += 1
                    continue
                
                # Otherwise, try to merge consecutive entries with same value and no op
                merged_entry = {
                    "time_range": list(current_entry["time_range"]),  # copy
                    "current_value": current_entry["current_value"],
                }
                if "attribute_type" in current_entry:
                    merged_entry["attribute_type"] = current_entry["attribute_type"]
                j = i + 1
                
                # Merge consecutive entries with no op and same current_value
                while j < len(timeline):
                    next_entry = timeline[j]
                    # Stop if next entry has an op (change)
                    if "op" in next_entry:
                        break
                    
                    # Check if current_value is the same (deep comparison for lists/dicts)
                    current_val = current_entry["current_value"]
                    next_val = next_entry["current_value"]
                    
                    # Simple comparison - for complex objects, we'd need deep comparison
                    # But for our use case, this should work
                    if current_val != next_val:
                        break
                    
                    # Check if time ranges are consecutive
                    current_end = merged_entry["time_range"][1]
                    next_start = next_entry["time_range"][0]
                    
                    # Parse dates to check if consecutive
                    try:
                        current_end_date = datetime.strptime(current_end, "%Y-%m-%d")
                        next_start_date = datetime.strptime(next_start, "%Y-%m-%d")
                        # Check if next window starts the day after current ends
                        if next_start_date != current_end_date + timedelta(days=1):
                            break
                    except (ValueError, TypeError):
                        # If date parsing fails, just check if they're adjacent strings
                        # This is a fallback
                        pass
                    
                    # Merge: extend time_range to include next entry
                    merged_entry["time_range"][1] = next_entry["time_range"][1]
                    j += 1
                
                merged_timeline.append(merged_entry)
                i = j
            
            # Update the timeline with merged version
            reorganized[state_type][key_name]["timeline"] = merged_timeline
    
    return reorganized


def _identify_stable_states(resolved_windows: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    Identify states that stay constant (no op, same value) across all windows.
    Returns a mapping per state_type -> state_name -> metadata.
    """
    stable_states: Dict[str, Dict[str, Dict]] = {
        "user_attributes_state": {},
        "habits_state": {},
        "preferences_state": {},
    }
    if not resolved_windows:
        return stable_states

    window_ids = [
        window.get("window_id") for window in resolved_windows if window.get("window_id")
    ]
    num_windows = len(window_ids)
    tracker: Dict[str, Dict[str, Dict[str, object]]] = {
        "user_attributes_state": {},
        "habits_state": {},
        "preferences_state": {},
    }

    for window_state in resolved_windows:
        window_id = window_state.get("window_id")
        if not window_id:
            continue
        for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
            for item in window_state.get(state_type, []) or []:
                name = item.get("name")
                attr_type = item.get("attribute_type")
                if not name:
                    continue
                state_key = (
                    f"{attr_type}.{name}"
                    if state_type == "user_attributes_state" and attr_type
                    else name
                )
                record = tracker[state_type].setdefault(
                    state_key,
                    {
                        "entries": [],
                        "has_op": False,
                        "attribute_type": attr_type,
                    },
                )
                record["entries"].append(
                    {"window_id": window_id, "value": item.get("current_value")}
                )
                if item.get("op"):
                    record["has_op"] = True

    for state_type, states in tracker.items():
        for name, info in states.items():
            entries = info.get("entries") or []
            if len(entries) != num_windows:
                continue
            if info.get("has_op"):
                continue
            normalized_values = [
                _normalize_attribute_value(entry["value"]) for entry in entries
            ]
            if len(set(normalized_values)) == 1:
                stable_states[state_type][name] = {
                    "value": entries[0]["value"],
                    "window_ids": [entry["window_id"] for entry in entries],
                }
                if info.get("attribute_type"):
                    stable_states[state_type][name]["attribute_type"] = info.get(
                        "attribute_type"
                    )

    return stable_states


def _plan_stable_state_reveals(
    resolved_windows: List[Dict],
    *,
    probability: float,
    seed: int | None = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """
    Decide in which windows a stable state should be revealed to semantic-event generation.
    Ensures every stable state is revealed in at least one window.
    """
    probability = max(0.0, min(1.0, probability))
    rng = random.Random(seed)
    stable_states = _identify_stable_states(resolved_windows)
    window_order = [
        window.get("window_id") for window in resolved_windows if window.get("window_id")
    ]
    plan: Dict[str, Dict[str, Dict[str, object]]] = {
        "user_attributes_state": {},
        "habits_state": {},
        "preferences_state": {},
        "_metadata": {
            "probability": probability,
            "seed": seed,
        },
    }

    for state_type, states in stable_states.items():
        for name, info in states.items():
            eligible_windows = [
                window_id for window_id in window_order if window_id in info["window_ids"]
            ]
            reveal_in = [
                window_id
                for window_id in eligible_windows
                if rng.random() < probability
            ]
            if not reveal_in and eligible_windows:
                reveal_in = [eligible_windows[0]]
            plan[state_type][name] = {
                "value": info["value"],
                "reveal_in_windows": reveal_in,
            }

    return plan


def _filter_window_state_for_semantic_events(
    window_state: Dict,
    reveal_plan: Dict[str, Dict[str, Dict[str, object]]],
) -> Dict:
    """Remove stable states that are not selected to be revealed in this window."""
    filtered = deepcopy(window_state)
    window_id = window_state.get("window_id")
    for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
        plan_for_type = reveal_plan.get(state_type, {})
        filtered_state: List[Dict[str, object]] = []
        for item in window_state.get(state_type, []) or []:
            name = item.get("name")
            attr_type = item.get("attribute_type")
            state_key = (
                f"{attr_type}.{name}"
                if state_type == "user_attributes_state" and attr_type
                else name
            )
            plan_entry = plan_for_type.get(state_key)
            if plan_entry:
                reveal_windows = plan_entry.get("reveal_in_windows") or []
                if window_id not in reveal_windows:
                    continue
            filtered_state.append(item)
        filtered[state_type] = filtered_state
    return filtered


def _extract_dynamic_profiles_initial_state(
    dynamic_profiles: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Pull out the initial_state section for each domain. Used as seed context
    for life context baseline generation.
    """
    initial_state_by_domain: Dict[str, Dict] = {}
    for domain_name, profile in (dynamic_profiles or {}).items():
        if isinstance(profile, dict):
            initial_state_by_domain[domain_name] = deepcopy(
                profile.get("initial_state") or {}
            )
    return initial_state_by_domain


def _apply_life_context_delta(
    previous_context: Dict, delta_payload: Dict
) -> Dict:
    """
    Apply a delta payload produced by Life_Context_Delta_Prompt onto the
    previous life context using generic patch ops. Falls back to a provided
    updated_life_context if no patches are present.
    """
    base = deepcopy(previous_context) if isinstance(previous_context, dict) else {}
    if not isinstance(delta_payload, dict):
        return base

    delta_section = delta_payload.get("life_context_delta") or {}
    patches = delta_section.get("patches") or delta_payload.get("patches") or []
    patched = _apply_patch_ops(base, patches) if patches else base

    updated_context = delta_payload.get("updated_life_context")
    if patches:
        return patched
    if isinstance(updated_context, dict):
        return deepcopy(updated_context)
    return patched


def _format_life_context_for_prompt(
    window_id: str | None, life_context: Dict | None, *, time_range: List[str] | None
) -> str:
    """Serialize life context into a compact string for semantic event prompts."""
    context = deepcopy(life_context or {})
    if isinstance(context, dict):
        context.pop("rationale", None)
        context.pop("reasoning", None)
    return json.dumps(context, indent=2, ensure_ascii=False)


def _build_life_contexts(
    *,
    dynamic_profiles: Dict[str, Dict],
    user_basic_profile: Dict,
    world_background_by_window: Dict[str, str],
    user_full_state_summaries: Dict[str, Dict],
    llm_client: GeminiJSONClient,
    output_dir: Path,
) -> tuple[Dict, Dict[str, Dict], Dict[str, Dict]]:
    """
    Build life context baseline (from initial states) and per-window contexts via
    delta application for downstream semantic event generation.
    """
    initial_state_by_domain = _extract_dynamic_profiles_initial_state(dynamic_profiles)

    ## save initial_state_by_domain
    _write_json(output_dir / "initial_state_by_domain.json", initial_state_by_domain)

    baseline_result = generate_life_context_baseline(
        llm_client,
        LifeContextBaselineRequest(
            user_basic_profile=user_basic_profile,
            dynamic_profiles_initial_state=initial_state_by_domain,
        ),
    )
    _write_text(
        output_dir / "life_context_baseline_prompt.txt", baseline_result.prompt
    )
    baseline_payload = (
        baseline_result.data if isinstance(baseline_result.data, dict) else {}
    )
    _write_json(
        output_dir / "life_context_baseline_result.json",
        baseline_payload,
    )

    baseline_context: Dict = {}
    if isinstance(baseline_payload, dict):
        if "life_context" in baseline_payload:
            baseline_context = deepcopy(baseline_payload.get("life_context") or {})
        elif "life_context_baseline" in baseline_payload:
            baseline_context = deepcopy(baseline_payload.get("life_context_baseline") or {})
        else:
            baseline_context = deepcopy(baseline_payload)

    usage: Dict[str, Dict] = {"baseline": baseline_result.usage}
    contexts_by_window: Dict[str, Dict] = {}

    for window_id in sorted(user_full_state_summaries.keys()):
        window_entry = user_full_state_summaries[window_id]
        if not isinstance(window_entry, dict):
            continue
        time_range = window_entry.get("time_range")
        window_profile_summary = window_entry.get("window_profile_summary", "")
        window_description = window_entry.get("window_description", "")
        window_world_background = world_background_by_window.get(window_id, "")
        window_state_by_domain = window_entry.get("domains", {})

        delta_result = generate_life_context_delta(
            llm_client,
            LifeContextDeltaRequest(
                user_basic_profile=user_basic_profile,
                time_range=time_range,
                window_id=window_id,
                window_description=window_description,
                window_profile_summary=window_profile_summary,
                world_background=window_world_background,
                life_context_baseline=baseline_context,
                previous_life_context=None,
                window_state_by_domain=window_state_by_domain,
            ),
        )
        usage[window_id] = delta_result.usage

        _write_text(
            output_dir / f"life_context_delta_{window_id}_prompt.txt",
            delta_result.prompt,
        )
        _write_json(
            output_dir / f"life_context_delta_{window_id}_result.json",
            delta_result.data,
        )
        delta_payload = (
            delta_result.data if isinstance(delta_result.data, dict) else {}
        )
        window_context = {
            "window_id": window_id,
            "time_range": time_range or [],
            "life_context": deepcopy(baseline_context),
            "life_context_delta": delta_payload,
            "window_profile_summary": window_profile_summary,
            "window_description": window_description,
            "world_background": window_world_background,
        }
        contexts_by_window[window_id] = window_context

    _write_json(
        output_dir / "life_context_by_window.json",
        {"baseline": baseline_context, "by_window": contexts_by_window},
    )

    return baseline_context, contexts_by_window, usage


def _build_user_spatiotemporal_constraints(
    user_full_state_summaries: Dict[str, Dict],
    world_background_by_window: Dict[str, str],
    llm_client: GeminiJSONClient,
    *,
    output_dir: Path,
    user_basic_profile: Dict,
) -> tuple[Dict[str, Dict[str, object]], Dict]:
    """
    Build per-window spatiotemporal constraints via LLM using merged
    cross-domain window summaries and per-window world background.
    """
    constraints: Dict[str, Dict[str, object]] = {}
    usage: Dict[str, Dict] = {}

    for window in sorted(
        user_full_state_summaries.values(), key=lambda w: w.get("window_id", "")
    ):
        window_id = window.get("window_id")
        if not window_id:
            continue

        time_range = window.get("time_range") or []
        window_profile_summary = window.get("window_profile_summary", "")
        window_description = window.get("window_description") or window_profile_summary
        world_bg = world_background_by_window.get(window_id, "")

        result = generate_spatiotemporal_constraints(
            llm_client,
            SpatiotemporalConstraintsRequest(
                time_range=time_range,
                user_basic_profile=user_basic_profile,
                window_profile_summary=window_profile_summary,
                window_description=window_description,
                world_background=world_bg,
            ),
        )
        usage[window_id] = result.usage
        _write_text(
            output_dir / f"spatiotemporal_constraints_{window_id}_prompt.txt",
            result.prompt,
        )

        payload = result.data if isinstance(result.data, dict) else {}
        base: Dict[str, object] = payload if isinstance(payload, dict) else {}
        base.setdefault("window_id", window_id)
        base.setdefault("time_range", time_range)
        base.setdefault("whereabouts", [])
        base.setdefault("home_base", "")
        base.setdefault("window_profile_summary", window_profile_summary)
        base.setdefault("window_description", window_description)
        constraints[window_id] = base

    return constraints, {"spatiotemporal_constraints": usage}


def _build_user_full_state_summaries(
    dynamic_profiles: Dict[str, Dict],
    llm_client: GeminiJSONClient | None = None,
) -> Dict[str, Dict]:
    """
    Build a cross-domain per-window view by stitching together each domain's window state.

    Output is keyed by window_id with an aggregated summary built from all domains.
    """
    summaries: Dict[str, Dict] = {}
    for domain_name, profile in dynamic_profiles.items():
        for window_state in _resolve_window_states(profile):
            window_id = window_state.get("window_id")
            if not window_id:
                continue

            entry = summaries.setdefault(
                window_id,
                {
                    "window_id": window_id,
                    "time_range": window_state.get("time_range"),
                    "domains": {},
                    "_summary_parts_by_domain": {},
                    "_window_description_parts_by_domain": {},
                    "summary_by_domain": {},
                    "window_description_by_domain": {},
                },
            )

            window_description = window_state.get("window_description") or ""
            window_summary = window_state.get("summary") or window_description
            entry["_summary_parts_by_domain"][domain_name] = window_summary or ""
            entry["summary_by_domain"][domain_name] = window_summary or ""
            entry["_window_description_parts_by_domain"][domain_name] = (
                window_description or ""
            )
            entry["window_description_by_domain"][domain_name] = window_description or ""

            entry.setdefault("domains", {})[domain_name] = {
                "window_description": window_state.get("window_description", ""),
                "summary": window_summary or "",
                "user_attributes_state": window_state.get("user_attributes_state", []),
                "habits_state": window_state.get("habits_state", []),
                "preferences_state": window_state.get("preferences_state", []),
            }

    for entry in summaries.values():
        summary_parts_by_domain = entry.pop("_summary_parts_by_domain", {})
        summary_parts = [
            f"{domain}: {summary}"
            for domain, summary in summary_parts_by_domain.items()
            if summary
        ]
        entry["window_profile_summary"] = "\n\n".join(summary_parts)
        # NOTE: Do NOT persist a redundant top-level "summary" field.
        # Use "window_profile_summary" and "summary_by_domain" instead.

        description_parts_by_domain = entry.pop("_window_description_parts_by_domain", {})
        description_parts = [
            f"{domain}: {desc}"
            for domain, desc in description_parts_by_domain.items()
            if desc
        ]
        combined_description = "\n\n".join(description_parts)
        entry["window_description"] = (
            combined_description or entry.get("window_profile_summary") or ""
        )

    return summaries


def generate_time_windows(
    start_date: str, end_date: str, num_windows: int
) -> List[Dict[str, str | List[str]]]:
    """
    Generate time windows by evenly splitting the date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        num_windows: Number of windows to create
        
    Returns:
        List of window dictionaries with window_id and time_range
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1  # +1 to include both start and end dates
    
    if num_windows < 1:
        raise ValueError("num_windows must be at least 1")
    
    windows = []
    days_per_window = total_days / num_windows
    
    for i in range(num_windows):
        # Calculate window start: round down to ensure no gaps
        window_start_offset = int(i * days_per_window)
        window_start = start + timedelta(days=window_start_offset)
        
        if i == num_windows - 1:
            # Last window should always end at the specified end_date
            window_end = end
        else:
            # Calculate window end: round down and subtract 1 to avoid overlap
            # This ensures each day belongs to exactly one window
            window_end_offset = int((i + 1) * days_per_window) - 1
            window_end = start + timedelta(days=window_end_offset)
        
        windows.append({
            "window_id": f"w{i + 1}",
            "time_range": [
                window_start.strftime("%Y-%m-%d"),
                window_end.strftime("%Y-%m-%d"),
            ],
        })
    
    return windows


def load_domains_from_file(path: str | Path) -> List[Domain]:
    path = Path(path)
    data = json.loads(path.read_text())
    domains: List[Domain] = []
    for entry in data:
        domain_name = entry.get("domain_name") or entry.get("name")
        if not domain_name:
            raise ValueError(f"domain entry missing name: {entry}")
        domains.append(
            Domain(
                domain_name=domain_name,
                domain_scope_definition=entry.get("domain_scope_definition"),
            )
        )
    return domains



class GenerationPipeline:
    def __init__(
        self,
        llm_client: GeminiJSONClient,
        output_dir: Path,
        timeline: TimelineConfig | None = None,
        semantic_config: SemanticEventsConfig | None = None,
        atomic_config: AtomicEventsConfig | None = None,
        real_data_config: RealDataConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.timeline = timeline or TimelineConfig()
        self.semantic_config = semantic_config or SemanticEventsConfig()
        self.atomic_config = atomic_config or AtomicEventsConfig()
        self.real_data_config = real_data_config or RealDataConfig()
        self.output_dir = Path(output_dir)
        _ensure_dir(self.output_dir)

    def build_user_basic_profile(
        self,
        *,
        user_description: str,
    ) -> tuple[Dict, Dict]:
        """
        Reason about the user_description and convert it into a structured basic profile.

        Returns the parsed profile and a usage dict (empty if loaded from cache).
        """
        basic_profile_path = self.output_dir / "user_basic_profile.json"

        basic_profile_result = generate_basic_profile(
            self.llm_client,
            BasicProfileRequest(user_description=user_description),
        )
        _write_json(basic_profile_path, basic_profile_result.data)
        _write_text(
            self.output_dir / "user_basic_profile_prompt.txt",
            basic_profile_result.prompt,
        )
        usage = {"basic_profile": basic_profile_result.usage}
        return basic_profile_result.data, usage

    def resolve_cross_domain_conflicts(
        self,
        dynamic_profiles: Dict[str, Dict],
        *,
        user_basic_profile: Dict | None = None,
    ) -> tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, object], List[Dict[str, object]]]:
        """
        Detect and resolve cross-domain conflicts using the LLM.
        Step 1: LLM aligns keys.
        Step 2: Auto-detect attribute key collisions across domains.
        Step 3: LLM resolves conflicts with full dynamic profiles.

        Returns:
            resolved_profiles: Updated dynamic profiles per domain
            usage: Usage metadata (empty if no conflicts)
            resolution_payload: Raw LLM resolution output (empty if no conflicts)
            conflicts_summary: Detected conflicts summary (from code + LLM detected_conflicts)
        """

        key_alignment_path = self.output_dir / "key_alignment_result.json"
        if key_alignment_path.exists():
            key_alignment = json.loads(key_alignment_path.read_text())
            alignment_usage: Dict = {}
        else:
            alignment_result = generate_key_alignment(
                self.llm_client,
                KeyAlignmentRequest(
                    dynamic_profiles=dynamic_profiles,
                    user_basic_profile=user_basic_profile,
                ),
            )
            key_alignment = alignment_result.data
            _write_text(
                self.output_dir / "key_alignment_prompt.txt", alignment_result.prompt
            )
            _write_json(self.output_dir / "key_alignment_result.json", key_alignment)
            alignment_usage = {"key_alignment": alignment_result.usage}
        dynamic_profiles, _key_alignment_mapping, _canonical_key_descriptions = _apply_key_alignment(
            dynamic_profiles, key_alignment
        )
        _write_json(
            self.output_dir / "dynamic_profiles_key_aligned.json", dynamic_profiles
        )

        import pdb; pdb.set_trace()

        attribute_conflicts = _collect_cross_domain_attribute_conflicts(
            dynamic_profiles
        )
        _write_json(
            self.output_dir / "auto_detected_attribute_conflicts.json",
            {"attribute_conflicts": attribute_conflicts},
        )
        resolved_profiles = deepcopy(dynamic_profiles)
        attribute_resolution_payload: Dict[str, object] = {}
        attribute_resolution_usage: Dict = {}
        all_conflicts_summary: List[Dict[str, object]] = list(attribute_conflicts)

        if attribute_conflicts:
            attr_resolution = generate_attribute_conflict_resolution(
                self.llm_client,
                ConflictResolutionRequest(
                    user_basic_profile=user_basic_profile,
                    dynamic_profiles=resolved_profiles,
                    detected_attribute_conflicts=attribute_conflicts,
                ),
            )
            attribute_resolution_payload = attr_resolution.data or {}
            attribute_resolution_usage = {"attribute_resolution": attr_resolution.usage or {}}
            _write_text(
                self.output_dir / "conflict_resolution_attribute_prompt.txt",
                attr_resolution.prompt,
            )
            _write_json(
                self.output_dir / "cross_domain_conflict_resolution_attribute.json",
                attribute_resolution_payload,
            )
            import pdb; pdb.set_trace()
            resolved_profiles = _apply_conflict_resolution_to_profiles(
                resolved_profiles,
                attribute_resolution_payload,
            )
            detected_attr_by_llm = (
                attribute_resolution_payload.get("conflicts_and_resolutions")
                if isinstance(attribute_resolution_payload, dict)
                else None
            )
            if isinstance(detected_attr_by_llm, list):
                all_conflicts_summary.extend(detected_attr_by_llm)

        temporal_conflicts = detect_temporal_conflicts(resolved_profiles)
        temporal_conflict_entries = (
            temporal_conflicts.get("conflicts")
            if isinstance(temporal_conflicts, dict)
            else temporal_conflicts
        ) or []
        _write_json(
            self.output_dir / "auto_detected_temporal_conflicts.json",
            {"temporal_conflicts": temporal_conflicts},
        )
        all_conflicts_summary.extend(temporal_conflict_entries)
        per_iteration_payloads: Dict[str, object] = {}
        per_iteration_usage: Dict[str, Dict] = {}
        habit_lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        window_range_lookup: Dict[Tuple[str, str], object] = {}
        final_payload: Dict[str, object] = {}

        for domain_name, profile in resolved_profiles.items():
            for snapshot in _materialize_habit_snapshots_for_conflicts(profile or {}):
                raw_window_id = snapshot.get("window_id") or "unknown_window"
                normalized_window_id = _normalize_window_id_label(raw_window_id)
                window_range_lookup[(domain_name, normalized_window_id)] = snapshot.get("time_range")
                for habit_name, habit_data in (snapshot.get("habits") or {}).items():
                    habit_lookup[(domain_name, normalized_window_id, habit_name)] = deepcopy(
                        habit_data
                    )

        def _select_focus_habit(conflicts_obj: Dict[str, object]) -> Optional[Dict[str, object]]:
            graph = conflicts_obj.get("graph_by_window") or {}
            nodes: List[Dict[str, object]] = []
            for window_id, data in graph.items():
                for node in data.get("top_nodes", []):
                    node = dict(node)
                    node["window_id"] = window_id
                    nodes.append(node)
            nodes = sorted(
                nodes,
                key=lambda x: (
                    -int(x.get("degree", 0)),
                    x.get("window_id", ""),
                    x.get("domain", ""),
                    x.get("habit", ""),
                ),
            )
            return nodes[0] if nodes else None

        def _subset_conflicts(conflicts_list: List[Dict[str, object]], focus: Dict[str, object] | None) -> List[Dict[str, object]]:
            if not focus:
                return []
            focus_key = (focus.get("domain"), focus.get("habit"))
            subset = []
            for c in conflicts_list:
                ha = c.get("habit_a") or {}
                hb = c.get("habit_b") or {}
                key_a = (ha.get("domain"), ha.get("habit"))
                key_b = (hb.get("domain"), hb.get("habit"))
                if key_a == focus_key or key_b == focus_key:
                    subset.append(c)
            return subset

        def _enrich_focus_habit(focus: Dict[str, object] | None, conflicts_for_focus: List[Dict[str, object]]) -> Dict[str, object]:
            if not focus:
                return {}
            enriched = dict(focus)
            focus_domain = enriched.get("domain")
            focus_name = enriched.get("habit") or enriched.get("habit_name")
            focus_window_id = enriched.get("window_id")
            focus_key = (focus_domain, focus_name)
            for entry in conflicts_for_focus:
                for habit_key in ("habit_a", "habit_b"):
                    habit_info = entry.get(habit_key) or {}
                    if (habit_info.get("domain"), habit_info.get("habit")) != focus_key:
                        continue
                    for field in ("priority", "timing", "window_id"):
                        if habit_info.get(field) and not enriched.get(field):
                            enriched[field] = habit_info[field]
                    if entry.get("window_range") and not enriched.get("window_range"):
                        enriched["window_range"] = entry.get("window_range")
            focus_window_id = enriched.get("window_id") or focus_window_id
            if focus_domain and focus_name:
                focus_detail = habit_lookup.get(
                    (focus_domain, focus_window_id or "unknown_window", focus_name)
                )
                if focus_detail:
                    enriched["habit_detail"] = focus_detail
                    for field in ("location", "schedule", "timing", "priority"):
                        if focus_detail.get(field) and not enriched.get(field):
                            enriched[field] = focus_detail[field]
                window_range = window_range_lookup.get(
                    (focus_domain, focus_window_id or "unknown_window")
                )
                if window_range and not enriched.get("window_range"):
                    enriched["window_range"] = window_range
            return enriched

        def _collect_conflict_times_for_focus(focus: Dict[str, object], conflicts_for_focus: List[Dict[str, object]]) -> List[Dict[str, object]]:
            conflict_times: List[Dict[str, object]] = []
            seen = set()
            focus_key = (focus.get("domain"), focus.get("habit"))
            for entry in conflicts_for_focus:
                ha = entry.get("habit_a") or {}
                hb = entry.get("habit_b") or {}
                key_a = (ha.get("domain"), ha.get("habit"))
                key_b = (hb.get("domain"), hb.get("habit"))
                other = hb if key_a == focus_key else ha
                conflict_window_id = entry.get("window_id")
                other_window_id = other.get("window_id") or conflict_window_id
                summary = {
                    "window_id": conflict_window_id,
                    "window_range": entry.get("window_range"),
                    "against": {
                        "domain": other.get("domain"),
                        "habit": other.get("habit"),
                        "priority": other.get("priority"),
                        "timing": other.get("timing"),
                        "window_id": other_window_id,
                    },
                    "occurrences": entry.get("occurrences"),
                    "sample_dates": (
                        (entry.get("sample_dates") or [])[:3]
                        if isinstance(entry.get("sample_dates"), list)
                        else entry.get("sample_dates")
                    ),
                    "overlap_examples": (
                        (entry.get("overlap_examples") or [])[:3]
                        if isinstance(entry.get("overlap_examples"), list)
                        else entry.get("overlap_examples")
                    ),
                }
                other_detail = habit_lookup.get(
                    (
                        other.get("domain"),
                        other_window_id or "unknown_window",
                        other.get("habit"),
                    )
                )
                if other_detail:
                    summary["against"]["habit_detail"] = other_detail
                    for field in ("location", "schedule", "timing", "priority"):
                        if other_detail.get(field) and not summary["against"].get(field):
                            summary["against"][field] = other_detail[field]
                if not summary.get("window_range"):
                    summary["window_range"] = window_range_lookup.get(
                        (focus.get("domain"), conflict_window_id or "unknown_window")
                    ) or window_range_lookup.get(
                        (other.get("domain"), conflict_window_id or "unknown_window")
                    )
                key = (
                    summary["window_id"],
                    summary["against"]["domain"],
                    summary["against"]["habit"],
                )
                if key not in seen:
                    seen.add(key)
                    conflict_times.append(summary)
            return conflict_times

        def _find_window_graph_key(graph_by_window: Dict[str, object] | None, target_norm: str) -> str | None:
            for key in graph_by_window or {}:
                if _normalize_window_id_label(key) == target_norm:
                    return key
            return None

        def _build_top_nodes_from_entries(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
            degrees: Counter[Tuple[str, str]] = Counter()
            for entry in entries:
                occ = int(entry.get("occurrences") or 1)
                for habit_key in ("habit_a", "habit_b"):
                    habit_info = entry.get(habit_key) or {}
                    key = (habit_info.get("domain") or "", habit_info.get("habit") or "")
                    degrees[key] += occ
            return sorted(
                (
                    {"domain": dom, "habit": habit, "degree": deg}
                    for (dom, habit), deg in degrees.items()
                ),
                key=lambda x: (-x["degree"], x["domain"], x["habit"]),
            )

        def _ordered_window_ids_from_profiles(profiles: Dict[str, Dict]) -> List[str]:
            ordered: List[str] = []
            seen = set()

            def _add(window_id: object) -> None:
                norm = _normalize_window_id_label(window_id)
                if norm not in seen:
                    seen.add(norm)
                    ordered.append(norm)

            _add("initial")
            for profile in profiles.values():
                for window in profile.get("time_windows") or []:
                    _add(window.get("window_id"))
            return ordered

        iteration_counter = 0
        window_resolution_order = _ordered_window_ids_from_profiles(resolved_profiles)

        def _resolve_window_conflicts(
            target_window_label: str,
            current_profiles: Dict[str, Dict],
            current_counter: int
        ) -> tuple[Dict[str, Dict], int]:
            """
            Resolve conflicts for a specific window.
            Returns: (updated_profiles, updated_counter)
            """
            normalized_target = _normalize_window_id_label(target_window_label)
            updated_profiles = current_profiles
            updated_counter = current_counter

            for _ in range(100):
                
                window_conflicts = detect_temporal_conflicts(updated_profiles)
                window_conflict_entries = [
                    c for c in (window_conflicts.get("conflicts") or [])
                    if _normalize_window_id_label(c.get("window_id")) == normalized_target
                ]

                _write_json(
                    self.output_dir / f"window_conflicts_{target_window_label}.json",
                    window_conflicts,
                )

                if not window_conflict_entries:
                    print(f"All conflicts resolved in {target_window_label}")
                    break

                # Always rebuild top_nodes from current window's conflicts to ensure we select
                # the habit with the most conflicts within this specific window (not global conflicts)
                top_nodes = _build_top_nodes_from_entries(window_conflict_entries)
                if not top_nodes:
                    break

                window_graph_data: Dict[str, object] = {
                    target_window_label: {"top_nodes": top_nodes}
                }

                focus_habit = _select_focus_habit({"graph_by_window": window_graph_data})

                if not focus_habit:
                    break

                focus_window_id = window_conflict_entries[0].get("window_id") or target_window_label
                focus_habit["window_id"] = focus_window_id
                focus_conflicts = _subset_conflicts(window_conflict_entries, focus_habit)
                focus_habit = _enrich_focus_habit(focus_habit, focus_conflicts)

                conflict_hints = {
                    "focus_habit": focus_habit,
                    "conflicts_for_focus": focus_conflicts,
                    "conflict_times": _collect_conflict_times_for_focus(focus_habit, focus_conflicts),
                }

                updated_counter += 1
                resolution_result = generate_time_conflict_resolution(
                    self.llm_client,
                    ConflictResolutionRequest(
                        user_basic_profile=user_basic_profile,
                        dynamic_profiles=updated_profiles,  # Pass full dynamic profiles
                        detected_temporal_conflicts=conflict_hints,
                        target_window_id=target_window_label,
                    ),
                    iteration_index=updated_counter,
                )

                payload = resolution_result.data or {}
                per_iteration_payloads[f"iteration_{updated_counter}_{target_window_label}"] = payload
                per_iteration_usage[f"iteration_{updated_counter}_{target_window_label}"] = resolution_result.usage or {}

                _write_text(
                    self.output_dir / f"conflict_resolution_temporal_{target_window_label}_iter{updated_counter}_prompt.txt",
                    resolution_result.prompt,
                )
                _write_json(
                    self.output_dir / f"cross_domain_conflict_resolution_temporal_{target_window_label}_iter{updated_counter}.json",
                    payload,
                )

                updated_profiles = _apply_conflict_resolution_to_profiles(
                    updated_profiles,
                    payload,
                )

                detected_by_llm = payload.get("conflicts_and_resolutions") if isinstance(payload, dict) else None
                if isinstance(detected_by_llm, list):
                    all_conflicts_summary.extend(detected_by_llm)

                _write_json(
                    self.output_dir / f"dynamic_profiles_conflict_resolved_{target_window_label}_iter{updated_counter}.json",
                    updated_profiles,
                )
                import pdb; pdb.set_trace()

            return updated_profiles, updated_counter

        for window_label in window_resolution_order:
            if not window_label or window_label == "unknown":
                continue
            print(f"\n=== Processing conflicts in {window_label} ===")
            resolved_profiles, iteration_counter = _resolve_window_conflicts(
                window_label, resolved_profiles, iteration_counter
            )

        # # Final comprehensive pass using full conflict resolver
        # final_resolution = generate_conflict_resolution(
        #     self.llm_client,
        #     ConflictResolutionRequest(
        #         user_basic_profile=user_basic_profile,
        #         dynamic_profiles=resolved_profiles,
        #         detected_attribute_conflicts=[],
        #         detected_temporal_conflicts=temporal_conflicts,
        #     ),
        # )
        # final_payload = final_resolution.data
        # _write_text(
        #     self.output_dir / "conflict_resolution_prompt.txt",
        #     final_resolution.prompt,
        # )
        _write_json(
            self.output_dir / "cross_domain_conflict_resolution.json",
            {
                "attribute": attribute_resolution_payload,
                "temporal_iters": per_iteration_payloads,
            },
        )

        usage = {
            "conflict_resolution": {
                "attribute": attribute_resolution_usage,
                "temporal_iters": per_iteration_usage,
            },
        }
        if alignment_usage:
            usage.update(alignment_usage)
        return resolved_profiles, usage, {"attribute": attribute_resolution_payload, "temporal_iters": per_iteration_payloads, "final": final_payload}, all_conflicts_summary

    def _validate_final_rules(
        self,
        dynamic_profiles: Dict[str, Dict],
        domains: Sequence[Domain],
        *,
        user_basic_profile: Dict | None = None,
    ) -> tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Final sanity check: validate Rules 1-2 for each domain after cross-domain conflict resolution.
        This prevents cascade errors from introducing new rule violations.

        Returns:
            updated_profiles: Dynamic profiles with rule fixes applied
            final_rule_usage: Usage metadata for rule fixes
        """
        print("\n=== Final Rule Validation (Rules 1-2) for Each Domain ===")
        final_rule_fixes: Dict[str, Dict] = {}
        final_rule_usage: Dict[str, Dict] = {}

        # Get user_profile text for rule fixes
        user_profile_text = json.dumps(user_basic_profile or {}, indent=2, ensure_ascii=False)

        # Create domain lookup by name
        domain_lookup = {domain.domain_name: domain for domain in domains}

        resolved_profiles = deepcopy(dynamic_profiles)

        for domain_name, profile in resolved_profiles.items():
            print(f"\nValidating domain: {domain_name}")
            domain_fixes: Dict[str, object] = {}
            domain_usage: Dict = {}
            current_profile = profile

            # Get the actual Domain object
            domain_obj = domain_lookup.get(domain_name)
            if domain_obj is None:
                # This should not happen if domains were properly loaded
                print(f"  WARNING: Domain '{domain_name}' not found in domains list. Skipping validation.")
                continue

            # Rule 1: Required fields
            rule1_issues = _detect_rule1_required_field_issues(current_profile)
            if rule1_issues:
                print(f"  - Rule 1: Found {len(rule1_issues)} required field issues")
                fix_result, usage = _fix_rule1_violations(
                    self.llm_client, domain_obj, user_profile_text, current_profile, rule1_issues
                )
                if fix_result:
                    domain_fixes["rule1"] = fix_result
                    current_profile = _apply_profile_revision(current_profile, fix_result)
                    domain_usage.update(usage)

            # Rule 2: Prior existence
            rule2_issues = _detect_rule2_prior_existence_issues(current_profile)
            if rule2_issues:
                print(f"  - Rule 2: Found {len(rule2_issues)} prior existence issues")
                fix_result, usage = _fix_rule2_violations(
                    self.llm_client, domain_obj, user_profile_text, current_profile, rule2_issues
                )
                if fix_result:
                    domain_fixes["rule2"] = fix_result
                    current_profile = _apply_profile_revision(current_profile, fix_result)
                    domain_usage.update(usage)

            # Update resolved_profiles with the final cleaned profile
            if domain_fixes:
                resolved_profiles[domain_name] = current_profile
                final_rule_fixes[domain_name] = domain_fixes
                final_rule_usage[domain_name] = domain_usage
                print(f"  ✓ Applied {len(domain_fixes)} rule fixes for {domain_name}")

        # Save final rule validation results
        if final_rule_fixes:
            _write_json(
                self.output_dir / "final_rule_validation_fixes.json",
                final_rule_fixes,
            )
            _write_json(
                self.output_dir / "dynamic_profiles_final_clean.json",
                resolved_profiles,
            )
            print("\n✓ Final rule validation completed. All domains are now clean.")

        return resolved_profiles, final_rule_usage

    ## Need Refactor
    # def debug_resolve_conflicts_from_file(
    #     self,
    #     domain_level_fixed_dynamic_profiles_path: str | Path,
    #     *,
    #     user_basic_profile: Dict | None = None,
    # ) -> tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, object], List[Dict[str, object]]]:
    #     """
    #     Convenience helper to re-run conflict resolution on an existing dynamic_profiles_raw.json
    #     without regenerating per-domain profiles. If in-domain fixes exist, they are applied
    #     before resolving cross-domain conflicts.
    #     """
    #     domain_level_fixed_dynamic_profiles_path = Path(domain_level_fixed_dynamic_profiles_path)
    #     if domain_level_fixed_dynamic_profiles_path.exists():
    #         dynamic_profiles = json.loads(domain_level_fixed_dynamic_profiles_path.read_text())
    #     else:
    #         ## use {domain}_all_fixed to generate combined dynamic profiles
    #         dynamic_profiles = {}
    #         # import pdb; pdb.set_trace()
    #         for domain_name in domain_level_fixed_dynamic_profiles_path.parent.glob("*_dynamic_profile_all_fixes.json"):
    #             # import pdb; pdb.set_trace()
    #             dynamic_profiles[domain_name.stem.replace("_dynamic_profile_all_fixes", "")] = json.loads(domain_name.read_text())
    #         _write_json(domain_level_fixed_dynamic_profiles_path, dynamic_profiles)
    #     import pdb; pdb.set_trace()
    #     # Load cached basic profile if not provided.
    #     if user_basic_profile is None:
    #         basic_profile_path = self.output_dir / "user_basic_profile.json"
    #         if basic_profile_path.exists():
    #             user_basic_profile = json.loads(basic_profile_path.read_text())

    #     # Persist the input for traceability, then run resolution.
    #     _write_json(self.output_dir / "dynamic_profiles_domain_level_fixes_applied.json", dynamic_profiles)
    #     return self.resolve_cross_domain_conflicts(
    #         dynamic_profiles, user_basic_profile=user_basic_profile
    #     )

    def _prepare_basic_profile(
        self, user_description: str | None
    ) -> tuple[Dict, Dict, str]:
        """
        Load or build the basic profile and return it with usage + serialized text.
        """
        basic_profile_path = self.output_dir / "user_basic_profile.json"
        if user_description is None and not basic_profile_path.exists():
            raise ValueError("user_description or user_profile is required to run.")

        if basic_profile_path.exists():
            user_basic_profile = json.loads(basic_profile_path.read_text())
            basic_usage: Dict = {}
        else:
            user_basic_profile, basic_usage = self.build_user_basic_profile(
                user_description=user_description  # validated above
            )

        user_profile_text = json.dumps(
            user_basic_profile, indent=2, ensure_ascii=False
        )
        return user_basic_profile, basic_usage, user_profile_text

    def _prepare_dynamic_profiles(
        self,
        domains: Sequence[Domain],
        *,
        user_profile_text: str,
        world_background_text: str,
        user_basic_profile: Dict | None,
    ) -> tuple[
        Dict[str, Dict],
        Dict[str, Dict],
        Dict[str, object],
        List[Dict[str, object]],
    ]:
        """
        Generate (or load) dynamic profiles, optionally resolve conflicts, and
        return profiles plus usage and conflict metadata.
        """
        dynamic_profiles: Dict[str, Dict] = {}
        aggregate_usage: Dict[str, Dict] = {}
        conflict_resolution_payload: Dict[str, object] = {}
        conflicts_summary: List[Dict[str, object]] = []
        # conflict_cache_used = False

        conflict_resolved_path = self.output_dir / "dynamic_profiles_conflict_resolved.json"
        life_domain_list = [
            f"{domain.domain_name}"
            for domain in domains
        ]

        conflicts_summary = []
        need_cross_domain_conflict_resolution = True
        # import pdb; pdb.set_trace()
        cached_resolved_profiles: Dict[str, Dict] | None = None
        if conflict_resolved_path.exists():
            cached_resolved_profiles = json.loads(conflict_resolved_path.read_text())
            need_cross_domain_conflict_resolution = False
            # conflict_cache_used = True
        # import pdb; pdb.set_trace()
        usage = {"dynamic_profile": {}}
        # domains = 
        # import pdb; pdb.set_trace()
        domain_level_fixed_dynamic_profiles_path = self.output_dir / "dynamic_profiles_domain_level_fixes_applied.json"
        if domain_level_fixed_dynamic_profiles_path.exists():
            cached_resolved_profiles = json.loads(domain_level_fixed_dynamic_profiles_path.read_text())
            need_cross_domain_conflict_resolution = True
        for domain in domains:
            import pdb; pdb.set_trace()
            cached_profile = None
            if cached_resolved_profiles is not None:
                cached_profile = cached_resolved_profiles.get(domain.domain_name)

            if cached_profile is not None:
                dynamic_profile = cached_profile
                
            else:
                slug = _slugify(domain.domain_name)
                # revised_result_path = (
                #     self.output_dir / f"{slug}_dynamic_profile_revised_result.json"
                # )    
                # if revised_result_path.exists():
                #     dynamic_profile = json.loads(revised_result_path.read_text())
                dynamic_profile = None
                if (self.output_dir / f"{slug}_dynamic_profile.json").exists():
                    dynamic_profile = json.loads((self.output_dir / f"{slug}_dynamic_profile.json").read_text())
                                        # import pdb; pdb.set_trace()
                if not dynamic_profile:
                    dynamic_profile, usage = self.generate_dynamic_profile_for_domain(
                        domain,
                        user_profile=user_profile_text,
                        world_background=world_background_text,
                        life_domain_list=life_domain_list,  
                    )
                ## record raw dynamic profile (domain level)
                _write_json(self.output_dir / f"{slug}_raw_dynamic_profile.json", dynamic_profile)
                # Always use the in-domain revised profile downstream.
                import pdb; pdb.set_trace()
                _reviewed_result, revised_dynamic_profile, review_usage = (
                    self.review_revise_dynamic_profile_for_domain(
                        domain,
                        dynamic_profile=dynamic_profile,
                        user_profile=user_profile_text,
                    )
                )
                usage.update(review_usage)
                dynamic_profile = revised_dynamic_profile
                # _write_json(self.output_dir / f"{slug}_domain_fixed_dynamic_profile.json", dynamic_profile)
                # new_dynamic_profile_generated = True

            dynamic_profiles[domain.domain_name] = dynamic_profile
            aggregate_usage[domain.domain_name] = usage

        ## [debug1, set to True to force conflict resolution]
        # raw_dynamic_profiles_path = self.output_dir / "dynamic_profiles_raw.json"
        # with open(raw_dynamic_profiles_path, "r", encoding="utf-8") as f:
        #     dynamic_profiles = json.load(f)
        # import pdb; pdb.set_trace()
        # new_dynamic_profile_generated = True
        
        # if new_dynamic_profile_generated:
        if need_cross_domain_conflict_resolution:
            # Persist the freshly generated, per-domain dynamic profiles before conflict resolution.
            
            ## [debug1, skip persisting raw dynamic profiles]
            # domain_level_fixed_dynamic_profiles_path = self.output_dir / "dynamic_profiles_domain_level_fixes_applied.json"
            # _write_json(domain_level_fixed_dynamic_profiles_path, dynamic_profiles)
            ## [debug1 end]

            (
                dynamic_profiles,
                conflict_usage,
                conflict_resolution_payload,
                conflicts_summary,
            ) = self.resolve_cross_domain_conflicts(
                dynamic_profiles, user_basic_profile=user_basic_profile
            )
            if conflict_usage:
                aggregate_usage["_conflict_resolution"] = conflict_usage

            # Final Rule Validation (sanity check after cross-domain conflict resolution)
            # This prevents cascade errors from introducing new rule violations
            import pdb; pdb.set_trace()
            dynamic_profiles, final_rule_usage = self._validate_final_rules(
                dynamic_profiles, domains, user_basic_profile=user_basic_profile
            )
            if final_rule_usage:
                aggregate_usage["_final_rule_validation"] = final_rule_usage

        return (
            dynamic_profiles,
            aggregate_usage,
            conflict_resolution_payload,
            conflicts_summary,
        )

    def _prepare_cross_domain_context(
        self,
        dynamic_profiles: Dict[str, Dict],
        *,
        world_background: str | Dict[str, str],
        user_basic_profile: Dict,
    ) -> tuple[Dict[str, Dict], Dict, Dict[str, str], Dict]:
        """
        Build cross-domain summaries, life context (baseline + window deltas), and per-window world background.
        """
        user_full_state_summaries_path = (
            self.output_dir / "user_full_state_summaries.json"
        )
        if user_full_state_summaries_path.exists():
            
            user_full_state_summaries = json.loads(
                user_full_state_summaries_path.read_text()
            )
            # needs_rebuild = any(
            #     not isinstance(entry, dict)
            #     or "summary_by_domain" not in entry
            #     or "window_description" not in entry
            #     or "window_description_by_domain" not in entry
            #     for entry in user_full_state_summaries.values()
            # )
            # if needs_rebuild:
            #     user_full_state_summaries = _build_user_full_state_summaries(
            #         dynamic_profiles, self.llm_client
            #     )
            #     _write_json(
            #         user_full_state_summaries_path,
            #         user_full_state_summaries,
            #     )
        else:
            user_full_state_summaries = _build_user_full_state_summaries(
                dynamic_profiles, self.llm_client
            )
            _write_json(
                user_full_state_summaries_path,
                user_full_state_summaries,
            )

        window_ids_all = sorted(user_full_state_summaries.keys())
        world_background_by_window = _map_world_background_to_windows(
            world_background, window_ids_all
        )

        life_context_by_window_path = (
            self.output_dir / "life_context_by_window.json"
        )
        life_context_usage: Dict = {}
        life_context_baseline: Dict = {}
        life_context_by_window: Dict[str, Dict] = {}
        if life_context_by_window_path.exists():
            payload = json.loads(life_context_by_window_path.read_text())
            if isinstance(payload, dict):
                life_context_baseline = payload.get("baseline", {}) or {}
                life_context_by_window = payload.get("by_window", {}) or {}
        else:
            (
                life_context_baseline,
                life_context_by_window,
                life_context_usage,
            ) = _build_life_contexts(
                dynamic_profiles=dynamic_profiles,
                user_basic_profile=user_basic_profile,
                world_background_by_window=world_background_by_window,
                user_full_state_summaries=user_full_state_summaries,
                llm_client=self.llm_client,
                output_dir=self.output_dir,
            )

        user_life_contexts = {
            "baseline": life_context_baseline,
            "by_window": life_context_by_window,
        }
        return (
            user_full_state_summaries,
            user_life_contexts,
            world_background_by_window,
            life_context_usage,
        )

    def _generate_semantic_events_for_domains(
        self,
        domains: Sequence[Domain],
        *,
        dynamic_profiles: Dict[str, Dict],
        user_basic_profile: Dict,
        user_life_contexts: Dict,
        user_full_state_summaries: Dict[str, Dict],
        world_background_by_window: Dict[str, str],
        aggregate_usage: Dict[str, Dict],
    ) -> tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Generate semantic events for all domains and return assembled payload + usage.
        """
        assembled: Dict[str, Dict] = {}
        # Aggregate initial summaries across domains for window1 prompts.
        initial_summaries_parts: List[str] = []
        for domain_name, profile in dynamic_profiles.items():
            init_summary = (
                (profile.get("initial_state") or {}).get("summary") or ""
            )
            if init_summary:
                initial_summaries_parts.append(f"{domain_name}: {init_summary}")
        initial_all_domains_summary = "\n\n".join(initial_summaries_parts)
        if not initial_all_domains_summary:
            first_window = None
            for window_id in sorted(user_full_state_summaries.keys()):
                entry = user_full_state_summaries.get(window_id)
                if isinstance(entry, dict):
                    first_window = entry
                    break
            if first_window:
                initial_all_domains_summary = (
                    first_window.get("window_profile_summary", "") or ""
                )

        for domain in domains:
            events_chain, usage = self.generate_semantic_events_for_domain(
                domain=domain,
                user_basic_profile=user_basic_profile,
                dynamic_profiles=dynamic_profiles,
                user_life_contexts=user_life_contexts,
                user_full_state_summaries=user_full_state_summaries,
                world_background=world_background_by_window,
                usage=aggregate_usage.get(domain.domain_name),
                initial_all_domains_summary=initial_all_domains_summary,
            )
            aggregate_usage[domain.domain_name] = usage
            assembled[domain.domain_name] = {
                "dynamic_profile": dynamic_profiles[domain.domain_name],
                "events_chain": events_chain,
            }

        return assembled, aggregate_usage

    def _generate_real_data_for_domains(
        self,
        domains: Sequence[Domain],
        *,
        dynamic_profiles: Dict[str, Dict],
        user_basic_profile: Dict,
        user_life_contexts: Dict,
        user_full_state_summaries: Dict[str, Dict],
        events_chain_by_domain: Dict[str, List[Dict]],
        aggregate_usage: Dict[str, Dict],
    ) -> tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
        """
        Generate real data for all domains using previously generated semantic events.
        """
        assembled: Dict[str, List[Dict]] = {}

        for domain in domains:
            events_chain_windows = (
                events_chain_by_domain.get(domain.domain_name) or []
            )
            if not events_chain_windows:
                continue

            usage = aggregate_usage.get(domain.domain_name)
            real_windows, usage = self.generate_real_data_for_domain(
                domain=domain,
                dynamic_profiles=dynamic_profiles,
                events_chain_windows=events_chain_windows,
                user_basic_profile=user_basic_profile,
                user_life_contexts=user_life_contexts,
                user_full_state_summaries=user_full_state_summaries,
                usage=usage,
            )
            aggregate_usage[domain.domain_name] = usage
            assembled[domain.domain_name] = real_windows

        return assembled, aggregate_usage

    def _assemble_final_payload(
        self,
        *,
        raw_user_description: str | None,
        user_basic_profile: Dict,
        user_profile_text: str,
        world_background_text: str,
        world_background_by_window: Dict[str, str],
        user_life_contexts: Dict,
        user_full_state_summaries: Dict[str, Dict],
        assembled_domains: Dict[str, Dict],
        conflicts_summary: List[Dict[str, object]],
        conflict_resolution_payload: Dict[str, object],
        usage: Dict[str, Dict],
        usage_summary: Dict[str, Dict[str, int]] | None = None,
        usage_pricing: Dict[str, object] | None = None,
    ) -> Dict:
        return {
            "user_description": raw_user_description,
            "user_basic_profile": user_basic_profile,
            # Structured profile serialized for prompts (backwards-compatible key name).
            "user_profile": user_profile_text,
            "world_background": world_background_text,
            "world_background_by_window": world_background_by_window,
            "user_life_contexts": user_life_contexts,
            # Legacy alias to avoid breaking downstream readers; remove once migrated.
            "general_environment": user_life_contexts,
            "user_full_state_summaries": user_full_state_summaries,
            "timeline_defaults": asdict(self.timeline),
            "domains": assembled_domains,
            "conflict_resolution": {
                "summary": conflicts_summary,
                "resolution": conflict_resolution_payload,
            },
            "usage": usage,
            "usage_summary": usage_summary or _summarize_usage(usage),
            "usage_pricing": usage_pricing,
        }

    def generate_dynamic_profile_inputs(
        self,
        domains: Sequence[Domain],
        *,
        user_description: str | None,
        world_background: str | Dict[str, str],
    ) -> tuple[Dict, str, Dict[str, Dict], Dict[str, Dict], Dict[str, object], List[Dict[str, object]]]:
        """
        Generate the basic profile and dynamic profiles stage of the pipeline.
        """
        user_basic_profile, basic_usage, user_profile_text = self._prepare_basic_profile(
            user_description
        )
        aggregate_usage: Dict[str, Dict] = {}
        if basic_usage:
            aggregate_usage["basic_profile"] = basic_usage

        world_background_text = (
            world_background
            if isinstance(world_background, str)
            else json.dumps(world_background, ensure_ascii=False)
        )

        (
            dynamic_profiles,
            dynamic_usage,
            conflict_resolution_payload,
            conflicts_summary,
        ) = self._prepare_dynamic_profiles(
            domains,
            user_profile_text=user_profile_text,
            world_background_text=world_background_text,
            user_basic_profile=user_basic_profile,
        )
        aggregate_usage.update(dynamic_usage)

        return (
            user_basic_profile,
            user_profile_text,
            dynamic_profiles,
            aggregate_usage,
            conflict_resolution_payload,
            conflicts_summary,
        )

    def prepare_context_for_semantic_events_generation(
        self,
        dynamic_profiles: Dict[str, Dict],
        *,
        world_background: str | Dict[str, str],
        user_basic_profile: Dict,
    ) -> tuple[Dict[str, Dict], Dict, Dict[str, str], Dict]:
        """
        Prepare cross-domain context needed for semantic events generation.
        """
        return self._prepare_cross_domain_context(
            dynamic_profiles,
            world_background=world_background,
            user_basic_profile=user_basic_profile,
        )

    def run(
        self,
        domains: Sequence[Domain],
        *,
        user_description: str | None = None,
        world_background: str | Dict[str, str],
    ) -> Dict:
        """
        Full pipeline run using separated dynamic-profile and context stages.
        """
        world_background_text = (
            world_background
            if isinstance(world_background, str)
            else json.dumps(world_background, ensure_ascii=False)
        )
        (
            user_basic_profile,
            user_profile_text,
            dynamic_profiles,
            aggregate_usage,
            conflict_resolution_payload,
            conflicts_summary,
        ) = self.generate_dynamic_profile_inputs(
            domains,
            user_description=user_description,
            world_background=world_background,
        )

        (
            user_full_state_summaries,
            user_life_contexts,
            world_background_by_window,
            life_context_usage,
        ) = self.prepare_context_for_semantic_events_generation(
            dynamic_profiles,
            world_background=world_background,
            user_basic_profile=user_basic_profile,
        )
        if life_context_usage:
            aggregate_usage["life_context"] = life_context_usage

        conflict_info = {
            "summary": conflicts_summary,
            "resolution": conflict_resolution_payload,
        }

        assembled_domains, aggregate_usage = self._generate_semantic_events_for_domains(
            domains,
            dynamic_profiles=dynamic_profiles,
            user_basic_profile=user_basic_profile,
            user_life_contexts=user_life_contexts,
            user_full_state_summaries=user_full_state_summaries,
            world_background_by_window=world_background_by_window,
            aggregate_usage=aggregate_usage,
        )

        events_chain_by_domain = {
            domain_name: payload.get("events_chain") or []
            for domain_name, payload in assembled_domains.items()
        }
        real_data_by_domain, aggregate_usage = self._generate_real_data_for_domains(
            domains,
            dynamic_profiles=dynamic_profiles,
            user_basic_profile=user_basic_profile,
            user_life_contexts=user_life_contexts,
            user_full_state_summaries=user_full_state_summaries,
            events_chain_by_domain=events_chain_by_domain,
            aggregate_usage=aggregate_usage,
        )
        for domain_name, real_windows in real_data_by_domain.items():
            assembled_domains.setdefault(domain_name, {})
            assembled_domains[domain_name]["real_data"] = real_windows

        usage_summary, usage_pricing = _persist_usage_artifacts(
            aggregate_usage,
            self.output_dir,
            model_name=getattr(self.llm_client, "model_name", None),
        )

        final_payload = self._assemble_final_payload(
            raw_user_description=user_description,
            user_basic_profile=user_basic_profile,
            user_profile_text=user_profile_text,
            world_background_text=world_background_text,
            world_background_by_window=world_background_by_window,
            user_life_contexts=user_life_contexts,
            user_full_state_summaries=user_full_state_summaries,
            assembled_domains=assembled_domains,
            conflicts_summary=conflict_info["summary"],
            conflict_resolution_payload=conflict_info["resolution"],
            usage=aggregate_usage,
            usage_summary=usage_summary,
            usage_pricing=usage_pricing,
        )
        _write_json(self.output_dir / "pipeline_output.json", final_payload)
        return final_payload

    def generate_dynamic_profile_for_domain(
        self,
        domain: Domain,
        *,
        user_profile: str,
        world_background: str,
        life_domain_list: List[str] | str | None = None,
    ) -> tuple[Dict, Dict]:
        """
        Generate latent state for an domain.
        
        Returns:
            Tuple of (dynamic_profile_data, usage_dict)
        """
        timeline = domain.resolve_timeline(self.timeline)
        slug = _slugify(domain.domain_name)

        # Pre-generate time windows
        time_windows = generate_time_windows(
            timeline.start_date,
            timeline.end_date,
            timeline.num_windows,
        )

        latent_result = generate_dynamic_profile(
            self.llm_client,
            DynamicProfileRequest(
                domain_name=domain.domain_name,
                domain_scope_definition=domain.domain_scope_definition,
                world_background=world_background,
                user_profile=user_profile,
                time_windows=time_windows,
                life_domain_list=life_domain_list,
            ),
        )
        # latent_path = self.output_dir / f"{slug}_dynamic_profile.json"
        # _write_json(latent_path, latent_result.data)

        usage = {"dynamic_profile": latent_result.usage}
        return latent_result.data, usage

    def review_revise_dynamic_profile_for_domain(
        self,
        domain: Domain,
        *,
        dynamic_profile: Dict,
        user_profile: str,
    ) -> tuple[Dict, Dict, Dict]:
        """
        Run a structured audit + repair pass on a generated dynamic profile.
        Each rule is detected and fixed in CASCADE mode:

        Rule 1: Required fields (structural integrity)
            → detect on original profile → fix → revised_profile_1
        Rule 2: Prior existence (operation legality)
            → detect on revised_profile_1 → fix → revised_profile_2
        Rule 3: Essential initialization (semantic reasonableness)
            → detect on revised_profile_2 → fix → revised_profile_3
        Rule 4: Short-term followups (semantic consistency)
            → detect on revised_profile_3 → fix → revised_profile_4
        Rule 5: Time conflicts (final feasibility)
            → detect on revised_profile_4 → fix → final_profile

        Only rules with violations are sent to the LLM for fixing.
        """
        slug = _slugify(domain.domain_name)

        # Initialize cascade: start with original profile
        revised_profile = deepcopy(dynamic_profile)

        # Track all detected issues for reporting
        autodetected_payload = {}
        all_fixes = {}
        aggregate_usage = {}

        # ========== Rule 1: Required fields (structural integrity) ==========
        # Detect on current profile (initially the original dynamic_profile)
        rule1_issues = _detect_rule1_required_field_issues(revised_profile)
        autodetected_payload["rule1_required_fields"] = rule1_issues

        if rule1_issues:
            # import pdb; pdb.set_trace()
            fix_result, usage = _fix_rule1_violations(
                self.llm_client, domain, user_profile, revised_profile, rule1_issues
            )
            if fix_result:
                all_fixes["rule1_required_fields"] = fix_result
                revised_profile = _apply_profile_revision(revised_profile, fix_result)
                aggregate_usage.update(usage)
                _write_json(
                    self.output_dir / f"{slug}_rule1_required_fields_fix.json",
                    fix_result
                )
                _write_json(
                    self.output_dir / f"{slug}_rule1_required_fields_fixed_result.json",
                    revised_profile
                )
                _write_json(
                    self.output_dir / f"{slug}_rule1_required_fields_autodetected_issues.json",
                    rule1_issues
                )

        # ========== Rule 2: Prior existence (operation legality) ==========
        # Detect on revised_profile (after Rule 1 fixes applied)
        rule2_issues = _detect_rule2_prior_existence_issues(revised_profile)
        autodetected_payload["rule2_prior_existence"] = rule2_issues

        if rule2_issues:
            # import pdb; pdb.set_trace() 
            fix_result, usage = _fix_rule2_violations(
                self.llm_client, domain, user_profile, revised_profile, rule2_issues
            )
            if fix_result:
                all_fixes["rule2_prior_existence"] = fix_result
                revised_profile = _apply_profile_revision(revised_profile, fix_result)
                aggregate_usage.update(usage)
                _write_json(
                    self.output_dir / f"{slug}_rule2_prior_existence_fix.json",
                    fix_result
                )
                _write_json(
                    self.output_dir / f"{slug}_rule2_prior_existence_fixed_result.json",
                    revised_profile
                )
                _write_json(
                    self.output_dir / f"{slug}_rule2_prior_existence_autodetected_issues.json",
                    rule2_issues
                )

        # ========== Rule 3: Essential initialization (semantic reasonableness) ==========
        # Detect on revised_profile (after Rule 1-2 fixes applied)
        rule3_issues = _detect_rule3_first_add_issues(revised_profile)
        autodetected_payload["rule3_essential_initialization"] = rule3_issues

        if rule3_issues:
            # import pdb; pdb.set_trace()
            fix_result, usage = _fix_rule3_violations(
                self.llm_client, domain, user_profile, revised_profile, rule3_issues
            )
            if fix_result:
                all_fixes["rule3_essential_initialization"] = fix_result
                revised_profile = _apply_profile_revision(revised_profile, fix_result)
                aggregate_usage.update(usage)
                _write_json(
                    self.output_dir / f"{slug}_rule3_essential_initialization_fix.json",
                    fix_result
                )
                _write_json(
                    self.output_dir / f"{slug}_rule3_essential_initialization_fixed_result.json",
                    revised_profile
                )
                _write_json(
                    self.output_dir / f"{slug}_rule3_essential_initialization_autodetected_issues.json",
                    rule3_issues
                )

        # ========== Rule 4: Short-term followups (semantic consistency) ==========
        # Detect on revised_profile (after Rule 1-3 fixes applied)
        rule4_issues = _detect_rule4_short_term_issues(revised_profile)
        autodetected_payload["rule4_short_term_followups"] = rule4_issues

        if rule4_issues:
            # import pdb; pdb.set_trace()
            fix_result, usage = _fix_rule4_violations(
                self.llm_client, domain, user_profile, revised_profile, rule4_issues
            )
            if fix_result:
                all_fixes["rule4_short_term_followups"] = fix_result
                revised_profile = _apply_profile_revision(revised_profile, fix_result)
                aggregate_usage.update(usage)
                _write_json(
                    self.output_dir / f"{slug}_rule4_short_term_followups_fix.json",
                    fix_result
                )
                _write_json(
                    self.output_dir / f"{slug}_rule4_short_term_followups_fixed_result.json",
                    revised_profile
                )
                _write_json(
                    self.output_dir / f"{slug}_rule4_short_term_followups_autodetected_issues.json",
                    rule4_issues
                )

        # ========== Rule 5: Time conflicts (final feasibility) ==========
        # Detect on revised_profile (after Rule 1-4 fixes applied)
        rule5_result = _detect_rule5_time_conflict_issues(revised_profile)
        rule5_conflicts = rule5_result.get("conflicts") if isinstance(rule5_result, dict) else []
        autodetected_payload["rule5_time_conflicts"] = rule5_result

        if rule5_conflicts:
            # import pdb; pdb.set_trace()
            fix_result, usage, prompt = _fix_rule5_violations(
                self.llm_client, domain, user_profile, revised_profile, rule5_result
            )
            if fix_result:
                all_fixes["rule5_time_conflicts"] = fix_result
                revised_profile = _apply_profile_revision(revised_profile, fix_result)
                aggregate_usage.update(usage)
                _write_json(
                    self.output_dir / f"{slug}_rule5_time_conflicts_fix.json",
                    fix_result
                )
                _write_json(
                    self.output_dir / f"{slug}_rule5_time_conflicts_fixed_result.json",
                    revised_profile
                )
                _write_json(
                    self.output_dir / f"{slug}_rule5_time_conflicts_autodetected_issues.json",
                    rule5_result
                )
                _write_json(
                    self.output_dir / f"{slug}_rule5_time_conflicts_prompt.json",
                    prompt
                )

        # Save all auto-detected issues (cascade detection results)
        autodetected_path = self.output_dir / f"{slug}_dynamic_profile_autodetected_issues.json"
        _write_json(autodetected_path, autodetected_payload)

        # Save all fixes summary
        all_fixes_path = self.output_dir / f"{slug}_dynamic_profile_all_fixes.json"
        _write_json(all_fixes_path, all_fixes)

        revised_path = self.output_dir / f"{slug}_dynamic_profile_revised_result.json"
        _write_json(revised_path, revised_profile)

        return all_fixes, revised_profile, aggregate_usage

    # # Backward-compatible alias
    # def _review_revise_dynamic_profile_for_domain(
    #     self,
    #     domain: Domain,
    #     *,
    #     dynamic_profile: Dict,
    #     user_profile: str,
    #     world_background: str,
    # ) -> tuple[Dict, Dict]:
    #     return self.review_revise_dynamic_profile_for_domain(
    #         domain,
    #         dynamic_profile=dynamic_profile,
    #         user_profile=user_profile,
    #         world_background=world_background,
    #     )
    
    def review_revise_dynamic_profile_from_file(
        self,
        *,
        domain_profile_path: str | Path,
        domain: Domain,
        user_profile_text: str | Dict | None = None,
    ) -> tuple[Dict, Dict]:
        """
        Load an existing dynamic profile JSON, run the audit/revise prompt, and
        save the corrected version for manual inspection.
        """
        with open(domain_profile_path, "r", encoding="utf-8") as f:
            dynamic_profile = json.load(f)

        reviewed_result, revised_profile, usage = self.review_revise_dynamic_profile_for_domain(
            domain,
            dynamic_profile=dynamic_profile,
            user_profile=user_profile_text,
        )

        # reviewed_path = self.output_dir / f"{slug}_dynamic_profile_checked_result.json"
        # _write_json(reviewed_path, reviewed_result)

        # revised_path = self.output_dir / f"{slug}_dynamic_profile_revised_result.json"
        # _write_json(revised_path, revised_profile)

        return reviewed_result, revised_profile, usage

    def generate_semantic_events_for_domain(
        self,
        *,
        domain: Domain,
        user_basic_profile: Dict,
        dynamic_profiles: Dict[str, Dict],
        user_life_contexts: Dict | None = None,
        user_full_state_summaries: Dict[str, Dict] | None = None,
        world_background: str | Dict[str, str] | None = None,
        usage: Dict | None = None,
        initial_all_domains_summary: str | None = None,
    ) -> tuple[List[Dict], Dict]:
        """
        Generate semantic events (events chain) for a domain based on existing latent state.
        
        Args:
            domain: domain specification
            dynamic_profiles: All latent state data across domains (from generate_dynamic_profile_for_domain)
            user_full_state_summaries: Cross-domain per-window summaries for previous-window context
            usage: Optional existing usage dict to update
            world_background: Optional global context used to keep events consistent (per-window or shared)
            
        Returns:
            Tuple of (events_chain_windows_list, updated_usage_dict)
        """
        if usage is None:
            usage = {"events_chain": {}}
        elif "events_chain" not in usage:
            usage["events_chain"] = {}

        domain_profile = (dynamic_profiles or {}).get(domain.domain_name)
        if not isinstance(domain_profile, dict):
            raise ValueError(f"Dynamic profile for domain {domain.domain_name} is missing.")

        user_life_contexts = user_life_contexts or {}
        user_full_state_summaries = user_full_state_summaries or {}
        life_context_baseline = {}
        life_context_by_window: Dict[str, Dict] = {}

        ## user_life_contexts is a dict with two keys: "baseline" and "by_window"
        if isinstance(user_life_contexts, dict):
            life_context_baseline = user_life_contexts.get("baseline") or {}
            life_context_by_window = user_life_contexts.get("by_window") or {}

        # import pdb; pdb.set_trace()     

        slug = _slugify(domain.domain_name)
        events_chain_windows: List[Dict] = []

        ## resolved_windows is a list of dicts, each dict is a window state
        resolved_windows = _resolve_window_states(domain_profile)
        resolved_windows_path = self.output_dir / f"{slug}_resolved_windows.json"
        _write_json(resolved_windows_path, resolved_windows)

        window_ids = [w.get("window_id") for w in resolved_windows if w.get("window_id")]
        world_background_map = _map_world_background_to_windows(
            world_background, window_ids
        )
        summary_all_by_window: Dict[str, str] = {}
        summary_by_window_domain: Dict[str, str] = {}

        for window_id, entry in user_full_state_summaries.items():
            if not window_id or not isinstance(entry, dict):
                continue
            summary_all_by_window[window_id] = entry.get("window_profile_summary", "") or ""
            domain_summary_val = (
                entry.get("summary_by_domain", {}).get(domain.domain_name)
                or entry.get("window_description_by_domain", {}).get(domain.domain_name, "")
            )
            summary_by_window_domain[window_id] = domain_summary_val

        ## for window 1
        domain_initial_summary = (
            (domain_profile.get("initial_state") or {}).get("summary") or ""
        )
        # import pdb; pdb.set_trace()
        initial_window_summary = initial_all_domains_summary or ""
        

        resolved_window_map = {
            w.get("window_id"): w for w in resolved_windows if w.get("window_id")
        }
        rng = random.Random(self.semantic_config.stable_state_reveal_seed)
        stale_sample_probability = max(
            0.0, min(1.0, self.semantic_config.stable_state_reveal_probability)
        )
        conversion_flags: Dict[str, Dict[str, bool]] = {
            "user_attributes_state": {},
            "habits_state": {},
            "preferences_state": {},
        }

        def _default_metadata() -> Dict[str, object]:
            return {
                "updated_this_window": False,
                "freshness": False,
                "already_converted_to_semantic_events": False,
                "should_convert_to_semantic_events": False,
                "reason": None,
            }

        def _make_item_key(value: object) -> str:
            try:
                return json.dumps(value, sort_keys=True, ensure_ascii=False)
            except TypeError:
                return str(value)

        def _init_state_tracker() -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
            tracker: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {
                "user_attributes_state": {},
                "habits_state": {},
                "preferences_state": {},
            }
            initial_state = domain_profile.get("initial_state") or {}
            initial_sections = {
                "user_attributes_state": (initial_state.get("user_attributes_state") or {}).get("initial"),
                "habits_state": initial_state.get("habits_state"),
                "preferences_state": initial_state.get("preferences_state"),
            }
            for state_type, section in initial_sections.items():
                if not isinstance(section, dict):
                    continue
                for name, value in section.items():
                    values = value if isinstance(value, list) else [value]
                    for val in values:
                        key = _make_item_key(val)
                        tracker[state_type].setdefault(name, {})[key] = {
                            "value": val,
                            "metadata": _default_metadata(),
                        }
            return tracker

        state_tracker = _init_state_tracker()

        def _materialize_state_table_from_tracker() -> Dict[str, List[Dict[str, object]]]:
            """Convert current tracker snapshot into a state_table list structure."""
            table: Dict[str, List[Dict[str, object]]] = {
                "user_attributes_state": [],
                "habits_state": [],
                "preferences_state": [],
            }
            for state_type, names in state_tracker.items():
                for name, entries in names.items():
                    for entry in entries.values():
                        table[state_type].append(
                            {
                                "name": name,
                                "current_value": entry["value"],
                                "op": "stable",
                                "metadata": deepcopy(entry["metadata"]),
                            }
                        )
            return table

        # Emit an initial baseline state_table snapshot before processing windows.
        initial_state_table_payload = {
            "window_id": "initial",
            "time_range": (domain_profile.get("initial_state") or {}).get("time_range"),
            "state_table": _materialize_state_table_from_tracker(),
        }
        _write_json(
            self.output_dir / f"{slug}_domain_window_state_payload_initial.json",
            initial_state_table_payload,
        )

        for idx, window_state in enumerate(resolved_windows):
            window_id = window_state.get("window_id")
            if not window_id:
                continue

            window_full_state = user_full_state_summaries.get(window_id, {})
            if not isinstance(window_full_state, dict):
                window_full_state = {}
            if idx == 0:
                previous_all_summary_raw = initial_window_summary
                previous_domain_summary_raw = domain_initial_summary
            else:
                prev_window_id = resolved_windows[idx - 1].get("window_id")
                previous_all_summary_raw = summary_all_by_window.get(prev_window_id, "")
                previous_domain_summary_raw = summary_by_window_domain.get(prev_window_id, "")
            user_previous_window_summary = (
                "<previous window summary (all domains)>\n"
                f"{previous_all_summary_raw or 'No previous window summary available for this window.'}\n"
                "</previous window summary (all domains)>"
            )
            user_domain_previous_window_summary = (
                f"<previous window summary in {domain.domain_name}>\n"
                f"{previous_domain_summary_raw or 'No previous window summary available for this window.'}\n"
                f"</previous window summary in {domain.domain_name}>"
            )
            life_context_for_window: Dict = {}
            life_context_entry = life_context_by_window.get(window_id, {}) or {}
            base_context = life_context_entry.get("life_context") or life_context_baseline
            delta_payload = life_context_entry.get("life_context_delta") or {}
            life_context_for_window = _apply_life_context_delta(
                base_context, delta_payload
            )

            if not life_context_for_window and isinstance(life_context_baseline, dict):
                life_context_for_window = life_context_baseline
            life_context_prompt_str = _format_life_context_for_prompt(
                window_id, life_context_for_window, time_range=window_state.get("time_range")
            )
            domain_window_description = (
                window_state.get("window_description")
            )
            user_this_window_description = (
                f"<this window description in {domain.domain_name}>\n"
                f"{domain_window_description or 'No description available for this domain in this window.'}\n"
                f"</this window description in {domain.domain_name}>"
            )

            resolved_window_state = resolved_window_map.get(window_id, {})

            conversion_targets: Dict[str, List[Dict[str, object]]] = {
                "user_attributes_state": [],
                "habits_state": [],
                "preferences_state": [],
            }
            selected_tracker_keys: Dict[str, List[tuple[str, str]]] = {
                "user_attributes_state": [],
                "habits_state": [],
                "preferences_state": [],
            }

            for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
                window_items = window_state.get(state_type) or []
                seen_keys: Dict[str, set[str]] = {}

                for item in window_items:
                    name = item.get("name")
                    if not name:
                        continue

                    item_op = item.get("op")
                    change_reason = item.get("change_reason")
                    previous_value = item.get("previous_value")
                    current_value = item.get("current_value")
                    item_op_lower = (item_op or "").lower()

                    values = (
                        current_value
                        if state_type == "user_attributes_state" and isinstance(current_value, list)
                        else [current_value]
                    )

                    for val in values:
                        key = _make_item_key(val)
                        tracker_entries = state_tracker[state_type].setdefault(name, {})
                        existing_entry = tracker_entries.get(key)

                        already_converted = bool(
                            existing_entry
                            and existing_entry["metadata"].get("already_converted_to_semantic_events")
                        )
                        updated_this_window = existing_entry is None
                        freshness = updated_this_window
                        reason = change_reason if updated_this_window else None

                        should_convert = False
                        if state_type == "habits_state":
                            should_convert = True  # Habits are always converted each window (no sampling).
                        elif freshness:
                            should_convert = True
                        elif not already_converted and rng.random() < stale_sample_probability:
                            should_convert = True

                        metadata = {
                            "updated_this_window": updated_this_window,
                            "freshness": freshness,
                            "already_converted_to_semantic_events": already_converted,
                            "should_convert_to_semantic_events": should_convert,
                            "reason": reason if freshness else None,
                        }

                        tracker_entries[key] = {"value": val, "metadata": deepcopy(metadata)}
                        seen_keys.setdefault(name, set()).add(key)

                        if should_convert:
                            entry: Dict[str, object] = {
                                "name": name,
                                "current_value": val,
                                "op": item_op if freshness and item_op else "stable",
                                "metadata": metadata,
                            }
                            if updated_this_window and change_reason:
                                entry["change_reason"] = change_reason
                            if updated_this_window:
                                if item_op_lower in {"add", "acquire"}:
                                    entry["previous_value"] = None
                                elif previous_value is not None:
                                    entry["previous_value"] = previous_value

                            conversion_targets[state_type].append(entry)
                            selected_tracker_keys[state_type].append((name, key))

                # Drop tracker entries that no longer exist in the current snapshot (e.g., removals).
                for name, entries in list(state_tracker[state_type].items()):
                    keep_keys = seen_keys.get(name, set())
                    for key in list(entries.keys()):
                        if key not in keep_keys:
                            entries.pop(key, None)
                    if not entries:
                        state_tracker[state_type].pop(name, None)

            domain_window_state_payload = {
                "window_id": window_id,
                "time_range": window_state.get("time_range"),
                "state_table": conversion_targets,
            }
            # Do not leak internal metadata into the prompt.
            domain_window_state_payload_for_prompt = deepcopy(domain_window_state_payload)
            for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
                entries = domain_window_state_payload_for_prompt["state_table"].get(state_type) or []
                for entry in entries:
                    entry.pop("metadata", None)

            domain_window_state = (
                f"<user detailed state in {domain.domain_name}>\n"
                f"{json.dumps(domain_window_state_payload_for_prompt, indent=2, ensure_ascii=False)}\n"
                f"</user detailed state in {domain.domain_name}>"
            )
            ## save domain_window_state_payload
            domain_window_state_payload_path = self.output_dir / f"{slug}_domain_window_state_payload_{window_id}.json"
            _write_json(domain_window_state_payload_path, domain_window_state_payload)
            domain_window_state_path = self.output_dir / f"{slug}_domain_window_state_{window_id}.json"
            _write_json(domain_window_state_path, domain_window_state_payload_for_prompt)
            window_world_background = world_background_map.get(
                window_id, "No world background available for this window."
            )
            user_basic_profile_str = json.dumps(user_basic_profile, indent=2, ensure_ascii=False)
            semantic_result = generate_semantic_events(
                self.llm_client,
                SemanticEventsRequest(
                    domain_name=domain.domain_name,
                    user_basic_profile=user_basic_profile_str,
                    user_life_context=life_context_prompt_str,
                    user_previous_window_summary=user_previous_window_summary,
                    user_domain_previous_window_summary=user_domain_previous_window_summary,
                    user_this_window_description=user_this_window_description,
                    world_background=window_world_background,
                    domain_window_state=domain_window_state,
                ),
            )
            events_chain_windows.append(semantic_result.data)
            usage["events_chain"][window_id] = semantic_result.usage

            ## save prompt
            prompt_path = self.output_dir / f"{slug}_events_chain_{window_id}_prompt.txt"
            _write_text(prompt_path, semantic_result.prompt)

            semantic_path = self.output_dir / f"{slug}_events_chain_{window_id}.json"
            _write_json(semantic_path, semantic_result.data)
            legacy_semantic_path = self.output_dir / f"{slug}_semantic_events_{window_id}.json"
            _write_json(legacy_semantic_path, semantic_result.data)

            # Mark converted items so we don't repeatedly force conversion in later windows.
            for state_type, entries in selected_tracker_keys.items():
                for name, key in entries:
                    meta_entry = state_tracker[state_type].get(name, {}).get(key)
                    if meta_entry:
                        meta_entry["metadata"]["already_converted_to_semantic_events"] = True
                        meta_entry["metadata"]["should_convert_to_semantic_events"] = False

        windows_by_id = {
            entry.get("window_id"): entry
            for entry in events_chain_windows
            if isinstance(entry, dict) and entry.get("window_id")
        }
        semantic_path = self.output_dir / f"{slug}_events_chain.json"
        payload = {
            "domain": domain.domain_name,
            "windows_by_id": windows_by_id,
        }
        _write_json(semantic_path, payload)
        legacy_semantic_path = self.output_dir / f"{slug}_semantic_events.json"
        _write_json(legacy_semantic_path, payload)

        return events_chain_windows, usage

    def generate_atomic_events_for_domain(
        self,
        domain: Domain,
        *,
        dynamic_profile_data: Dict,
        events_chain_windows: List[Dict],
        usage: Dict | None = None,
    ) -> tuple[List[Dict], Dict]:
        """
        Generate atomic events for a domain based on existing latent state and events chain.
        
        Args:
            spec: domain specification
            dynamic_profile_data: The latent state data
            events_chain_windows: List of events chain entries for each window
            usage: Optional existing usage dict to update
            
        Returns:
            Tuple of (atomic_events_windows_list, updated_usage_dict)
        """
        if usage is None:
            usage = {"atomic_events": {}}
        elif "atomic_events" not in usage:
            usage["atomic_events"] = {}

        slug = _slugify(domain.domain_name)
        atomic_windows: List[Dict] = []

        # Create a mapping from window_id to resolved window state and semantic events
        window_states = {
            w.get("window_id"): w for w in _resolve_window_states(dynamic_profile_data)
        }
        import pdb; pdb.set_trace()
        for events_chain_data in events_chain_windows:
            window_id = events_chain_data.get("window_id")
            if not window_id:
                continue

            window_state = window_states.get(window_id)
            if not window_state:
                continue

            events_payload = (
                events_chain_data.get("events_chain")
                or events_chain_data.get("semantic_events")
                or events_chain_data
            )

            atomic_result = generate_atomic_events(
                self.llm_client,
                AtomicEventsRequest(
                    domain_name=domain.domain_name,
                    window_id=window_id,
                    window_time_range=_format_window_range(window_state),
                    current_window_state=window_state,
                    semantic_events=events_payload,
                    min_atomic_per_semantic=self.atomic_config.min_per_semantic,
                    max_atomic_per_semantic=self.atomic_config.max_per_semantic,
                ),
            )
            atomic_windows.append(atomic_result.data)
            usage["atomic_events"][window_id] = atomic_result.usage

        atomic_path = self.output_dir / f"{slug}_atomic_events.json"
        _write_json(
            atomic_path,
            {"domain": domain.domain_name, "windows": atomic_windows},
        )

        return atomic_windows, usage

    def generate_real_data_for_domain(
        self,
        *,
        domain: Domain,
        dynamic_profiles: Dict[str, Dict],
        events_chain_windows: List[Dict],
        user_basic_profile: Dict,
        user_life_contexts: Dict | None = None,
        user_full_state_summaries: Dict[str, Dict] | None = None,
        usage: Dict | None = None,
    ) -> tuple[List[Dict], Dict]:
        """
        Generate real data for a domain directly from semantic events (evidence chains).
        """
        if usage is None:
            usage = {"real_data": {}}
        elif "real_data" not in usage:
            usage["real_data"] = {}

        domain_profile = (dynamic_profiles or {}).get(domain.domain_name)
        if not isinstance(domain_profile, dict):
            raise ValueError(f"Dynamic profile for domain {domain.domain_name} is missing.")

        slug = _slugify(domain.domain_name)
        real_windows: List[Dict] = []

        resolved_windows = _resolve_window_states(domain_profile)
        window_states = {
            w.get("window_id"): w for w in resolved_windows if w.get("window_id")
        }
        window_order = [w.get("window_id") for w in resolved_windows if w.get("window_id")]
        prev_window_map = {
            window_order[i]: window_order[i - 1] for i in range(1, len(window_order))
        }

        life_context_baseline = {}
        life_context_by_window: Dict[str, Dict] = {}
        if isinstance(user_life_contexts, dict):
            life_context_baseline = user_life_contexts.get("baseline") or {}
            life_context_by_window = user_life_contexts.get("by_window") or {}

        summary_by_window_domain: Dict[str, str] = {}
        for window_id, entry in (user_full_state_summaries or {}).items():
            if not window_id or not isinstance(entry, dict):
                continue
            domain_summary_val = (
                entry.get("summary_by_domain", {}).get(domain.domain_name)
                or entry.get("window_description_by_domain", {}).get(domain.domain_name, "")
            )
            summary_by_window_domain[window_id] = domain_summary_val

        domain_initial_summary = (
            (domain_profile.get("initial_state") or {}).get("summary") or ""
        )
        user_basic_profile_str = json.dumps(
            user_basic_profile, indent=2, ensure_ascii=False
        )

        import pdb; pdb.set_trace() 
        for events_chain_data in events_chain_windows:
            window_id = events_chain_data.get("window_id")
            if not window_id:
                continue

            window_state = window_states.get(window_id, {})
            window_time_range = (
                events_chain_data.get("time_range") or window_state.get("time_range")
            )
            window_description = window_state.get("window_description") or ""

            previous_window_id = prev_window_map.get(window_id)
            previous_domain_summary_raw = (
                domain_initial_summary
                if previous_window_id is None
                else summary_by_window_domain.get(previous_window_id, "")
            )
            if previous_domain_summary_raw:
                previous_summary_text = (
                    f"Window {previous_window_id or 'initial'} summary in {domain.domain_name}: "
                    f"{previous_domain_summary_raw}"
                )
            else:
                previous_summary_text = (
                    f"No previous window summary available for window {window_id} in {domain.domain_name}."
                )

            life_context_entry = life_context_by_window.get(window_id, {}) or {}
            base_context = life_context_entry.get("life_context") or life_context_baseline
            delta_payload = life_context_entry.get("life_context_delta") or {}
            life_context_for_window = _apply_life_context_delta(
                base_context, delta_payload
            )
            if not life_context_for_window and isinstance(life_context_baseline, dict):
                life_context_for_window = life_context_baseline
            life_context_prompt_str = _format_life_context_for_prompt(
                window_id, life_context_for_window, time_range=window_time_range
            )

            evidence_chains = events_chain_data.get("evidence_chains") or []
            if not evidence_chains:
                continue

            window_real_data: Dict[str, object] = {
                "window_id": window_id,
                "time_range": window_time_range,
                "real_data": [],
            }

            for chain_idx, chain in enumerate(evidence_chains):
                chain_id = chain.get("evidence_chain_id") or f"{window_id}_chain_{chain_idx+1:02d}"
                chain_payload = deepcopy(chain)
                chain_payload.setdefault("evidence_chain_id", chain_id)
                chain_payload.setdefault("window_id", window_id)
                chain_payload.setdefault("time_range", window_time_range)
                chain_payload.setdefault("domain_name", domain.domain_name)

                import pdb; pdb.set_trace()

                real_result = generate_real_data(
                    self.llm_client,
                    RealDataRequest(
                        user_basic_profile=user_basic_profile_str,
                        life_context=life_context_prompt_str,
                        previous_window_summary_this_domain=previous_summary_text,
                        current_window_description=window_description
                        or "No description available for this domain in this window.",
                        evidence_chain=chain_payload,
                        window_id=window_id,
                        window_time_range=window_time_range,
                        domain_name=domain.domain_name,
                    ),
                )
                window_real_data["real_data"].append(real_result.data)
                usage["real_data"].setdefault(window_id, {})[chain_id] = real_result.usage

                prompt_path = (
                    self.output_dir
                    / f"{slug}_real_data_{window_id}_{_slugify(chain_id)}_prompt.txt"
                )
                _write_text(prompt_path, real_result.prompt)

                _write_json(
                    self.output_dir / f"{slug}_real_data_{window_id}_{_slugify(chain_id)}.json",
                    real_result.data,
                )

                ## record usage
                _write_json(
                    self.output_dir / f"{slug}_real_data_{window_id}_{_slugify(chain_id)}_usage.json",
                    real_result.usage,
                )

            real_windows.append(window_real_data)

        real_path = self.output_dir / f"{slug}_real_data.json"
        _write_json(
            real_path,
            {"domain": domain.domain_name, "windows": real_windows},
        )

        return real_windows, usage

    def load_dynamic_profile_from_file(
        self, domain: Domain
    ) -> Dict | None:
        """Load latent state from file if it exists."""
        slug = _slugify(domain.domain_name)
        latent_path = self.output_dir / f"{slug}_dynamic_profile.json"
        if latent_path.exists():
            return json.loads(latent_path.read_text())
        return None

    def load_events_chain_from_file(
        self, domain: Domain
    ) -> List[Dict] | None:
        """Load events chain (semantic events) from file if it exists."""
        slug = _slugify(domain.domain_name)
        events_chain_path = self.output_dir / f"{slug}_events_chain.json"
        legacy_semantic_path = self.output_dir / f"{slug}_semantic_events.json"
        path_to_use = None
        if events_chain_path.exists():
            path_to_use = events_chain_path
        elif legacy_semantic_path.exists():
            path_to_use = legacy_semantic_path

        if path_to_use:
            data = json.loads(path_to_use.read_text())
            windows_by_id = data.get("windows_by_id")
            if isinstance(windows_by_id, dict):
                try:
                    return [windows_by_id[k] for k in sorted(windows_by_id.keys())]
                except Exception:
                    return list(windows_by_id.values())
            windows = data.get("windows")
            if isinstance(windows, list):
                return windows
            if isinstance(windows, dict):
                try:
                    return [windows[k] for k in sorted(windows.keys())]
                except Exception:
                    return list(windows.values())
        return None
    def load_atomic_events_from_file(
        self, domain: Domain
    ) -> List[Dict] | None:
        """Load atomic events from file if it exists."""
        slug = _slugify(domain.domain_name)
        atomic_path = self.output_dir / f"{slug}_atomic_events.json"
        if atomic_path.exists():
            data = json.loads(atomic_path.read_text())
            return data.get("windows", [])
        return None


def _load_or_sample_elite_personas(
    sample_path: str | Path = DEFAULT_ELITE_SAMPLE_PATH,
    *,
    sample_size: int = DEFAULT_ELITE_SAMPLE_SIZE,
    seed: int = ELITE_SAMPLE_SEED,
) -> List[Dict]:
    """
    Ensure we have a local elite persona sample and return it.
    """
    sample_path = Path(sample_path)
    if not sample_path.exists():
        sample_elite_personas(
            output_path=sample_path,
            sample_size=sample_size,
            seed=seed,
        )
    return load_sampled_personas(sample_path)


def batch_generate_dynamic_profiles_from_elite_personas(
    *,
    sample_path: str | Path = DEFAULT_ELITE_SAMPLE_PATH,
    max_personas: int = DEFAULT_ELITE_SAMPLE_SIZE,
    output_root: str | Path | None = None,
    model_name: str | None = None,
    skip_existing: bool = True,
) -> None:
    """
    Generate dynamic profiles for a batch of elite personas.

    This only runs stages up to dynamic profiles (no semantic/real data) to keep
    runtime manageable while benchmarking persona coverage.
    """
    base_dir = Path(__file__).resolve().parent
    world_background = (base_dir / "context_world_background_2024.txt").read_text().strip()
    domains_path = base_dir / "domains.json"
    domains = load_domains_from_file(domains_path)

    target_sample_size = max(max_personas, DEFAULT_ELITE_SAMPLE_SIZE)
    personas = _load_or_sample_elite_personas(
        sample_path=sample_path,
        sample_size=target_sample_size,
        seed=ELITE_SAMPLE_SEED,
    )
    if max_personas:
        personas = personas[:max_personas]

    output_root_path = (
        Path(output_root) if output_root is not None else base_dir / "generated_outputs_elite_sample"
    )
    _ensure_dir(output_root_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    resolved_model_name = (
        model_name
        or os.getenv("GEMINI_MODEL_NAME")
        or os.getenv("GENERATION_MODEL_NAME")
        or "gemini-3-flash-preview"
    )
    client = GeminiJSONClient(api_key=api_key, model_name=resolved_model_name)

    total = len(personas)
    for idx, persona in enumerate(personas):
        user_description = persona.get("persona") or persona.get("user_description")
        if not user_description:
            # Skip malformed rows.
            continue

        persona_dir = output_root_path / f"{idx:03d}_{_slugify(user_description)[:48]}"
        prepare_path = persona_dir / "prepare_inputs_result.json"
        resolved_path = persona_dir / "dynamic_profiles_conflict_resolved.json"
        if skip_existing and prepare_path.exists() and resolved_path.exists():
            print(f"[{idx + 1}/{total}] skip existing {persona_dir}")
            continue

        pipeline = GenerationPipeline(
            client,
            output_dir=persona_dir,
        )
        (
            user_basic_profile,
            user_profile_text,
            dynamic_profiles,
            aggregate_usage,
            conflict_resolution_payload,
            conflicts_summary,
        ) = pipeline.generate_dynamic_profile_inputs(
            domains=domains,
            user_description=user_description,
            world_background=world_background,
        )
        conflict_info = {
            "summary": conflicts_summary,
            "resolution": conflict_resolution_payload,
        }
        intermediate = {
            "raw_user_description": user_description,
            "user_basic_profile": user_basic_profile,
            "user_profile_text": user_profile_text,
            "world_background_by_window": {},
            "user_life_contexts": {"baseline": {}, "by_window": {}},
            "general_environment": {},
            "user_full_state_summaries": {},
            "dynamic_profiles": dynamic_profiles,
        }

        _write_text(persona_dir / "raw_user_description.txt", user_description)
        _write_json(persona_dir / "persona_record.json", persona)
        _write_json(persona_dir / "aggregate_usage.json", aggregate_usage)
        _write_json(persona_dir / "prepare_inputs_result.json", intermediate)
        print(f"[{idx + 1}/{total}] generated dynamic profiles at {persona_dir}")


def example_usage() -> None:
    """Small helper so the module can be run directly."""
    api_key = os.getenv("GOOGLE_API_KEY")
    client = GeminiJSONClient(api_key=api_key)
    base_dir = Path(__file__).resolve().parent
    # base_dir = "/export/scratch_large/wenya/mem_bench/behavior_and_conversation"
    domains_path = base_dir / "domains.json"
    domains = load_domains_from_file(domains_path)
    # selected, selection_meta = select_domains_for_user(domains, optional_probability=0.5, seed=42)
    pipeline = GenerationPipeline(
        client,
        output_dir=base_dir / "generated_outputs_v2",
    )
    world_background = (base_dir / "context_world_background_2024.txt").read_text().strip()
    user_description = (base_dir / "context_user_description.txt").read_text().strip()
    pipeline.run(
        domains,
        user_description=user_description,
        world_background=world_background,
    )

def debug_dynamic_profile_generation() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = (
        os.getenv("GEMINI_MODEL_NAME")
        or os.getenv("GENERATION_MODEL_NAME")
        or "gemini-2.5-flash-lite"
    )

    model_name="gemini-3-flash-preview"
    # model_name="gemini-3-pro-preview"
    # model_name="gemini-2.5-flash-lite"
    client = GeminiJSONClient(api_key=api_key, model_name=model_name)
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "generated_outputs_debug_v14" / _slugify(model_name)
    domains_path = base_dir / "domains.json"
    # domains_path = base_dir / "domains_test.json"
    domains = load_domains_from_file(domains_path)
    pipeline = GenerationPipeline(
        client,
        output_dir=output_dir,
    )

    world_background = (base_dir / "context_world_background_2024.txt").read_text().strip()
    user_description = (base_dir / "context_user_description.txt").read_text().strip()

    (
        user_basic_profile,
        user_profile_text,
        dynamic_profiles,
        aggregate_usage,
        conflict_resolution_payload,
        conflicts_summary,
    ) = pipeline.generate_dynamic_profile_inputs(
        domains=domains,
        user_description=user_description,
        world_background=world_background,
    )
    conflict_info = {
        "summary": conflicts_summary,
        "resolution": conflict_resolution_payload,
    }
    intermediate = {
        "raw_user_description": user_description,
        "user_basic_profile": user_basic_profile,
        "user_profile_text": user_profile_text,
        "world_background": world_background,
        "world_background_by_window": {},
        "user_life_contexts": {"baseline": {}, "by_window": {}},
        "general_environment": {},
        "user_full_state_summaries": {},
        "dynamic_profiles": dynamic_profiles,
    }
    print(intermediate)
    print(aggregate_usage)
    print(conflict_info)

    usage_summary, usage_pricing = _persist_usage_artifacts(
        aggregate_usage,
        output_dir,
        basename="debug_token_usage",
        model_name=model_name,
    )

    # record intermediate, usage, conflict info to per-model debug directory
    _write_json(
        output_dir / "debug_dynamic_profile_generation_intermediate.json",
        intermediate,
    )
    _write_json(
        output_dir / "debug_dynamic_profile_generation_aggregate_usage.json",
        aggregate_usage,
    )
    _write_json(
        output_dir / "debug_dynamic_profile_generation_aggregate_usage_summary.json",
        usage_summary,
    )
    if usage_pricing:
        _write_json(
            output_dir / "debug_dynamic_profile_generation_aggregate_usage_pricing.json",
            usage_pricing,
        )
    _write_json(
        output_dir / "debug_dynamic_profile_generation_conflict_info.json",
        conflict_info,
    )

def debug_resolve_conflicts_from_file() -> None:
    base_dir = Path(__file__).resolve().parent
    model_name = "gemini-3-flash-preview"
    output_dir = base_dir / "generated_outputs_debug_v11" / _slugify(model_name)
    domain_level_fixed_dynamic_profiles_path = base_dir / "generated_outputs_debug_v11" / _slugify(model_name) / "dynamic_profiles_domain_level_fixes_applied.json"
    user_basic_profile_path = base_dir / "generated_outputs_debug_v11" / _slugify(model_name) / "user_basic_profile.json"
    user_basic_profile = json.loads(user_basic_profile_path.read_text())
    client = GeminiJSONClient(model_name=model_name)
    pipeline = GenerationPipeline(client, output_dir=output_dir)

    resolved, usage, payload = pipeline.debug_resolve_conflicts_from_file(domain_level_fixed_dynamic_profiles_path=domain_level_fixed_dynamic_profiles_path, user_basic_profile=user_basic_profile)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

def debug_review_revise_dynamic_profile_from_file() -> None:
    base_dir = Path(__file__).resolve().parent
    model_name = "gemini-3-flash-preview"
    output_dir = base_dir / "generated_outputs_debug_v10" / _slugify(model_name)
    client = GeminiJSONClient(model_name=model_name)


    pipeline = GenerationPipeline(
        llm_client=client,
        output_dir=output_dir
    )
    # domain_name = "Health & Self-care"
    # domain_name = "Leisure & Media Consumption"
    # domain_scope_definition="Encompasses users' physical and mental well-being, including health conditions, lifestyle habits, and self-care practices such as exercise, diet, sleep, and healthcare-seeking behavior. It describes how users manage and optimize their health over time."
    # domain_scope_definition="Captures users' recreational activities and content preferences, including entertainment, hobbies, travel, and consumption of digital media such as videos, music, games, and books. It reflects how users spend discretionary time and pursue enjoyment."
    # domain_name = "Health & Self-care"
    # domain_scope_definition="Encompasses users' physical and mental well-being, including health conditions, lifestyle habits, and self-care practices such as exercise, diet, sleep, and healthcare-seeking behavior. It describes how users manage and optimize their health over time."

    domain_name = "Family & Close Relationships"
    domain_scope_definition = "Describes users' family structure and intimate relationships, such as partnerships, parenting roles, and household responsibilities. It captures close interpersonal bonds that shape daily routines, obligations, and life decisions."

    domain = Domain(domain_name=domain_name, domain_scope_definition=domain_scope_definition)
    slug = _slugify(domain.domain_name)

    with open(output_dir  / "user_basic_profile.json", "r") as f:
        user_basic_profile = json.load(f)

    with open(output_dir / f"{slug}_dynamic_profile.json", "r", encoding="utf-8") as f:
        dynamic_profile = json.load(f)

    # import pdb; pdb.set_trace()

    reviewed_result, revised_profile, usage = pipeline.review_revise_dynamic_profile_from_file(
        domain_profile_path=output_dir / f"{slug}_dynamic_profile.json",
        domain=domain,
        user_profile_text=user_basic_profile,  # or pass the basic profile JSON string/dict
    )
    # revised_profile = _apply_profile_revision(dynamic_profile, reviewed)
    # revised_path = output_dir / f"{slug}_dynamic_profile_revised_result.json"
    # _write_json(revised_path, revised_profile)
    # print("Reviewed saved to:", reviewed_path)
    # print("Revised saved to:", revised_path)
    print("Usage:", usage)

def debug_prepare_context_for_semantic_events_generation() -> None: 
    base_dir = Path(__file__).resolve().parent
    model_name = "gemini-3-flash-preview"
    output_dir = base_dir / "generated_outputs_debug_v6" / _slugify(model_name)
    client = GeminiJSONClient(model_name=model_name)
    pipeline = GenerationPipeline(client, output_dir=output_dir)
    with open(base_dir / "context_world_background_2024.txt", "r") as f:
        world_background = f.read()
    with open(output_dir / "user_basic_profile.json", "r") as f:
        user_basic_profile = json.load(f)

    with open(output_dir / "dynamic_profiles_conflict_resolved.json", "r") as f:
        dynamic_profiles = json.load(f)
    (
        user_full_state_summaries,
        user_life_contexts,
        world_background_by_window,
        life_context_usage,
    ) = pipeline.prepare_context_for_semantic_events_generation(dynamic_profiles, world_background=world_background, user_basic_profile=user_basic_profile)
    print(user_full_state_summaries)
    print(user_life_contexts)
    print(world_background_by_window)
    print(life_context_usage)

def debug_generate_semantic_events() -> None:
    base_dir = Path(__file__).resolve().parent
    model_name = "gemini-3-flash-preview"
    output_dir = base_dir / "generated_outputs_debug_v6" / _slugify(model_name)
    client = GeminiJSONClient(model_name=model_name)
    pipeline = GenerationPipeline(client, output_dir=output_dir)
    with open(output_dir / "user_basic_profile.json", "r") as f:
        user_basic_profile = json.load(f)
    with open(base_dir / "context_world_background_2024.txt", "r") as f:
        world_background = f.read()
    with open(output_dir / "dynamic_profiles_conflict_resolved.json", "r") as f:
        dynamic_profiles = json.load(f)
    with open(output_dir / "life_context_by_window.json", "r") as f:
        user_life_contexts = json.load(f)
    with open(output_dir / "user_full_state_summaries.json", "r") as f:
        user_full_state_summaries = json.load(f)

    # Build cross-domain initial summary for w1 previous-window context.
    initial_summaries_parts: List[str] = []
    for d_name, profile in dynamic_profiles.items():
        init_summary = (profile.get("initial_state") or {}).get("summary") or ""
        if init_summary:
            initial_summaries_parts.append(f"{d_name}: {init_summary}")
    initial_all_domains_summary = "\n\n".join(initial_summaries_parts)
    if not initial_all_domains_summary:
        first_window = None
        for window_id in sorted(user_full_state_summaries.keys()):
            entry = user_full_state_summaries.get(window_id)
            if isinstance(entry, dict):
                first_window = entry
                break
        if first_window:
            initial_all_domains_summary = first_window.get("window_profile_summary", "") or ""
    # domain_name = "Leisure & Media Consumption"
    # # domain_scope_definition="Encompasses users' physical and mental well-being, including health conditions, lifestyle habits, and self-care practices such as exercise, diet, sleep, and healthcare-seeking behavior. It describes how users manage and optimize their health over time."
    # domain_scope_definition="Captures users' recreational activities and content preferences, including entertainment, hobbies, travel, and consumption of digital media such as videos, music, games, and books. It reflects how users spend discretionary time and pursue enjoyment."
    domain_name = "Health & Self-care"
    domain_scope_definition = "Encompasses users' physical and mental well-being, including health conditions, lifestyle habits, and self-care practices such as exercise, diet, sleep, and healthcare-seeking behavior. It describes how users manage and optimize their health over time."


    domain = Domain(domain_name=domain_name, domain_scope_definition=domain_scope_definition)
 
    # NOTE: generate_semantic_events_for_domain expects the full multi-domain dynamic_profiles dict.
    events_chain_windows = pipeline.generate_semantic_events_for_domain(
        domain=domain,
        user_basic_profile=user_basic_profile,
        dynamic_profiles=dynamic_profiles,
        user_life_contexts=user_life_contexts,
        world_background=world_background,
        user_full_state_summaries=user_full_state_summaries,
        initial_all_domains_summary=initial_all_domains_summary,
    )
    print(events_chain_windows)

def debug_generate_real_data() -> None:
    base_dir = Path(__file__).resolve().parent
    model_name = "gemini-3-flash-preview"
    output_dir = base_dir / "generated_outputs_debug_v6" / _slugify(model_name)
    client = GeminiJSONClient(model_name=model_name)
    pipeline = GenerationPipeline(client, output_dir=output_dir)

    with open(output_dir / "user_basic_profile.json", "r") as f:
        user_basic_profile = json.load(f)
    with open(output_dir / "dynamic_profiles_conflict_resolved.json", "r") as f:
        dynamic_profiles = json.load(f)
    with open(output_dir / "life_context_by_window.json", "r") as f:
        user_life_contexts = json.load(f)
    with open(output_dir / "user_full_state_summaries.json", "r") as f:
        user_full_state_summaries = json.load(f)

    domain_name = "Health & Self-care"
    domain_scope_definition = "Encompasses users' physical and mental well-being, including health conditions, lifestyle habits, and self-care practices such as exercise, diet, sleep, and healthcare-seeking behavior. It describes how users manage and optimize their health over time."
    domain = Domain(domain_name=domain_name, domain_scope_definition=domain_scope_definition)

    events_chain_windows = pipeline.load_events_chain_from_file(domain) or []
    aggregate_usage: Dict[str, Dict] = {}
    import pdb; pdb.set_trace()
    if not events_chain_windows:
        with open(base_dir / "context_world_background_2024.txt", "r") as f:
            world_background = f.read()

        # Build cross-domain initial summary for window1 context.
        initial_summaries_parts: List[str] = []
        for d_name, profile in dynamic_profiles.items():
            init_summary = (profile.get("initial_state") or {}).get("summary") or ""
            if init_summary:
                initial_summaries_parts.append(f"{d_name}: {init_summary}")
        initial_all_domains_summary = "\n\n".join(initial_summaries_parts)
        if not initial_all_domains_summary:
            first_window = None
            for window_id in sorted(user_full_state_summaries.keys()):
                entry = user_full_state_summaries.get(window_id)
                if isinstance(entry, dict):
                    first_window = entry
                    break
            if first_window:
                initial_all_domains_summary = (
                    first_window.get("window_profile_summary", "") or ""
                )

        events_chain_windows, usage = pipeline.generate_semantic_events_for_domain(
            domain=domain,
            user_basic_profile=user_basic_profile,
            dynamic_profiles=dynamic_profiles,
            user_life_contexts=user_life_contexts,
            user_full_state_summaries=user_full_state_summaries,
            world_background=world_background,
            usage={},
            initial_all_domains_summary=initial_all_domains_summary,
        )
        aggregate_usage[domain.domain_name] = usage

    if not events_chain_windows:
        raise FileNotFoundError("No semantic events found; generate them before real data.")

    usage_for_real = aggregate_usage.get(domain.domain_name)
    real_data_windows, usage_for_real = pipeline.generate_real_data_for_domain(
        domain=domain,
        dynamic_profiles=dynamic_profiles,
        events_chain_windows=events_chain_windows,
        user_basic_profile=user_basic_profile,
        user_life_contexts=user_life_contexts,
        user_full_state_summaries=user_full_state_summaries,
        usage=usage_for_real,
    )
    aggregate_usage[domain.domain_name] = usage_for_real

    print(json.dumps(real_data_windows, indent=2, ensure_ascii=False))
    print(json.dumps(aggregate_usage, indent=2, ensure_ascii=False))

def debug_cross_domain_conflict_resolution_temporal() -> None:
    ## use /export/scratch_large/wenya/mem_bench/behavior_and_conversation/generated_outputs_debug_v11/gemini_3_flash_preview/cross_domain_conflict_resolution_temporal_claude.json
    ## apply resolution to dynamic profiles
    ## save the resolved dynamic profiles to /export/scratch_large/wenya/mem_bench/behavior_and_conversation/generated_outputs_debug_v11/gemini_3_flash_preview/cross_domain_conflict_resolution_temporal_iter1.json
    base_dir = Path(__file__).resolve().parent
    model_name = "gemini-3-flash-preview"
    output_dir = base_dir / "generated_outputs_debug_v11" / _slugify(model_name)
    with open(output_dir / "cross_domain_conflict_resolution_temporal_claude.json", "r") as f:
        data = json.load(f)
    dynamic_profiles_path = output_dir / "dynamic_profiles_domain_level_fixes_applied.json"
    with open(dynamic_profiles_path, "r") as f:
        dynamic_profiles = json.load(f)
    resolved_profiles = _apply_conflict_resolution_to_profiles(dynamic_profiles, data)
    _write_json(output_dir / "dynamic_profiles_conflict_resolved_claude.json", resolved_profiles)

if __name__ == "__main__":
    # debug_cross_domain_conflict_resolution_temporal()
    # example_usage()
    debug_dynamic_profile_generation()
    # debug_resolve_conflicts_from_file()
    # debug_resolve_conflicts_from_file()
    # debug_review_revise_dynamic_profile_from_file()
    # debug_review_revise_dynamic_profile_from_file()
    # debug_resolve_conflicts_from_file()

    # ===== batch generate dynamic profiles from elite personas =====
    # base_dir = Path(__file__).resolve().parent
    # batch_generate_dynamic_profiles_from_elite_personas(
    #     sample_path=base_dir / "persona_data" / "elite_personas_sample_seed42.jsonl",
    #     max_personas=10,
    #     output_root=base_dir / "generated_outputs_elite_sample",
    #     model_name="gemini-3-flash-preview",
    # )
    # ===== batch generate dynamic profiles from elite personas =====

    # ===== prepare context for semantic events generation =====
    # debug_prepare_context_for_semantic_events_generation()
    # debug_generate_semantic_events()
    # debug_generate_real_data()
    # ===== prepare context for semantic events generation =====


    # debug_review_revise_dynamic_profile_from_file()
