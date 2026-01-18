from __future__ import annotations

from jinja2 import Template


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


app_log_prompt = Template("""## Task Overview

Your task is to generate a realistic API call log (input and output) for a specific user interaction with an app. You are simulating what a real user would input and what the app would return based on the user's profile, context, and current app state.

---

## Input Context

You will receive the following information:

### 1. API Schema
The structure of the API call, showing input and output fields with their types:
```
{{ api_schema }}
```

### 2. User Profile
Background information about the user, including:
- Basic demographics and characteristics
- Domain-specific context (work, interests, expertise)
- Relevant personality traits and preferences

```
{{ user_profile }}
```

### 3. Current App State
The state of this app before this API call, which may include:
- Search history
- Previous interactions
- Saved items, playlists, notes, etc.
- Order history, conversations, etc.

```
{{ app_state }}
```

### 4. Event Payload
Details about this specific event:
- Time specification (reference time, NOT exact timestamp to use)
- User intent (why this interaction is happening)
- Related state items (which user attributes/habits/preferences this demonstrates)
- Chain context

```
{{ event_payload }}
```

### 5. Previous App Logs in This Chain
Previously generated app logs from the same event chain (in chronological order). Use these to maintain consistency in:
- Ongoing conversations or sessions
- Product/item references (use same IDs, names)
- Narrative continuity
- Building on previous interactions

```
{{ previous_chain_logs }}
```

---

## Generation Guidelines

### Rule 1: Follow Schema Exactly

- **Input:** Include ALL required fields specified in the schema
- **Output:** Include ALL fields specified in the schema
- **Types:** Respect field types (string, number, boolean, array, object)
- **Structure:** Match the nesting structure exactly

### Rule 2: Generate Realistic User Input

**Characteristics of realistic user input:**

**Natural and concise**
- Users typically use short queries and natural language
- Not overly verbose or formal
- May include typos, abbreviations, or casual phrasing (when appropriate)

**Reflects user expertise**
- Technical users naturally use technical terminology
- Domain experts use domain-specific language
- Adjust specificity based on user's knowledge level

**Context-aware**
- Reference previous interactions when relevant (using app state)
- Show progression (e.g., more specific searches after general ones)
- Reflect current user intent

**Sometimes "lazy" or efficient**
- Users may use minimal words to get the job done
- Not every query needs full sentences
- Users reuse successful patterns

**Examples:**

For a Senior Software Engineer searching for AI tools:
- Good: `"github copilot vs chatgpt for C++"`
- Good: `"AI code completion tools"`
- Too formal: `"Please provide a comprehensive comparison of artificial intelligence-powered code completion tools suitable for enterprise C++ development"`
- Too vague: `"coding help"`

For the same user creating a note:
- Good: Title: `"AI Tool Evaluation Notes"`, Content: `"Tested Copilot on state machine refactor - works well with modern C++ patterns. Need to try on legacy code next."`
- Too polished: Content looks like a published article with perfect formatting and structure

### Rule 3: Generate Semantically Rich Output

**Characteristics of semantically rich output:**

**Detailed and informative**
- Include descriptions, specifications, features
- Provide realistic metadata (ratings, counts, dates)
- Add context that makes output useful

**Personalized to user**
- Search results should align with user's profile and interests
- Recommendations should match user's preferences
- Content should be relevant to user's expertise level

**Realistic and plausible**
- Use real product names, song titles, artist names when possible
- If fictional, make them realistic (e.g., realistic book titles, company names)
- Prices, ratings, and counts should be plausible
- Dates and timestamps should be logical

**Contextually appropriate**
- Match the user's intent
- Consider the user's previous interactions (from app state)
- Reflect the domain context

**Examples:**

For `SearchProducts` with query "noise canceling headphones" (Senior Software Engineer, likes focus music):
```json
{
  "products": [
    {
      "name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
      "price": 399.99,
      "rating": 4.7,
      "description": "Industry-leading noise cancellation with exceptional sound quality. 30-hour battery life, multipoint connection. Ideal for focused work and long coding sessions."
    },
    {
      "name": "Bose QuietComfort 45 Bluetooth Headphones",
      "price": 329.00,
      "rating": 4.6,
      "description": "Premium comfort with excellent ANC. Perfect for all-day wear in office or remote work. Balanced audio with clear mids for voice calls."
    },
    {
      "name": "Apple AirPods Max",
      "price": 549.00,
      "rating": 4.5,
      "description": "High-fidelity audio with adaptive EQ and active noise cancellation. Seamless integration with Apple ecosystem. Premium build quality."
    }
  ],
  "total_results": 847
}
```

Note how the output:
- Uses real product names and realistic prices
- Includes descriptions that mention "focused work" and "office" (aligned with user profile)
- Has plausible ratings and result counts
- Provides enough detail to be useful

### Rule 4: Maintain Consistency and Continuity

**Reference previous interactions:**
- When input requires referring to previous actions, use information from app state
- Maintain continuity within event chains (e.g., product viewed in previous event should be the same product added to cart)
- Use consistent IDs, names, and references

**Examples:**

If app state shows:
```json
{
  "search_history": ["noise canceling headphones"],
  "viewed_products": ["Sony WH-1000XM5"]
}
```

Then for `AddToCart` API:
```json
{
  "input": {
    "product_name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
    "quantity": 1
  }
}
```
(References the product from viewed_products)

**Apply Realistic Timestamp Variation:**
The `time_specification` in the event payload provides a **reference time**, not an exact timestamp. Real-world interactions have natural timing variation. When generating timestamps:

- **Add realistic randomness**: A user who "usually works out at 7:00 AM" might actually start at 6:52, 7:08, or 7:15 on different days
- **±15-30 minute variation is normal** for most activities (meetings, workouts, daily habits)
- **±5-10 minute variation** for more time-sensitive activities (appointments, scheduled calls)
- **Consider the activity context**:
  - Morning routines vary based on sleep quality, traffic, mood
  - Evening activities vary based on work day length, energy levels
  - Weekend timing is typically more relaxed than weekday
- **Duration variation**: If an activity has start_time and end_time, the actual duration can vary too
- **Maintain logical ordering**: If you add variation, ensure timestamps still make sense sequentially

Example: If time_specification shows `"start_time": "09:00:00"`:
- Good: Generate timestamp as "09:07:23" or "08:54:15" (realistic variation)
- Bad: Use exactly "09:00:00" (too precise, unrealistic)

**Respect temporal logic:**
- Estimated delivery dates should be in the future
- Order numbers, IDs should be unique and follow realistic patterns

**Maintain chain coherence:**
- If event is part of a chain (check `event_payload.chain`), ensure the API call fits the narrative
- Related state items provide context for what this interaction demonstrates
- User intent explains the "why" — let it guide the content

**Use Previous Chain Logs for Consistency:**
When `previous_chain_logs` is provided, use it to maintain consistency:
- **Reference same entities**: If a previous log searched for "Sony WH-1000XM5", subsequent add-to-cart should use the same product name
- **Continue conversations**: If previous LLM chat discussed a topic, new messages should be coherent continuations
- **Build on context**: User may reference things done earlier in the chain ("like I mentioned earlier", "continuing from yesterday")
- **Maintain session IDs**: If a session_id or conversation_id was established, continue using it
- **Show progression**: Later events in a chain often show refinement, follow-up, or completion of earlier actions

Example chain coherence:
- Event 1: Search for "noise canceling headphones" → Results include Sony WH-1000XM5
- Event 2: View product → Should be Sony WH-1000XM5 (same product from search)
- Event 3: Add to cart → Should be Sony WH-1000XM5 (same product)
- Event 4: Checkout → Order should contain Sony WH-1000XM5

### Rule 5: Simulate Realistic App Behavior

**Search and Discovery:**
- Search results should be relevant but not perfect
- Include variety in results (different brands, price points)
- Total result counts should be realistic
- Order results by relevance (considering user profile)

**Recommendations:**
- Personalize based on user profile and app state
- Mix popular items with niche items matching user interests
- Consider user's expertise level

**User-Generated Content:**
- Notes, messages, reviews should match user's personality
- Technical users write technical content
- Casual users write casual content
- Content reflects user's knowledge and interests

**System-Generated Data:**
- Order numbers: realistic formats (e.g., "AMZ-2024-01-15-8372")
- Tracking events: realistic status updates
- Statistics: plausible numbers aligned with user's activity level

---

## Complete Examples

### Example 1: Amazon.SearchProducts

**Context:**
- User: Senior Software Engineer, interested in AI tools and productivity
- Intent: "Attribute acquisition: researching noise-canceling headphones for better focus during work"
- Timestamp: 2024-01-20 19:30:00
- App State: `{"search_history": [], "viewed_products": [], "cart": [], "order_history": []}`

**API Schema:**
```json
{
  "input": {"query": "string"},
  "output": {
    "products": [{"name": "string", "price": "number", "rating": "number", "description": "string"}],
    "total_results": "integer"
  }
}
```

**Generated Log:**
```json
{
  "input": {
    "query": "noise canceling headphones"
  },
  "output": {
    "products": [
      {
        "name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        "price": 399.99,
        "rating": 4.7,
        "description": "Industry-leading noise cancellation with 30-hour battery. Perfect for focused work with superior sound quality and multipoint connection for seamless device switching."
      },
      {
        "name": "Bose QuietComfort 45 Bluetooth Headphones",
        "price": 329.00,
        "rating": 4.6,
        "description": "Premium comfort for all-day wear with excellent ANC. Balanced audio profile ideal for calls and music. Trusted quality for professionals."
      },
      {
        "name": "Apple AirPods Max",
        "price": 549.00,
        "rating": 4.5,
        "description": "High-fidelity audio with computational audio and adaptive EQ. Seamless Apple ecosystem integration. Premium build with aluminum ear cups."
      },
      {
        "name": "Sennheiser Momentum 4 Wireless",
        "price": 379.95,
        "rating": 4.6,
        "description": "Audiophile-grade sound with adaptive noise cancellation. 60-hour battery life. Exceptional clarity for both music and conference calls."
      }
    ],
    "total_results": 1247
  }
}
```

**Why this is good:**
- Input is concise and natural
- Products are real and relevant to a professional user
- Descriptions mention work/focus (aligned with intent)
- Prices and ratings are realistic
- Total results is plausible
- Variety in brands and price points

---

### Example 2: LLM.Chat

**Context:**
- User: Senior Software Engineer, expertise in C++ embedded systems
- Intent: "Change reason awareness: exploring how AI tools handle legacy C++ codebases"
- Timestamp: 2024-01-03 20:45:00
- App State: `{"conversation_history": []}`

**API Schema:**
```json
{
  "input": {"message": "string"},
  "output": {"response": "string"}
}
```

**Generated Log:**
```json
{
  "input": {
    "message": "How well do AI coding tools like Copilot and ChatGPT handle legacy C++ codebases with C++17/20 mixed usage?"
  },
  "output": {
    "response": "AI coding tools have varying capabilities with legacy C++ code. GitHub Copilot generally handles C++17/20 features well and can be helpful for:\n\n1. Autocompletion of modern patterns (structured bindings, if constexpr, etc.)\n2. Generating boilerplate for common patterns\n3. Suggesting refactorings to newer standards\n\nHowever, challenges with legacy code include:\n- May struggle with older pre-C++11 patterns mixed with modern code\n- Limited context window can miss critical dependencies in large codebases\n- Sometimes suggests overly modern solutions for compatibility-constrained environments\n\nChatGPT (like this conversation) is better for:\n- Explaining migration strategies\n- Discussing trade-offs between approaches\n- Generating focused code snippets with specific constraints\n\nFor embedded systems specifically, I'd recommend testing both on representative code sections. Copilot excels at day-to-day completion, while ChatGPT is better for architectural decisions and complex refactoring strategies. Many teams use both complementarily."
  }
}
```

**Why this is good:**
- Input reflects user's technical expertise (mentions C++17/20, legacy code)
- Input is natural and conversational, not overly formal
- Output is detailed and technical, matching user's level
- Response specifically addresses embedded systems (from user profile)
- Provides practical, actionable information
- Tone is professional but conversational

---

### Example 3: SimpleNote.CreateNote

**Context:**
- User: Senior Software Engineer, methodical and prefers documentation
- Intent: "Post-acquisition usage: documenting Python analysis script for recurring telemetry review"
- Timestamp: 2024-02-09 13:45:00
- App State: `{"notes": [{"title": "AI Tool Evaluation", "tags": ["work", "tools"]}]}`

**API Schema:**
```json
{
  "input": {"title": "string", "content": "string", "tags": ["string"]},
  "output": {"note_id": "string", "title": "string", "created_at": "string", "tags": ["string"]}
}
```

**Generated Log:**
```json
{
  "input": {
    "title": "Telemetry Analysis Script - Python",
    "content": "Quick reference for the telemetry log parser I built:\n\nPurpose: Parse robot motor performance logs from test runs, identify anomalies\n\nKey libraries:\n- pandas for CSV parsing\n- numpy for statistical analysis\n- matplotlib for visualization (optional)\n\nUsage:\n```\npython analyze_telemetry.py --input logs/motor_test_20240209.csv --threshold 0.95\n```\n\nOutput: Generates summary stats and flags any performance drops >5%\n\nNote: Script assumes standard test log format (timestamp, motor_id, rpm, current, temp). Need to update parser if log format changes.\n\nNext: Add automated alerting if anomalies detected.",
    "tags": ["work", "python", "automation"]
  },
  "output": {
    "note_id": "note_1707484700",
    "title": "Telemetry Analysis Script - Python",
    "created_at": "2024-02-09 13:45:00",
    "tags": ["work", "python", "automation"]
  }
}
```

**Why this is good:**
- Title is clear and descriptive (engineer's style)
- Content is technical but concise (not over-polished)
- Includes code snippet in a realistic format
- "Quick reference" tone matches the "documenting for recurring use" intent
- Tags are relevant and consistent with existing notes
- Note structure reflects how a developer actually documents
- Mentions domain-specific terms (motor, telemetry, rpm)

---

### Example 4: Spotify.PlaySong (Repeated Event)

**Context:**
- User: Senior Software Engineer, prefers instrumental focus music
- Intent: "Combined usage: using new headphones with preferred lofi music during deep work"
- Timestamp: 2024-01-25 09:15:00 (first of multiple dates)
- App State: `{"playlists": [{"name": "Focus Coding", "tracks": []}], "recently_played": [], "favorite_genres": []}`

**API Schema:**
```json
{
  "input": {"track": "string", "playlist": "string"},
  "output": {"now_playing": "string", "artist": "string", "duration": "integer"}
}
```

**Generated Log:**
```json
{
  "input": {
    "track": "Snowfall",
    "playlist": "Lofi Hip Hop Beats"
  },
  "output": {
    "now_playing": "Snowfall",
    "artist": "Øneheart",
    "duration": 178
  }
}
```

**Why this is good:**
- Track and artist are real lofi artists/tracks
- Input references a lofi playlist (matches preference)
- Duration is realistic (under 3 minutes, typical for lofi)
- Simple input reflects actual Spotify usage
- Aligned with "instrumental focus music" preference

---

### Example 5: Amazon.Checkout (With Chain Context)

**Context:**
- User: Senior Software Engineer
- Intent: "Attribute acquisition: purchasing selected noise-canceling headphones"
- Timestamp: 2024-01-21 11:00:00
- Related state: Previous events in chain include searching and viewing products
- App State: 
```json
{
  "cart": [
    {"product_name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones", "quantity": 1, "price": 399.99}
  ],
  "order_history": []
}
```

**API Schema:**
```json
{
  "input": {},
  "output": {
    "order_number": "string",
    "items": ["object"],
    "total_price": "number",
    "timestamp": "string",
    "estimated_delivery": "string"
  }
}
```

**Generated Log:**
```json
{
  "input": {},
  "output": {
    "order_number": "AMZ-2024-01-21-4729",
    "items": [
      {
        "product_name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        "quantity": 1,
        "price": 399.99
      }
    ],
    "total_price": 399.99,
    "timestamp": "2024-01-21 11:00:00",
    "estimated_delivery": "2024-01-24"
  }
}
```

**Why this is good:**
- Items match what's in cart (consistency with app state)
- Order number follows realistic format with date
- Timestamp uses event timestamp
- Estimated delivery is 3 days later (realistic for standard shipping)
- Total price matches item price

---

### Example 6: Fitness.LogWorkout (Habit Execution with Timestamp Variation)

**Context:**
- User: Mid-30s professional, building morning workout habit
- Intent: "Habit execution: daily morning workout, building consistency and routine"
- Time specification: start_time "07:00:00" (reference time for morning workout habit)
- App State: 
```json
{
  "workouts": [
    {"date": "2024-01-04", "type": "strength", "duration": 25, "logged_at": "2024-01-04 07:12:00"},
    {"date": "2024-01-05", "type": "cardio", "duration": 20, "logged_at": "2024-01-05 06:48:00"}
  ],
  "user_baseline": {"fitness_level": "beginner-intermediate"}
}
```

**API Schema:**
```json
{
  "input": {"workout_type": "string", "duration": "integer", "intensity": "string", "notes": "string"},
  "output": {"workout_id": "string", "date": "string", "calories_burned": "integer", "logged_at": "string"}
}
```

**Generated Log:**
```json
{
  "input": {
    "workout_type": "strength",
    "duration": 30,
    "intensity": "moderate",
    "notes": "Full body circuit - push-ups, squats, planks. Felt good, added 5 extra squats."
  },
  "output": {
    "workout_id": "workout_20240108_070823",
    "date": "2024-01-08",
    "calories_burned": 185,
    "logged_at": "2024-01-08 07:08:23"
  }
}
```

**Why this is good:**
- Workout type and duration align with habit pattern (30min morning session)
- Notes sound authentic (casual, brief, mentions small progress)
- Intensity matches beginner-intermediate level
- Calories burned is realistic for 30min moderate strength training
- Shows gradual progression (added 5 extra squats) fitting a new habit
- **logged_at has realistic variation (07:08:23 instead of exact 07:00:00)** — real people don't start workouts at exactly the scheduled time

---

## Common Pitfalls to Avoid

❌ **Over-formal or artificial input**
- Bad: `"I would like to inquire about high-quality noise cancellation headphones"`
- Good: `"noise canceling headphones"`

❌ **Generic or template-like output**
- Bad: All products have identical generic descriptions
- Good: Each product has unique, relevant details

❌ **Ignoring user profile**
- Bad: Recommending beginner tutorials to an expert
- Good: Content matches user's expertise level

❌ **Inconsistent references**
- Bad: Adding product to cart that was never searched/viewed
- Good: Cart contains product from previous ShowProduct call

❌ **Unrealistic data**
- Bad: Product priced at $10,000 or rated 5.0 with 2 million reviews
- Good: Plausible prices and realistic rating distributions

❌ **Ignoring user intent**
- Bad: Generic search results when intent specifies acquisition for specific purpose
- Good: Results and descriptions aligned with stated intent

❌ **Overly precise timestamps**
- Bad: Using exactly "07:00:00" when time_specification says "07:00:00" (too robotic)
- Good: Using "07:08:23" or "06:54:17" (realistic human variation)

❌ **Breaking temporal logic**
- Bad: Timestamps going backwards within a session
- Good: Timestamps progress logically (start < end, sequential events ordered)

❌ **Missing semantic richness**
- Bad: Product description: `"Good headphones"`
- Good: Product description: `"Industry-leading noise cancellation with 30-hour battery. Perfect for focused work..."`

---

## Final Checklist

Before returning your JSON, verify:

- [ ] Input includes all required schema fields
- [ ] Output includes all required schema fields  
- [ ] Input is realistic and natural for this user
- [ ] Output is semantically rich and detailed
- [ ] Data aligns with user profile and intent
- [ ] References to previous interactions are consistent with app state
- [ ] **Timestamps have realistic variation** (not exactly matching time_specification)
- [ ] **Previous chain logs are used for consistency** (same products, continuing conversations, etc.)
- [ ] IDs and references follow realistic patterns
- [ ] No markdown fences or extra text
- [ ] Valid JSON syntax

---

## Now Generate

Based on all the context provided, generate the API call log as a valid JSON object.

**CRITICAL: Your output MUST have EXACTLY this structure:**

```json
{
  "input": { ... },
  "output": { ... }
}
```

- Use **"input"** for API input (NOT "request")
- Use **"output"** for API output (NOT "response")
- Do NOT include any other top-level keys (no "event_id", "timestamp", "app_name", etc.)
- The "input" and "output" values should match the API schema provided above

""")


