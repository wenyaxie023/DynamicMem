import json
from dataclasses import dataclass

from jinja2 import Template

from mem_bench.behavior_and_conversation.llm_client import (
    GeminiJSONClient,
    LLMResult,
)

# 
# events_chain_template = Template("""You are generating **semantic events** for ONE domain within ONE time window.

# 所有具体事件都必须能被映射到dialogue事件或者是app事件
# semantic events主要能够交代整个事情的因果 以及 可能同一个目标下的多个事件 比如 一个user attribute 显示用户增加了一个新的设备，那么一个semantic event交代了一个完整的故事链 例如说 用户先在dialogue中提起想要 看了一个 (app 事件1),
# A **semantic event** is:
# > A semantically meaningful unit of user activity that provides observable evidence for the current window’s user_attributes_state, habits_state, or preferences_state.
# > Each semantic event can later be expanded into multiple concrete logs (app data, dialogue).

# ### Domain
# {{ domain_name }}

# ### Time window state
# Below is the state for this domain in this time window (piecewise-stable):

# {{ window_state_json }}

# ### Definitions

# - **User Attribute semantic event**:
#   - Reveals that some object/resource is acquired, upgraded, maintained, or abandoned.
#   - Evidence for user_attributes_state.
# - **Habit semantic event**:
#   - Shows a recurring routine or typical context consistent with the habits_state.
#   - Evidence for habits_state.
# - **Preference semantic event**:
#   - Gives explicit or strong implicit evidence about what the user prefers within this domain.
#   - Evidence for preferences_state.

# ### Requirements

# 1. Generate semantic events for this (domain, window).
# 2. All events must:
#    - Be consistent with the given user_attributes_state, habits_state, and preferences_state.
#    - Be plausible given the world background for this window.
#    - Not change the latent state themselves; they are **evidence** of the current state.
# 3. Each semantic event must:
#    - Have a unique ID within this window.
#    - Have a short, precise description.
#    - Label its **event_type** ∈ {"user_attribute", "habit", "preference"}.
#    - Indicate which parts of user_attributes_state, habits_state, and preferences_state it supports (as short strings).

# Return **valid JSON only**, with this schema:

# {
#   "domain": "{{ domain_name }}",
#   "window_id": "{{ window_id }}",
#   "events_chain": [
#     {
#       "semantic_event_id": "se_01",
#       "event_type": "user_attribute",  // or "habit" or "preference"
#       "description": "<one-sentence description>",
#       "evidence_for": {
#         "user_attributes_state": "<which user attributes this supports, or empty string>",
#         "habits_state": "<which habits this supports, or empty>",
#         "preferences_state": "<which preference dimensions this supports, or empty>"
#       }
#     },
#     ...
#   ]
# }

# Examples:

# """
# )
# events_chain_template = Template("""
# You are a User Behavior Simulator responsible for constructing a timeline of "Semantic Events" for one user in one time window.

# 1. Inputs

# - Time window: {{ window_state_json['time_range'][0] }} to {{ window_state_json['time_range'][1] }}
# - Window description: {{ window_state_json['window_description'] }}
# - Life domain: {{ domain_name }}
# - Full user profile state in this window (attributes, habits, preferences, including any fields with "op"): 
# {{ window_state_json }}

# 2. Task

# A "Semantic Event" is a high-level, human-interpretable episode, such as:
# - "Researching new laptops for travel"
# - "Weekly grocery run"
# - "Revising budget after discovering new sports streaming services"

# You will output a list of such events that will later be turned into detailed app logs and dialogues.

# 3. Requirements

# 3.1 Transition coverage (fields with "op")

# - In the input JSON, some profile entries contain an "op" field (for example: "modify", "acquire", "adjust", "drop", "shift", "amplify", "attenuate").
# - For EVERY profile entry that has an "op" field, you MUST create at least one event of type "TRANSITION" that explicitly explains or demonstrates this change.
# - The event’s "related_profile_fields" must include the exact "name" of each changed field that the event explains.
# - It is allowed for one TRANSITION event to involve multiple related_profile_fields when the changes are causally related.
# - The narrative must show a clear cause → action → outcome that makes the change realistic.

# Example:
# - Data: user_monthly_tech_gadget_budget has "op": "modify" with change_reason about major summer sporting events and tech purchases.
# - Event: the user learns about streaming options and new viewing equipment for the Olympics, reviews their monthly budget, and decides to increase the tech gadget budget.

# 3.2 Routine coverage (stable state)

# - For profile entries without an "op" field (or for the new post-change state of entries that do have "op"), generate events of type "ROUTINE" that show the user’s typical behavior in this window.
# - Use the frequencies implied by habits_state. For example, if the user "Shops online 3–4 times per week", there should be multiple shopping-related events spread across the window.
# - Ensure all events are consistent with:
#   - The occupation (for example, Software Engineer),
#   - The income level (for example, mid-to-high income),
#   - The time window and its description (for example, major summer sporting events, peak summer, impulse purchases, etc.).

# 3.3 Time and diversity

# - "approx_date" must be an ISO date string "YYYY-MM-DD".
# - "approx_date" MUST fall between {{ window_state_json['time_range'][0] }} and {{ window_state_json['time_range'][1] }} (inclusive).
# - Distribute events across the whole window; avoid putting all events on the same day unless this is clearly justified by the narrative.
# - Avoid redundant or almost-duplicate events. Each event should add new information or context about the user.

# 3.4 Grounding plan

# For each event, specify how it would be observed in downstream data:

# - "primary_evidence" MUST be one of: "App Logs", "Dialogue", or "Both".
#   - Use "App Logs" for behaviors mainly reflected in app / service activity (e-commerce, banking, streaming, fitness, etc.).
#   - Use "Dialogue" for things mainly expressed in natural language conversations.
#   - Use "Both" when both are natural and informative.
# - "supporting_evidence" is a short free-text note, such as:
#   - "Chat with friend about budget increase"
#   - "Mention in casual conversation about watching the Olympics"
#   - "None"

# 4. Output format (STRICT)

# Return ONLY a JSON array of objects, with no additional explanation, comments, or markdown.

# Each event object MUST follow this schema:

# [
#   {
#     "event_id": "evt_{{ window_state_json['window_id'] }}_001",   // unique within this window
#     "type": "TRANSITION" or "ROUTINE",
#     "related_profile_fields": ["field_name_1", "field_name_2"], // list of profile 'name' strings
#     "approx_date": "YYYY-MM-DD",
#     "summary": "Short one-line description of the event",
#     "narrative": "A detailed, coherent mini-story that explains the event (cause → action → result) and is consistent with the profile and window description.",
#     "grounding_plan": {
#       "primary_evidence": "App Logs" or "Dialogue" or "Both",
#       "supporting_evidence": "Short note or 'None'"
#     }
#   }

#   // ... more events
# ]

# 5. Event count

# Generate exactly {{ num_events }} events.

# - First, ensure that every profile entry with an "op" field is covered by at least one TRANSITION event. You may cover multiple related fields in a single TRANSITION event by listing them all in "related_profile_fields".
# - After all "op" changes are covered, use the remaining events for ROUTINE behavior that reflects:
#   - the stable parts of the profile,
#   - the habits frequencies,
#   - and the time-window description (for example, peak summer, major sporting events, impulse purchases).
# """)


# events_chain_template = Template("""You are a model that converts ONE user profile dimension timeline into a list of semantic events.

# ## Task

# You receive a request with the following fields:

# - domain_name: a string describing the business / product domain. You MAY use it for style or minor assumptions, but you MUST NOT hard-code domain-specific logic.
# - user_profile_dimension_category: one of
#   - "user_attributes_state"
#   - "habits_state"
#   - "preferences_state"

# - user_profile_dimension_name: a string key for this dimension, e.g.:
#   - "user_income"
#   - "user_investment_portfolio"
#   - "user_budget_tracking"
#   - "technology_spending"
#   etc.

# - user_profile_dimension_data: an object that ALWAYS contains a `timeline` field.

# The shape of `user_profile_dimension_data` is:

# {
#   "timeline": [
#     {
#       "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
#       "current_value": <ANY JSON value describing the state in this period>,
#       "op": "add" | "modify" | "adjust" | "shift" | "drop" | "acquire" | "none" (optional),
#       "previous_value": <ANY JSON value> (optional),
#       "change_reason": "<string>" (optional)
#     },
#     ...
#   ]
# }

# The `timeline` is time-ordered and describes how this ONE profile dimension evolved over time.

# Your job is to:

# - Summarize this evolution as a small list of high-quality semantic events.
# - Capture both:
#   - meaningful CHANGES (e.g. weekly → daily tracking, tech → AI ETFs, habit dropped, preference shifted)
#   - important STABLE patterns (e.g. income unchanged all year, long-term absence of a habit, stable preference).

# You MUST only consider the ONE dimension described by:
# - user_profile_dimension_category
# - user_profile_dimension_name
# - user_profile_dimension_data.timeline


# ## Output format

# You MUST output ONLY a JSON array (no comments, no explanations), where each element has the structure:

# {
#   "semantic_event_type": "habit_change" | "preference_change" | "user_attribute_change" | "stable_user_attribute_revealed" | "stable_habit_revealed" | "stable_preference_revealed",
#   "semantic_event_id": "<string, unique within this output>",
#   "user_profile_dimension_category": "<copy from input>",
#   "user_profile_dimension_name": "<copy from input>",
#   "value": {
#     "previous_value": <JSON or null>,
#     "new_value": <JSON or string summarizing the new state>,
#     "change_reason": "<string summary; use input change_reason if available, otherwise infer a concise reason or put null>"
#   },
#   "semantic_event_time": {
#     "from_when": "YYYY-MM-DD or null",
#     "to_when": "YYYY-MM-DD or null"
#   },
#   "semantic_event_narrative": "<short natural language description of what this event means about the user>",
#   "grounding_plan": [
#     {
#       "from_when": "YYYY-MM-DD",
#       "to_when": "YYYY-MM-DD or null",
#       "source_from": "app_log" | "dialogue" | "both",
#       "narrative": "<how this event is grounded in observed data for this period>"
#     }
#   ]
# }

# ### Rules for `semantic_event_type`

# 1. If `user_profile_dimension_category` == "habits_state":
#    - Use `habit_change` for meaningful habit changes:
#      - start / stop a habit
#      - change in frequency (daily / weekly / monthly)
#      - change in timing or context (e.g. before bed, weekends only)
#    - Use `stable_habit_revealed` for long-term stable habits or stable absence of a habit.

# 2. If `user_profile_dimension_category` == "preferences_state":
#    - Use `preference_change` for shifts in preferences:
#      - e.g. gadgets → workstation performance → energy efficiency.
#    - Use `stable_preference_revealed` when a preference clearly holds across long time windows.

# 3. If `user_profile_dimension_category` == "user_attributes_state":
#    - Use `user_attribute_change` for changes in user attributes:
#      - income, savings rate, investment portfolio composition, residence type, etc.
#    - Use `stable_user_attribute_revealed` when the attribute remains effectively unchanged across all timeline entries (or across a long period that is clearly stable).


# ## Rules for event selection

# - Do NOT create one semantic event per timeline entry by default.
# - Focus on SEMANTIC events:
#   - significant changes
#   - long-lived stable states that matter for understanding the user.

# - If the dimension stays effectively the same across all time windows:
#   - Create exactly ONE `stable_*_revealed` event summarizing this stability.

# - If there are explicit change operations (entries with `op` + `previous_value`):
#   - Create at least one `*_change` event per meaningful change.
#   - Use `previous_value`, `current_value`, and `change_reason` as primary evidence.

# - If there are long stretches of time with the same `current_value` and no `op`:
#   - You MAY summarize them with a `stable_*_revealed` event
#     (e.g. long-term absence of budget tracking, stable investment theme, stable preference).

# - Prefer a small number of high-information semantic events over many noisy ones.


# ## Rules for time handling

# - Use `time_range` from each timeline entry as your main temporal information.
# - For each semantic event:
#   - `semantic_event_time.from_when` should align with the start date of the first relevant `time_range`.
#   - `semantic_event_time.to_when` should align with the end date of the last relevant `time_range`,
#     or be null if the pattern is ongoing / open-ended.

# - For `grounding_plan`:
#   - Include one or more entries with `from_when` / `to_when` taken from relevant `time_range` intervals.
#   - `source_from`:
#     - use `"app_log"` when it is natural that the evidence comes from app activity
#       (e.g. repeated logs, tracking events, portfolio changes).
#     - use `"dialogue"` when it is more natural that this comes from conversation.
#     - use `"both"` when both are plausible sources.
#   - The `narrative` should describe observable facts in that time range
#     (e.g. "Daily expense logs recorded in budgeting app", "User mentioned tax-loss harvesting at year end").


# ## Rules for narratives

# - `semantic_event_narrative`:
#   - Short, high-level, human-readable.
#   - Explain what the event reveals about the user (discipline, burnout, risk appetite, tax awareness, etc.).

# - `grounding_plan[n].narrative`:
#   - Describe what was actually observed in that period, not an abstract conclusion.


# ## Output constraints

# - You MUST output valid JSON.
# - You MUST output a JSON array (e.g. `[...]`).
# - You MUST NOT include any text outside the JSON (no explanations, no comments).
# - The number of events should be modest and focused. Avoid over-fragmentation.


# ## Example (shape only; your actual content may differ)

# Input (conceptual):

# - user_profile_dimension_category: "habits_state"
# - user_profile_dimension_name: "user_budget_tracking"
# - user_profile_dimension_data.timeline: [ ... ]

# Output style:

# [
#   {
#     "semantic_event_type": "habit_change",
#     "semantic_event_id": "budget_tracking_intensified_2024Q1",
#     "user_profile_dimension_category": "habits_state",
#     "user_profile_dimension_name": "user_budget_tracking",
#     "value": {
#       "previous_value": {...},
#       "new_value": {...},
#       "change_reason": "..."
#     },
#     "semantic_event_time": {
#       "from_when": "2024-01-01",
#       "to_when": "2024-03-31"
#     },
#     "semantic_event_narrative": "...",
#     "grounding_plan": [
#       {
#         "from_when": "2024-01-01",
#         "to_when": "2024-03-31",
#         "source_from": "app_log",
#         "narrative": "..."
#       }
#     ]
#   }
# ]
# """)

events_chain_template = Template("""
You are an expert at generating event chains that demonstrate user behaviors based on their dynamic profile state.

### Your Task:
Given a user's dynamic profile for a specific domain and time window, generate a sequence of realistic events that would naturally occur based on their attributes, habits, and preferences. Each event must specify which app/API it uses and the user intent.

---

## Context Information

### Life Context
{{ user_life_context }}

### World Background
{{ world_background }}

### User Basic Profile
{{ user_basic_profile }}

### Previous Window Summary (User's state summary for all user's lives)
{{ user_previous_window_summary }}

### Previous Window Summary (This Domain)
{{ user_domain_previous_window_summary }}

### Current Window Description (This Domain)
{{ user_this_window_description }}

---

## Domain State for Current Window

{{ domain_window_state }}

### How to read domain_window_state (STRUCTURED INPUT)
- It is a dict with keys: window_id, time_range, and state_table.
- state_table has three lists: user_attributes_state, habits_state, preferences_state.
- Each state item includes: name, current_value, op, and optional change_reason/previous_value.
- For habits_state, current_value may include schedule/timing/context/schedule_dates; use these to set time_specification.
- For preferences_state, convert current_value.signals into observable behaviors.
- If change_reason exists (especially for op add/modify/acquire/adjust/refine/shift/drop), include events that make the reason observable.

---

## Available Apps and APIs

You must specify which app and API each event uses. Available apps and APIs:

### Amazon App
- **Login**: authenticate and start a session
- **SearchProducts**: search catalog with filters and sort
- **ShowProduct**: view details for a specific product
- **Checkout**: place an order for selected items
- **ShowOrders**: list recent orders and statuses

### Spotify App
- **Login**: authenticate and start a session
- **SearchSongs**: search tracks by query
- **PlaySong**: start playback of a track
- **ShowPlaylists**: list user playlists
- **ShowRecentlyPlayed**: list recent listens

### SimpleNote App
- **Login**: authenticate and start a session
- **ShowNotes**: list notes, optionally by tag
- **ShowNote**: open a specific note
- **CreateNote**: create a new note

### LLM App
- **Chat**: ask a question or request advice

### Google APP
- **Search**: web search for information

### Fitness APP
- **Login**: authenticate and start a session
- **LogWorkout**: record a workout
- **ShowDailyStats**: fetch daily fitness summary

### Calendar APP
- **CreateEvent**: create a calendar event
- **UpdateEvent**: modify an existing event
- **DeleteEvent**: remove an event
- **ShowEvents**: list events in a time window
- **ShowEvent**: show details for a specific event
- **RespondToInvite**: respond to an event invitation

### Message APP
- **SendMessage**: send a message to a user
- **SearchMessages**: search message history
- **GetMessages**: fetch messages in a thread
- **CreateGroup**: create a group chat

---

## Event Types and App Usage

Each event in your event chain must specify:
1. **app_name**: Which app is used (Amazon, Spotify, SimpleNote, LLM, Google, Fitness, Calendar, Message)
2. **api_name**: Which specific API is called
3. **user_intent**: Why this event occurs (user intent tied to attributes/habits/preferences)
4. **time_specification**: Explicit schedule_dates with time or start_time/end_time

---

## Time Specification (IMPORTANT)

Time is represented ONLY by explicit schedule dates (no time ranges or cadence).

### A) Single occurrence
Use when the event happens once:
- Required fields:
  - schedule_dates: ["YYYY-MM-DD"] (single date)
  - time: "HH:MM:SS"
- Do NOT include start_time/end_time.

### B) Repeated occurrences (habits)
Use when one JSON event represents many similar occurrences across a window.
- schedule_dates must list every concrete date (no cadence/days_of_week)
- schedule_dates must fall within the top-level window time_range
- Use habit schedule/timing from domain_window_state when available

How to align with habits_state.current_value:
- If schedule_dates exists, set:
  - schedule_dates = the full list
  - start_time = timing.start_time
  - end_time = timing.end_time
- If schedule_dates is missing, expand schedule.frequency_type + days_of_week into explicit dates within the window.

Required/Recommended fields:
- schedule_dates: ["YYYY-MM-DD", "YYYY-MM-DD", ...]
- start_time: "HH:MM:SS"
- end_time: "HH:MM:SS"
- note (optional): explain what repeats and what each occurrence logs

---

## Event Generation Guidelines

### 1. Generate Events Based on User State

For each element in the user's state (attributes, habits, preferences), generate realistic events:

**For User Attributes:**
- If an attribute was added/modified: generate events showing the acquisition or change process
- If an attribute is stable: generate events showing ongoing usage/interaction

**For Habits:**
- If habit operation is acquire/adjust: MUST include:
  1) 1–2 events showing how the user obtains/sets up the habit (research/decision/setup)
  2) If change_reason is provided, include at least one event that makes that reason observable
  3) Then show ongoing daily/weekly usage as either:
     - multiple single-date events, OR
     - ONE event that lists all schedule_dates across the window (preferred for daily habits)
- If habit operation is stable: generate recurring executions matching frequency, using either multiple single-date events or one event with schedule_dates.

**For Preferences:**
- Generate events that demonstrate the user's choices and decisions
- Show preference through actual behaviors (what they choose, not just what they say)

### 2. Dependency Constraints
- Keep dependencies realistic (search → show → purchase, search → play, login → actions).

---

## Output Format (JSON ONLY)

Return ONLY a valid JSON object with this structure:

{
  "window_id": "copy from domain_window_state",
  "domain_name": "{{ domain_name }}",
  "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "event_chains": [
    {
      "chain_id": "unique_identifier",
      "related_state_items": [
        {
          "state_category": "user_attributes_state | habits_state | preferences_state",
          "state_name": "exact_name_from_state",
          "operation": "add | modify | acquire | adjust | drop | shift | refine | stable"
        }
      ],
      "events": [
        {
          "event_id": "unique_identifier",
          "time_specification": {
            "schedule_dates": ["YYYY-MM-DD", "YYYY-MM-DD"],
            "time": "HH:MM:SS",
            "start_time": "HH:MM:SS",
            "end_time": "HH:MM:SS",
            "note": "string"
          },
          "app_name": "Amazon | Spotify | SimpleNote | LLM | Google | Fitness | Calendar | Message",
          "api_name": "Login | SearchProducts | Chat | SearchSongs | ShowNotes | ShowNote | CreateNote | Search | etc.",
          "user_intent": "Why this event occurs (attribute acquisition, habit execution, preference demonstration, etc.)"
        }
      ]
    }
  ]
}

### time_specification rules
- Use schedule_dates only.
- For single events: schedule_dates length = 1 and include time.
- For repeated events: schedule_dates list all dates and include start_time/end_time.
- All schedule_dates must fall within the top-level time_range window.

---

## Generation Instructions

### 1. Coverage
- Cover ALL state items: every attribute, habit, and preference in the current window must be represented in at least one chain
- One item per chain by default: only group 2-3 items if they have direct causal relationship

### 2. Event Generation Strategy
**For Attributes (add/modify):**
Generate 2-4 events showing: research/discovery → decision → acquisition → usage

**For Habits (acquire/adjust):**
Generate 1-2 events for acquisition/setup, then:
- daily/weekly executions via one event with schedule_dates listing all occurrences

**For Habits (stable):**
Recurring execution events matching frequency (multiple single events OR one event with schedule_dates)

**For Preferences (stable):**
Generate 2-3 events demonstrating the preference through behavior

**For Preferences (shift/refine):**
Generate 1-2 events showing the preference shift/refinement process and generate 2-3 events demonstrating the preference through behavior

### 3. Realistic Timing
- Spread events across the time window
- Avoid clustering everything on one day (except truly urgent actions)
- Respect habit schedules (morning ~07:00, evening ~20:00, etc.)
- Use realistic timestamps

---

## Examples (VALID JSON SNIPPETS)

### Example 1: Habit acquire → then daily executions via schedule_dates
{
  "chain_id": "w1_health_001",
  "related_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "morning_workout",
      "operation": "acquire"
    }
  ],
  "events": [
    {
      "event_id": "w1_health_001_e001",
      "time_specification": {
        "schedule_dates": ["2024-01-02"],
        "time": "21:10:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "research and planning before starting the habit."
    },
    {
      "event_id": "w1_health_001_e002",
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "20:45:00"
      },
      "app_name": "Calendar",
      "api_name": "CreateEvent",
      "user_intent": "Habit acquisition: setup and commitment via calendar reminder."
    },
    {
      "event_id": "w1_health_001_e003",
      "time_specification": {
        "schedule_dates": [
          "2024-01-04",
          "2024-01-05",
          "2024-01-06",
          "2024-01-07",
          "2024-01-08",
          "2024-01-09",
          "2024-01-10"
        ],
        "start_time": "07:00:00",
        "end_time": "07:20:00",
        "note": "Each morning the user completes a short workout and logs it in Fitness.LogWorkout."
      },
      "app_name": "Fitness",
      "api_name": "LogWorkout",
      "user_intent": "Habit execution: recurring daily behavior and consistent API logging."
    }
  ]
}

### Example 2: Attribute add (new device) → purchase flow with single-date events
{
  "chain_id": "w1_shopping_002",
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "owns_noise_canceling_headphones",
      "operation": "add"
    }
  ],
  "events": [
    {
      "event_id": "w1_shopping_002_e001",
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "12:30:10"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Attribute acquisition: research and decision support."
    },
    {
      "event_id": "w1_shopping_002_e002",
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "12:45:40"
      },
      "app_name": "Amazon",
      "api_name": "SearchProducts",
      "user_intent": "Attribute acquisition: product discovery with constraints."
    },
    {
      "event_id": "w1_shopping_002_e003",
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "12:48:05"
      },
      "app_name": "Amazon",
      "api_name": "ShowProduct",
      "user_intent": "Attribute acquisition: evaluation before purchase."
    },
    {
      "event_id": "w1_shopping_002_e004",
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "12:55:22"
      },
      "app_name": "Amazon",
      "api_name": "Checkout",
      "user_intent": "Attribute acquisition: completing ownership change via transaction."
    }
  ]
}

### Example 3: Preference stable → repeated behavior captured as schedule_dates
{
  "chain_id": "w1_focusmusic_003",
  "related_state_items": [
    {
      "state_category": "preferences_state",
      "state_name": "prefers_lofi_for_deep_work",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "event_id": "w1_focusmusic_003_e001",
      "time_specification": {
        "schedule_dates": ["2024-03-11"],
        "time": "09:05:00"
      },
      "app_name": "Spotify",
      "api_name": "SearchSongs",
      "user_intent": "Preference demonstration: selecting content consistent with preference."
    },
    {
      "event_id": "w1_focusmusic_003_e002",
      "time_specification": {
        "schedule_dates": [
          "2024-03-11",
          "2024-03-12",
          "2024-03-13",
          "2024-03-14",
          "2024-03-15"
        ],
        "start_time": "09:10:00",
        "end_time": "11:00:00",
        "note": "During weekday work sessions, the user plays lofi tracks while working."
      },
      "app_name": "Spotify",
      "api_name": "PlaySong",
      "user_intent": "Preference demonstration: repeated selection and playback behavior."
    }
  ]
}

---

## Critical Rules

1) Return ONLY valid JSON (no markdown, no comments, no extra text)
2) Cover all state items in the current window
3) Specify app_name and api_name for every event
4) Do NOT include an event-level "description" field; use user_intent instead
5) Use time_specification with schedule_dates for every event (no time_range/cadence)
6) All schedule_dates must be within the window time_range
7) Create narrative coherence (logical story per chain)
8) Demonstrate preferences through behavior, not just statements
""")

@dataclass
class EventsChainRequest:
    user_basic_profile: str
    domain_name: str
    domain_window_state: str
    user_life_context: str
    user_previous_window_summary: str
    user_domain_previous_window_summary: str
    user_this_window_description: str
    world_background: str


def render_events_chain_prompt(request: EventsChainRequest) -> str:
    return events_chain_template.render(
        user_basic_profile=request.user_basic_profile,
        domain_name=request.domain_name,
        user_life_context=request.user_life_context,
        user_previous_window_summary=request.user_previous_window_summary,
        user_domain_previous_window_summary=request.user_domain_previous_window_summary,
        user_this_window_description=request.user_this_window_description,
        world_background=request.world_background,
        domain_window_state=request.domain_window_state,
    )


def generate_events_chain(
    llm_client: GeminiJSONClient, request: EventsChainRequest
) -> LLMResult:
    prompt = render_events_chain_prompt(request)
    return llm_client.generate_json(prompt)

# OLD TEMPLATE CONTENT BELOW - TO BE REMOVED
# """

# ## Example 1: Two Related Items Grouped Together (Valid Grouping)

# State items to be grouped:
# ```json
# [
#   {
#     "name": "user_health_tracking_devices",
#     "current_value": "Whoop 4.0",
#     "op": "add",
#     "change_reason": "Influenced by CES 2024 tech trends"
#   },
#   {
#     "name": "wellness_philosophy", 
#     "current_value": "Strongly prefers highly granular, AI-enhanced data metrics",
#     "op": "amplify",
#     "change_reason": "Adoption of the Whoop strap increases reliance on algorithmic health insights"
#   }
# ]
# ```

# Why grouped: These 2 items are directly causally connected - acquiring Whoop (attribute) directly amplifies the data-driven preference (preference). This is a valid grouping.

# Evidence chain:
# ```json
# {
#   "evidence_chain_id": "w1_health_group_001",
#   "proving_state_items": [
#     {
#       "state_category": "user_attributes_state",
#       "state_name": "user_health_tracking_devices",
#       "state_value": "Whoop 4.0 (screenless wearable for recovery tracking)",
#       "operation": "add",
#       "change_reason": "Influenced by CES 2024 tech trends"
#     },
#     {
#       "state_category": "preferences_state",
#       "state_name": "wellness_philosophy",
#       "state_value": "Strongly prefers highly granular, AI-enhanced data metrics",
#       "operation": "amplify",
#       "change_reason": "Adoption of the Whoop strap increases reliance on algorithmic health insights"
#     }
#   ],
#   "evidence_story": "User's interest in Whoop began from watching CES coverage (app log), but the decision to purchase was driven by detailed discussion with AI about algorithmic recovery insights (dialogue). The dialogue reveals user's preference for data-driven approaches, which explains why they chose Whoop over simpler alternatives. Subsequent app logs show daily engagement with Whoop's granular metrics, while dialogue discussions about interpreting the data demonstrate deepening reliance on algorithmic guidance.",
#   "multi_source_dependency_rationale": "Without dialogue, we'd only see that user bought and uses Whoop, but not why (preference for AI-enhanced metrics over simple tracking). Without app logs, we'd only know user is interested in data tracking, but not that they actually acquired the device or use it consistently.",
#   "evidence_sequence": [
#     {
#       "evidence_id": "w1_health_group_001_ev001",
#       "behavior_type": "information_seeking",
#       "data_source": "app_log",
#       "time_specification": {
#         "type": "specific_time",
#         "specific_time": "2024-01-20 22:15:33"
#       },
#       "description": "User watches 'CES 2024: Best Health Wearables' video (18 min) on YouTube from The Verge, pauses and replays the Whoop 4.0 segment multiple times"
#     },
#     {
#       "evidence_id": "w1_health_group_001_ev002",
#       "behavior_type": "decision_making",
#       "data_source": "dialogue",
#       "time_specification": {
#         "type": "specific_time",
#         "specific_time": "2024-01-22 20:31:09"
#       },
#       "description": "User asks AI to compare Whoop 4.0 vs Apple Watch Series 8 specifically for recovery algorithms and data granularity. User explains they already have Apple Watch but want 'more scientific, AI-driven insights into when to push hard vs rest'. Conversation reveals preference for algorithmic guidance over simple metrics."
#     },
#     {
#       "evidence_id": "w1_health_group_001_ev003",
#       "behavior_type": "transactional",
#       "data_source": "app_log",
#       "time_specification": {
#         "type": "specific_time",
#         "specific_time": "2024-01-25 19:47:33"
#       },
#       "description": "User subscribes to Whoop annual membership ($239) and device ships. Transaction made within days of AI conversation, selecting annual plan (showing commitment)."
#     },
#     {
#       "evidence_id": "w1_health_group_001_ev004",
#       "behavior_type": "usage_tracking",
#       "data_source": "app_log",
#       "time_specification": {
#         "type": "time_range",
#         "time_range": {
#           "start": "2024-02-01",
#           "end": "2024-03-31",
#           "frequency": "daily"
#         }
#       },
#       "description": "User checks Whoop app every morning (7:15-7:30 AM) to view recovery score, HRV, and strain recommendations before deciding workout intensity. Average session duration 3-5 minutes, views detailed metrics breakdown."
#     },
#     {
#       "evidence_id": "w1_health_group_001_ev005",
#       "behavior_type": "self_reporting",
#       "data_source": "dialogue",
#       "time_specification": {
#         "type": "specific_time",
#         "specific_time": "2024-02-28 19:22:15"
#       },
#       "description": "User discusses with AI how they now rely on Whoop's recovery algorithm to decide whether to climb hard or take rest day. Mentions 'trusting the data more than my own feeling' and asks AI to explain the HRV science behind recommendations."
#     }
#   ]
# }
# ```

# ---

# ## Example 2: Independent Habit (Stable)

# State item:
# ```json
# {
#   "name": "morning_hydration_routine",
#   "current_value": {
#     "action": "drink_water_and_electrolyte_mix",
#     "frequency": "daily",
#     "timing": "immediately after waking (7:00 AM - 7:15 AM)"
#   },
#   "op": "stable"
# }
# ```

# Evidence chain:
# ```json
# {
#   "evidence_chain_id": "w1_health_habit_002",
#   "proving_state_items": [
#     {
#       "state_category": "habits_state",
#       "state_name": "morning_hydration_routine",
#       "state_value": {
#         "action": "drink_water_and_electrolyte_mix",
#         "frequency": "daily",
#         "timing": "immediately after waking (7:00 AM - 7:15 AM)"
#       },
#       "operation": "stable",
#       "change_reason": null
#     }
#   ],
#   "evidence_story": "User maintains consistent morning hydration habit tracked via a hydration app (app log), but the specific choice of adding electrolytes (not just plain water) was based on AI conversation about optimal hydration for climbing training. App logs show daily execution, dialogue reveals the reasoning behind the specific method.",
#   "multi_source_dependency_rationale": "Without app logs, we'd know user believes in electrolyte hydration but not that they actually do it daily. Without dialogue, we'd see hydration logs but not understand why they specifically use electrolyte mix rather than plain water.",
#   "evidence_sequence": [
#     {
#       "evidence_id": "w1_health_habit_002_ev001",
#       "behavior_type": "information_seeking",
#       "data_source": "dialogue",
#       "time_specification": {
#         "type": "specific_time",
#         "specific_time": "2024-01-08 20:45:11"
#       },
#       "description": "User asks AI whether adding electrolytes to morning water helps with climbing performance and recovery. AI explains benefits of sodium/potassium for athletes, user asks about simple implementation (pinch of sea salt vs commercial products)."
#     },
#     {
#       "evidence_id": "w1_health_habit_002_ev002",
#       "behavior_type": "usage_tracking",
#       "data_source": "app_log",
#       "time_specification": {
#         "type": "time_range",
#         "time_range": {
#           "start": "2024-01-10",
#           "end": "2024-03-31",
#           "frequency": "daily"
#         }
#       },
#       "description": "User logs water intake in WaterMinder app every morning at 7:05-7:15 AM, consistently noting '500ml water + sea salt' or '500ml water + electrolyte mix'. Over 80% completion rate across the period."
#     }
#   ]
# }
# ```

# ---

# ## Example 3: Preference Revealed Through Decision

# State item:
# ```json
# {
#   "name": "exercise_environment",
#   "current_value": "Prefers indoor, climate-controlled environments for exercise to maintain consistency regardless of weather",
#   "op": "stable"
# }
# ```

# Evidence chain:
# ```json
# {
#   "evidence_chain_id": "w1_health_pref_001",
#   "proving_state_items": [
#     {
#       "state_category": "preferences_state",
#       "state_name": "exercise_environment",
#       "state_value": "Prefers indoor, climate-controlled environments for exercise to maintain consistency regardless of weather",
#       "operation": "stable",
#       "change_reason": null
#     }
#   ],
#   "evidence_story": "User's preference for indoor exercise is revealed through consistent gym check-ins even on mild weather days (app logs) and a conversation with AI where user explicitly explains choosing indoor climbing because outdoor weather in Chicago is unreliable and they prioritize training consistency.",
#   "multi_source_dependency_rationale": "Without app logs, we'd only have user's stated preference but no behavioral proof. Without dialogue, we'd see indoor gym attendance but couldn't distinguish between preference vs lack of outdoor options - the dialogue reveals this is a deliberate choice for consistency, not just convenience.",
#   "evidence_sequence": [
#     {
#       "evidence_id": "w1_health_pref_001_ev001",
#       "behavior_type": "usage_tracking",
#       "data_source": "app_log",
#       "time_specification": {
#         "type": "time_range",
#         "time_range": {
#           "start": "2024-01-01",
#           "end": "2024-03-31",
#           "frequency": "weekly"
#         }
#       },
#     },
#     {
#       "evidence_id": "w1_health_pref_001_ev002",
#       "behavior_type": "decision_making",
#       "data_source": "dialogue",
#       "time_specification": {
#         "type": "specific_time",
#         "specific_time": "2024-03-15 18:30:44"
#       },
#       "description": "Friend mentions trying outdoor climbing spot, user discusses with AI why they prefer indoor climbing. User explains: 'Chicago weather is too unpredictable, I need consistent training conditions to progress. Indoor gym means I never skip sessions due to rain or cold.' Reveals preference is about training reliability, not comfort."
#     }
#   ]
# }
# ```

# ## Example 4: INVALID Grouping Examples

# ### Example 4A: Temporal Proximity (WRONG)

# **DON'T DO THIS:**
# ```json
# {
#   "proving_state_items": [
#     {"state_name": "morning_hydration_routine"},    // Happens at 7:00 AM
#     {"state_name": "sleep_tracking_review"}         // Happens at 7:15 AM
#   ],
#   "evidence_story": "User does both in the morning routine..."
# }
# ```

# **Why this is WRONG:**
# - These are two independent habits that just happen sequentially
# - No causal relationship: hydrating doesn't cause sleep tracking review
# - Evidence for hydration (drinking water log) is completely separate from evidence for sleep review (checking Whoop app)
# - Just because they happen in the same 30-minute window doesn't make them one event

# **CORRECT APPROACH:**
# - Chain 1: morning_hydration_routine (1 item) - Prove with water intake logs + dialogue about electrolyte choice
# - Chain 2: sleep_tracking_review (1 item) - Prove with app usage logs + dialogue about sleep optimization

# ### Example 4B: Thematic Similarity (WRONG)

# **DON'T DO THIS:**
# ```json
# {
#   "proving_state_items": [
#     {"state_name": "user_dietary_supplements"},     // Vitamin D
#     {"state_name": "user_health_tracking_devices"}, // Whoop
#     {"state_name": "morning_mindfulness_meditation"} // Meditation
#   ],
#   "evidence_story": "User adopted multiple wellness practices in Q1..."
# }
# ```

# **Why this is WRONG:**
# - All three are "wellness" related but completely independent
# - Buying Vitamin D has no causal relationship with buying Whoop or starting meditation
# - Each has its own acquisition story, evidence, and timeline
# - "All happened in Q1" is not a valid grouping criterion

# **CORRECT APPROACH:**
# - Chain 1: user_dietary_supplements (1 item)
# - Chain 2: user_health_tracking_devices (1 item) 
# - Chain 3: morning_mindfulness_meditation (1 item)

# ### Example 4C: Valid Grouping (CORRECT)

# **DO THIS:**
# ```json
# {
#   "proving_state_items": [
#     {
#       "state_name": "user_health_tracking_devices",  // Whoop device
#       "operation": "add"
#     },
#     {
#       "state_name": "wellness_philosophy",            // Data-driven preference
#       "operation": "amplify"
#     }
#   ]
# }
# ```

# **Why this is CORRECT:**
# - Direct causal relationship: acquiring Whoop amplifies data-driven preference
# - change_reason explicitly links them: "Adoption of the Whoop strap increases reliance on algorithmic health insights"
# - Evidence naturally overlaps: discussions about wanting better data → purchase → using data features
# - Cannot be fully separated: the preference explains the purchase, the purchase enables the preference


# ---

# ## Critical Rules

# 1. **MANDATORY: Cover all state items** - Each item must appear in exactly one evidence chain
# 2. **DEFAULT: One state item per chain** - Only group if ALL three criteria are met (causal + inseparable + shared story)
# 3. **FORBIDDEN: Grouping by temporal proximity, theme, or domain** - These are invalid reasons
# 4. **MANDATORY: Every evidence chain MUST include BOTH app_log AND dialogue sources**
# 5. **MANDATORY: Multi-source dependency** - Neither source alone should be sufficient to prove the state
# 6. **Minimal evidence only** - No redundant or repetitive evidence pieces
# 7. **Maximize chain count** - More chains (with fewer items each) is better than fewer chains (with many items)
# 8. **Reveal preferences through decisions** - Show choices, not just statements
# 9. **Use time_range for recurring patterns** - Don't list many individual instances
# 10. **All times must fall within window's time_range**
# 11. **Return ONLY valid JSON** - No comments or extra text

# **Self-check before grouping items:**
# - [ ] Does Item A directly cause Item B? (not just "happen before")
# - [ ] Is the evidence for A inherently overlapping with evidence for B?
# - [ ] Are they part of ONE acquisition/change event, not just the same time period?
# - If any answer is "no", keep items separate.

# """)
@dataclass
class EventsChainRequest:
    user_basic_profile: str
    domain_name: str
    domain_window_state: str
    user_life_context: str
    user_previous_window_summary: str
    user_domain_previous_window_summary: str
    user_this_window_description: str
    world_background: str


def render_events_chain_prompt(request: EventsChainRequest) -> str:
    return events_chain_template.render(
        user_basic_profile=request.user_basic_profile,
        domain_name=request.domain_name,
        user_life_context=request.user_life_context,
        user_previous_window_summary=request.user_previous_window_summary,
        user_domain_previous_window_summary=request.user_domain_previous_window_summary,
        user_this_window_description=request.user_this_window_description,
        world_background=request.world_background,
        domain_window_state=request.domain_window_state,
    )


def generate_events_chain(
    llm_client: GeminiJSONClient, request: EventsChainRequest
) -> LLMResult:
    prompt = render_events_chain_prompt(request)
    return llm_client.generate_json(prompt)
