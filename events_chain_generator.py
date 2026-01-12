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

events_chain_template = Template("""# Event Chain Generation Prompt

## 1. Overview & Objective

You are an expert at generating realistic event chains that demonstrate user behaviors based on their dynamic profile state.

### Your Task
Given a user's state for a specific time window, generate a sequence of realistic events that would naturally occur based on their attributes, habits, and preferences. Each event must specify which app and API it uses, along with the user intent.

### What Makes a Good Event Chain
- **Observable**: Every state item is demonstrated through concrete app interactions
- **Realistic**: Events follow natural human behavior patterns and realistic timing
- **Diverse**: Avoid repetitive patterns; users take different paths to similar outcomes
- **Coherent**: Events within a chain tell a logical story
- **Complete**: Change reasons are made observable through events; both the process and the result are shown

---

## 2. Core Concepts

### 2.1 Observable State

State items must be made observable through events:

**For Stable State:**
- Generate events showing ongoing usage, execution, or demonstration of the state
- Example: `habit: morning_workout (stable)` → recurring workout logging events

**For Changed State (add/modify/acquire/adjust/refine/shift):**
- Generate events showing HOW the change happened (process events)
- Generate events showing the change result (usage events)
- If `change_reason` exists, include events that make the reason observable
- Example: `attribute: owns_headphones (add)` → research events + purchase event + usage events

### 2.2 Event Chain Organization

**Default Rule: One state item → One event chain**

**Exception: Multiple state items can share one chain when:**
1. They naturally interweave in actual usage (e.g., a habit execution demonstrates both an attribute and a preference)
2. Maximum 2 state items per merged chain
3. Merging scenarios:
   - Preference demonstrated during attribute usage
   - Preference demonstrated during habit execution  
   - Habit execution naturally involves an attribute

**Note:** Multiple state items with the same `change_reason` do NOT automatically merge. They may demonstrate the reason through different independent events.

### 2.3 Time Distribution Principles

- **Spread events across the time window** to avoid unrealistic clustering
- **Respect temporal dependencies:**
  - Acquisition/setup events should occur before usage events
  - If an attribute is added early in the window, related habit acquisition can follow later
  - Change reason events should occur near the timing of the state change
- **Use realistic timestamps:**
  - Work activities during work hours (09:00-18:00)
  - Personal activities in evenings/weekends
  - Respect habit schedules from state data

### 2.4 Behavior Diversity Principles

Real users exhibit diverse behavior patterns. Avoid generating the same "search → view → purchase" pattern for every acquisition.

**Key Principle:** Different users (and even the same user at different times) take different paths to achieve similar outcomes.

Diversity sources:
- **Information gathering intensity:** quick vs. thorough research
- **Decision timeline:** impulsive vs. deliberate
- **Social validation:** consulting others vs. independent decision
- **Trial/experimentation:** trying before committing vs. direct purchase
- **Problem awareness:** reactive (solving a problem) vs. proactive (seeking improvement)

*Detailed patterns and examples are provided in Section 6: Generation Strategy.*

---

## 3. Input Schema

### 3.1 Domain Window State Structure

```json
{
  "window_id": "w1",
  "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "state_table": {
    "user_attributes_state": [...],
    "habits_state": [...],
    "preferences_state": [...]
  }
}
```

### 3.2 How to Read State Items

#### User Attributes State
```json
{
  "name": "attribute_name",
  "current_value": "description",
  "op": "add | modify | stable",
  "change_reason": "why this changed (if op is add/modify)",
  "previous_value": "old value (if op is modify)"
}
```

#### Habits State
```json
{
  "name": "habit_name",
  "current_value": {
    "schedule": {
      "frequency_type": "daily | weekly | biweekly",
      "days_of_week": [0-6],
      "start_date": "YYYY-MM-DD"
    },
    "timing": {
      "start_time": "HH:MM",
      "end_time": "HH:MM"
    },
    "location": "where it happens",
    "priority": "high | medium | low",
    "schedule_dates": ["YYYY-MM-DD", ...] // explicit dates when habit occurs
  },
  "op": "acquire | adjust | stable",
  "change_reason": "why this changed (if op is acquire/adjust)",
  "previous_value": {...}
}
```

**Important for Habits:**
- If `schedule_dates` exists, use it directly for event timing
- If missing, expand `frequency_type + days_of_week` into explicit dates within the window
- Use `timing.start_time` and `timing.end_time` for event time specifications

#### Preferences State
```json
{
  "name": "preference_name",
  "current_value": {
    "statement": "description of the preference"
  },
  "op": "refine | shift | stable",
  "change_reason": "why this changed (if op is refine/shift)",
  "previous_value": {...}
}
```

**Note:** The `signals` field has been removed. Generate observable behaviors based solely on the `statement` and available apps/APIs.

---

## 4. Output Schema

### 4.1 Event Chain JSON Structure

```json
{
  "window_id": "copy from input",
  "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "event_chains": [
    {
      "related_state_items": [
        {
          "state_category": "user_attributes_state | habits_state | preferences_state",
          "state_name": "exact_name_from_state",
          "operation": "add | modify | acquire | adjust | drop | shift | refine | stable"
        }
      ],
      "events": [
        {
          "time_specification": {
            "schedule_dates": ["YYYY-MM-DD", ...],
            "time": "HH:MM:SS",           // for single occurrence
            "start_time": "HH:MM:SS",     // for repeated occurrences
            "end_time": "HH:MM:SS",       // for repeated occurrences
            "note": "optional explanation for repeated events"
          },
          "app_name": "Amazon | Spotify | SimpleNote | LLM | Google | Fitness | Calendar | Message | Finance",
          "api_name": "specific API name from Section 5",
          "user_intent": "why this event occurs, tied to state item operation"
        }
      ]
    }
  ]
}
```

### 4.2 Time Specification Rules

**Rule 1: Use `schedule_dates` for all events (no time ranges or cadence strings)**

**Rule 2: For single occurrence events:**
```json
{
  "schedule_dates": ["2024-01-15"],
  "time": "14:30:00"
}
```
- `schedule_dates` contains exactly one date
- Include `time` field
- Do NOT include `start_time` or `end_time`

**Rule 3: For repeated occurrence events (habits):**
```json
{
  "schedule_dates": ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"],
  "start_time": "07:00:00",
  "end_time": "07:30:00",
  "note": "Each Saturday morning the user completes a workout and logs it."
}
```
- `schedule_dates` lists ALL dates when the event occurs
- Include `start_time` and `end_time` 
- Optionally include `note` to explain the repetition pattern
- Do NOT include single-event `time` field

**Rule 4: All `schedule_dates` must fall within the window `time_range`**

**Rule 5: For habits, align with state data:**
- If `habits_state.current_value.schedule_dates` exists, use those dates
- Use `timing.start_time` and `timing.end_time` from habit state

---

## 5. Available Apps & APIs

Each event must specify `app_name` and `api_name`. Below are the available apps and their APIs.

### 5.1 Amazon App (Shopping & Product Research)

**SearchProducts**
- Search product catalog with filters and sorting
- Typical use: Initial exploration, finding options, comparison shopping
- Behavior types: information_seeking, decision_making

**ShowProduct**  
- View detailed information for a specific product
- Typical use: Evaluation, reading specifications, checking compatibility
- Behavior types: information_seeking, decision_making

**ReadReviews**
- Browse customer reviews and ratings for a product
- Typical use: Social validation, quality assessment, learning from others' experiences
- Behavior types: information_seeking, decision_making

**AddToCart**
- Add item to shopping cart (not yet purchased)
- Typical use: Saving items for later, building a cart before checkout, comparing total prices
- Behavior types: decision_making, transactional

**Checkout**
- Complete purchase transaction
- Typical use: Final acquisition, converting intent to ownership
- Behavior types: transactional

**ShowOrders**
- View order history and current order statuses
- Typical use: Tracking purchases, reviewing past decisions, monitoring deliveries
- Behavior types: monitoring

**TrackOrder**
- Check delivery status for a specific order
- Typical use: Anticipating arrival, ensuring timely delivery
- Behavior types: monitoring

### 5.2 Spotify App (Music & Audio)

**SearchSongs**
- Search for tracks, artists, or albums
- Typical use: Discovering new music, finding specific songs, exploring genres
- Behavior types: information_seeking

**ShowArtist**
- View artist profile and discography
- Typical use: Learning about artist, exploring their catalog
- Behavior types: information_seeking

**PlaySong**
- Start playback of a track or playlist
- Typical use: Listening, background music, focused work, relaxation
- Behavior types: consumption

**CreatePlaylist**
- Create a new playlist
- Typical use: Organizing music by mood/activity, curating collections
- Behavior types: creation, maintenance

**AddToPlaylist**
- Add track to an existing playlist
- Typical use: Building collections, saving favorites
- Behavior types: maintenance

**ShowPlaylists**
- List user's playlists
- Typical use: Choosing what to listen to, reviewing collections
- Behavior types: monitoring

**ShowRecentlyPlayed**
- View listening history
- Typical use: Re-finding songs, tracking listening habits
- Behavior types: monitoring

### 5.3 SimpleNote App (Note-taking & Documentation)

**ShowNotes**
- List notes with optional filtering by tags
- Typical use: Finding notes, reviewing notes, organizing thoughts
- Behavior types: monitoring, maintenance

**ShowNote**
- Open and read a specific note
- Typical use: Reviewing information, referencing past notes
- Behavior types: consumption

**CreateNote**
- Create a new note
- Typical use: Capturing ideas, documenting information, planning
- Behavior types: creation

**EditNote**
- Modify existing note content
- Typical use: Updating information, refining ideas, adding details
- Behavior types: maintenance

**SearchNotes**
- Search note content by keywords
- Typical use: Finding specific information, retrieving past notes
- Behavior types: information_seeking

**TagNote**
- Add or modify tags on a note
- Typical use: Organizing notes, categorizing information
- Behavior types: maintenance

### 5.4 LLM App (AI Assistant)

**Chat**
- Ask questions, request advice, generate content, solve problems
- Typical use: Research, decision support, learning, brainstorming, code generation, writing assistance
- Behavior types: information_seeking, decision_making, creation

### 5.5 Google App (Web Search & Research)

**Search**
- Web search for general information
- Typical use: Finding information, researching topics, fact-checking
- Behavior types: information_seeking

**SearchNews**
- Search for recent news articles
- Typical use: Staying informed, tracking current events, researching recent developments
- Behavior types: information_seeking

### 5.6 Fitness App (Health & Exercise Tracking)

**LogWorkout**
- Record a completed workout session
- Typical use: Tracking exercise, maintaining workout history, building consistency
- Behavior types: self_reporting, usage_tracking

**LogActivity**
- Record daily activity metrics (steps, calories, etc.)
- Typical use: Tracking daily movement, monitoring activity levels
- Behavior types: self_reporting, usage_tracking

**ShowDailyStats**
- View today's fitness summary
- Typical use: Checking progress, monitoring activity, staying motivated
- Behavior types: monitoring

**ShowWeeklyStats**
- View weekly fitness trends and summaries
- Typical use: Reviewing progress, identifying patterns, adjusting goals
- Behavior types: monitoring

**SetGoal**
- Set fitness or health goals
- Typical use: Defining targets, planning improvement, establishing commitment
- Behavior types: decision_making

**ShowProgress**
- View progress toward goals
- Typical use: Tracking achievement, staying motivated, adjusting approach
- Behavior types: monitoring

### 5.7 Calendar App (Scheduling & Time Management)

**CreateEvent**
- Schedule a new calendar event
- Typical use: Planning activities, blocking time, setting reminders, establishing commitments
- Behavior types: decision_making, maintenance

**ShowEvents**
- View calendar schedule
- Typical use: Checking schedule, planning day, avoiding conflicts
- Behavior types: monitoring

**EditEvent**
- Modify existing calendar event
- Typical use: Rescheduling, updating details, adjusting plans
- Behavior types: maintenance

**SetReminder**
- Add reminder to an event
- Typical use: Ensuring timely action, preventing forgetting, building habits
- Behavior types: maintenance

### 5.8 Message App (Communication & Collaboration)

**SendMessage**
- Send a message to an individual or group
- Typical use: Communication, sharing information, coordination, asking questions
- Behavior types: social_interaction

**GetMessages**
- Fetch and read messages in a conversation thread
- Typical use: Staying informed, following conversations, responding to others
- Behavior types: consumption, social_interaction

**SearchMessages**
- Search message history by keywords
- Typical use: Finding past information, retrieving shared links/files, referencing discussions
- Behavior types: information_seeking

**CreateGroup**
- Create a group chat
- Typical use: Organizing team communication, coordinating activities
- Behavior types: social_interaction, maintenance

**ReactToMessage**
- Add reaction (emoji/like) to a message
- Typical use: Acknowledging receipt, expressing agreement, lightweight feedback
- Behavior types: social_interaction

### 5.9 Finance App (Financial Management & Tracking)

**ShowAccounts**
- View list of financial accounts and balances
- Typical use: Monitoring finances, checking balances, reviewing accounts
- Behavior types: monitoring

**ShowTransactions**
- View transaction history for accounts
- Typical use: Reviewing spending, tracking expenses, finding specific transactions
- Behavior types: monitoring

**CreateBudget**
- Set up a spending budget with categories and limits
- Typical use: Financial planning, controlling spending, establishing limits
- Behavior types: decision_making, maintenance

**LogExpense**
- Manually record an expense
- Typical use: Tracking cash spending, categorizing expenses, maintaining records
- Behavior types: self_reporting, usage_tracking

**ShowBudgetProgress**
- View spending against budget limits
- Typical use: Monitoring spending, staying within budget, adjusting habits
- Behavior types: monitoring

**SetFinancialGoal**
- Define a financial goal (savings, debt reduction, etc.)
- Typical use: Planning financial future, establishing targets, motivating saving
- Behavior types: decision_making

**ShowInvestments**
- View investment portfolio and performance
- Typical use: Monitoring investments, tracking returns, reviewing allocation
- Behavior types: monitoring

**TransferMoney**
- Transfer funds between accounts
- Typical use: Managing cash flow, paying bills, moving money to savings
- Behavior types: transactional

---

## 6. Generation Strategy

### 6.1 Strategy Overview Table

| State Category | Operation | Event Pattern | Typical APIs |
|---------------|-----------|---------------|--------------|
| **user_attributes_state** | add | Acquisition: research → decision → acquisition → usage | LLM.Chat, Google.Search, Amazon.SearchProducts, Amazon.Checkout, etc. |
| | modify | Transition: awareness → adjustment → new usage | LLM.Chat, various usage APIs |
| | stable | Ongoing usage | Usage APIs relevant to attribute |
| **habits_state** | acquire | Establishment: exploration → setup → execution (repeated) | LLM.Chat, Calendar.CreateEvent, habit-specific APIs |
| | adjust | Adaptation: reason awareness → modification → adjusted execution | LLM.Chat, Calendar.EditEvent, habit-specific APIs |
| | stable | Execution (repeated) | Habit-specific APIs with schedule_dates |
| **preferences_state** | refine/shift | Evolution: change reason awareness → experimentation → new demonstration | Various choice-making APIs |
| | stable | Demonstration through choices | Various choice-making APIs |

### 6.2 User Attributes Generation

#### 6.2.1 Attributes with `add` Operation

**Goal:** Show HOW the user acquired this attribute + demonstrate its usage

**Base Pattern:** Research → Decision → Acquisition → Usage

**Behavior Diversity - Common Acquisition Paths:**

1. **Quick & Direct Path:**
   - User knows what they want
   - Minimal research, fast decision
   - Example: SearchProducts → ShowProduct → Checkout

2. **Research-Heavy Path:**
   - Thorough comparison and evaluation
   - Multiple information sources
   - Example: LLM.Chat (research) → Google.Search → Amazon.SearchProducts → Amazon.ReadReviews → Amazon.ShowProduct (multiple products) → Checkout

3. **Social Validation Path:**
   - Consulting others or reading reviews extensively
   - Example: Message.SendMessage (asking friends) → Amazon.ReadReviews → Amazon.ShowProduct → Checkout

4. **Trial-First Path:**
   - User tries before full commitment (where applicable)
   - Example: Spotify.SearchSongs (trial song) → Spotify.PlaySong → LLM.Chat (evaluate) → [subsequent acquisition]

5. **Problem-Driven Path:**
   - Reactive acquisition due to specific need
   - Example: Google.Search (problem symptoms) → LLM.Chat (solutions) → Amazon.SearchProducts → Checkout

6. **Gradual/Delayed Path:**
   - User gathers information over time before deciding
   - Example: Amazon.SearchProducts (Day 1) → Amazon.ShowProduct (Day 3) → Amazon.ReadReviews (Day 5) → Amazon.AddToCart (Day 7) → Checkout (Day 10)

**Generation Guidelines:**
- **MUST include change_reason demonstration:** If `change_reason` exists, include 1-2 events that make this reason observable
- **Vary your paths:** Don't always use research-heavy; match path to user personality and attribute type
- **Realistic timing:** Spread acquisition events appropriately (quick decisions in hours, deliberate ones over days/weeks)
- **Include post-acquisition usage:** After acquisition, show 2-3 usage events demonstrating the attribute in action

**Example:**
```json
{
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "technical_skills_inventory",
      "operation": "add"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "21:15:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Change reason awareness: researching AI tools for productivity gains in C++ development"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-05"],
        "time": "10:30:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Decision support: evaluating GitHub Copilot capabilities for legacy codebase"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "14:20:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Initial usage: using ChatGPT to generate unit tests for firmware module"
    }
  ]
}
```

#### 6.2.2 Attributes with `modify` Operation

**Goal:** Show the transition from old to new value + demonstrate new usage

**Base Pattern:** Awareness of need to change → Adjustment process → New usage

**Generation Guidelines:**
- Show events demonstrating the OLD value (if still present early in window)
- Show transition/adjustment events
- Show events demonstrating the NEW value
- Include change_reason demonstration if provided

#### 6.2.3 Attributes with `stable` Operation

**Goal:** Demonstrate ongoing usage/possession

**Base Pattern:** Regular usage events spread across the window

**Generation Guidelines:**
- Generate 3-5 usage events showing the attribute in action
- Vary the contexts/scenarios where the attribute is relevant
- Spread across the time window

### 6.3 Habits Generation

#### 6.3.1 Habits with `acquire` Operation

**Goal:** Show HOW the habit was established + demonstrate recurring execution

**Base Pattern:** Exploration/Research → Decision/Setup → Recurring Execution

**Behavior Diversity - Common Establishment Paths:**

1. **Planned & Prepared:**
   - User researches habit formation strategies
   - Creates calendar events and reminders
   - Example: LLM.Chat (habit research) → Calendar.CreateEvent → Calendar.SetReminder → [executions]

2. **Inspiration-Driven:**
   - User gets inspired and starts immediately
   - Minimal planning
   - Example: Google.SearchNews (inspiring article) → Fitness.SetGoal → [executions start next day]

3. **Problem-Solving:**
   - Habit formed to solve a specific issue
   - Example: LLM.Chat (addressing problem) → Google.Search (solutions) → Calendar.CreateEvent → [executions]

4. **Gradual Build-Up:**
   - User eases into the habit
   - Early executions are exploratory
   - Example: [single trial execution] → LLM.Chat (evaluate) → Calendar.CreateEvent (commit) → [regular executions]

5. **Social-Influenced:**
   - Habit adopted due to social influence
   - Example: Message.GetMessages (friend recommendation) → Google.Search (research) → [executions]

**Generation Guidelines:**
- **MUST include change_reason demonstration:** Include 1-2 events showing why the habit was acquired
- **Generate establishment events (1-3 events):** These should occur BEFORE the first `schedule_dates` entry
- **Generate ONE recurring execution event:** Use a single event with `schedule_dates` listing all occurrence dates, `start_time`, `end_time`, and optional `note`
- **User intent for recurring event should describe evolution:** Indicate how the habit execution evolves over time (early phase → mid phase → late phase), OR keep it simple if the habit is uniform throughout
- **Align with state data:** Use `schedule_dates` from `habits_state.current_value` if available

**Example:**
```json
{
  "related_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "ai_tool_experimentation",
      "operation": "acquire"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "20:45:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Change reason awareness: exploring how AI tools handle legacy C++ codebases"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "14:30:00"
      },
      "app_name": "Calendar",
      "api_name": "CreateEvent",
      "user_intent": "Habit establishment: blocking Friday afternoons for dedicated AI tool evaluation"
    },
    {
      "time_specification": {
        "schedule_dates": [
          "2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26",
          "2024-02-02", "2024-02-09", "2024-02-16", "2024-02-23",
          "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22", "2024-03-29"
        ],
        "start_time": "15:00:00",
        "end_time": "16:30:00",
        "note": "Weekly Friday afternoon sessions. Early weeks focus on GitHub Copilot evaluation with simple refactoring tasks. Mid-period sessions test ChatGPT for unit test generation. Later weeks experiment with AI-assisted debugging on legacy modules."
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Habit execution: regular AI tool experimentation evolving from basic code completion to complex legacy codebase interactions"
    }
  ]
}
```

#### 6.3.2 Habits with `adjust` Operation

**Goal:** Show the adjustment process + demonstrate adjusted execution

**Base Pattern:** Awareness of need to adjust → Modification → Adjusted execution

**Generation Guidelines:**
- Include events showing why adjustment was needed (change_reason)
- Show adjustment actions (e.g., Calendar.EditEvent, LLM.Chat for new approach)
- Generate recurring execution events with the new pattern

#### 6.3.3 Habits with `stable` Operation

**Goal:** Demonstrate consistent recurring execution

**Base Pattern:** Recurring execution events

**Generation Guidelines:**
- Generate ONE event with `schedule_dates` listing all occurrences
- Use `start_time` and `end_time` from habit state
- User intent can describe the habit's purpose or indicate evolution if the habit naturally evolves (e.g., "early sessions focus on X, later sessions include Y")
- If the habit is truly uniform, keep user_intent simple and focused on the habit's core purpose

**Example:**
```json
{
  "related_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "firmware_code_review",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": [
          "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
          "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
          // ... (all 64 weekday dates in the window)
          "2024-03-27", "2024-03-28", "2024-03-29"
        ],
        "start_time": "09:30:00",
        "end_time": "10:30:00",
        "note": "Daily morning code review sessions on all weekdays."
      },
      "app_name": "SimpleNote",
      "api_name": "CreateNote",
      "user_intent": "Daily habit execution: documenting code review findings and maintaining review checklist. Early weeks focus on familiarizing with new team members' coding styles; later weeks incorporate mentoring notes for junior engineers."
    }
  ]
}
```

### 6.4 Preferences Generation

#### 6.4.1 Preferences with `refine` or `shift` Operation

**Goal:** Show the evolution process + demonstrate the new preference through behaviors

**Base Pattern:** Change awareness → Experimentation/Transition → New preference demonstration

**Behavior Diversity - Preference Evolution Paths:**

1. **Sudden Shift:**
   - User encounters a compelling reason and shifts quickly
   - Example: Google.SearchNews (new insight) → LLM.Chat (evaluating new approach) → [new behaviors immediately]

2. **Gradual Refinement:**
   - User tries new approach alongside old, then transitions
   - Example: [mix of old and new behaviors] → LLM.Chat (reflection) → [predominantly new behaviors]

3. **Problem-Prompted:**
   - User changes preference due to dissatisfaction with old approach
   - Example: SimpleNote.CreateNote (documenting frustration) → LLM.Chat (exploring alternatives) → [new behaviors]

**Generation Guidelines:**
- **MUST demonstrate change_reason:** Include 1-2 events that show why the preference evolved
- **Show the new preference through choices:** Generate 3-5 events where user's decisions reflect the new preference
- **Avoid stating preference directly:** Demonstrate through behavior, not through notes that say "I prefer X"
- If `previous_value` exists, optionally show 1-2 events early in window demonstrating the old preference (for contrast)

**Example:**
```json
{
  "related_state_items": [
    {
      "state_category": "preferences_state",
      "state_name": "learning_style",
      "operation": "refine"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-06"],
        "time": "16:45:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Change reason: discovering that AI-assisted coding accelerates project-based learning for new technologies"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-15"],
        "time": "21:00:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "New preference demonstration: using AI to refactor complex state machine instead of reading documentation"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-03"],
        "time": "15:30:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "New preference demonstration: prompt-engineering test scripts via AI rather than searching StackOverflow"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-20"],
        "time": "19:15:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "New preference demonstration: building toy project with AI assistance to test code generation limits on ARM assembly"
    }
  ]
}
```

#### 6.4.2 Preferences with `stable` Operation

**Goal:** Demonstrate the preference through consistent choices

**Base Pattern:** Multiple events showing decisions aligned with the preference

**Behavior Diversity - Preference Demonstration Approaches:**

1. **Choice-Based Demonstration:**
   - User repeatedly chooses option A over option B
   - Example: Spotify.SearchSongs (lofi) + Spotify.PlaySong (lofi) [multiple times]

2. **Avoidance-Based Demonstration:**
   - User avoids certain options or patterns
   - Example: User creates detailed docs (SimpleNote.CreateNote) to avoid meetings (implied by absence of Calendar.CreateEvent for calls)

3. **Active Organization:**
   - User organizes activities/content according to preference
   - Example: Creating specialized playlists, tagging notes with specific structure

4. **Consistency Across Contexts:**
   - Preference shows up in different contexts
   - Example: Preference for async communication shown in Message.SendMessage (detailed messages) + SimpleNote.CreateNote (documentation) + Email (if available)

**Generation Guidelines:**
- **Generate 3-5 events demonstrating preference**
- **Show preference through ACTIONS, not statements**
- **Vary contexts:** Show preference in different situations
- **Be subtle:** Preference should be inferable from choices, not explicitly stated

**Example:**
```json
{
  "related_state_items": [
    {
      "state_category": "preferences_state",
      "state_name": "collaboration_modality",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-11"],
        "time": "10:15:00"
      },
      "app_name": "Message",
      "api_name": "SendMessage",
      "user_intent": "Preference demonstration: sending detailed technical message with diagrams rather than requesting a meeting"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-18"],
        "time": "14:45:00"
      },
      "app_name": "SimpleNote",
      "api_name": "CreateNote",
      "user_intent": "Preference demonstration: authoring 10-page technical specification to clarify design instead of scheduling design meeting"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-07"],
        "time": "16:30:00"
      },
      "app_name": "Message",
      "api_name": "SendMessage",
      "user_intent": "Preference demonstration: requesting written agenda before accepting sync meeting invitation"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-03-05"],
        "time": "11:20:00"
      },
      "app_name": "SimpleNote",
      "api_name": "EditNote",
      "user_intent": "Preference demonstration: updating team status in shared note instead of giving verbal briefing"
    }
  ]
}
```

### 6.5 Chain Merging Rules

**When to Merge Multiple State Items into One Chain:**

**Condition 1: Natural Interweaving**
- The state items naturally appear together in actual user behavior
- They are NOT just coincidentally related; one directly involves or demonstrates the other

**Condition 2: Maximum 2 Items**
- Never merge more than 2 state items into a single chain

**Condition 3: Specific Merging Scenarios**

Scenario A: **Preference demonstrated during attribute usage**
- Example: User acquires noise-canceling headphones (attribute add) + user prefers instrumental music for focus (preference stable)
- The headphones are used to listen to instrumental music
- Events: headphones purchase + repeated listening to instrumental music

Scenario B: **Preference demonstrated during habit execution**
- Example: User has morning reading habit (habit stable) + user prefers technical blogs over books (preference stable)
- The habit execution naturally involves preference-aligned choices
- Events: repeated morning reading sessions choosing blogs

Scenario C: **Habit execution involves an attribute**
- Example: User has data analysis habit (habit stable) + user owns Python data skills (attribute stable)
- The habit directly uses the attribute
- Events: repeated data analysis sessions using Python

**When NOT to Merge:**

- ❌ Multiple items have the same `change_reason` but are acquired through independent events
- ❌ Items are thematically related but don't directly interweave in actual usage
- ❌ More than 2 items are involved

**Example of Merged Chain:**
```json
{
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "owns_noise_canceling_headphones",
      "operation": "add"
    },
    {
      "state_category": "preferences_state",
      "state_name": "prefers_instrumental_music_for_focus",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-20"],
        "time": "19:30:00"
      },
      "app_name": "Amazon",
      "api_name": "SearchProducts",
      "user_intent": "Attribute acquisition: researching noise-canceling headphones for better focus during work"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-21"],
        "time": "12:15:00"
      },
      "app_name": "Amazon",
      "api_name": "Checkout",
      "user_intent": "Attribute acquisition: purchasing noise-canceling headphones"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-24"],
        "time": "09:00:00"
      },
      "app_name": "Spotify",
      "api_name": "SearchSongs",
      "user_intent": "Combined usage: finding instrumental focus music to use with new headphones"
    },
    {
      "time_specification": {
        "schedule_dates": [
          "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-29", "2024-01-30",
          "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-08"
        ],
        "start_time": "09:15:00",
        "end_time": "11:30:00",
        "note": "Morning deep work sessions using new headphones with instrumental music playlists."
      },
      "app_name": "Spotify",
      "api_name": "PlaySong",
      "user_intent": "Combined usage: attribute (headphones) enables preference (instrumental music) during focused work"
    }
  ]
}
```

---

## 7. Critical Rules & Constraints

### 7.1 JSON Format Requirements

**MANDATORY:**
- Return ONLY valid JSON
- No markdown code fences (no ```json)
- No comments or extra text outside JSON
- All strings must be properly escaped
- All arrays and objects properly closed

### 7.2 Coverage Requirements

**MANDATORY:**
- EVERY state item in `state_table` MUST appear in at least one event chain
- Count user_attributes_state items: N1
- Count habits_state items: N2
- Count preferences_state items: N3
- Total chains must cover all N1 + N2 + N3 items (accounting for merges)

**Verification Checklist:**
- [ ] All user_attributes_state items are covered?
- [ ] All habits_state items are covered?
- [ ] All preferences_state items are covered?

### 7.3 Time Constraints

**MANDATORY:**
- All `schedule_dates` must fall within the window `time_range`
- For `add`/`acquire` operations: acquisition/setup events should occur early enough in the window to allow usage events afterward
- For habits: establishment events must occur BEFORE the first `schedule_dates` entry
- Spread events across the window; avoid clustering everything in the first week unless contextually appropriate

### 7.4 Mandatory Fields

**For every event:**
- ✅ `time_specification` (with `schedule_dates`)
- ✅ `app_name` (must match Section 5)
- ✅ `api_name` (must match Section 5)
- ✅ `user_intent` (clear explanation tied to state item)

**For every chain:**
- ✅ `related_state_items` (at least one, max two)
- ✅ `events` (at least one event)

**Do NOT include:**
- ❌ Event-level `description` field (use `user_intent` instead)
- ❌ Time ranges or cadence strings in `time_specification`
- ❌ Apps or APIs not listed in Section 5

### 7.5 Anti-Patterns to Avoid

**Pattern Repetition:**
- ❌ Don't use "search → view → purchase" for every attribute acquisition
- ❌ Don't use "research → calendar setup → execution" for every habit
- ✅ Vary acquisition paths based on user context and attribute type

**Unrealistic Clustering:**
- ❌ Don't put 10 events on the same day unless they're genuinely related
- ✅ Spread events naturally across the window

**Over-Explaining:**
- ❌ Don't write novel-length `user_intent` descriptions
- ✅ Be concise: one sentence explaining why this event occurs

**Ignoring Change Reasons:**
- ❌ Don't skip demonstrating `change_reason` when it exists
- ✅ Always include events that make the reason observable

**Generic User Intents:**
- ❌ "User is using the app"
- ❌ "Executing the habit"
- ✅ "Habit execution: daily code review with focus on low-latency patterns, documenting findings for team knowledge base"

---

## 8. Complete Examples

### Example 1: Attribute Add - Research-Heavy Path

```json
{
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "technical_skills_inventory",
      "operation": "add"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "20:30:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Change reason awareness: researching AI-assisted development tools for productivity gains"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "21:00:00"
      },
      "app_name": "Google",
      "api_name": "SearchNews",
      "user_intent": "Information seeking: checking recent developments in AI code generation tools"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-05"],
        "time": "14:15:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Decision support: evaluating GitHub Copilot vs ChatGPT for C++ legacy codebases"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-10"],
        "time": "16:30:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Skill acquisition: experimenting with AI-generated unit test for firmware module"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-22"],
        "time": "11:00:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Post-acquisition usage: using AI to refactor complex state machine implementation"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-14"],
        "time": "15:45:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Post-acquisition usage: generating boilerplate code for new sensor integration module"
    }
  ]
}
```

### Example 2: Attribute Add - Quick Path

```json
{
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "python_data_analysis_skills",
      "operation": "add"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "19:15:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Change reason awareness: exploring Python for processing robotics telemetry logs more efficiently than current tools"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-12"],
        "time": "21:30:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Skill acquisition: learning NumPy basics for log analysis via interactive examples"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-18"],
        "time": "10:00:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Post-acquisition usage: using Pandas to parse and visualize motor performance data from test runs"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-09"],
        "time": "13:45:00"
      },
      "app_name": "SimpleNote",
      "api_name": "CreateNote",
      "user_intent": "Post-acquisition usage: documenting Python analysis script for recurring telemetry review"
    }
  ]
}
```

### Example 3: Habit Acquire - Planned Establishment

```json
{
  "related_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "ai_tool_experimentation",
      "operation": "acquire"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-02"],
        "time": "21:00:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Change reason awareness: researching how to systematically evaluate AI tools for legacy C++ codebases"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "20:15:00"
      },
      "app_name": "Calendar",
      "api_name": "CreateEvent",
      "user_intent": "Habit establishment: scheduling recurring Friday afternoon block for AI tool experimentation"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "20:18:00"
      },
      "app_name": "Calendar",
      "api_name": "SetReminder",
      "user_intent": "Habit establishment: adding reminder to maintain consistency and track progress"
    },
    {
      "time_specification": {
        "schedule_dates": [
          "2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26",
          "2024-02-02", "2024-02-09", "2024-02-16", "2024-02-23",
          "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22", "2024-03-29"
        ],
        "start_time": "15:00:00",
        "end_time": "16:30:00",
        "note": "Weekly Friday experimentation sessions. January: focus on GitHub Copilot with basic refactoring. February: testing ChatGPT for unit test generation. March: advanced experiments with AI-assisted debugging on legacy modules."
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Habit execution: weekly AI tool evaluation sessions evolving from simple code completion to complex legacy codebase challenges"
    }
  ]
}
```

### Example 4: Habit Stable - Long-term Execution with Evolution

```json
{
  "related_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "industry_tech_reading",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": [
          "2024-01-02", "2024-01-04", "2024-01-09", "2024-01-11",
          "2024-01-16", "2024-01-18", "2024-01-23", "2024-01-25",
          "2024-01-30", "2024-02-01", "2024-02-06", "2024-02-08",
          "2024-02-13", "2024-02-15", "2024-02-20", "2024-02-22",
          "2024-02-27", "2024-02-29", "2024-03-05", "2024-03-07",
          "2024-03-12", "2024-03-14", "2024-03-19", "2024-03-21",
          "2024-03-26", "2024-03-28"
        ],
        "start_time": "20:30:00",
        "end_time": "21:30:00",
        "note": "Tuesday and Thursday evening reading sessions."
      },
      "app_name": "Google",
      "api_name": "SearchNews",
      "user_intent": "Stable habit execution: staying current with embedded systems and industrial automation developments through biweekly evening reading"
    }
  ]
}
```

### Example 5: Preference Refine - Evolution with Demonstration

```json
{
  "related_state_items": [
    {
      "state_category": "preferences_state",
      "state_name": "learning_style",
      "operation": "refine"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-07"],
        "time": "16:20:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "Change reason awareness: discovering AI-assisted coding significantly accelerates project-based learning"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-15"],
        "time": "21:15:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "New preference demonstration: using AI-pair programming to refactor state machine rather than reading manuals"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-28"],
        "time": "19:45:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "New preference demonstration: prompt-engineering test script instead of searching StackOverflow"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-18"],
        "time": "14:30:00"
      },
      "app_name": "LLM",
      "api_name": "Chat",
      "user_intent": "New preference demonstration: building toy ARM assembly project with AI to test code generation limits through doing"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-03-10"],
        "time": "11:00:00"
      },
      "app_name": "SimpleNote",
      "api_name": "CreateNote",
      "user_intent": "New preference demonstration: documenting learning insights from AI-assisted experimentation rather than passive reading"
    }
  ]
}
```

### Example 6: Preference Stable - Demonstration Through Choices

```json
{
  "related_state_items": [
    {
      "state_category": "preferences_state",
      "state_name": "collaboration_modality",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-11"],
        "time": "10:30:00"
      },
      "app_name": "Message",
      "api_name": "SendMessage",
      "user_intent": "Preference demonstration: sending comprehensive async technical explanation with diagrams instead of scheduling meeting"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-19"],
        "time": "15:00:00"
      },
      "app_name": "SimpleNote",
      "api_name": "CreateNote",
      "user_intent": "Preference demonstration: authoring detailed technical specification document to replace need for multi-hour design session"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "09:15:00"
      },
      "app_name": "Message",
      "api_name": "SendMessage",
      "user_intent": "Preference demonstration: requesting written agenda before accepting sync meeting invitation"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-22"],
        "time": "16:45:00"
      },
      "app_name": "SimpleNote",
      "api_name": "EditNote",
      "user_intent": "Preference demonstration: providing detailed status update in shared documentation rather than verbal standup"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-03-14"],
        "time": "11:20:00"
      },
      "app_name": "Message",
      "api_name": "SendMessage",
      "user_intent": "Preference demonstration: answering technical question with thorough async writeup including code examples"
    }
  ]
}
```

### Example 7: Merged Chain - Attribute + Preference

```json
{
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "owns_noise_canceling_headphones",
      "operation": "add"
    },
    {
      "state_category": "preferences_state",
      "state_name": "prefers_lofi_for_deep_work",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-20"],
        "time": "12:30:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Attribute acquisition: researching noise-canceling headphones for better focus during coding"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-20"],
        "time": "12:50:00"
      },
      "app_name": "Amazon",
      "api_name": "SearchProducts",
      "user_intent": "Attribute acquisition: comparing noise-canceling headphones with good audio quality for music"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-20"],
        "time": "13:05:00"
      },
      "app_name": "Amazon",
      "api_name": "ReadReviews",
      "user_intent": "Attribute acquisition: checking reviews for comfort during extended coding sessions"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-21"],
        "time": "11:00:00"
      },
      "app_name": "Amazon",
      "api_name": "Checkout",
      "user_intent": "Attribute acquisition: purchasing selected noise-canceling headphones"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-25"],
        "time": "09:00:00"
      },
      "app_name": "Spotify",
      "api_name": "SearchSongs",
      "user_intent": "Combined usage: finding lofi playlists to test new headphones for deep work sessions"
    },
    {
      "time_specification": {
        "schedule_dates": [
          "2024-01-25", "2024-01-26", "2024-01-29", "2024-01-30",
          "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06",
          "2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13"
        ],
        "start_time": "09:15:00",
        "end_time": "11:30:00",
        "note": "Morning focused coding sessions with headphones and lofi music."
      },
      "app_name": "Spotify",
      "api_name": "PlaySong",
      "user_intent": "Combined usage: using new headphones with preferred lofi music during deep work, demonstrating both attribute and preference"
    }
  ]
}
```

---

## 9. Generation Checklist

Before finalizing your output, verify:

**Coverage:**
- [ ] Every user_attributes_state item is covered
- [ ] Every habits_state item is covered
- [ ] Every preferences_state item is covered

**Quality:**
- [ ] Each event has clear `user_intent` tied to state item
- [ ] Change reasons are made observable through events
- [ ] Behavior patterns are diverse (not all following the same template)
- [ ] Time distribution is realistic and spread across window

**Format:**
- [ ] Valid JSON with no syntax errors
- [ ] All `schedule_dates` within window `time_range`
- [ ] All `app_name` and `api_name` match Section 5
- [ ] No forbidden fields (e.g., event-level `description`)

**Realism:**
- [ ] Event sequences tell coherent stories
- [ ] Timestamps match realistic daily schedules
- [ ] Habit executions align with provided schedules
- [ ] Acquisition events occur before usage events

---

## 10. Final Instructions

1. **Read the input `domain_window_state` carefully**
2. **Identify all state items** that need coverage
3. **Decide on chain organization** (independent vs. merged)
4. **Generate diverse event patterns** for each chain
5. **Verify coverage and format** using the checklist
6. **Output ONLY the JSON** - no extra text, no markdown fences

**Remember:** Your goal is to create realistic, diverse, observable event sequences that bring the user's state to life through concrete app interactions.
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
