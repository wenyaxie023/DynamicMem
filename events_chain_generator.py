import json
from dataclasses import dataclass

from jinja2 import Template

from mem_bench.behavior_and_conversation.app_catalog import APP_CATALOG
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

### Core Principle: Lossless State-to-Event Conversion

**CRITICAL:** Every state item must be converted into concrete, observable events WITHOUT information loss. This means:

1. **Semantic Completeness**: The generated events must fully capture the MEANING of each state item. If a state says "user prefers async communication over meetings," the events must demonstrate BOTH the preference for async AND the avoidance of sync—not just one side.

2. **Behavioral Fidelity**: Events should simulate ACTUAL user behavior with realistic details:
   - What exactly would the user type/click/search?
   - What specific content would they create or consume?
   - What would trigger this action at this moment?

3. **Intent Transparency**: Each event's `user_intent` must clearly explain:
   - WHY this action happens (motivation)
   - HOW it connects to the state item (observable link)
   - WHAT it reveals about the user (insight)

4. **Process + Outcome**: For any state change, show BOTH:
   - The process that led to the change (causation events)
   - The result of the change (demonstration events)

**Anti-pattern to avoid:** Generating shallow, generic events that technically "cover" a state item but lose its richness and specificity.

---

Context Information You Will Receive:

**Life Context:**
{{ user_life_context }}

**User Basic Profile:** Contains demographic information, personality traits, and stable characteristics.
{{ user_basic_profile }}

**Previous Window Summary (All Domains):**
{{ user_previous_window_summary }}

**Previous Window Summary (Current Domain):** Describes what happened in earlier time windows.
{{ user_domain_previous_window_summary }}

**Current Window Description (Current Domain):** Provides an overview of this window's themes and major developments.
{{ user_this_window_description }}

**Domain Window State (Current Window):**
{{ domain_window_state }}

---

## 2. Lossless Conversion Framework

### 2.1 What "Lossless" Means in Practice

For each state item, ask yourself: **"If I only saw these events (without the original state), could I accurately reconstruct the state item?"**

| State Aspect | Must Be Observable Through Events |
|--------------|-----------------------------------|
| **Attribute value** | Concrete usage demonstrating possession/capability |
| **Attribute acquisition** | Research, decision-making, and purchase/setup process |
| **Habit schedule** | Events occurring at specified times on specified days |
| **Habit evolution** | Early executions differ from later ones (learning curve) |
| **Preference direction** | Choices that favor X over alternatives |
| **Preference strength** | Consistency and frequency of preference-aligned choices |
| **Change reason** | Events that make the trigger/catalyst observable |

### 2.2 Event Detail Requirements

Each event must be specific enough to be "executable" — meaning you could actually perform this action in the real app.

**Insufficient Detail (BAD):**
```json
{
  "app_name": "Amazon",
  "api_name": "SearchProducts",
  "user_intent": "Searching for headphones"
}
```

**Sufficient Detail (GOOD):**
```json
{
  "app_name": "Amazon",
  "api_name": "SearchProducts",
  "user_intent": "Searching 'noise canceling headphones for programming' after experiencing concentration difficulties in open office; prioritizing comfort for 8+ hour daily wear over audio quality for music"
}
```

The good example captures:
- Exact search query context
- Triggering situation (concentration difficulties)
- Specific use case (programming, not music)
- Key decision criteria (comfort > audio quality)
- Usage pattern (8+ hours daily)

### 2.3 User Intent Composition

Every `user_intent` should follow this structure:

```
[Action Context]: [Specific Action] + [Motivation/Trigger] + [Decision Criteria/Constraints]
```

**Examples:**
- "Habit execution: logging 45-minute morning run (5.2km at 5:45/km pace) before 7am standup; tracking pace improvement from last week's 6:10/km average"
- "Preference demonstration: selecting 'lofi hip hop radio' playlist over 'Top 50 Global' for deep work session; avoiding lyrics that compete with code reading"
- "Change reason awareness: reading HackerNews thread about AI coding assistants after colleague mentioned 3x productivity gains; skeptical but curious about claims"

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

### 3.2 State Item Interpretation Guide

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

**Conversion Requirements by Operation:**
- `add`: Must show acquisition journey (research → decision → acquisition) + multiple usage instances
- `modify`: Must show old value usage → transition trigger → new value usage
- `stable`: Must show ongoing usage across different contexts within the window

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
    "schedule_dates": ["YYYY-MM-DD", ...]
  },
  "op": "acquire | adjust | stable",
  "change_reason": "why this changed (if op is acquire/adjust)",
  "previous_value": {...}
}
```

**Conversion Requirements by Operation:**
- `acquire`: Must show motivation discovery → habit design → initial struggles → stabilization
- `adjust`: Must show dissatisfaction with old pattern → adjustment reasoning → new pattern execution
- `stable`: Must show consistent execution with natural variation (not robotic repetition)

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

**Conversion Requirements by Operation:**
- `refine/shift`: Must show old preference in action → catalyst event → experimentation → new preference dominance
- `stable`: Must show preference through CHOICES (selecting A over B), not just using A

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
            "time": "HH:MM:SS",
            "start_time": "HH:MM:SS",
            "end_time": "HH:MM:SS",
            "note": "optional explanation for repeated events"
          },
          "app_name": "from available apps",
          "api_name": "from available APIs",
          "user_intent": "detailed explanation following the composition structure"
        }
      ]
    }
  ]
}
```

### 4.2 Time Specification Rules

**Rule 1: Use `schedule_dates` for all events**

**Rule 2: For single occurrence events:**
```json
{
  "schedule_dates": ["2024-01-15"],
  "time": "14:30:00"
}
```

**Rule 3: For repeated occurrence events:**
```json
{
  "schedule_dates": ["2024-01-05", "2024-01-12", "2024-01-19"],
  "start_time": "07:00:00",
  "end_time": "07:30:00",
  "note": "Weekly Saturday morning sessions with progressive intensity increase"
}
```

**Rule 4: All dates must fall within the window `time_range`**

---

## 5. Available Apps & APIs

{{ app_catalog_json }}

---

## 6. Generation Strategy by State Type

### 6.1 User Attributes Generation

#### Operation: `add`

**Goal:** Demonstrate the complete acquisition journey and subsequent integration into user's life.

**Required Event Phases:**

1. **Trigger/Awareness Phase** (1-2 events)
   - What made the user realize they need/want this?
   - Observable through: searches, conversations, problem encounters

2. **Research/Evaluation Phase** (1-4 events, varies by user personality)
   - How did they evaluate options?
   - Observable through: product comparisons, reviews, asking others, AI consultations

3. **Decision/Acquisition Phase** (1-2 events)
   - The actual acquisition moment
   - Observable through: checkout, signup, download, subscription

4. **Integration/Usage Phase** (2-4 events)
   - How does this attribute show up in daily life?
   - Observable through: repeated usage across different contexts

**Behavioral Diversity Consideration:**
Different users have different acquisition styles. Match to user personality:
- Analytical users: heavy research phase
- Impulsive users: short trigger-to-acquisition gap
- Social users: friend consultation before decision
- Cautious users: trial period before commitment

#### Operation: `modify`

**Goal:** Show the transition narrative from old to new value.

**Required Event Phases:**
1. Old value in use (1-2 events early in window)
2. Trigger for change (1-2 events)
3. Transition process (1-2 events)
4. New value in use (2-3 events)

#### Operation: `stable`

**Goal:** Demonstrate ongoing presence through varied usage contexts.

**Requirements:**
- Generate 3-5 usage events spread across the window
- Each event should show the attribute in a DIFFERENT context/scenario
- Avoid repetitive events; real usage is contextually varied

### 6.2 Habits Generation

#### Operation: `acquire`

**Goal:** Show how the habit was born and became established.

**Required Event Phases:**

1. **Motivation Discovery** (1-2 events)
   - What problem or aspiration triggered habit consideration?
   - Observable through: searches, AI conversations, social discussions

2. **Habit Design** (1-2 events)
   - How did user structure the habit?
   - Observable through: calendar/reminder setup, tracker creation, goal setting

3. **Execution Events** (single event with multiple dates)
   - The recurring execution of the habit
   - Use `schedule_dates` array with `start_time` and `end_time`
   - Include `note` describing evolution: early phase (struggle/learning) → mid phase (stabilizing) → late phase (automatic/optimizing)

**Critical:** Establishment events must occur BEFORE the first `schedule_dates` entry.

#### Operation: `adjust`

**Goal:** Show why and how the habit pattern changed.

**Required Event Phases:**
1. Old pattern execution (early dates showing previous pattern)
2. Adjustment trigger (what prompted the change)
3. Modification action (updating schedule/approach)
4. New pattern execution (later dates showing adjusted pattern)

#### Operation: `stable`

**Goal:** Show consistent but human execution.

**Requirements:**
- Single event with `schedule_dates` covering all occurrences
- `user_intent` should describe what happens during execution with enough detail to understand the habit's nature
- Optional `note` can describe natural evolution even within stable habits

### 6.3 Preferences Generation

#### Operation: `refine` or `shift`

**Goal:** Make the preference evolution visible through changing choices.

**Required Event Phases:**

1. **Old Preference Demonstration** (optional, 1-2 events if previous_value exists)
   - Show choices aligned with old preference early in window

2. **Catalyst Event** (1-2 events)
   - What triggered the preference change?
   - Must make `change_reason` observable

3. **Experimentation Phase** (1-2 events)
   - Trying the new approach, possibly mixed with old

4. **New Preference Demonstration** (2-4 events)
   - Consistent choices aligned with new preference
   - Show preference in multiple contexts

#### Operation: `stable`

**Goal:** Demonstrate preference through observable choice patterns.

**Critical Principle:** Preferences are revealed through CHOICES, not statements.

**Requirements:**
- Generate 4-6 events showing preference-aligned choices
- Each event should be a decision point where alternatives existed
- `user_intent` should make clear what was chosen OVER what alternative
- Show preference consistency across different contexts

**Example of Preference Demonstration:**
- Instead of: "Using Notion for notes"
- Show: "Creating detailed async project update in Notion instead of requesting a sync meeting; including annotated screenshots to preempt questions"

---

## 7. Chain Organization Rules

### 7.1 Default: One State Item → One Event Chain

### 7.2 Merging Conditions (Maximum 2 items per chain)

Merge ONLY when:
1. **Natural Interweaving:** The items genuinely co-occur in actual behavior
2. **Demonstrable Connection:** One item's usage naturally demonstrates the other

**Valid Merge Scenarios:**
- Attribute enables habit execution (e.g., owns running shoes + morning run habit)
- Preference demonstrated during attribute usage (e.g., owns headphones + prefers instrumental music)
- Preference shapes habit execution (e.g., learning habit + preference for hands-on over reading)

**Invalid Merge (keep separate):**
- Same `change_reason` but independent acquisition paths
- Thematically related but not behaviorally intertwined
- More than 2 items involved

---

## 8. Comprehensive Examples

### Example 1: Attribute Add — Full Acquisition Journey

**State Item:**
```json
{
  "name": "ai_coding_assistant_subscription",
  "current_value": "GitHub Copilot Business subscription, integrated with VS Code and JetBrains IDEs",
  "op": "add",
  "change_reason": "Team lead mandated AI tool adoption after Q4 productivity review showed 20% lag behind industry benchmarks"
}
```

**Generated Event Chain:**
```json
{
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "ai_coding_assistant_subscription",
      "operation": "add"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "16:45:00"
      },
      "app_name": "Gmail",
      "api_name": "ReadEmail",
      "user_intent": "Change reason trigger: reading team lead's email summarizing Q4 productivity review results; noting the 20% lag statistic and mandatory AI tool adoption directive effective Q1"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "21:15:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Initial research: searching 'best AI coding assistants 2024 comparison' to understand landscape before team discussion; skeptical about productivity claims but recognizing need to comply with directive"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "12:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Deep evaluation: asking Claude to compare GitHub Copilot vs Cursor vs Amazon CodeWhisperer specifically for C++ embedded systems development; concerned about legacy codebase compatibility and offline functionality for secure environments"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "14:00:00"
      },
      "app_name": "WhatsApp",
      "api_name": "SendMessage",
      "user_intent": "Social validation: messaging senior colleague who adopted Copilot last quarter asking about real-world experience with firmware codebases; specifically asking about false positive rate in suggestions and learning curve"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-05"],
        "time": "10:30:00"
      },
      "app_name": "WhatsApp",
      "api_name": "GetMessages",
      "user_intent": "Gathering peer input: reading colleague's detailed response about Copilot experience; noting their recommendation to start with simple boilerplate generation before trusting complex suggestions"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "09:15:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Acquisition preparation: searching 'GitHub Copilot Business setup JetBrains CLion' to understand integration process before requesting license from IT"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "11:00:00"
      },
      "app_name": "Gmail",
      "api_name": "SendEmail",
      "user_intent": "Acquisition action: emailing IT department to request GitHub Copilot Business license activation; cc'ing team lead to document compliance with Q1 directive"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-10"],
        "time": "14:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Initial integration usage: testing Copilot on low-stakes task—asking Claude how to write effective Copilot prompts for generating unit test boilerplate for existing sensor calibration module"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-15"],
        "time": "10:00:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Growing competence usage: using AI assistant to debug a Copilot-generated state machine that had subtle race condition; learning to verify AI suggestions rather than blindly accepting"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-22"],
        "time": "15:45:00"
      },
      "app_name": "Notion",
      "api_name": "CreatePage",
      "user_intent": "Integration reflection: documenting personal 'Copilot best practices' learned over two weeks—noting that it excels at boilerplate but requires careful review for timing-critical code; planning to share with team"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "11:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Mature usage: confidently using AI to generate complete test harness for new motor control module; has developed intuition for when to trust vs verify suggestions based on code complexity"
    }
  ]
}
```

**Why This Example Demonstrates Lossless Conversion:**
- Change reason is explicitly observable (email about Q4 review)
- Research phase shows realistic skepticism and specific concerns (legacy codebase, offline)
- Social validation matches user seeking peer experience
- Acquisition process is concrete (IT request email)
- Usage evolution shows learning curve (cautious → competent → confident)
- Integration into workflow is demonstrated (documentation, sharing intent)

---

### Example 2: Habit Acquire — From Motivation to Establishment

**State Item:**
```json
{
  "name": "weekly_ai_experimentation",
  "current_value": {
    "schedule": {
      "frequency_type": "weekly",
      "days_of_week": [4],
      "start_date": "2024-01-05"
    },
    "timing": {
      "start_time": "15:00",
      "end_time": "17:00"
    },
    "location": "home office",
    "priority": "medium",
    "schedule_dates": ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26", "2024-02-02", "2024-02-09", "2024-02-16", "2024-02-23", "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22", "2024-03-29"]
  },
  "op": "acquire",
  "change_reason": "Recognized need to systematically evaluate AI tools rather than ad-hoc exploration after seeing inconsistent results from sporadic attempts"
}
```

**Generated Event Chain:**
```json
{
  "related_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "weekly_ai_experimentation",
      "operation": "acquire"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-02"],
        "time": "20:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Change reason crystallization: reflecting on past month's scattered AI tool attempts with Claude; realizing that trying Copilot on Monday, ChatGPT on Thursday, and Cursor randomly led to no clear conclusions about which tool works best for embedded C++ work"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "12:15:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Habit design research: searching 'systematic approach to evaluating developer tools' to find frameworks for structured experimentation; wanting to avoid previous mistake of unstructured exploration"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "21:00:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Habit structure planning: asking Claude to help design a 13-week AI tool evaluation protocol with specific test tasks for each week; deciding on Friday afternoons when energy for creative work is typically higher and weekend proximity allows extended sessions if needed"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "19:45:00"
      },
      "app_name": "Notion",
      "api_name": "CreateDatabaseEntry",
      "user_intent": "Habit infrastructure setup: creating 'AI Tool Evaluation Tracker' database with columns for date, tool tested, task type, success metrics, and learnings; blocking Friday 3-5pm as recurring 'AI Lab' time in personal schedule"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "19:55:00"
      },
      "app_name": "Notion",
      "api_name": "CreatePage",
      "user_intent": "Habit protocol documentation: writing evaluation criteria document specifying test tasks (unit test generation, code refactoring, documentation writing, bug fixing) and scoring rubric (accuracy, time saved, learning curve)"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26", "2024-02-02", "2024-02-09", "2024-02-16", "2024-02-23", "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22", "2024-03-29"],
        "start_time": "15:00:00",
        "end_time": "17:00:00",
        "note": "Weekly Friday AI experimentation sessions. Weeks 1-4 (January): Focus on GitHub Copilot—testing basic code completion (week 1), unit test generation (week 2), refactoring suggestions (week 3), documentation generation (week 4). Weeks 5-8 (February): Shift to ChatGPT/Claude comparison—same four task categories to enable direct comparison. Weeks 9-13 (March): Advanced scenarios—testing on actual legacy codebase modules with complex dependencies, timing-critical sections, and hardware abstraction layers. By late March, sessions feel routine and user has developed clear preferences for which tool suits which task type."
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Habit execution: dedicated Friday afternoon AI tool evaluation sessions following established protocol; each session focuses on specific tool-task combination, documents findings in Notion tracker, and iteratively refines understanding of AI capabilities for embedded systems development workflow"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-02"],
        "time": "17:15:00"
      },
      "app_name": "Notion",
      "api_name": "UpdatePage",
      "user_intent": "Mid-habit reflection: updating evaluation tracker with month-one insights after completing Copilot deep-dive; noting that Copilot excels at boilerplate but struggles with hardware-specific code patterns; adjusting February protocol to test this hypothesis with ChatGPT"
    }
  ]
}
```

**Why This Example Demonstrates Lossless Conversion:**
- Change reason explicitly shown (reflecting on past scattered attempts)
- Habit design process is observable (research, AI-assisted planning)
- Infrastructure setup is concrete (Notion database, protocol document)
- Recurring execution uses `note` to show realistic evolution across 13 weeks
- Mid-habit adjustment shows living, adaptive behavior (not robotic execution)

---

### Example 3: Preference Shift — From Old to New Through Catalyst

**State Item:**
```json
{
  "name": "learning_approach",
  "current_value": {
    "statement": "Strongly prefers hands-on project-based learning with AI assistance over traditional documentation reading; believes building something reveals gaps faster than reading about concepts"
  },
  "op": "shift",
  "change_reason": "After struggling for 3 weeks to understand a complex state machine from documentation alone, completed equivalent learning in 2 days by building a simplified version with AI pair programming",
  "previous_value": {
    "statement": "Prefers comprehensive documentation reading before attempting implementation; believes thorough understanding prevents costly mistakes"
  }
}
```

**Generated Event Chain:**
```json
{
  "related_state_items": [
    {
      "state_category": "preferences_state",
      "state_name": "learning_approach",
      "operation": "shift"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "20:00:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Old preference demonstration: searching 'finite state machine design patterns embedded systems PDF' to find comprehensive documentation before implementing motor control FSM; following established pattern of thorough reading before coding"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-10"],
        "time": "21:30:00"
      },
      "app_name": "Goodreads",
      "api_name": "SearchBooks",
      "user_intent": "Old preference demonstration: searching for books on state machine design after online documentation proved insufficient; believing a comprehensive textbook will provide the deep understanding needed"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-15"],
        "time": "19:45:00"
      },
      "app_name": "Notion",
      "api_name": "CreatePage",
      "user_intent": "Frustration documentation: creating note titled 'FSM Learning Struggle' after two weeks of documentation reading still leaves confusion about hierarchical state handling; beginning to question whether reading-first approach is working for this topic"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-22"],
        "time": "14:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Catalyst moment: desperately asking Claude 'can you help me build a simple hierarchical FSM step by step?' after 3 weeks of documentation produced more confusion than clarity; abandoning reading-first approach out of frustration"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-22"],
        "time": "16:45:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Revelation experience: continuing AI pair programming session, now on third iteration of FSM implementation; suddenly understanding hierarchical state transitions through building rather than reading—what took 3 weeks to NOT understand is becoming clear in hours"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-23"],
        "time": "10:00:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Preference consolidation: asking Claude to extend yesterday's FSM with edge cases; completing what would have been weeks of documentation study in second day of hands-on building; explicitly recognizing this as better learning approach"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-24"],
        "time": "20:30:00"
      },
      "app_name": "Notion",
      "api_name": "UpdatePage",
      "user_intent": "Preference articulation: updating 'FSM Learning Struggle' note with breakthrough summary; explicitly writing 'building reveals gaps faster than reading'—the exact principle that now guides learning approach"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "15:00:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "New preference application: starting to learn CAN bus protocol by asking Claude to help build a simple message parser instead of reading protocol specification; deliberately applying hands-on-first approach validated by FSM experience"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-12"],
        "time": "21:15:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "New preference reinforcement: learning I2C driver implementation through iterative building with AI assistance; skipping the 50-page specification document entirely in favor of 'build and discover gaps' approach"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-03-01"],
        "time": "19:00:00"
      },
      "app_name": "WhatsApp",
      "api_name": "SendMessage",
      "user_intent": "New preference evangelism: advising junior colleague struggling with SPI protocol to 'just start building with ChatGPT instead of reading the spec—you'll learn 10x faster'; confidently sharing shifted learning philosophy"
    }
  ]
}
```

**Why This Example Demonstrates Lossless Conversion:**
- Old preference is demonstrated through concrete actions (documentation search, book search)
- Struggle period is observable (frustration note after 2 weeks)
- Catalyst moment is precise (the desperate pivot to AI pair programming)
- Revelation is captured (understanding in hours what didn't click in weeks)
- New preference is applied to multiple subsequent contexts (CAN bus, I2C)
- Preference strength shown through social sharing (advising colleague)

---

### Example 4: Merged Chain — Attribute + Preference Naturally Intertwined

**State Items:**
```json
{
  "name": "noise_canceling_headphones",
  "current_value": "Sony WH-1000XM5, used daily for focused coding sessions",
  "op": "add",
  "change_reason": "Open office environment making deep work impossible; needed isolation solution"
}
```
```json
{
  "name": "work_music_preference",
  "current_value": {
    "statement": "Strongly prefers instrumental lofi/ambient over lyrical music during coding; finds lyrics compete with internal verbalization of code logic"
  },
  "op": "stable"
}
```

**Generated Event Chain:**
```json
{
  "related_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "noise_canceling_headphones",
      "operation": "add"
    },
    {
      "state_category": "preferences_state",
      "state_name": "work_music_preference",
      "operation": "stable"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "11:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Attribute change reason: venting to Claude about open office concentration problems after third interrupted deep work session this week; asking for solutions to focus in noisy environment"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "20:15:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Attribute research: searching 'best noise canceling headphones for programming 2024' with specific focus on comfort for all-day wear and ANC effectiveness against office chatter"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-09"],
        "time": "12:45:00"
      },
      "app_name": "Amazon",
      "api_name": "SearchProducts",
      "user_intent": "Attribute comparison: searching 'Sony WH-1000XM5' after multiple recommendations in research; prioritizing ANC quality over audiophile sound since primary use is blocking noise, not music appreciation"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-09"],
        "time": "12:52:00"
      },
      "app_name": "Amazon",
      "api_name": "ShowProduct",
      "user_intent": "Attribute evaluation: reading detailed reviews focusing on comments about comfort during long coding sessions and ANC effectiveness in office environments; noting reviewer who mentioned 8-hour daily use without fatigue"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-09"],
        "time": "13:10:00"
      },
      "app_name": "Amazon",
      "api_name": "Checkout",
      "user_intent": "Attribute acquisition: purchasing Sony WH-1000XM5 with expedited shipping; justifying $350 expense as investment in productivity given daily usage expectation"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-12"],
        "time": "09:00:00"
      },
      "app_name": "Spotify",
      "api_name": "SearchSongs",
      "user_intent": "Combined first use: searching 'lofi coding beats instrumental' to test new headphones with preferred focus music; preference for instrumental immediately relevant as testing whether lyrics remain distracting even with ANC"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-12"],
        "time": "09:05:00"
      },
      "app_name": "Spotify",
      "api_name": "PlaySong",
      "user_intent": "Combined preference validation: playing 'lofi hip hop radio - beats to code/relax to' through new headphones; confirming that instrumental-only preference still holds—ANC blocks external noise but internal lyric processing would still compete with code thinking"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26"],
        "start_time": "09:00:00",
        "end_time": "11:30:00",
        "note": "Daily morning deep work sessions combining attribute (headphones) with preference (instrumental music). Early sessions (Jan 12-15) involve experimenting with ANC levels and playlist selection. Mid-sessions (Jan 16-22) establish routine: ANC on max, lofi playlist on shuffle, 2.5 hour focus block. Later sessions (Jan 23-26) become automatic ritual."
      },
      "app_name": "Spotify",
      "api_name": "PlaySong",
      "user_intent": "Combined habitual usage: daily morning focus ritual using noise-canceling headphones with carefully curated instrumental lofi playlist; headphones enable isolation while instrumental preference ensures music aids rather than competes with coding cognition"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-19"],
        "time": "14:30:00"
      },
      "app_name": "Spotify",
      "api_name": "AddToPlaylist",
      "user_intent": "Preference curation: adding newly discovered ambient electronic track to 'Deep Work' playlist after it maintained flow state for full pomodoro; excluding similar track with subtle vocals that broke concentration—demonstrating preference precision"
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "10:15:00"
      },
      "app_name": "WhatsApp",
      "api_name": "SendMessage",
      "user_intent": "Combined evangelism: responding to colleague's complaint about open office noise with enthusiastic recommendation of Sony headphones + instrumental music combination; specifically warning against lyrical playlists for coding based on personal experience"
    }
  ]
}
```

**Why This Merge Works:**
- The headphones exist primarily to enable focused music listening during work
- The music preference directly shapes how the headphones are used
- Every usage event naturally involves BOTH the attribute AND the preference
- Separating them would create artificial, redundant events

---

## 9. Validation Checklist

Before finalizing output, verify:

### Coverage Validation
- [ ] Every `user_attributes_state` item appears in at least one chain
- [ ] Every `habits_state` item appears in at least one chain
- [ ] Every `preferences_state` item appears in at least one chain

### Lossless Conversion Validation
For EACH state item, confirm:
- [ ] Could someone reconstruct the state item from ONLY the events?
- [ ] Is the `change_reason` (if present) observable through at least one event?
- [ ] For `add/acquire`: Are both process AND outcome events present?
- [ ] For `stable`: Is the state demonstrated across multiple contexts?
- [ ] For preferences: Are CHOICES shown (A over B), not just usage of A?

### Event Quality Validation
- [ ] Each `user_intent` follows the composition structure: [Context]: [Action] + [Motivation] + [Criteria]
- [ ] Each event is specific enough to be "executable" in the real app
- [ ] No generic/shallow events that technically cover but don't demonstrate

### Temporal Validation
- [ ] All `schedule_dates` fall within window `time_range`
- [ ] Acquisition events precede usage events
- [ ] Habit establishment events precede first `schedule_dates` entry
- [ ] Events are spread across the window (not clustered unrealistically)

### Format Validation
- [ ] Valid JSON with no syntax errors
- [ ] All `app_name` values match available apps
- [ ] All `api_name` values match available APIs for their app
- [ ] No forbidden fields (e.g., event-level `description`)

---

## 10. Final Instructions

1. **Read the input state carefully** — understand each item's full meaning
2. **Plan the conversion** — for each state item, what events would make it observable?
3. **Generate detailed events** — each event should pass the "executable" test
4. **Verify lossless conversion** — could someone reconstruct state from events alone?
5. **Check coverage** — every state item must be represented
6. **Output ONLY the JSON** — no markdown fences, no commentary

**Remember:** Your goal is to create a behavioral trace so realistic and detailed that the original state items become fully observable through the events. Information should flow FROM state TO events with zero loss.
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
        app_catalog_json=json.dumps(APP_CATALOG, indent=2),
        user_basic_profile=request.user_basic_profile,
        domain_name=request.domain_name,
        user_life_context=request.user_life_context,
        user_previous_window_summary=request.user_previous_window_summary,
        user_domain_previous_window_summary=request.user_domain_previous_window_summary,
        user_this_window_description=request.user_this_window_description,
        domain_window_state=request.domain_window_state,
        world_background=request.world_background,
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
