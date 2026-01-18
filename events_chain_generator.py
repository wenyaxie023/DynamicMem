import json
from dataclasses import dataclass

from jinja2 import Template

from mem_bench.behavior_and_conversation.app_catalog import APP_CATALOG
from mem_bench.behavior_and_conversation.llm_client import (
    GeminiJSONClient,
    LLMResult,
)

events_chain_template = Template("""You are an expert at generating realistic event chains that demonstrate user behaviors based on their dynamic profile state.

### Your Task
Given a user's state for a specific time window, generate a sequence of realistic events that would naturally occur based on their attributes, habits, and preferences. Each event must specify which app and API it uses, along with the user intent.

### Core Principle: Lossless State-to-Event Conversion

**CRITICAL:** Every state item must be converted into concrete, observable events WITHOUT information loss. This means:

1. **Semantic Completeness**: The generated events must fully capture the MEANING of each state item. If a state says "user prefers async communication over meetings," the events must demonstrate BOTH the preference for async AND the avoidance of sync—not just one side.

2. **Behavioral Fidelity**: Events should simulate ACTUAL user behavior with realistic details:
   - What exactly would the user type/click/search?
   - What specific content would they create or consume?
   - What would trigger this action at this moment?

3. **Intent as Generation Context**: Each event's `user_intent` serves two purposes:
   - **Motivation**: Why the user is performing this action (the trigger or need)
   - **Content Direction**: Provides sufficient context for downstream generation of concrete inputs (e.g., what query to type in Google Search, what message to send in WhatsApp, what question to ask the LLM)
   
   The `user_intent` should be detailed enough that someone could generate realistic, specific content for that app/API based on it.

4. **Process + Outcome**: For any state change, show BOTH:
   - The process that led to the change (causation events)
   - The result of the change (demonstration events)

**Anti-pattern to avoid:** Generating shallow, generic events that technically "cover" a state item but lose its richness and specificity.

---

## 1. Context Information You Will Receive

**World Background:** Provides a detailed description of the world background in this window.
{{ world_background }}

**User Basic Profile:** Contains demographic information, personality traits, and stable characteristics.
{{ user_basic_profile }}

**Previous Window Summary (All Domains):**
{{ user_previous_window_summary }}

**Previous Window Summary (Current Domain):** Describes what happened in earlier time windows.
{{ user_domain_previous_window_summary }}

**Current Window Description (Current Domain):** Provides an overview of this window's themes and major developments.
{{ user_this_window_description }}

**Domain Window State (Current Window):** Contains the state items you need to convert into events.
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

`user_intent` should provide enough context for downstream content generation. A good user_intent typically includes:

1. **What the user wants to do** (the goal/action)
2. **Why they want to do it** (motivation/trigger/context)
3. **Any specific constraints or criteria** (optional, but helpful for realistic generation)

**Examples:**

- "Searching for noise canceling headphones for programming after experiencing concentration difficulties in open office; prioritizing comfort for 8+ hour daily wear over audio quality"
  → Enables generating search query: "noise canceling headphones comfortable long wear programming"

- "Asking Claude to help build a simple hierarchical FSM step by step after 2 weeks of documentation reading produced more confusion than clarity"
  → Enables generating LLM prompt about FSM implementation with step-by-step guidance request

- "Messaging senior colleague about their real-world experience with GitHub Copilot on firmware codebases; specifically asking about false positive rate and learning curve"
  → Enables generating WhatsApp message asking specific questions about Copilot

- "Logging 45-minute morning run (5.2km at 5:45/km pace) before 7am standup; tracking pace improvement from last week's 6:10/km average"
  → Enables generating Fitbit workout log with specific metrics

Note: In the following examples, the `user_intent` includes a prefix label (e.g., "Initial research:", "Catalyst moment:") 
to indicate the event's narrative role in the chain. This helps maintain coherent storytelling while 
the rest of the intent provides context for downstream content generation.

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
  "change_type": "add | modify | unchanged",
  "change_reason": "why this changed (if change_type is add/modify)",
  "previous_value": "old value (if change_type is modify)",
  "required_observable_fields": ["field1", "field2"]
}
```

**Conversion Requirements by Change Type:**
- `add`: Must show acquisition journey (research → decision → acquisition) + multiple usage instances
- `modify`: Must show old value usage → transition trigger → new value usage
- `unchanged`: Must show ongoing usage across different contexts within the window

#### Habits State
```json
{
  "name": "habit_name",
  "current_value": {
    "schedule": {
      "frequency_type": "daily | weekly | biweekly | monthly_by_date | monthly_nth_weekday",
      "...": "required fields based on frequency_type"
    },
    "timing": {
      "start_time": "HH:MM",
      "end_time": "HH:MM"
    },
    "location": "where it happens",
    "priority": "high | medium | low",
    "schedule_dates": ["YYYY-MM-DD", ...]
  },
  "change_type": "acquire | adjust | drop | unchanged",
  "change_reason": "why this changed (if change_type is acquire/adjust/drop)",
  "previous_value": {...},
  "required_observable_fields": ["field1", "field2"]
}
```

**Conversion Requirements by Change Type:**
- `acquire`: Must show motivation discovery → habit design → initial struggles → stabilization
- `adjust`: Must show dissatisfaction with old pattern → adjustment reasoning → new pattern execution
- `drop`: Must show discontinuation trigger → no more pattern execution
- `unchanged`: Must show consistent execution with natural variation (not robotic repetition)

#### Preferences State
```json
{
  "name": "preference_name",
  "current_value": {
    "statement": "description of the preference"
  },
  "change_type": "refine | shift | unchanged",
  "change_reason": "why this changed (if change_type is refine/shift)",
  "previous_value": {...},
  "required_observable_fields": ["field1", "field2"]
}
```

**Conversion Requirements by Change Type:**
- `refine/shift`: Must show old preference in action → catalyst event → experimentation → new preference dominance
- `unchanged`: Must show preference through CHOICES (selecting A over B), not just using A

---

## 4. Output Schema

### 4.1 Event Chain JSON Structure
```json
{
  "window_id": "copy from input",
  "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "event_chains": [
    {
      "converted_state_items": [
        {
          "state_category": "user_attributes_state | habits_state | preferences_state",
          "state_name": "exact name from input state",
          "change_type": "the operation type from input (add | modify | unchanged | acquire | adjust | drop | refine | shift)",
          "current_value": "copy the full current_value from input state",
          "previous_value": "copy the full previous_value from input state if exists, otherwise null",
          "change_reason": "copy the change_reason from input state if exists, otherwise null",
          "required_observable_fields": ["list of fields that MUST be evidenced by events"]
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
          "app_name": "from available apps (for single-variation events)",
          "api_name": "from available APIs (for single-variation events)",
          "app_api_variations": [
            {"app_name": "App1", "api_name": "API1"},
            {"app_name": "App2", "api_name": "API2"}
          ],
          "user_intent": "detailed explanation following the composition structure",
          "evidence_for_states": [
            {
              "state_name": "name of the state this event supports",
              "evidenced_fields": ["field1", "field2"]
            }
          ]
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

### 4.3 App/API Specification Rules

There are two ways to specify which app and API an event uses:

#### Option A: Single App/API (for uniform events)
Use when the event always uses the same app and API on every occurrence:
```json
{
  "app_name": "Fitbit",
  "api_name": "SyncDevice",
  "user_intent": "..."
}
```

#### Option B: App/API Variations Set (for events with natural variation)
Use when the habit/behavior naturally manifests through different apps/APIs on different occasions. Provide a SET of valid app-api combinations that represent how this event can occur:

```json
{
  "app_api_variations": [
    {"app_name": "LinkedIn", "api_name": "GetFeed"},
    {"app_name": "Google", "api_name": "Search"}
  ],
  "user_intent": "..."
}
```

**Important Rules:**
1. Use EITHER `app_name`+`api_name` OR `app_api_variations`, never both
2. `app_api_variations` is a SET (unique combinations only, no duplicates)
3. Each dict in `app_api_variations` must have a valid app-api pairing (the api must belong to that app)
4. The downstream code will sample from this set to assign specific app/api to each `schedule_dates` entry
5. The `user_intent` should explain how different app/api combinations serve the habit's purpose

### 4.4 Realistic Habit Variation Guidelines

**Real humans don't perform habits identically every time.** A habit like "industry tech reading" might manifest as:
- Some days: scrolling LinkedIn feed for industry updates
- Other days: searching Google for specific technical content
- Occasionally: reading a book on Goodreads about the topic

**When to use `app_api_variations` vs single `app_name`/`api_name`:**

| Use Single App/API | Use `app_api_variations` |
|--------------------|--------------------------|
| Daily step sync to Fitbit (truly uniform) | News/industry reading (multiple sources) |
| Checking bank balance (specific need) | Learning new skills (different modalities) |
| Posting to specific platform | Social connection maintenance (multi-platform) |
| Single purchase event | Financial monitoring (checking different accounts) |

**Common variation patterns for habits:**

| Habit Type | Realistic Variations |
|------------|---------------------|
| News/Industry reading | `[{LinkedIn, GetFeed}, {Google, Search}]` |
| Learning/Skill development | `[{LLM Assistant, ContinueConversation}, {Google, Search}, {Goodreads, SearchBooks}]` |
| Social connection maintenance | `[{WhatsApp, SendMessage}, {Instagram, LikePost}, {LinkedIn, CommentOnPost}]` |
| Financial monitoring | `[{Chase, GetBalance}, {Chase, GetTransactions}, {Robinhood, GetPortfolio}]` |
| Content consumption | `[{Netflix, PlayContent}, {Spotify, PlaySong}]` |

---

### 4.5 Required Observable Fields

The input `domain_window_state` already contains `required_observable_fields` for each state item. Your task is to ensure that **every field listed in `required_observable_fields` is evidenced by at least one event**.

#### 4.5.1 Field Reference Format

Fields use dot notation to reference nested structures in the state item:

| Field Reference | What It Points To | Example Value |
|-----------------|-------------------|---------------|
| `current_value` | The entire current_value (for simple string values) | `"GitHub Copilot Business subscription..."` |
| `current_value.timing` | The timing object in a habit | `{"start_time": "13:00", "end_time": "14:30"}` |
| `current_value.schedule` | The schedule object in a habit | `{"frequency_type": "weekly", "days_of_week": [0]}` |
| `current_value.location` | The location field in a habit | `"home office desk"` |
| `current_value.statement` | The statement field in a preference | `"Prefers a balanced approach..."` |
| `previous_value` | The entire previous_value (for simple values) | `"9 months of essential living expenses"` |
| `previous_value.timing` | The previous timing (for habit adjustments) | `{"start_time": "20:00", "end_time": "21:30"}` |
| `previous_value.statement` | The previous preference statement | `"Prefers purchasing high-quality..."` |
| `change_reason` | Why this state changed | `"Rescheduled to resolve scheduling conflicts..."` |

#### 4.5.2 Interpreting Required Fields by State Category

**User Attributes State:**
```json
{
  "name": "financial_buffer_status",
  "current_value": "Liquid cash reserves increased from 9 to 12 months...",
  "change_type": "modify",
  "change_reason": "Deliberately increased liquidity to mitigate potential market volatility...",
  "previous_value": "ING Savings account holding approximately 9 months...",
  "required_observable_fields": ["current_value", "change_reason"]
}
```
→ Events must show: (1) the new 12-month buffer in action, (2) the trigger related to market volatility concerns

**Habits State:**
```json
{
  "name": "financial_reconciliation",
  "current_value": {
    "schedule": {"frequency_type": "monthly_nth_weekday", "week_of_month": 1, "day_of_week": 0},
    "timing": {"start_time": "13:00", "end_time": "14:30"},
    "location": "home office desk",
    "priority": "high",
    "schedule_dates": ["2024-10-07", "2024-11-04", "2024-12-02"]
  },
  "change_type": "adjust",
  "change_reason": "Rescheduled to the first Monday of the month (13:00-14:30) to resolve scheduling conflicts...",
  "previous_value": {
    "timing": {"start_time": "20:00", "end_time": "21:30"},
    ...
  },
  "required_observable_fields": ["current_value.timing", "current_value.schedule", "change_reason"]
}
```
→ Events must show: (1) the new 13:00-14:30 timing, (2) the first-Monday-of-month schedule pattern, (3) why the schedule was changed

**Preferences State:**
```json
{
  "name": "spending_logic",
  "current_value": {
    "statement": "Prefers a balanced approach between high-quality material acquisitions and intentional charitable giving..."
  },
  "change_type": "shift",
  "change_reason": "A year of significant global events and the symbolic reopening of Notre Dame shifted his perspective...",
  "previous_value": {
    "statement": "Prefers purchasing high-quality, durable electronics and appliances..."
  },
  "required_observable_fields": ["current_value.statement", "change_reason"]
}
```
→ Events must show: (1) choices demonstrating the new balanced approach (material + charitable), (2) the catalyst related to global events/Notre Dame

#### 4.5.3 Copy Required Fields to Output

When generating `converted_state_items`, copy the `required_observable_fields` exactly from the input:
```json
{
  "converted_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "financial_reconciliation",
      "change_type": "adjust",
      "current_value": { ... },
      "previous_value": { ... },
      "change_reason": "Rescheduled to the first Monday...",
      "required_observable_fields": ["current_value.timing", "current_value.schedule", "change_reason"]
    }
  ]
}
```

### 4.6 Evidence Annotation Rules

Each event must specify which state(s) and which fields it provides evidence for via the `evidence_for_states` array.

#### 4.6.1 Structure
```json
"evidence_for_states": [
  {
    "state_name": "exact state name from converted_state_items",
    "evidenced_fields": ["field1", "field2", ...]
  }
]
```

#### 4.6.2 Rules

1. **Every event must have at least one entry** in `evidence_for_states`
2. **Field names must exactly match** those in `required_observable_fields`
3. **An event can evidence multiple fields** from the same state
4. **An event can evidence fields from multiple states** (in merged chains)
5. **All `required_observable_fields` must be covered** by at least one event

#### 4.6.3 Field Evidence Guidelines

| Field Type | How to Evidence | Example App/API |
|------------|-----------------|-----------------|
| `change_reason` | Event showing the trigger or catalyst | Gmail:ReadEmail, Google:Search, LLM Assistant:ContinueConversation |
| `current_value` (attribute) | Event demonstrating possession/usage | Any app showing the attribute in use |
| `current_value.timing` | Event occurring at the specified time | Fitbit:LogWorkout, Notion:CreateDatabaseEntry, Chase:GetTransactions |
| `current_value.schedule` | Events on the correct schedule_dates pattern | Repeated execution events matching frequency |
| `current_value.location` | Event with location context | Fitbit:LogWorkout, Fitbit:RecordActivity, Google Maps:CheckIn, Instagram:CreatePost, UberEats:PlaceOrder |
| `current_value.statement` (preference) | Event showing choice of A over B | Any event where alternative existed but wasn't chosen |
| `previous_value` (any) | Event showing old behavior BEFORE the change | Events early in window demonstrating old pattern |
| `previous_value.timing` | Event at the OLD time (for habit adjustments) | Early window events at previous schedule |
| `previous_value.statement` | Event showing old preference in action | Early window choices aligned with old preference |

#### 4.6.4 Location Evidence Apps

When you need to evidence a `location` field, use these app/api combinations:

| Scenario | App | API | Example user_intent |
|----------|-----|-----|---------------------|
| Workout location | Fitbit | LogWorkout | "Logging 30-min strength training session at Basic-Fit Eindhoven Strijp" |
| Outdoor activity route | Fitbit | RecordActivity | "Recording 5km morning run through Vondelpark, Amsterdam" |
| General check-in | Google Maps | CheckIn | "Checking in at WeWork Metropool coworking space" |
| Photo with location | Instagram | CreatePost | "Posting gym selfie at Virgin Active with location tag" |
| Food delivery | UberEats | PlaceOrder | "Ordering lunch to home office at [address]" |
| Navigation arrival | Google Maps | GetDirections | "Getting directions from home to office gym for lunch workout" |

#### 4.6.5 Examples

**Single state, single field:**
```json
{
  "app_name": "Gmail",
  "api_name": "ReadEmail",
  "user_intent": "Trigger: reading doctor's email recommending cardio exercise after annual checkup showed elevated resting heart rate",
  "evidence_for_states": [
    {
      "state_name": "morning_run_habit",
      "evidenced_fields": ["change_reason"]
    }
  ]
}
```

**Single state, multiple fields:**
```json
{
  "app_name": "Fitbit",
  "api_name": "LogWorkout",
  "user_intent": "Habit execution: logging 30-minute morning run (5.2km) at Vondelpark before 7am standup",
  "evidence_for_states": [
    {
      "state_name": "morning_run_habit",
      "evidenced_fields": ["current_value.schedule.timing", "current_value.location"]
    }
  ]
}
```

**Multiple states (merged chain):**
```json
{
  "app_name": "LLM Assistant",
  "api_name": "CreateConversation",
  "user_intent": "Catalyst moment: starting interactive session to refactor state machine, choosing to 'build with AI' rather than reading legacy documentation",
  "evidence_for_states": [
    {
      "state_name": "technical_skills_inventory",
      "evidenced_fields": ["current_value"]
    },
    {
      "state_name": "learning_style",
      "evidenced_fields": ["change_reason", "current_value.statement"]
    }
  ]
}
```

## 5. Available Apps & APIs

{{ app_catalog_json }}

---

## 6. Generation Strategy by State Type

### 6.1 User Attributes Generation

#### Change Type: `add`

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

#### Change Type: `modify`

**Goal:** Show the transition narrative from old to new value.

**Required Event Phases:**
1. Old value in use (1-2 events early in window)
2. Trigger for change (1-2 events)
3. Transition process (1-2 events)
4. New value in use (2-3 events)

#### Change Type: `unchanged`

**Goal:** Demonstrate ongoing presence through varied usage contexts.

**Requirements:**
- Generate 3-5 usage events spread across the window
- Each event should show the attribute in a DIFFERENT context/scenario
- Avoid repetitive events; real usage is contextually varied

### 6.2 Habits Generation

#### Change Type: `acquire`

**Goal:** Show how the habit was born and became established.

**Required Event Phases:**

1. **Motivation Discovery** (1-2 events)
   - What problem or aspiration triggered habit consideration?
   - Observable through: searches, AI conversations, social discussions

2. **Habit Design** (1-2 events)
   - How did user structure the habit?
   - Observable through: calendar/reminder setup, tracker creation, goal setting

3. **Execution Events** (single event with multiple dates AND realistic variation)
   - The recurring execution of the habit
   - Use `schedule_dates` array with `start_time` and `end_time`
   - **Use `app_api_variations` to define the set of ways this habit can manifest**
   - Include `note` describing evolution: early phase (struggle/learning) → mid phase (stabilizing) → late phase (automatic/optimizing)

**Critical:** Establishment events must occur BEFORE the first `schedule_dates` entry.

#### Change Type: `adjust`

**Goal:** Show why and how the habit pattern changed.

**Required Event Phases:**
1. Old pattern execution (early dates showing previous pattern)
2. Adjustment trigger (what prompted the change)
3. Modification action (updating schedule/approach)
4. New pattern execution (later dates showing adjusted pattern)

#### Change Type: `drop`

**Goal:** Show the habit being discontinued.

**Required Event Phases:**
1. Discontinuation trigger (what prompted the change)
2. Discontinuation action (stopping the habit)

#### Change Type: `unchanged`

**Goal:** Show consistent but human execution with natural variation.

**Requirements:**
- Single event with `schedule_dates` covering all occurrences
- **Use `app_api_variations` when the habit naturally involves different tools/methods**
- `user_intent` should describe what happens during execution with enough detail to understand the habit's nature AND explain why different apps/apis serve different aspects
- `note` should describe any natural evolution within the unchanged habit

**CRITICAL for unchanged habits:** Do NOT generate robotic, identical repetitions. Real humans:
- Read news from different sources on different days
- Exercise with different activities within a fitness routine
- Check finances through different apps depending on what they need to know
- Learn through different modalities (reading, watching, doing, discussing)

### 6.3 Preferences Generation

#### Change Type: `refine` or `shift`

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

#### Change Type: `unchanged`

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

### 7.3 Evidence Coverage for Merged Chains

When merging multiple state items into one chain:

1. **Each state's `required_observable_fields` must be fully covered**
2. **One event can evidence fields from multiple states** — use multiple entries in `evidence_for_states`
3. **Verify coverage separately for each state** — merging doesn't reduce evidence requirements

Example validation for a merged chain with 2 states:
```
State A required_observable_fields: [field1, field2, field3]
State B required_observable_fields: [fieldX, fieldY]

Event 1: evidences A.field1, B.fieldX
Event 2: evidences A.field2
Event 3: evidences A.field3, B.fieldY

✓ State A fully covered: field1 ✓, field2 ✓, field3 ✓
✓ State B fully covered: fieldX ✓, fieldY ✓
```

---

## 8. Comprehensive Examples
### Example 1: Attribute Add — Full Acquisition Journey with Evidence Annotation

**Input State Item:**
```json
{
  "name": "ai_coding_assistant_subscription",
  "current_value": "GitHub Copilot Business subscription, integrated with VS Code and JetBrains IDEs",
  "op": "add",
  "change_reason": "Team lead mandated AI tool adoption after Q4 productivity review showed 20% lag behind industry benchmarks",
  "previous_value": null
}
```

**Generated Event Chain:**
```json
{
  "converted_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "ai_coding_assistant_subscription",
      "change_type": "add",
      "current_value": "GitHub Copilot Business subscription, integrated with VS Code and JetBrains IDEs",
      "previous_value": null,
      "change_reason": "Team lead mandated AI tool adoption after Q4 productivity review showed 20% lag behind industry benchmarks",
      "required_observable_fields": ["current_value", "change_reason"]
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
      "user_intent": "Change reason trigger: reading team lead's email summarizing Q4 productivity review results; noting the 20% lag statistic and mandatory AI tool adoption directive effective Q1",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["change_reason"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-03"],
        "time": "21:15:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Initial research: searching 'best AI coding assistants 2024 comparison' to understand landscape before team discussion; skeptical about productivity claims but recognizing need to comply with directive",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["current_value"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "12:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Deep evaluation: asking Claude to compare GitHub Copilot vs Cursor vs Amazon CodeWhisperer specifically for C++ embedded systems development; concerned about legacy codebase compatibility and offline functionality for secure environments",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["current_value"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "14:00:00"
      },
      "app_name": "WhatsApp",
      "api_name": "SendMessage",
      "user_intent": "Social validation: messaging senior colleague who adopted Copilot last quarter asking about real-world experience with firmware codebases; specifically asking about false positive rate in suggestions and learning curve",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["current_value"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-08"],
        "time": "11:00:00"
      },
      "app_name": "Gmail",
      "api_name": "SendEmail",
      "user_intent": "Acquisition action: emailing IT department to request GitHub Copilot Business license activation; cc'ing team lead to document compliance with Q1 directive",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["current_value"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-15"],
        "time": "10:00:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Integration usage: using AI assistant to debug a Copilot-generated state machine that had subtle race condition; learning to verify AI suggestions rather than blindly accepting",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["current_value"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-02-05"],
        "time": "11:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Mature usage: confidently using AI to generate complete test harness for new motor control module; has developed intuition for when to trust vs verify suggestions based on code complexity",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["current_value"]
        }
      ]
    }
  ]
}
```

**Evidence Coverage Verification:**
| Required Field | Evidenced By |
|----------------|--------------|
| `change_reason` | w1_e001 ✓ |
| `current_value` | w1_e002, w1_e003, w1_e004, w1_e005, w1_e006, w1_e007 ✓ |

---

### Example 2: Habit Adjust — Schedule Change with Timing and Reason Evidence

**Input State Item:**
```json
{
  "name": "financial_reconciliation",
  "current_value": {
    "schedule": {
      "frequency_type": "monthly_nth_weekday",
      "week_of_month": 1,
      "day_of_week": 0
    },
    "timing": {
      "start_time": "13:00",
      "end_time": "14:30"
    },
    "location": "home office desk",
    "priority": "high",
    "schedule_dates": ["2024-10-07", "2024-11-04", "2024-12-02"]
  },
  "change_type": "adjust",
  "change_reason": "Rescheduled the monthly financial reconciliation to the first Monday of the month (13:00-14:30) to resolve scheduling conflicts with evening leisure, professional reading, and family traditions.",
  "previous_value": {
    "schedule": {
      "frequency_type": "monthly_by_date",
      "days_of_month": [1]
    },
    "timing": {
      "start_time": "20:00",
      "end_time": "21:30"
    },
    "location": "home office desk",
    "priority": "high",
    "schedule_dates": ["2024-10-01", "2024-11-01", "2024-12-01"]
  },
  "required_observable_fields": ["current_value.timing", "current_value.schedule", "change_reason"]
}
```

**Generated Event Chain:**
```json
{
  "converted_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "financial_reconciliation",
      "change_type": "adjust",
      "current_value": {
        "schedule": {
          "frequency_type": "monthly_nth_weekday",
          "week_of_month": 1,
          "day_of_week": 0
        },
        "timing": {
          "start_time": "13:00",
          "end_time": "14:30"
        },
        "location": "home office desk",
        "priority": "high",
        "schedule_dates": ["2024-10-07", "2024-11-04", "2024-12-02"]
      },
      "previous_value": {
        "schedule": {
          "frequency_type": "monthly_by_date",
          "days_of_month": [1]
        },
        "timing": {
          "start_time": "20:00",
          "end_time": "21:30"
        },
        "location": "home office desk",
        "priority": "high",
        "schedule_dates": ["2024-10-01", "2024-11-01", "2024-12-01"]
      },
      "change_reason": "Rescheduled the monthly financial reconciliation to the first Monday of the month (13:00-14:30) to resolve scheduling conflicts with evening leisure, professional reading, and family traditions.",
      "required_observable_fields": ["current_value.timing", "current_value.schedule", "change_reason"]
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-10-01"],
        "time": "19:45:00"
      },
      "app_name": "Notion",
      "api_name": "UpdatePage",
      "user_intent": "Schedule conflict trigger: attempting to start monthly financial reconciliation at 20:00 but realizing tonight conflicts with the season premiere of a show planned with partner; noting this is the third month in a row where the 1st-of-month evening slot created friction with personal plans.",
      "evidence_for_states": [
        {
          "state_name": "financial_reconciliation",
          "evidenced_fields": ["change_reason"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-10-02"],
        "time": "12:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Schedule optimization: asking Claude to help find a better recurring time slot for monthly financial review that avoids evening leisure time, professional reading habits, and potential holiday conflicts; considering moving to a weekend afternoon or weekday lunch break.",
      "evidence_for_states": [
        {
          "state_name": "financial_reconciliation",
          "evidenced_fields": ["change_reason"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-10-07", "2024-11-04", "2024-12-02"],
        "start_time": "13:00:00",
        "end_time": "14:30:00",
        "note": "New schedule: first Monday of each month, 13:00-14:30. October session focuses on setting up the new routine; November and December sessions run smoothly with no conflicts."
      },
      "app_name": "Chase",
      "api_name": "GetTransactions",
      "user_intent": "Habit execution with new schedule: performing monthly financial reconciliation during Monday lunch break (13:00-14:30) at home office desk; reviewing Chase transactions, categorizing expenses, and updating budget tracker in Notion. The new daytime slot eliminates evening conflicts and allows uninterrupted focus.",
      "evidence_for_states": [
        {
          "state_name": "financial_reconciliation",
          "evidenced_fields": ["current_value.timing", "current_value.schedule"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-10-07", "2024-11-04", "2024-12-02"],
        "start_time": "13:30:00",
        "end_time": "14:30:00",
        "note": "Part of the reconciliation session - updating records after reviewing transactions"
      },
      "app_name": "Notion",
      "api_name": "UpdatePage",
      "user_intent": "Reconciliation documentation: updating monthly budget tracker with categorized expenses from Chase review; comparing actual spending against budget allocations and noting any variances for next month's planning.",
      "evidence_for_states": [
        {
          "state_name": "financial_reconciliation",
          "evidenced_fields": ["current_value.timing", "current_value.schedule"]
        }
      ]
    }
  ]
}
```

**Evidence Coverage Verification:**
| Required Field | Evidenced By |
|----------------|--------------|
| `current_value.timing` | w4_e003 (13:00-14:30), w4_e004 (13:30-14:30) ✓ |
| `current_value.schedule` | w4_e003, w4_e004 (first Monday pattern: Oct 7, Nov 4, Dec 2) ✓ |
| `change_reason` | w4_e001 (conflict trigger), w4_e002 (optimization discussion) ✓ |

**Why This Example Works:**
- Shows the OLD pattern conflict (w4_e001: attempting at 20:00 on the 1st)
- Shows the reasoning process (w4_e002: asking for help finding better slot)
- Shows the NEW pattern execution (w4_e003, w4_e004: 13:00-14:30 on first Monday)
- `change_reason` explicitly mentions evening conflicts → w4_e001 demonstrates this

---

### Example 3: Unchanged Habit with Realistic Variation using `app_api_variations`

**Input State Item:**
```json
{
  "name": "industry_tech_reading",
  "current_value": {
    "schedule": {
      "frequency_type": "weekly",
      "days_of_week": [1, 3]
    },
    "timing": {
      "start_time": "20:30",
      "end_time": "21:30"
    },
    "location": "home living room",
    "priority": "medium",
    "schedule_dates": ["2024-01-02", "2024-01-04", "2024-01-09", "2024-01-11", "2024-01-16", "2024-01-18", "2024-01-23", "2024-01-25", "2024-01-30"]
  },
  "change_type": "unchanged",
  "change_reason": null,
  "previous_value": null
}
```

**Generated Event Chain:**
```json
{
  "converted_state_items": [
    {
      "state_category": "habits_state",
      "state_name": "industry_tech_reading",
      "change_type": "unchanged",
      "current_value": {
        "schedule": {
          "frequency_type": "weekly",
          "days_of_week": [1, 3]
        },
        "timing": {
          "start_time": "20:30",
          "end_time": "21:30"
        },
        "location": "home living room",
        "priority": "medium",
        "schedule_dates": ["2024-01-02", "2024-01-04", "2024-01-09", "2024-01-11", "2024-01-16", "2024-01-18", "2024-01-23", "2024-01-25", "2024-01-30"]
      },
      "previous_value": null,
      "change_reason": null,
      "required_observable_fields": [
        "current_value.schedule.timing",
        "current_value.schedule.frequency",
        "current_value.location"
      ]
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": [
          "2024-01-02", "2024-01-04", "2024-01-09", "2024-01-11",
          "2024-01-16", "2024-01-18", "2024-01-23", "2024-01-25", "2024-01-30"
        ],
        "start_time": "20:30:00",
        "end_time": "21:30:00",
        "note": "Bi-weekly evening tech reading ritual from the living room couch. Sessions alternate organically between LinkedIn for industry pulse and Google for deeper technical searches."
      },
      "app_api_variations": [
        {"app_name": "LinkedIn", "api_name": "GetFeed"},
        {"app_name": "Google", "api_name": "Search"}
      ],
      "user_intent": "Habit execution: bi-weekly evening technical reading sessions from the living room couch. LinkedIn sessions focus on industry pulse—scrolling through posts from embedded systems engineers, catching announcements about Zephyr RTOS updates and industrial automation trends. Google sessions pivot to deep technical content—searching for specific topics like 'ARM Cortex-M7 cache optimization techniques' or 'FreeRTOS vs Zephyr comparison 2024'.",
      "evidence_for_states": [
        {
          "state_name": "industry_tech_reading",
          "evidenced_fields": ["current_value.schedule.timing", "current_value.schedule.frequency", "current_value.location"]
        }
      ]
    }
  ]
}
```

**Evidence Coverage Verification:**
| Required Field | Evidenced By |
|----------------|--------------|
| `current_value.schedule.timing` | w1_e001 (start_time/end_time) ✓ |
| `current_value.schedule.frequency` | w1_e001 (schedule_dates pattern) ✓ |
| `current_value.location` | w1_e001 (user_intent: "living room couch") ✓ |

---

### Example 4: Merged Chain — Two States with Shared Events

**Input State Items:**
```json
[
  {
    "name": "technical_skills_inventory",
    "current_value": "AI-assisted development (GitHub Copilot and ChatGPT-4 for unit test generation)",
    "op": "add",
    "change_reason": "Exploring productivity gains through AI tools and improving data processing capabilities for robotics telemetry.",
    "previous_value": null
  },
  {
    "name": "learning_style",
    "current_value": {
      "statement": "Strongly prefers hands-on project-based learning, now specifically favoring interactive AI-pair programming as a primary exploration method for new technologies"
    },
    "op": "refine",
    "change_reason": "Discovered that AI-assisted coding significantly accelerates the 'doing' phase of project-based learning.",
    "previous_value": {
      "statement": "Prefers hands-on, hardware-in-the-loop experimentation over purely theoretical documentation or video-based tutorials"
    }
  }
]
```

**Generated Event Chain:**
```json
{
  "converted_state_items": [
    {
      "state_category": "user_attributes_state",
      "state_name": "technical_skills_inventory",
      "change_type": "add",
      "current_value": "AI-assisted development (GitHub Copilot and ChatGPT-4 for unit test generation)",
      "previous_value": null,
      "change_reason": "Exploring productivity gains through AI tools and improving data processing capabilities for robotics telemetry.",
      "required_observable_fields": ["current_value", "change_reason"]
    },
    {
      "state_category": "preferences_state",
      "state_name": "learning_style",
      "change_type": "refine",
      "current_value": {
        "statement": "Strongly prefers hands-on project-based learning, now specifically favoring interactive AI-pair programming as a primary exploration method for new technologies"
      },
      "previous_value": {
        "statement": "Prefers hands-on, hardware-in-the-loop experimentation over purely theoretical documentation or video-based tutorials"
      },
      "change_reason": "Discovered that AI-assisted coding significantly accelerates the 'doing' phase of project-based learning.",
      "required_observable_fields": ["current_value.statement", "previous_value.statement", "change_reason"]
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-02"],
        "time": "10:15:00"
      },
      "app_name": "LinkedIn",
      "api_name": "GetFeed",
      "user_intent": "Trigger moment: scrolling through CES 2024 recap posts on LinkedIn and noticing the heavy emphasis on Generative AI for industrial code generation; sparking curiosity about how these tools could optimize embedded firmware workflow and improve productivity.",
      "evidence_for_states": [
        {
          "state_name": "technical_skills_inventory",
          "evidenced_fields": ["change_reason"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-02"],
        "time": "20:30:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Initial research: searching 'GitHub Copilot for embedded C++ performance optimization' and 'ChatGPT-4 unit test generation for real-time systems' to evaluate if AI tools can handle deterministic RTOS constraints.",
      "evidence_for_states": [
        {
          "state_name": "technical_skills_inventory",
          "evidenced_fields": ["current_value"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-04"],
        "time": "19:00:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Old preference demonstration: searching 'finite state machine design patterns embedded systems PDF' to find comprehensive documentation before implementing motor control FSM; following established pattern of reading documentation first before hands-on implementation.",
      "evidence_for_states": [
        {
          "state_name": "learning_style",
          "evidenced_fields": ["previous_value.statement"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-05"],
        "time": "15:30:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "CreateConversation",
      "user_intent": "Catalyst moment for preference shift: starting an interactive session to refactor a complex state machine implementation; choosing to 'build with AI' rather than spending hours reading the legacy documentation for the old motion control module—a deliberate departure from usual documentation-first approach.",
      "evidence_for_states": [
        {
          "state_name": "technical_skills_inventory",
          "evidenced_fields": ["current_value"]
        },
        {
          "state_name": "learning_style",
          "evidenced_fields": ["change_reason", "current_value.statement"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-01-19"],
        "time": "16:00:00"
      },
      "app_name": "LLM Assistant",
      "api_name": "ContinueConversation",
      "user_intent": "Demonstrating new preference and skill: prompt-engineering a complex Python test script using ChatGPT to process robot telemetry instead of searching StackOverflow; finding that the interactive dialogue accelerates the learning of NumPy much faster than static tutorials.",
      "evidence_for_states": [
        {
          "state_name": "technical_skills_inventory",
          "evidenced_fields": ["current_value", "change_reason"]
        },
        {
          "state_name": "learning_style",
          "evidenced_fields": ["current_value.statement", "change_reason"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-03-01"],
        "time": "15:45:00"
      },
      "app_name": "Notion",
      "api_name": "CreatePage",
      "user_intent": "Outcome documentation: creating a 'Toy Project' log where documenting the limits of AI-generated ARM assembly; reflecting on how the shift to AI-pair programming has halved the time from 'idea' to 'running code' on hardware—confirming that this interactive approach accelerates the 'doing' phase of learning.",
      "evidence_for_states": [
        {
          "state_name": "technical_skills_inventory",
          "evidenced_fields": ["current_value", "change_reason"]
        },
        {
          "state_name": "learning_style",
          "evidenced_fields": ["current_value.statement", "change_reason"]
        }
      ]
    }
  ]
}
```

**Evidence Coverage Verification:**

**State: technical_skills_inventory**
| Required Field | Evidenced By |
|----------------|--------------|
| `current_value` | w1_e002, w1_e004, w1_e005, w1_e006 ✓ |
| `change_reason` | w1_e001 (trigger: productivity gains), w1_e005 (telemetry processing), w1_e006 (outcome) ✓ |

**State: learning_style**
| Required Field | Evidenced By |
|----------------|--------------|
| `current_value.statement` | w1_e004, w1_e005, w1_e006 ✓ |
| `previous_value.statement` | w1_e003 ✓ |
| `change_reason` | w1_e004 (catalyst), w1_e005 (discovery), w1_e006 (reflection) ✓ |

### Example 5: Preference Shift — From Material-Only to Balanced Spending

**Input State Item:**
```json
{
  "name": "spending_logic",
  "current_value": {
    "statement": "Prefers a balanced approach between high-quality material acquisitions and intentional charitable giving, valuing social impact alongside durability"
  },
  "change_type": "shift",
  "change_reason": "A year of significant global events and the symbolic reopening of Notre Dame shifted his perspective toward more balanced spending that includes intentional charitable impact.",
  "previous_value": {
    "statement": "Prefers purchasing high-quality, durable electronics and appliances over budget-friendly but short-lived alternatives"
  },
  "required_observable_fields": ["current_value.statement", "change_reason"]
}
```

**Generated Event Chain:**
```json
{
  "converted_state_items": [
    {
      "state_category": "preferences_state",
      "state_name": "spending_logic",
      "change_type": "shift",
      "current_value": {
        "statement": "Prefers a balanced approach between high-quality material acquisitions and intentional charitable giving, valuing social impact alongside durability"
      },
      "previous_value": {
        "statement": "Prefers purchasing high-quality, durable electronics and appliances over budget-friendly but short-lived alternatives"
      },
      "change_reason": "A year of significant global events and the symbolic reopening of Notre Dame shifted his perspective toward more balanced spending that includes intentional charitable impact.",
      "required_observable_fields": ["current_value.statement", "change_reason"]
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-12-07"],
        "time": "20:30:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Catalyst moment: searching 'Notre Dame cathedral reopening December 2024' after seeing news coverage; feeling moved by the global community effort to restore a cultural landmark and reflecting on the power of collective charitable action.",
      "evidence_for_states": [
        {
          "state_name": "spending_logic",
          "evidenced_fields": ["change_reason"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-12-08"],
        "time": "14:00:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Preference shift exploration: searching 'Friends of Notre Dame de Paris donation' to find official channels for contributing to the cathedral's ongoing preservation; wanting to participate in this historic moment rather than just observing.",
      "evidence_for_states": [
        {
          "state_name": "spending_logic",
          "evidenced_fields": ["current_value.statement", "change_reason"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-12-08"],
        "time": "14:30:00"
      },
      "app_name": "Chase",
      "api_name": "PayBill",
      "user_intent": "New preference demonstration: making commemorative donation to Friends of Notre Dame de Paris; consciously allocating funds that might have gone toward a material purchase (was considering new wireless earbuds) toward charitable impact instead.",
      "evidence_for_states": [
        {
          "state_name": "spending_logic",
          "evidenced_fields": ["current_value.statement"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-12-20"],
        "time": "19:00:00"
      },
      "app_name": "Google",
      "api_name": "Search",
      "user_intent": "Balanced approach in action: searching 'Stichting Vluchteling year-end donation' for holiday charitable giving; integrating humanitarian support into year-end spending alongside planned material gifts for family.",
      "evidence_for_states": [
        {
          "state_name": "spending_logic",
          "evidenced_fields": ["current_value.statement"]
        }
      ]
    },
    {
      "time_specification": {
        "schedule_dates": ["2024-12-20"],
        "time": "19:30:00"
      },
      "app_name": "Chase",
      "api_name": "PayBill",
      "user_intent": "New preference reinforcement: completing year-end donation to Stichting Vluchteling; reflecting that this year's holiday spending now balances quality gifts (material) with meaningful charitable contributions (social impact)—a shift from previous years' material-only focus.",
      "evidence_for_states": [
        {
          "state_name": "spending_logic",
          "evidenced_fields": ["current_value.statement"]
        }
      ]
    }
  ]
}
```

**Evidence Coverage Verification:**
| Required Field | Evidenced By |
|----------------|--------------|
| `current_value.statement` | w4_e002, w4_e003, w4_e004, w4_e005 (multiple events showing balanced material + charitable choices) ✓ |
| `change_reason` | w4_e001 (Notre Dame catalyst), w4_e002 (connecting to global events) ✓ |

**Why This Example Works:**
- `change_reason` mentions "Notre Dame reopening" → w4_e001, w4_e002 explicitly connect to this
- `change_reason` mentions "significant global events" → charitable giving demonstrates engagement with global issues
- `current_value.statement` mentions "balanced approach" → events show BOTH material considerations AND charitable giving
- w4_e003 explicitly shows the choice to donate instead of buying earbuds → demonstrates preference shift from material-only
---

## 9. Validation Checklist

Before finalizing output, verify:

### Coverage Validation
- [ ] Every `user_attributes_state` item appears in at least one chain
- [ ] Every `habits_state` item appears in at least one chain
- [ ] Every `preferences_state` item appears in at least one chain

### Converted State Items Validation
- [ ] Each `converted_state_items` entry contains: state_category, state_name, change_type, current_value, previous_value, change_reason, required_observable_fields
- [ ] `current_value` is copied exactly from input state
- [ ] `previous_value` is copied exactly from input state (or null if not present)
- [ ] `change_reason` is copied exactly from input state (or null if not present)
- [ ] `required_observable_fields` is copied exactly from input state
- [ ] `change_type` matches the input (add | modify | unchanged | acquire | adjust | drop | refine | shift)

### Lossless Conversion Validation
For EACH state item, confirm:
- [ ] Could someone reconstruct the state item from ONLY the events?
- [ ] Is the `change_reason` (if present) observable through at least one event?
- [ ] For `add/acquire`: Are both process AND outcome events present?
- [ ] For `drop`: Is the habit clearly discontinued with no more pattern execution?
- [ ] For `unchanged`: Is the state demonstrated across multiple contexts?
- [ ] For preferences: Are CHOICES shown (A over B), not just usage of A?

### Habit Variation Validation (CRITICAL)
- [ ] Unchanged/recurring habits with natural variation use `app_api_variations`
- [ ] Each entry in `app_api_variations` is a valid {app_name, api_name} pair
- [ ] No duplicate entries in `app_api_variations` (it's a set)
- [ ] The `user_intent` explains what each variation serves
- [ ] Truly uniform habits (same action every time) use single `app_name`/`api_name`

### App/API Specification Validation
- [ ] Each event uses EITHER (`app_name` + `api_name`) OR `app_api_variations`, never both
- [ ] All `app_name` values match available apps
- [ ] All `api_name` values match available APIs for their corresponding app
- [ ] In `app_api_variations`, each api_name belongs to its paired app_name

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
- [ ] No forbidden field combinations (e.g., both app_name and app_api_variations in same event)

---

## 10. Final Instructions

1. **Read the input state carefully** — understand each item's full meaning including current_value, previous_value, and change_reason
2. **Plan the conversion** — for each state item, what events would make it observable?
3. **For habits: determine if variation exists** — does this habit naturally manifest through different apps/apis? If yes, use `app_api_variations`
4. **Generate detailed events** — each event should pass the "executable" test
5. **Populate converted_state_items fully** — include all fields from the input state
6. **Verify lossless conversion** — could someone reconstruct state from events alone?
7. **Check coverage** — every state item must be represented
8. **Output ONLY the JSON** — no markdown fences, no commentary

**Remember:** Your goal is to create a behavioral trace so realistic and detailed that the original state items become fully observable through the events. Information should flow FROM state TO events with zero loss. Habits should show realistic human variation through `app_api_variations` when natural variation exists.

**Evidence Annotation Reminder:**
- Every event MUST have `evidence_for_states` populated
- Every field in `required_observable_fields` MUST be covered by at least one event
- Use the Evidence Coverage Verification table format shown in examples to double-check your work before outputting
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
