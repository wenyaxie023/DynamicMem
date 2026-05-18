import json
from dataclasses import dataclass

from jinja2 import Template

from app_catalog import APP_CATALOG
from llm_client import (
    GeminiJSONClient,
    LLMResult,
)
from typing import Optional


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

## 2. CRITICAL: Available Apps & APIs - STRICT COMPLIANCE REQUIRED

{{ app_catalog_json }}

### **MANDATORY APP/API USAGE RULES**

**YOU MUST ONLY USE APPS AND APIs FROM THE CATALOG ABOVE. THIS IS NON-NEGOTIABLE.**

1. **Every `app_name` in your output MUST exist in the provided catalog**
2. **Every `api_name` MUST belong to its specified `app_name` according to the catalog**
3. **In `app_api_variations`, every {app_name, api_name} pair MUST be valid per the catalog**
4. **Before using any app/API, verify it exists in the catalog**

### **Default Fallback for Missing Functionality**

**If the user behavior requires functionality not available in the catalog:**
- **USE "LLM Assistant" with "ContinueConversation" or "CreateConversation" as the default fallback**
- The user can express their need through conversational dialogue with the AI assistant
- Example: If there's no dedicated "Meditation App" but the user has a meditation habit, use:
  ```json
  {
    "app_name": "LLM Assistant",
    "api_name": "ContinueConversation",
    "user_intent": "Asking Claude for a guided 10-minute breathing meditation session..."
  }
  ```

**Why this matters:** Using apps/APIs outside the catalog will cause downstream errors. The LLM Assistant can serve as a flexible interface for many user needs through conversational interaction.

---

## 3. Lossless Conversion Framework

### 3.1 What "Lossless" Means in Practice

For each state item, ask yourself: **"If I only saw these events (without the original state), could I accurately reconstruct the state item?"**

**Why this matters:** The events are the observable trace of the user's life. If critical information from the state is missing in the events, we lose valuable behavioral data. The goal is to create a complete behavioral record where every aspect of the state manifests through concrete actions.

| State Aspect | Must Be Observable Through Events | Why It Matters |
|--------------|-----------------------------------|----------------|
| **Attribute value** | Concrete usage demonstrating possession/capability | Shows what the user actually HAS and USES, not just theoretical ownership |
| **Attribute acquisition** | Research, decision-making, and purchase/setup process | Reveals how users make decisions and integrate new things into their lives |
| **Habit schedule** | Events occurring at specified times on specified days | Proves the habit exists through temporal pattern, not just description |
| **Habit evolution** | Early executions differ from later ones (learning curve) | Captures realistic human behavior - we don't master habits instantly |
| **Preference direction** | Choices that favor X over alternatives | Preferences only exist in the context of choices - must see what wasn't chosen |
| **Preference strength** | Consistency and frequency of preference-aligned choices | Strong preferences show up repeatedly across contexts |
| **Change reason** | Events that make the trigger/catalyst observable | Understanding WHY changes happen is crucial for realistic simulation |

### 3.2 Event Detail Requirements

Each event must be specific enough to be "executable" — meaning you could actually perform this action in the real app.

**Why specificity matters:** Generic events like "searching for headphones" don't capture the user's actual mental state, decision criteria, or context. Real people don't just "search" - they search with specific questions, constraints, and goals in mind. This detail is what makes the simulation realistic.

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

### 3.3 User Intent Composition

`user_intent` should provide enough context for downstream content generation. A good user_intent typically includes:

1. **What the user wants to do** (the goal/action)
2. **Why they want to do it** (motivation/trigger/context)
3. **Any specific constraints or criteria** (optional, but helpful for realistic generation)

**Why this structure matters:** Downstream systems need to generate actual content (search queries, messages, prompts). Without sufficient context, they can't create realistic, specific content that matches the user's actual behavior.

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

## 4. Input Schema & State Item Interpretation

### 4.1 Domain Window State Structure

The input state JSON has the following structure:

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

### 4.2 State Item Change Types and Required Observable Fields

Each state item uses a **semantically appropriate structure** for its change type. The `required_observable_fields` specify which aspects of the state MUST be evidenced through events.

#### Unchanged State Item
```json
{
  "name": "item_name",
  "change_type": "unchanged",
  "current_value": {...},
  "required_observable_fields": ["current_value.field1", "current_value.field2"]
}
```
→ Must show ongoing usage/execution across different contexts

#### New Item (add/acquire)
```json
{
  "name": "item_name",
  "change_type": "add | acquire",
  "new_value": {...},
  "change_reason": "why this was added/acquired",
  "required_observable_fields": ["new_value", "change_reason"]
}
```
→ Must show: (1) acquisition journey and usage, (2) the trigger that motivated acquisition

#### Dropped Item (drop)
```json
{
  "name": "item_name",
  "change_type": "drop",
  "dropped_value": {...},
  "change_reason": "why this was dropped",
  "required_observable_fields": ["dropped_value", "change_reason"]
}
```
→ Must show: (1) what was dropped, (2) the trigger that caused discontinuation

#### Modified Item (modify/adjust/shift/refine)
```json
{
  "name": "item_name",
  "change_type": "modify | adjust | shift | refine",
  "delta": {
    "from": {...},
    "to": {...}
  },
  "change_reason": "why this change happened",
  "required_observable_fields": ["delta.to.field1", "delta.from.field2", "change_reason"]
}
```
→ Must show: (1) old value usage, (2) transition trigger, (3) new value usage

### 4.3 Field Reference Format (Dot Notation)

Fields use dot notation to reference nested structures:

| Field Reference | What It Points To | Example |
|-----------------|-------------------|---------|
| `current_value` | Entire current value (for simple strings) | `"GitHub Copilot Business..."` |
| `current_value.timing` | Timing object in a habit | `{"start_time": "13:00", ...}` |
| `current_value.schedule` | Schedule object in a habit | `{"frequency_type": "weekly", ...}` |
| `current_value.location` | Location field in a habit | `"home office desk"` |
| `current_value.statement` | Statement field in a preference | `"Prefers balanced approach..."` |
| `new_value` / `new_value.*` | New value and its nested fields | Same as current_value |
| `dropped_value` / `dropped_value.*` | Dropped value and its nested fields | Same as current_value |
| `delta.from` / `delta.from.*` | Previous value before change | Same as current_value |
| `delta.to` / `delta.to.*` | New value after change | Same as current_value |
| `change_reason` | Why the change happened | `"Team lead mandated..."` |

### 4.4 Conversion Requirements by State Category

#### User Attributes State
**Conversion Requirements by Change Type:**
- `add`: Must show acquisition journey (research → decision → acquisition) + multiple usage instances
  - **Why:** Acquisition is a process, not a moment. Real people research, evaluate, decide, then gradually integrate new things into their lives.
- `modify`: Must show old value usage → transition trigger → new value usage
  - **Why:** Modifications don't happen in a vacuum. There's always a "before" state and a catalyst that drives change.
- `unchanged`: Must show ongoing usage across different contexts within the window
  - **Why:** Possession without usage is meaningless. We need to see the attribute actively shaping the user's behavior in varied situations.

#### Habits State
**Conversion Requirements by Change Type:**
- `acquire`: Must show motivation discovery → habit design → initial struggles → stabilization
  - **Why:** New habits don't appear fully formed. There's a formation process with typical stages.
- `adjust`: Must show dissatisfaction with old pattern → adjustment reasoning → new pattern execution
  - **Why:** Habits change for reasons. We need to see what prompted the adjustment and how the new pattern addresses it.
- `drop`: Must show discontinuation trigger → no more pattern execution
  - **Why:** Habits don't just vanish. Something causes them to stop.
- `unchanged`: Must show consistent execution with natural variation (not robotic repetition)
  - **Why:** Real humans don't perform habits identically every time. There's natural variation while maintaining the core pattern.

#### Preferences State
**Conversion Requirements by Change Type:**
- `refine/shift`: Must show old preference in action → catalyst event → experimentation → new preference dominance
  - **Why:** Preference changes are gradual transitions, not instant flips.
- `unchanged`: Must show preference through CHOICES (selecting A over B), not just using A
  - **Why:** A preference is revealed only when alternatives exist.

### 4.5 CRITICAL: How to Evidence `change_reason`

**The `change_reason` field describes WHY a change happened, but it cannot be evidenced simply by executing the new pattern.** Instead, it must be made observable through user expressions or behaviors that reveal their motivations, frustrations, or catalysts.

**Valid ways to evidence `change_reason`:**
- **User conversations**: WhatsApp messages, LLM Assistant dialogues discussing the problem/motivation
- **User writings**: Notion entries, Gmail drafts, document notes reflecting on the issue
- **User searches**: Google searches revealing the problem they're trying to solve
- **User emails**: Receiving or sending emails that trigger the change
- **User social posts**: LinkedIn posts, comments expressing frustrations or discoveries

**INVALID way to evidence `change_reason`:**
- Simply executing the new pattern (e.g., using new schedule, using new tool, making new choice)
- The new behavior shows the OUTCOME, not the REASON

**Example - WRONG approach:**
```json
// State item
{
  "name": "financial_reconciliation",
  "change_type": "adjust",
  "delta": {
    "from": {"timing": {"start_time": "20:00", "end_time": "21:30"}},
    "to": {"timing": {"start_time": "13:00", "end_time": "14:30"}}
  },
  "change_reason": "Rescheduled to resolve scheduling conflicts with evening leisure",
  "required_observable_fields": ["delta.to.timing", "change_reason"]
}

// WRONG: This event executes the new pattern but doesn't show WHY
{
  "app_name": "Chase",
  "api_name": "GetTransactions",
  "user_intent": "Performing monthly reconciliation at new time 13:00-14:30",
  "evidence_for_states": [{
    "state_name": "financial_reconciliation",
    "evidenced_fields": ["delta.to.timing", "change_reason"]  // ❌ WRONG
  }]
}
```

**Example - CORRECT approach:**
```json
// Event showing the reason BEFORE the new pattern
{
  "app_name": "LLM Assistant",
  "api_name": "ContinueConversation",
  "user_intent": "Discussing with Claude about recurring conflicts between 20:00 monthly reconciliation and evening plans with partner; asking for help finding a better time slot that doesn't interfere with leisure time",
  "evidence_for_states": [{
    "state_name": "financial_reconciliation",
    "evidenced_fields": ["change_reason"]  // ✓ CORRECT
  }]
},
// Event showing the new pattern execution
{
  "app_name": "Chase",
  "api_name": "GetTransactions",
  "user_intent": "Performing monthly reconciliation at new time 13:00-14:30",
  "evidence_for_states": [{
    "state_name": "financial_reconciliation",
    "evidenced_fields": ["delta.to.timing"]  // ✓ Shows new timing, not reason
  }]
}
```

**Key principle:** The `change_reason` must be observable through events where the user **explicitly expresses or discusses** the motivation, problem, or catalyst—not just through events that demonstrate the changed behavior.

---

## 5. Output Schema

### 5.1 Event Chain JSON Structure

**IMPORTANT: Do NOT repeat the input state values in your output.** The output only needs to reference state items by name and track which fields are evidenced.

```json
{
  "window_id": "copy from input",
  "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "event_chains": [
    {
      "state_refs": [
        {
          "state_category": "user_attributes_state | habits_state | preferences_state",
          "state_name": "exact name from input state"
        }
      ],
      "events": [
        {
          "time_specification": {
            "schedule_dates": ["YYYY-MM-DD", ...],
            "time": "HH:MM:SS",
            "start_time": "HH:MM:SS",
            "end_time": "HH:MM:SS"
          },
          "app_name": "MUST exist in provided catalog",
          "api_name": "MUST exist in provided catalog and belong to app_name",
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

**Key differences from input:**
- `state_refs` replaces `converted_state_items` — it only contains `state_category` and `state_name` as references
- No need to copy `current_value`, `delta`, `change_type`, or other input fields
- The `evidence_for_states` in each event tracks which fields are demonstrated

### 5.2 Time Specification Rules

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
  "end_time": "07:30:00"
}
```

**Rule 4: All dates must fall within the window `time_range`**

### 5.3 App/API Specification Rules

**CRITICAL: All apps and APIs must be from the provided catalog (Section 2)**

There are two ways to specify which app and API an event uses:

#### Option A: Single App/API (for uniform events)
Use when the event always uses the same app and API on every occurrence:
```json
{
  "app_name": "Fitbit",  // MUST exist in catalog
  "api_name": "SyncDevice",  // MUST belong to Fitbit in catalog
  "user_intent": "..."
}
```

#### Option B: App/API Variations Set (for events with natural variation)
Use when the habit/behavior naturally manifests through different apps/APIs on different occasions. Provide a SET of valid app-api combinations:

```json
{
  "app_api_variations": [
    {"app_name": "LinkedIn", "api_name": "GetFeed"},  // Each pair MUST be valid per catalog
    {"app_name": "Google", "api_name": "Search"}
  ],
  "user_intent": "..."
}
```

**Important Rules:**
1. Use EITHER `app_name`+`api_name` OR `app_api_variations`, never both
2. `app_api_variations` is a SET (unique combinations only, no duplicates)
3. **Each app and api must exist in the provided catalog (Section 2)**
4. Each dict in `app_api_variations` must have a valid app-api pairing per the catalog
5. The downstream code will sample from this set to assign specific app/api to each `schedule_dates` entry
6. The `user_intent` should explain how different app/api combinations serve the behavior's purpose

### 5.4 Realistic Habit Variation Guidelines

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

**Common variation patterns for habits (VERIFY EACH PAIR EXISTS IN CATALOG):**

| Habit Type | Realistic Variations |
|------------|---------------------|
| News/Industry reading | `[{LinkedIn, GetFeed}, {Google, Search}]` |
| Learning/Skill development | `[{LLM Assistant, ContinueConversation}, {Google, Search}, {Goodreads, SearchBooks}]` |
| Social connection | `[{WhatsApp, SendMessage}, {Instagram, LikePost}, {LinkedIn, CommentOnPost}]` |
| Financial monitoring | `[{Chase, GetBalance}, {Chase, GetTransactions}, {Robinhood, GetPortfolio}]` |
| Content consumption | `[{Netflix, PlayContent}, {Spotify, PlaySong}]` |

### 5.5 Evidence Annotation Rules

Each event must specify which state(s) and which fields it provides evidence for via the `evidence_for_states` array.

**Structure:**
```json
"evidence_for_states": [
  {
    "state_name": "exact state name from state_refs",
    "evidenced_fields": ["field1", "field2", ...]
  }
]
```

**Rules:**
1. Every event must have at least one entry in `evidence_for_states`
2. Field names must exactly match those in `required_observable_fields`
3. An event can evidence multiple fields from the same state
4. An event can evidence fields from multiple states (in merged chains)
5. All `required_observable_fields` must be covered by at least one event

**Field Evidence Guidelines:**

| Field Type | How to Evidence | Example Apps/APIs (MUST BE FROM CATALOG) |
|------------|-----------------|------------------------------------------|
| `change_reason` | **User expressions/discussions about the motivation** | LLM Assistant:ContinueConversation, WhatsApp:SendMessage, Notion:CreatePage, Gmail:SendEmail, Google:Search (revealing problem) |
| `current_value` (attribute) | Usage events demonstrating possession | Any app showing the attribute in use |
| `current_value.timing` | Events occurring at the specified time | Fitbit:LogWorkout, Notion:CreateDatabaseEntry, Chase:GetTransactions |
| `current_value.schedule` | Events on the correct schedule pattern | Repeated execution matching frequency |
| `current_value.location` | Events with location context in user_intent | Fitbit:LogWorkout, Instagram:CreatePost, UberEats:PlaceOrder |
| `current_value.statement` (preference) | Choice events (A over B) | Any event showing selection among alternatives |
| `delta.from.*` | Events showing old behavior BEFORE change | Early window events with old pattern |
| `delta.to.*` | Events showing new behavior AFTER change | Later window events with new pattern |

**Examples:**

**Single state, single field:**
```json
{
  "app_name": "LLM Assistant",
  "api_name": "ContinueConversation",
  "user_intent": "Discussing frustration with current 20:00 reconciliation time conflicting with evening plans; asking for advice on better scheduling",
  "evidence_for_states": [
    {
      "state_name": "financial_reconciliation",
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
  "user_intent": "Logging 30-minute morning run (5.2km) at Vondelpark before 7am standup",
  "evidence_for_states": [
    {
      "state_name": "morning_run_habit",
      "evidenced_fields": ["current_value.timing", "current_value.location"]
    }
  ]
}
```

**Multiple states (merged chain):**
```json
{
  "app_name": "LLM Assistant",
  "api_name": "CreateConversation",
  "user_intent": "Starting interactive session to refactor state machine, choosing to 'build with AI' rather than reading documentation",
  "evidence_for_states": [
    {
      "state_name": "technical_skills_inventory",
      "evidenced_fields": ["current_value"]
    },
    {
      "state_name": "learning_style",
      "evidenced_fields": ["change_reason", "delta.to.statement"]
    }
  ]
}
```

---

## 6. Generation Strategy by State Type

### 6.1 User Attributes Generation

#### Change Type: `add`

**Goal:** Demonstrate the complete acquisition journey and subsequent integration.

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

#### Change Type: `modify`

**Goal:** Show the transition narrative from old to new value.

**Required Event Phases:**
1. Old value in use (1-2 events early in window)
2. Trigger for change (1-2 events) — **must be user expression/discussion, not just new usage**
3. Transition process (1-2 events)
4. New value in use (2-3 events)

#### Change Type: `unchanged`

**Goal:** Demonstrate ongoing presence through varied usage contexts.

**Requirements:**
- Generate 3-5 usage events spread across the window
- Each event should show the attribute in a DIFFERENT context/scenario
- Avoid repetitive events

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
   - **Use `app_api_variations` to define ways this habit manifests (VERIFY ALL PAIRS IN CATALOG)**
   - Include natural evolution over time in user_intent

**Critical:** Establishment events (phases 1-2) must occur BEFORE the first `schedule_dates` entry.

#### Change Type: `adjust`

**Goal:** Show why and how the habit pattern changed.

**Required Event Phases:**
1. Old pattern execution (early dates showing previous pattern)
2. Adjustment trigger (1-2 events) — **must be user expression/discussion about the problem**
3. Modification action (updating schedule/approach) — optional
4. New pattern execution (later dates showing adjusted pattern)

**CRITICAL:** The adjustment trigger must show the user **explicitly discussing or expressing** the problem with the old pattern. Simply executing the new pattern does NOT evidence `change_reason`.

#### Change Type: `drop`

**Goal:** Show the habit being discontinued.

**Required Event Phases:**
1. Discontinuation trigger (1-2 events) — **must be user expression/discussion**
2. Optional: final execution or conscious stopping action

#### Change Type: `unchanged`

**Goal:** Show consistent but human execution with natural variation.

**Requirements:**
- Single event with `schedule_dates` covering all occurrences
- **Use `app_api_variations` when natural variation exists (VERIFY ALL PAIRS IN CATALOG)**
- `user_intent` should describe execution details AND explain variation
- Show natural evolution within the unchanged habit

### 6.3 Preferences Generation

#### Change Type: `refine` or `shift`

**Goal:** Make the preference evolution visible through changing choices.

**Required Event Phases:**

1. **Old Preference Demonstration** (optional, 1-2 events if delta.from exists)
   - Show choices aligned with old preference

2. **Catalyst Event** (1-2 events)
   - What triggered the preference change?
   - **Must be user expression, realization, or external experience**
   - Must make `change_reason` observable

3. **Experimentation Phase** (1-2 events)
   - Trying the new approach, possibly mixed with old

4. **New Preference Demonstration** (2-3 events)
   - Consistent choices aligned with new preference across contexts

#### Change Type: `unchanged`

**Goal:** Demonstrate preference through observable choice patterns.

**Requirements:**
- Generate 4-6 events showing preference-aligned choices
- Each event should be a decision point where alternatives existed
- `user_intent` should make clear what was chosen OVER what alternative

---

## 7. Chain Organization Rules

### 7.1 Default: One State Item → One Event Chain

**This is strongly preferred.** Each state item should have its own dedicated event chain. Merging is the exception.

### 7.2 Merging Constraints

**CRITICAL RESTRICTIONS:**
1. **NEVER merge two habits together.** Each habit MUST have its own separate event chain.
2. **Maximum 2 items per chain** (when merging is allowed)
3. **Default is 1 item per chain** — only merge when there's genuine behavioral interweaving

**Allowed Merge Scenarios (rare):**
- Attribute enables habit execution (e.g., owns running shoes + morning run habit)
- Preference demonstrated during attribute usage (e.g., owns headphones + prefers instrumental music)

**NEVER Merge:**
- Two habits (regardless of how related)
- Items with only thematic similarity but independent execution
- More than 2 items

### 7.3 Evidence Coverage for Merged Chains

When merging:
1. Each state's `required_observable_fields` must be fully covered
2. One event can evidence fields from multiple states
3. Verify coverage separately for each state

---

## 8. Comprehensive Examples

### Example 1: Attribute Add — Full Acquisition Journey

**Input State Item:**
```json
{
  "name": "ai_coding_assistant_subscription",
  "change_type": "add",
  "new_value": "GitHub Copilot Business subscription, integrated with VS Code and JetBrains IDEs",
  "change_reason": "Team lead mandated AI tool adoption after Q4 productivity review showed 20% lag behind industry benchmarks",
  "required_observable_fields": ["new_value", "change_reason"]
}
```

**Generated Event Chain:**
```json
{
  "state_refs": [
    {
      "state_category": "user_attributes_state",
      "state_name": "ai_coding_assistant_subscription"
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
      "user_intent": "Initial research: searching 'best AI coding assistants 2024 comparison' to understand landscape before team discussion",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["new_value"]
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
      "user_intent": "Deep evaluation: asking Claude to compare GitHub Copilot vs Cursor vs CodeWhisperer for C++ embedded systems; concerned about legacy codebase compatibility",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["new_value"]
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
      "user_intent": "Acquisition action: emailing IT department to request GitHub Copilot Business license activation",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["new_value"]
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
      "user_intent": "Integration usage: using AI to debug a Copilot-generated state machine with race condition; learning to verify AI suggestions",
      "evidence_for_states": [
        {
          "state_name": "ai_coding_assistant_subscription",
          "evidenced_fields": ["new_value"]
        }
      ]
    }
  ]
}
```

---

### Example 2: Habit Adjust — Schedule Change with Proper change_reason Evidence

**Input State Item:**
```json
{
  "name": "financial_reconciliation",
  "change_type": "adjust",
  "delta": {
    "from": {
      "schedule": {"frequency_type": "monthly_by_date", "days_of_month": [1]},
      "timing": {"start_time": "20:00", "end_time": "21:30"},
      "schedule_dates": ["2024-10-01", "2024-11-01", "2024-12-01"]
    },
    "to": {
      "schedule": {"frequency_type": "monthly_nth_weekday", "week_of_month": 1, "day_of_week": 0},
      "timing": {"start_time": "13:00", "end_time": "14:30"},
      "schedule_dates": ["2024-10-07", "2024-11-04", "2024-12-02"]
    }
  },
  "change_reason": "Rescheduled to the first Monday of the month (13:00-14:30) to resolve scheduling conflicts with evening leisure, professional reading, and family traditions",
  "required_observable_fields": ["delta.to.timing", "delta.to.schedule", "change_reason"]
}
```

**Generated Event Chain:**
```json
{
  "state_refs": [
    {
      "state_category": "habits_state",
      "state_name": "financial_reconciliation"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-10-01"],
        "time": "19:30:00"
      },
      "app_name": "WhatsApp",
      "api_name": "SendMessage",
      "user_intent": "Expressing frustration to partner about missing tonight's season premiere because of monthly reconciliation; noting this is the third month where the 1st evening slot conflicts with plans",
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
      "user_intent": "Discussing scheduling problem with Claude: monthly reconciliation at 20:00 on the 1st keeps conflicting with evening leisure, reading time, and holiday traditions; asking for help finding a better recurring slot",
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
        "end_time": "14:30:00"
      },
      "app_name": "Chase",
      "api_name": "GetTransactions",
      "user_intent": "Performing monthly reconciliation during Monday lunch break at new time (13:00-14:30); reviewing transactions and updating budget tracker",
      "evidence_for_states": [
        {
          "state_name": "financial_reconciliation",
          "evidenced_fields": ["delta.to.timing", "delta.to.schedule"]
        }
      ]
    }
  ]
}
```

**Evidence Coverage Verification:**
| Required Field | Evidenced By |
|----------------|--------------|
| `delta.to.timing` | event 3 (13:00-14:30) ✓ |
| `delta.to.schedule` | event 3 (first Monday: Oct 7, Nov 4, Dec 2) ✓ |
| `change_reason` | events 1-2 (expressing conflicts) ✓ |

**Why This Works:**
- Events 1-2 show the user **explicitly expressing/discussing** the scheduling conflicts
- Event 3 shows the new pattern execution
- The `change_reason` is evidenced through user communication, NOT just by doing the new schedule

---

### Example 3: Unchanged Habit with `app_api_variations`

**Input State Item:**
```json
{
  "name": "industry_tech_reading",
  "change_type": "unchanged",
  "current_value": {
    "schedule": {"frequency_type": "weekly", "days_of_week": [1, 3]},
    "timing": {"start_time": "20:30", "end_time": "21:30"},
    "location": "home living room",
    "schedule_dates": ["2024-01-02", "2024-01-04", "2024-01-09", "2024-01-11", "2024-01-16", "2024-01-18"]
  },
  "required_observable_fields": ["current_value.timing", "current_value.schedule", "current_value.location"]
}
```

**Generated Event Chain:**
```json
{
  "state_refs": [
    {
      "state_category": "habits_state",
      "state_name": "industry_tech_reading"
    }
  ],
  "events": [
    {
      "time_specification": {
        "schedule_dates": ["2024-01-02", "2024-01-04", "2024-01-09", "2024-01-11", "2024-01-16", "2024-01-18"],
        "start_time": "20:30:00",
        "end_time": "21:30:00"
      },
      "app_api_variations": [
        {"app_name": "LinkedIn", "api_name": "GetFeed"},
        {"app_name": "Google", "api_name": "Search"}
      ],
      "user_intent": "Bi-weekly evening tech reading from living room couch. LinkedIn sessions focus on industry pulse—scrolling posts from embedded systems engineers, catching Zephyr RTOS updates. Google sessions dive into specific topics like 'ARM Cortex-M7 cache optimization' or 'FreeRTOS vs Zephyr 2024'",
      "evidence_for_states": [
        {
          "state_name": "industry_tech_reading",
          "evidenced_fields": ["current_value.timing", "current_value.schedule", "current_value.location"]
        }
      ]
    }
  ]
}
```

---

### Example 4: Preference Shift with Proper change_reason Evidence

**Input State Item:**
```json
{
  "name": "spending_logic",
  "change_type": "shift",
  "delta": {
    "from": {"statement": "Prefers purchasing high-quality, durable electronics over budget alternatives"},
    "to": {"statement": "Prefers balanced approach between material acquisitions and charitable giving"}
  },
  "change_reason": "Notre Dame reopening shifted perspective toward more balanced spending including social impact",
  "required_observable_fields": ["delta.to.statement", "change_reason"]
}
```

**Generated Event Chain:**
```json
{
  "state_refs": [
    {
      "state_category": "preferences_state",
      "state_name": "spending_logic"
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
      "user_intent": "Searching 'Notre Dame cathedral reopening December 2024' after seeing news; feeling moved by the global restoration effort",
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
        "time": "12:00:00"
      },
      "app_name": "Notion",
      "api_name": "CreatePage",
      "user_intent": "Writing reflection on Notre Dame news: 'The collective effort to restore this landmark made me realize I've been focused only on personal material quality. Maybe I should balance my spending to include meaningful charitable contributions'",
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
        "time": "14:30:00"
      },
      "app_name": "Chase",
      "api_name": "PayBill",
      "user_intent": "Making donation to Friends of Notre Dame; consciously allocating funds that might have gone toward new earbuds toward charitable impact instead",
      "evidence_for_states": [
        {
          "state_name": "spending_logic",
          "evidenced_fields": ["delta.to.statement"]
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
      "user_intent": "Year-end donation to Stichting Vluchteling; reflecting that this year's spending now balances quality gifts with charitable contributions—a shift from previous material-only focus",
      "evidence_for_states": [
        {
          "state_name": "spending_logic",
          "evidenced_fields": ["delta.to.statement"]
        }
      ]
    }
  ]
}
```

**Why This Works:**
- Event 1: Shows external catalyst (Notre Dame news)
- Event 2: Shows user **explicitly reflecting and expressing** the realization (evidences `change_reason`)
- Events 3-4: Show the new balanced behavior (evidences `delta.to.statement`)

---

## 9. Validation Checklist

Before finalizing output, verify:

### Coverage Validation
- [ ] Every state item appears in at least one chain
- [ ] Each `state_refs` entry has: state_category, state_name
- [ ] No input state values repeated in output (only references)

### App/API Catalog Compliance (CRITICAL)
- [ ] **Every `app_name` exists in catalog**
- [ ] **Every `api_name` belongs to its `app_name` per catalog**
- [ ] **All pairs in `app_api_variations` are valid per catalog**
- [ ] **LLM Assistant used as fallback when needed**

### Chain Organization Validation
- [ ] **NEVER merge two habits**
- [ ] Default is 1 item per chain
- [ ] Maximum 2 items when merging

### change_reason Evidence Validation (CRITICAL)
- [ ] **`change_reason` evidenced through user expressions/discussions**
- [ ] **NOT evidenced merely by executing new behavior**
- [ ] Valid evidence methods: conversations, writings, searches revealing motivation

### Lossless Conversion Validation
For EACH state item:
- [ ] Could someone reconstruct the state from ONLY the events?
- [ ] Is `change_reason` observable through explicit user expression?
- [ ] For `add/acquire`: Both process AND outcome present?
- [ ] For preferences: CHOICES shown (A over B)?

### Evidence Field Reference Validation
- [ ] Unchanged items: use `current_value.*`
- [ ] Add/acquire items: use `new_value.*`, `change_reason`
- [ ] Drop items: use `dropped_value.*`, `change_reason`
- [ ] Modified items: use `delta.to.*`, `delta.from.*`, `change_reason`

### Habit Variation Validation
- [ ] Natural variation uses `app_api_variations` with catalog-valid pairs
- [ ] No duplicates in `app_api_variations` (it's a set)
- [ ] `user_intent` explains variation

### Event Quality Validation
- [ ] Each `user_intent` includes: action + motivation + context
- [ ] Events are specific and "executable"
- [ ] No generic/shallow events

### Temporal Validation
- [ ] Acquisition precedes usage
- [ ] Events spread across window
- [ ] All dates within `time_range`

---

## 10. Final Instructions

1. **Read input state carefully** — understand each item's structure
2. **VERIFY APP/API CATALOG COMPLIANCE** — confirm all apps/APIs exist
3. **Plan conversion** — what events make state observable?
4. **Respect chain rules:**
   - Default: 1 state → 1 chain
   - NEVER merge two habits
5. **For habits: determine variation** — use `app_api_variations` when natural
6. **Generate detailed events** — pass the "executable" test
7. **Use state_refs correctly** — only `state_category` and `state_name`
8. **Use correct field references** in `evidence_for_states`
9. **Evidence change_reason properly** — through user expressions, NOT just new behavior
10. **Verify lossless conversion** — could state be reconstructed?
11. **Check coverage** — every state represented
12. **Output ONLY JSON** — single dict starting with `{` and ending with `}`

**Remember:**
- Create behavioral traces so realistic the state becomes fully observable
- **ONLY use apps/APIs from catalog**
- **Evidence `change_reason` through user expressions/discussions**
- Show realistic human variation in habits
- Every event MUST have `evidence_for_states`
- Every `required_observable_fields` field MUST be covered
""")

@dataclass
class EventsChainRequest:
    user_basic_profile: str
    domain_name: str
    domain_window_state: str
    user_previous_window_summary: str
    user_domain_previous_window_summary: str
    user_this_window_description: str
    world_background: str
    user_life_context: Optional[str] = None


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
