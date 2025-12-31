from dataclasses import dataclass
from typing import List, Dict

from jinja2 import Template

from mem_bench.behavior_and_conversation.llm_client import (
    GeminiJSONClient,
    LLMResult,
)


dynamic_profile_template = Template("""You are an expert simulator of long-horizon dynamic user profile trajectories.

### Your task:
Given the world background, the basic user profile, and ONE life domain, time windows, generate a **time-windowed user profile trajectory** for this life domain over the provided time windows.

### Input Provided:

**All life domains in this system:**
{{ life_domain_list }}

**Current focus domain:**
{{ life_domain }}: {{ scope_definition }}

**World background:**
{{ world_background }}

**Basic user profile:**
{{ user_profile }}

**Time windows:**
{{ time_windows_json }}

---

### Definitions

Before proceeding, you must understand these core concepts:

- **User Attributes**:
  Definition: Factual, discrete attributes about the user (e.g., occupation, family members, owned devices), which can be accurately stored and updated.

  Examples:
  - "user_occupation": [
      "Software Engineer"
    ]
  - "user_family_members": [
      "partner",
      "child (age 5)"
    ]
  - "user_smart_home_devices": [
      "Amazon Echo Dot",
      "Philips Hue bulbs"
    ]
  
  Constraints:
  - All user attribute values must be concrete and unambiguous.
    Bad: "Master's degree in technology-related field" (Too vague, should be a specific degree and field)
    Good (single-element list): ["Master's degree in Computer Science from a public university"]
    Bad: "Moderate income" (Too vague, should be a specific amount)
    Good: ["40000 USD per year before tax from primary job"]
    Bad: "High-end laptop (e.g., Dell XPS), smartphone (e.g., Google Pixel), smart home hub (e.g., Amazon Echo Dot)" (Too vague, should be a specific brand and model)
    Good:
      [
        "Dell XPS 13 (ultrabook used mainly for software development)",
        "Google Pixel 7 (Android smartphone used for daily communication)",
        "Amazon Echo Dot (voice assistant in the living room)"
      ]
  
  Allowed types of user attribute change:

  - **add**:
    Meaning:
      For this attribute_name, introduce new list elements that were not present
      in the previous window. This covers two cases:
      (1) The attribute already exists: append one or more new values to its list.
      (2) The attribute did not exist for this user in the previous window
          (i.e., was not recorded at all): start tracking this attribute by giving
          it a non-empty list as its initial value.

    Example (append to an existing attribute list):
      Before: "user_owned_devices": [
        "Dell XPS 13 (ultrabook used mainly for development)",
        "Google Pixel 7 (Android smartphone used for daily communication)"
      ]
      new_state: "user_owned_devices": [
        "Dell XPS 13 (ultrabook used mainly for development)",
        "Google Pixel 7 (Android smartphone used for daily communication)",
        "Apple Watch SE (entry-level smartwatch for fitness tracking)"
      ]

    Example (start tracking a previously absent attribute):
      Before: (no 'user_subscriptions' attribute recorded for this user)
      new_state: "user_subscriptions": [
        "O'Reilly Media subscription (used for technical learning and reading)"
      ]

  - **remove**:
    Meaning: for a specific attribute, delete an **existing value** that was present in the previous window.

  - **modify**:
    Meaning: change the list of values for an existing attribute.
    **CRITICAL**: You can ONLY use "modify" if this attribute already exists in a previous window (either in initial state or added in an earlier delta). You cannot modify an attribute that has never been defined.
    Example:
      Before: "user_residence": [
        "Lives in a rented studio apartment in downtown area"
      ]
      After: "user_residence": [
        "Lives in a recently purchased two-bedroom apartment in a suburban area"
      ]

- **Habits**:
  Definition: Recurring behavioral patterns (e.g., morning jogging routine, weekly meal prep), which can be identified and consolidated from repeated observations.
  Each habit MUST be represented as a structured JSON object (dictionary) capturing action, frequency, timing, and context.

  A habit object MUST have the following fields:
  - action: a concrete, unambiguous action label.
  - frequency: how often the habit occurs (e.g., "daily", "3_times_per_week", "weekdays_only").
  - timing: when it typically happens(e.g., "before work (6:00 AM–6:30 AM)", "after dinner (7:00 PM–7:30 PM)", "Sunday mornings (9:00 AM–11:00 AM)").
  - context: where / in what setting it happens (e.g., "in a small home gym", "walking around the neighborhood", "at a local coffee shop").
  - description: a short natural language description (5–20 words) summarizing the habit in human-readable form.

  Examples:
  - "user_walks_dog_morning": {
      "action": "walk_dog_on_leash",
      "frequency": "daily",
      "timing": "early morning before work",
      "context": "around the residential neighborhood for 20–30 minutes",
      "description": "User walks the dog every morning before work for 20–30 minutes."
    }

  Constraints:
  - All habit fields must be concrete and unambiguous.
  - Avoid vague action names such as "hiking", "walking", "training", or "going out" when the actor, purpose, or context is unclear.
    For example, use "dog_training" rather than "training" when the actor is a dog.
  - The frequency should be reasonable and realistic for a human user.

  Allowed types of habit change:
  - **acquire**:
    Meaning: a new habit appears in this window.
  - **drop**:
    Meaning: a **previously existing habit** ceases entirely.
  - **adjust**:
    Meaning: frequency, timing, duration, or context of a habit changes.
    **CRITICAL**: You can ONLY use "adjust" if this habit already exists in a previous window (either in initial state or acquired in an earlier delta). You cannot adjust a habit that has never been defined.

- **Preferences**:
  Definition: Subjective inclinations that guide choices (e.g., preferring window seats, favoring spicy food).
  It describes *what* the user tends to choose when multiple alternatives exist.
  Preference changes trigger when new experiences, significant events, world/seasonal factors, or new equipment/resources are encountered.

  For this domain, you MUST:
  - propose a small set of **2-4 distinct preference dimensions** that are most important for this domain and this user,
  - dimensions should be:
    - non-redundant,
    - interpretable,
    - sufficient to describe meaningful variation in this domain.
  - Values should reflect **relative inclination** (e.g., prefer A over B), not abstract intensity scores.

  Common triggers of preference change:
  - new experiences (e.g., first time photographing a solar eclipse)
  - significant events (e.g., pet illness → preference for higher-quality food)
  - world/seasonal factors (e.g., winter → preference for indoor activities)
  - new equipment/resources (e.g., prime lens → preference for street photography)

  Examples of preferences:
  - "investment_focus": "Prefers long-term investments in technology stocks and diversified index funds over short-term trading."

  Constraints:
  - All preference values must be concrete and unambiguous.
    Bad: "better", "worse", "moderate", "likes it a lot" (Too vague, should be a specific preference description)
    Good: "Prefers quiet solo activities over group-based social gatherings"

  Allowed types of preference change:
  - **shift**:
    Meaning: dominant preference changes from one option to another.
    **CRITICAL**: You can ONLY use "shift" if this preference already exists in a previous window (either in initial state or shifted in an earlier delta). You cannot shift a preference that has never been defined.

  - **amplify**:
    Meaning: preference strength increases.
    **CRITICAL**: You can ONLY use "amplify" if this preference already exists in a previous window (either in initial state or amplified in an earlier delta). You cannot amplify a preference that has never been defined.

  - **attenuate**:
    Meaning: preference strength weakens.
    **CRITICAL**: You can ONLY use "attenuate" if this preference already exists in a previous window (either in initial state or attenuated in an earlier delta). You cannot attenuate a preference that has never been defined.
  
---

### CRITICAL CONSTRAINTS

#### 1. NO BASIC PROFILE OVERLAP
You are generating the **domain-specific profile component** only.
You must **NOT** include or redefine attributes that belong to the Basic User Profile.
- **DO NOT** generate: Age, Gender, Location Type, Location Region, Languages, Occupation, Employment Status, Industry, Weekly Work Hours, Education Level, Income Band, Financial Buffer Months, Housing Status, Household Size, Household Composition, Caregiving Load, Health Constraint Level, Digital Literacy, Planning Orientation, Risk Tolerance.
- **DO** generate: Specific job titles, specific skills, specific assets, owned items, specific daily routines.

#### 2. STRICT DOMAIN FOCUS
You are building the profile for **{{ life_domain }}** ONLY.

**Rules for domain focus:**
- Focus **ONLY** on attributes, habits, and preferences that are **directly relevant** to {{ life_domain }}.
- Assume all other life domains have already been constructed separately in parallel.
- Do NOT generate content that clearly belongs to another domain.
- All changes should be motivated by and directly relevant to this specific domain.

**Examples of INCORRECT cross-domain content:**
- If building **Family & Close Relationships**, do NOT include:
  - "daily_ai_news_recap" habit with "listen to AI-focused podcast during commute" 
    → This belongs to **Professional Development** or **Learning & Personal Growth**
  - "user_work_laptop" attribute 
    → This belongs to **Professional Life & Career**
  - "morning_gym_routine" habit 
    → This belongs to **Health & Self-care**

- If building **Health & Self-care**, do NOT include:
  - "weekly_family_dinner" habit 
    → This belongs to **Family & Close Relationships**
  - "user_office_location" attribute 
    → This belongs to **Work & Education**

---

### Generation Approach

Now that you understand the core concepts and constraints, here's how to generate the trajectory:

**For the initial state (window 0):**
1. Define the baseline state for attributes, habits, and preferences in this domain
2. Write a summary describing the user's starting state

**For each subsequent time window (window 1, 2, 3, ...):**
1. **First, write the window_description**: Identify the salient external factors (from world background: seasons, events, circumstances) and the user's internal motivations/focus during this period. This is the **latent causal variable** that will drive coordinated changes across attributes, habits, and preferences.

2. **Then, generate the trajectory deltas**: Based on the window_description, determine how attributes, habits, and preferences evolve together in directionally coherent ways:
   - user_attributes_delta: what gets added, removed, or modified
   - habits_delta: what habits are acquired, adjusted, or dropped
   - preferences_delta: what preferences shift, amplify, or attenuate
   
3. **Finally, write the summary**: Synthesize all changes and explicitly connect them back to the window_description's driving factors (format: "Due to [context], the user [changed X, Y, Z]...")

**Key narrative components:**

- **window_description** (required for each time window):
  - A short context statement (1-3 sentences) that identifies:
    - External factors: world events, seasonal changes, circumstances from the world background
    - Internal factors: user's focus, goals, or motivations during this period
  - This is the **latent causal variable** that explains WHY changes happen together
  - Think of it as the "story engine" that drives coordinated evolution across the profile
  - Example: "Early spring brings milder weather and increased outdoor activity opportunities. User is motivated to improve cardiovascular fitness after a routine health checkup showed elevated blood pressure."

- **summary** (required for initial state and each time window):
  - For initial state: 2-4 sentences describing the user's baseline in this domain
  - For time windows: 2-4 sentences synthesizing what changed and connecting back to window_description
  - Should mention key attributes, habits, and preferences
  - Example: "User acquires morning jogging habit 3x/week, purchases running shoes and fitness tracker, and shifts preference toward cardio-focused activities—all motivated by health concerns and favorable weather."

---

### Critical Consistency Requirements

1. **Grounding in context**: Every change must be plausible given:
   - The world background (external events, seasons, societal trends, etc.)
   - The basic user profile (demographics, resources, constraints, etc.)
   - The prior state (previous windows' accumulated changes)

2. **Cross-window consistency**: 
   - Changes should form a coherent narrative arc across windows
   - Short-term changes (seasonal, event-driven) must be reversed/adjusted when the external factor ends
   - Long-term changes (lifestyle shifts, permanent acquisitions) should persist and compound
   - Avoid contradictions (e.g., don't drop a habit in window 2 then reacquire it in window 3 without clear motivation)

3. **Within-window coherence**:
   - The window_description should logically motivate ALL changes in that window
   - Attributes, habits, and preferences should co-evolve in directionally consistent ways
   - Example: new fitness goal (motivation) → buys running shoes (attribute) → starts jogging routine (habit) → prefers cardio activities (preference)

4. **Essential items planned to evolve must start in initial_state**:
   - **CRITICAL**: If an item is **essential for this domain** AND you plan to **evolve it in later windows** (modify, adjust, shift), it MUST be initialized in initial_state with a baseline value.
   - Essential = things a realistic person in this situation would naturally already have.
   
   **An example of a bad case (essential item not initialized before evolution):**
   - Finances & Material Living → NO smartphone in initial_state → adds first phone in window 3 (Unrealistic: should already own one)

5. **Short-term vs Long-term changes**:
   - Be aware of whether a change is driven by **short-term external factors** (e.g., seasonal changes, special events, temporary circumstances) or **long-term shifts** (e.g., lifestyle changes, permanent acquisitions)
   - **For short-term changes**, you MUST plan follow-up adjustments to revert or normalize the behavior once the external factor ends
   
   Examples of short-term changes requiring follow-up:
   - **Seasonal adjustment**: If a user starts a daily hydration habit in winter to combat dry air, this should be adjusted back (reduced or dropped) when summer arrives
   - **Event-driven habit**: If a user acquires a habit of watching Olympic events every evening during the Olympics, this habit MUST be dropped once the Olympics end
   - **Temporary circumstance**: If a user shifts to indoor exercise during a heatwave, they should shift back to outdoor exercise when weather normalizes
   
   **How to implement**:
   - When you introduce a change motivated by a short-term factor, note this in the "reason" field
   - In a subsequent window (when the factor ends or reverses), include a corresponding delta that reverts, adjusts, or drops the temporary change
   - Ensure the timeline is realistic (e.g., Olympics last ~2 weeks in summer; winter lasts ~3 months; heatwaves last days to weeks)

---

### Realism Requirements

All generated states and deltas MUST reflect **realistic, everyday human life**.
This means:

1. **No placeholders**:
   - DO NOT use vague placeholder names such as "brand A", "brand B", "model X",
     "food brand Y", "camera brand Z", or any generic label without real meaning.
   - If you introduce a concrete brand, product, or item, you must give a **realistic,
     human-like descriptive note** that explains what the item is.
     This description should be short (5-12 words) and express a useful property
     (e.g., budget level, common usage, defining feature), not marketing language.

   - Example format:
       "Blue Buffalo Life Protection (a mid-range dry dog food focused on digestion)"
       "Canon EF 50mm f/1.8 (an entry-level prime lens for portraits)"
       "iRobot Roomba 694 (a budget robot vacuum for small apartments)"

2. **No vague descriptions**:
   - Facts about the user must be stated definitively, not as guesses using words like “likely” or “probably.”

---
### Output Format (JSON only)

**CRITICAL FORMATTING REQUIREMENTS:**
- Every window object MUST include "window_description" and "summary" fields
- The initial_state object MUST include a "summary" field
- These fields are NOT optional - your output will be considered incomplete without them

Return strictly valid JSON with this schema:
{
  "life_domain": "{{ life_domain }}",
  "initial_state": {
      "user_attributes_state": {
        "initial": {
          {
            "<attribute_name>": <attribute value, list format of concrete values>,
          },
          ...
        }
      },
      "habits_state": {
        "initial": {
          "<habit_name>": {
              "action": "<concrete action label, e.g., 'walk_dog_on_leash'>",
              "frequency": "<frequency pattern, e.g., 'daily', '3_times_per_week'>",
              "timing": "<typical timing within the day, e.g., 'early morning before work (6:00 AM–6:30 AM)'>",
              "context": "<setting / environment, e.g., 'around the neighborhood for 20–30 minutes'>",
              "description": "<5–20 words natural language description of this habit>"
            }
          ...
        }
      },
      "preferences_state": {
        "initial": {
          {
            "<preference_name>": "<preference value, should be a concrete value, 5-20 words description of what this preference value means in natural language>",
          },
          ...
        }
      },
      "summary": "<summary of inital state, should include all the attributes, habits, and preferences in initial state>",
    },
  "time_windows": [
    {
      "window_id": "w1",
      "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "window_description": "<short description of the general conditions that motivate changes>",
      "user_attributes_delta": {
        "operations": [
          {
            "op": "add" | "remove" | "modify",
            "attribute_name": "<attribute name (e.g., 'user_occupation', 'user_family_members', 'user_owned_devices')>",
            "new_state": <new value or null, list format of concrete values>,
            "reason": "<short reason for this change>"
          }
        ]
      },
      "habits_delta": {
        "operations": [
          {
            "op": "acquire" | "adjust" | "drop",
            "habit_name": "<habit name (e.g., 'user_walks_dog_morning', 'user_evening_gym_sessions')>",
            "new_state": "<null if the habit is dropped, or a JSON object with fields {action, frequency, timing, context, description}; description must be a concrete 5–20 word explanation of what this habit means in natural language>",
            "reason": "<short reason>"
          }
        ]
      },
      "preferences_delta": {
        "operations": [
          {
            "op": "shift" | "amplify" | "attenuate",
            "preference_name": "<preference name (e.g., 'preferred_subject_to_photograph', 'preferred_type_of_pet', 'preferred_exercise_style')>",
            "new_state": "<new value, should be a concrete value, 5-20 words description of what this preference value means in natural language>",
            "reason": "<short reason (e.g., new experience / world factor)>"
          }
        ]
      },
      "summary": "<summary of this window, should include all the attributes, habits, and preferences within this window>",
    },
    ...
  ]
}
"""
)


@dataclass
class DynamicProfileRequest:
    domain_name: str
    world_background: str
    user_profile: str
    time_windows: List[Dict[str, str | List[str]]]
    domain_scope_definition: str | None = None
    life_domain_list: List[str] | str | None = None


def render_dynamic_profile_prompt(request: DynamicProfileRequest) -> str:
    import json
    # time_windows_json = json.dumps(request.time_windows, indent=4, ensure_ascii=False)
    time_windows_json = request.time_windows
    life_domain_list = request.life_domain_list

    return dynamic_profile_template.render(
        life_domain=request.domain_name,
        scope_definition=request.domain_scope_definition or "",
        world_background=request.world_background,
        user_profile=request.user_profile,
        time_windows_json=time_windows_json,
        domain_scope_definition=request.domain_scope_definition or "",
        life_domain_list=life_domain_list,
    )


def generate_dynamic_profile(
    llm_client: GeminiJSONClient, request: DynamicProfileRequest
) -> LLMResult:
    prompt = render_dynamic_profile_prompt(request)

    print(prompt)
    return llm_client.generate_json(prompt)
