from dataclasses import dataclass
from typing import List, Dict, Any

from jinja2 import Template

from trajectory_synthesis.llm_client import (
    GeminiJSONClient,
    LLMResult,
)
from trajectory_synthesis.prompt_templates import WORLD_BACKGROUND_GENERATION_PROMPT

dynamic_profile_template = Template("""You are an expert simulator of long-horizon dynamic user profile trajectories.

### Your task:
Given the world background, the basic user profile, and ONE life domain, time windows, generate a **time-windowed user profile trajectory** for this life domain over the provided time windows.

**Simulation approach**: We use an **initial state + incremental deltas** model:
- Define the baseline state at time 0 (initial_state)
- For each subsequent time window, specify only what CHANGED (deltas), using valid change_type to represent the change.


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

  **STRUCTURE**: Attributes are divided into two types:
  
  #### Type 1: Singular Attributes
  - **Semantics**: Dimensions that have exactly ONE value at any given time, but the value can change over time.
  - **Structure**: Simple key-value pairs where each key is an attribute name and the value is a concrete description string.
  
  Examples of singular attributes:
  "singular": {
    "primary_health_insurance": "Blue Cross Blue Shield PPO plan through employer (Family coverage, $300/month copay)",
    "primary_banking_institution": "Chase Bank checking account ending in 4521 (main account for direct deposit and bills)",
    "commute_mode": "Driving personal car to office (25-minute commute each way on weekdays)",
  }
  
  - **Delta Format**: Use modify change_type to represent the change.
  For singular attributes, only the **modify** change_type is valid, because it is a single value and cannot be added or removed.
  - Details of the **modify** change_type:
    - **modify**:
      Meaning: Replace the current value with a new value.
      **CRITICAL**: You can ONLY modify attributes that already exist in previous state.
      **DELTA FORMAT**: Provide the new value as a string.
      
      Example:
        Current state: {
          "primary_residence": "Rented studio apartment in downtown area"
        }
        Delta: {
          "change_type": "modify",
          "attribute_type": "singular",
          "attribute_name": "primary_residence",
          "delta": "Recently purchased two-bedroom apartment in suburban area",
          "reason": "Bought first home after saving for down payment"
        }
  
  #### Type 2: Collection Attributes
  - **Semantics**: Dimensions that can have MULTIPLE items simultaneously, where items can be added or removed independently.
  - **Structure**: Key-value pairs where each key is a collection name and the value is an **array of concrete description strings**.
  - **Important**: Each description string should be sufficiently specific to be uniquely identifiable (include brand, model, key details).
  
  Examples of collection attributes:
  "collections": {
    "cooking_equipment": [
      "KitchenAid Artisan stand mixer (5-quart capacity for baking)",
      "Lodge cast iron skillet 12-inch (seasoned, for high-heat cooking)",
      "Vitamix E310 blender (entry-level model for smoothies and soups)"
    ],
    "exercise_equipment": [
      "Bowflex SelectTech 552 adjustable dumbbells (5-52.5 lbs per hand)",
      "Manduka PRO yoga mat (6mm thick, for home practice)",
      "TRX Home2 suspension trainer (mounted in spare bedroom doorframe)"
    ],
    "professional_certifications": [
      "AWS Certified Solutions Architect - Associate (obtained March 2023, valid until March 2026)",
      "Project Management Professional PMP (obtained June 2022, requires renewal in 2025)",
      "Certified ScrumMaster CSM (obtained January 2024 from Scrum Alliance)"
    ]
  }

  - **Delta Format**: Use add and remove change_type to represent the change.
    For collection attributes, the **add** and **remove** change_type are allowed, because it is a multi-item collection and can be added or removed. Modify is not allowed, it can be represented as a combination of **add** and **remove**.
    - Details of the **add** and **remove** change_type:
      - **add**:
        Meaning: Add new items to a collection array. This covers:
          (1) The collection already exists: append new items to the existing array
          (2) The collection doesn't exist: create the collection with initial items

        **DELTA FORMAT**: Provide an array of new description strings to add
        
        Example (add to existing collection):
          Current state: {
            "fitness_gear": [
              "Nike Air Zoom Pegasus 39 running shoes (neutral support for road running)",
              "Garmin Forerunner 245 GPS watch (tracks runs and heart rate)"
            ]
          }
          Delta: {
            "change_type": "add",
            "attribute_type": "collections",
            "collection_name": "fitness_gear",
            "delta": [
              "Aftershokz OpenRun bone conduction headphones (for safe outdoor running with music)",
              "Nathan SpeedDraw Plus insulated water bottle (handheld 18oz for long runs)"
            ],
            "reason": "Purchased accessories to improve comfort and safety during longer training runs"
          }

        Example (create new collection):
          Current state: (no 'meal_prep_containers' exists)
          Delta: {
            "change_type": "add",
            "attribute_type": "collections",
            "collection_name": "meal_prep_containers",
            "delta": [
              "Prep Naturals 5-pack glass containers with lids (3-compartment, 36oz each for weekly meal prep)"
            ],
            "reason": "Started meal prepping on Sundays to save time and eat healthier during work week"
          }

      - **remove**:
        Meaning: Remove specific items from a collection array.
        **DELTA FORMAT**: Provide an array of description strings to remove (must match exactly).
        
        Example:
          Current state: {
            "photography_lenses": [
              "Canon EF-S 18-55mm f/3.5-5.6 IS STM (kit lens, general purpose)",
              "Canon EF 50mm f/1.8 STM (nifty fifty prime for portraits)",
              "Sigma 10-20mm f/3.5 EX DC HSM (ultra-wide angle for landscapes)"
            ]
          }
          Delta: {
            "change_type": "remove",
            "attribute_type": "collections",
            "collection_name": "photography_lenses",
            "delta": [
              "Canon EF-S 18-55mm f/3.5-5.6 IS STM (kit lens, general purpose)"
            ],
            "reason": "Sold kit lens after upgrading to better general-purpose zoom lens"
          }
        
        **Important**: The description strings in the delta must match the exact strings in the current state.

  **How to decide which type:**
  - Ask: "Can the user have multiple of these simultaneously?"
    - YES: Collection (e.g., devices, friends, hobbies, subscriptions, skills)
    - NO: Singular (e.g., primary residence, marital status, main job)
  - Ask: "Does it make sense to talk about 'adding' or 'removing' items?"
    - YES: Collection
    - NO: Singular (you modify the single value instead)
  
  **Constraints for all attributes:**
  - All attribute values must be concrete and unambiguous.
    Bad: "Master's degree in technology-related field" (Too vague)
    Good: "Master's degree in Computer Science from State University"
    Bad: "Moderate income" (Too vague)
    Good: "42000 USD per year before tax from primary software engineering job"
    Bad: "High-end laptop" (Too vague)
    Good: "Dell XPS 13 (ultrabook used mainly for software development)"
  
  - For collection items, each description must be specific enough to be uniquely identifiable.
    This typically means including: brand/name, model/type, and key distinguishing details.


- **Habits**:
  Definition: Recurring behavioral patterns (e.g., morning jogging routine, weekly meal prep), which can be identified and consolidated from repeated observations.
  Each habit MUST be represented as a structured JSON object capturing schedule, timing, location, and priority.

  **Required fields:**
  
  - **habit_name**: A unique identifier for this specific habit instance (used as dictionary key).
    - **Purpose**: Identifies the essential nature/identity of THIS particular habit
    - **Naming convention**: Use underscore_case, describe WHAT the habit is (not WHEN)
    - **Examples**:
      - "outdoor_run"
      - "gym_strength_training"
      - "dog_walk"
    - **Anti-examples (DO NOT use time/frequency words in habit_name)**:
      - "morning_jog" → "outdoor_jog" ✓
      - "weekly_meal_prep" → "meal_prep" ✓
      - "daily_meditation" → "meditation" ✓
  
  - **schedule**: When this habit occurs.
    **Format**: JSON object with "frequency_type" key plus additional required fields based on type.
    **Structure**: {"frequency_type": <type>, <additional_required_fields>}

    **Type 1: Daily**
    Format: {"frequency_type": "daily"}
    Example: {"frequency_type": "daily"}

    **Type 2: Weekly**
    Format: {"frequency_type": "weekly", "days_of_week": [<day_indices>]}
    Required fields:
      - frequency_type: "weekly"
      - days_of_week: Array of integers from 0-6, where 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    MUST specify exact days. NO vague "3 times per week".
    Can be one or multiple days.
    Examples:
      - {"frequency_type": "weekly", "days_of_week": [1, 3, 5]} // Tue/Thu/Sat
      - {"frequency_type": "weekly", "days_of_week": [5, 6]} // Weekends only
      - {"frequency_type": "weekly", "days_of_week": [0, 1, 2, 3, 4]} // Weekdays only

    **Type 3: Biweekly**
    Format: {"frequency_type": "biweekly", "days_of_week": [<day_index>], "start_date": "YYYY-MM-DD"}
    Required fields:
      - frequency_type: "biweekly"
      - days_of_week: Array with single integer 0-6 (same encoding as weekly)
      - start_date: The first occurrence date in YYYY-MM-DD format
    Repeats every 2 weeks from the start_date.
    Example: {"frequency_type": "biweekly", "days_of_week": [3], "start_date": "2024-03-07"} // Every other Thursday starting March 7, 2024

    **Type 4: Monthly by date**
    Format: {"frequency_type": "monthly_by_date", "days_of_month": [<day_numbers>]}
    Required fields:
      - frequency_type: "monthly_by_date"
      - days_of_month: Array of integers from 1-28 (avoid 29-31 to prevent skipping months)
    MUST specify exact dates. NO vague "once a month".
    Can be one or multiple dates.
    Examples:
      - {"frequency_type": "monthly_by_date", "days_of_month": [1]} // 1st of every month
      - {"frequency_type": "monthly_by_date", "days_of_month": [1, 15]} // 1st and 15th of every month

    **Type 5: Monthly by nth weekday**
    Format: {"frequency_type": "monthly_nth_weekday", "week_of_month": <1-4 or "last">, "day_of_week": <0-6>}
    Required fields:
      - frequency_type: "monthly_nth_weekday"
      - week_of_month: Integer 1-4 or string "last" (1=first week, 2=second week, "last"=last week)
      - day_of_week: Integer 0-6 (0=Mon, 1=Tue, ..., 6=Sun)
    Examples:
      - {"frequency_type": "monthly_nth_weekday", "week_of_month": 1, "day_of_week": 0} // First Monday of every month
      - {"frequency_type": "monthly_nth_weekday", "week_of_month": "last", "day_of_week": 4} // Last Friday of every month
      - {"frequency_type": "monthly_nth_weekday", "week_of_month": 3, "day_of_week": 5} // Third Saturday of every month

  - **timing**: Time window when the habit occurs.
    **Format**: JSON object with exactly two keys: "start_time" and "end_time"
    **Structure**: {"start_time": "HH:MM", "end_time": "HH:MM"}
    **Requirements**:
      - Both times must use 24-hour format (HH:MM)
      - end_time must be after start_time
      - Duration (end_time - start_time) MUST NOT exceed 3 hours
    **Examples**:
      - {"start_time": "06:30", "end_time": "07:00"} // 30-minute morning activity
      - {"start_time": "14:00", "end_time": "16:00"} // 2 hour afternoon activity (acceptable, under 3 hours)
  
  - **location**: Where it happens (string, 3-10 words). Pure location only.
    Examples: "neighborhood park", "24 Hour Fitness on Main St", "home office"
  
  - **priority**: Importance for scheduling (ENUM).
    MUST be one of: "critical" | "high" | "medium" | "low"

  **CRITICAL constraints:**
  
  1. **frequency_type is ENUM**: Only use "daily", "weekly", "biweekly", "monthly_by_date", or "monthly_nth_weekday"
  2. **All schedules fully specified**: List exact days/dates, not vague frequencies
  3. **Max duration**: 3 hours per occurrence (Strictly enforced)
     **Examples of INCORRECT duration:**
      - Bad: "childcare duties" from 07:00 to 19:00 (exceeds 3 hours and also not a discrete activity)
      - Bad: "studying" from 08:00 to 18:00 (exceeds 3 hours)
  4. **Fixed timing**: All timings are fixed (no flexibility parameter)
  5. **No time/frequency words in habit_name**: Use schedule field for timing info
  6. **No time conflicts**: Habits cannot overlap in time on any shared day; schedules must be realistic.
  7. **Travel buffer across locations**: If two habits occur on the same day in different locations, leave at least 30 minutes between end_time and the next start_time.

  **Complete examples:**
  "dog_walk": {
    "schedule": {"frequency_type": "daily"},
    "timing": {"start_time": "06:30", "end_time": "07:00"},
    "location": "neighborhood park",
    "priority": "high"
  }
  "budget_review": {
    "schedule": {"frequency_type": "monthly_by_date", "days_of_month": [1]},
    "timing": {"start_time": "19:00", "end_time": "20:00"},
    "location": "home",
    "priority": "medium"
  }

  **Allowed change_type:**
  User can acquire a new habit, drop an existing habit, or adjust an existing habit.
  
  - **acquire**: Start a new habit. Delta = complete habit object.
  
  - **drop**: Stop an existing habit. Delta = null.
  
  - **adjust**: Modify an existing habit without changing its core identity.
    - **Definition**: Use adjust when the habit remains fundamentally the same activity but details change.
    - Changes that qualify for adjust: schedule, timing, location, priority
    - **CRITICAL**: You can ONLY adjust habits that already exist in previous state.
    - **CRITICAL**: Must change at least one field (schedule, timing, location).
    - Delta = only the changed fields (partial object).
    
    Example (change schedule):
    {
      "change_type": "adjust",
      "habit_name": "outdoor_jog",
      "delta": {
        "schedule": {"frequency_type": "daily"}
      },
      "reason": "Building up cardiovascular endurance, increased from 3x/week to daily"
    }

    Example (change timing):
    {
      "change_type": "adjust",
      "habit_name": "reading",
      "delta": {
        "timing": {"start_time": "21:00", "end_time": "22:00"}
      },
      "reason": "Work schedule changed, shifted reading time one hour later"
    }

    Example (change location):
    {
      "change_type": "adjust",
      "habit_name": "outdoor_run",
      "delta": {
        "location": "state park trail"
      },
      "reason": "Discovered better running route with more scenery"
    }

- **Preferences**:
  Definition: Subjective inclinations that guide choices (e.g., preferring window seats, favoring spicy food).
  It describes *what* the user tends to choose when multiple alternatives exist.
  
  **IMPORTANT**: Preferences are **stable internal inclinations** that represent the user's underlying tendencies. They are distinct from habits in that they reflect what the user genuinely prefers or values, not just what they do. Preferences are relatively resistant to short-term fluctuations and typically require sustained experiences or significant events to change.
  
  Each preference MUST be represented as a structured object with a statement and supporting signals.

  **Structure:**
  ```json
  "preference_name": {
    "statement": "<concrete preference statement, 10-30 words>",
    "signals": [
      "<observable behavior or evidence 1>",
      "<observable behavior or evidence 2>",
      "<observable behavior or evidence 3>"
    ]
  }
  ```

  - **statement**: A clear, concrete preference statement describing what the user prefers.
    - Must use comparative language: "prefers A over B", "favors X rather than Y"
    - Can include intensity: "strongly prefers", "somewhat prefers", "slightly favors"
    - Should be specific to this domain and actionable
  
  - **signals**: Array of 2-4 observable behaviors or evidence that support this preference.
    - Must be concrete and verifiable (not abstract feelings)
    - **CRITICAL**: Must be observable within a single time window (typically 1-3 months)
    - Should demonstrate the preference through recent, specific actions or choices
    - Avoid cross-window references like "over the past year" or "has not done X in months"
    
    **Good signals (observable within window):**
    - "Chose solo morning runs over gym class invitations twice this month"
    - "Purchased running shoes and outdoor gear rather than gym membership"
    - "Declined three group fitness class invitations from coworkers"
    
    **Bad signals (too vague or cross-window):**
    - "Has not made individual stock trades in over a year" (cross-window)
    - "Consistently chooses running" (too vague, no concrete timeframe)
    - "Reads books on passive investing" (could mean read once or many times)

  **Examples:**
  ```json
  "preferences_state": {
    "initial": {
      "exercise_setting": {
        "statement": "Prefers solo outdoor activities like running and cycling over group gym classes",
        "signals": [
          "Chose solo morning runs over gym class invitations twice this month",
          "Declined coworker's invitation to join CrossFit group",
          "Purchased running shoes and bike gear rather than considering gym membership"
        ]
      },
      "investment_focus": {
        "statement": "Prefers long-term passive investments in diversified index funds over active stock trading",
        "signals": [
          "Allocated 90% of this quarter's savings to Vanguard index funds",
          "Chose to rebalance portfolio using low-cost ETFs rather than picking individual stocks",
          "Spent time reading 'The Simple Path to Wealth' instead of day-trading guides"
        ]
      },
      "learning_approach": {
        "statement": "Prefers hands-on project-based learning over passive video tutorials or reading",
        "signals": [
          "Started building a web scraper project immediately after learning basic Python syntax",
          "Skipped tutorial videos and went straight to documentation for new React library",
          "Created three practice projects this month to learn new frameworks"
        ]
      }
    }
  }
  ```

  **Constraints:**
  - For this domain, you MUST propose a small set of **2-4 distinct preference dimensions**
  - Dimensions should be: non-redundant, interpretable, and meaningful for this domain
  - All statement values must be concrete and unambiguous
    Bad: "better", "worse", "moderate", "likes it a lot" (Too vague)
    Good: "Prefers quiet solo activities over group-based social gatherings"
  - **Signals must be observable within the time window** (avoid "over the past year" or similar cross-window references)
  - Signals should be specific, recent actions or choices that clearly demonstrate the preference

  **Common triggers of preference change:**
  
  Preferences are stable and resistant to change. They typically change only through:
  
  1. **Sustained new experiences** (e.g., trying a new activity multiple times over weeks or months and discovering genuine enjoyment that challenges prior preferences)
  2. **Significant life events** (e.g., injury forcing permanent adaptation, becoming a parent, major career change)
  3. **Deliberate reflection** (e.g., reading research that challenges assumptions, extended mentorship, therapeutic insights)
  
  **What does NOT usually change preferences:**
  
  - Single seasonal shifts (winter does not make someone "prefer" indoors if they genuinely love nature)
  - One-time events (attending one concert does not shift music taste preference)
  - Temporary constraints (busy week does not change preference for exercise type)
  - Short-term external factors (heatwave, special event, temporary circumstance)
  
  **Season/weather should affect habits, not preferences:**
  
  CORRECT modeling:
  - User prefers outdoor running (preference remains stable)
  - Winter arrives: switches to indoor treadmill (habit drop + acquire)
  - Spring returns: resumes outdoor running (habit drop + acquire)
  - Preference never changed throughout
  
  INCORRECT modeling:
  - User prefers outdoor running (initial preference)
  - Winter arrives: preference shifts to indoor exercise (too shallow, unrealistic)

  **Allowed change_type:**

  - **shift**:
    Meaning: Preference direction changes (from preferring A to preferring B).
    **CRITICAL**: You can ONLY use "shift" if this preference already exists in a previous window.
    **CRITICAL**: Shifts require sustained experiences or significant events, not short-term factors.
    **DELTA FORMAT**: Provide the complete new preference object (statement + signals).
    
    Example (shift due to sustained experience):
    ```json
    // Previous state:
    "exercise_setting": {
      "statement": "Prefers solo outdoor activities like running and cycling over group gym classes",
      "signals": [
        "Chose solo morning runs over gym class invitations twice last month",
        "Declined multiple invitations to join group fitness classes",
        "Invested in running gear rather than gym equipment"
      ]
    }
    
    // Delta (shift change_type after sustained team sports experience):
    {
      "change_type": "shift",
      "preference_name": "exercise_setting",
      "delta": {
        "statement": "Prefers group fitness classes and team sports over solo outdoor activities",
        "signals": [
          "Joined recreational soccer league and attended 8 sessions over 2 months",
          "Voluntarily invited 3 friends to join after discovering enjoyment of team dynamics",
          "Declined solo hiking trip invitation to attend team practice",
          "Signed up for additional group yoga classes at local studio"
        ]
      },
      "reason": "After consistent participation in soccer league over 2 months, discovered deep enjoyment of social motivation and team camaraderie that outweighs previous preference for solitude"
    }
    ```

  - **refine**:
    Meaning: Preference in the same direction is refined or adjusted in strength/specificity.
    **CRITICAL**: You can ONLY use "refine" if this preference already exists in a previous window.
    **DELTA FORMAT**: Provide the complete new preference object (statement + signals).
    
    Example (strengthening):
    ```json
    // Previous state:
    "investment_focus": {
      "statement": "Prefers long-term investments in technology stocks over short-term trading",
      "signals": [
        "Purchased tech stocks with plan to hold for multiple years",
        "Checked portfolio only twice this month rather than daily",
        "Read annual reports rather than following daily market news"
      ]
    }
    
    // Delta (refine change_type - strengthening):
    {
      "change_type": "refine",
      "preference_name": "investment_focus",
      "delta": {
        "statement": "Strongly prefers long-term buy-and-hold investments in diversified index funds, actively avoiding individual stock picking and any short-term trading",
        "signals": [
          "Moved 95% of individual tech stocks into Vanguard Total Market Index Fund this month",
          "Set up automatic monthly contributions to avoid any active trading decisions",
          "Unsubscribed from stock-picking newsletters and installed portfolio app blocking for weekdays",
          "Purchased 'A Random Walk Down Wall Street' and highlighted key passive investing sections"
        ]
      },
      "reason": "After reading investment research and experiencing market volatility, conviction in passive long-term strategy strengthened significantly"
    }
    ```
    
    Example (weakening):
    ```json
    // Previous state:
    "learning_approach": {
      "statement": "Strongly prefers hands-on project-based learning, actively avoiding passive video tutorials or reading",
      "signals": [
          "Started three coding projects without watching any tutorials",
          "Skipped documentation and learned purely through trial and error",
          "Explicitly avoided video courses despite recommendations"
        ]
    }
    
    // Delta (refine change_type - weakening):
    {
      "change_type": "refine",
      "preference_name": "learning_approach",
      "delta": {
        "statement": "Prefers hands-on projects but now values video tutorials for quick skill acquisition before diving in",
        "signals": [
          "Watched 20-minute React tutorial before starting new web project this week",
          "Used TypeScript video course to understand basics before hands-on practice",
          "Still built two practice projects this month but preceded each with focused tutorial viewing"
        ]
      },
      "reason": "Found that video tutorials are effective for learning new tools quickly before hands-on practice"
    }
    ```
  
  **Note on change_type:**
  - **shift** = change preference direction (solo to group, active to passive, A to B)
  - **refine** = same direction, but adjust strength/specificity (prefer to strongly prefer, or vice versa)
  - Both change_type require more than short-term external factors; they need sustained experiences or significant events
  
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
  - "daily_industry_podcast" habit with "listen to tech podcast during commute" 
    (This belongs to **Work & Education**)
  - "morning_gym_routine" habit 
    (This belongs to **Health & Self-care**)

- If building **Health & Self-care**, do NOT include:
  - "weekly_family_dinner" habit 
    (This belongs to **Family & Close Relationships**)
  - "professional_certification_course" attribute 
    (This belongs to **Work & Education**)

---

### Generation Approach

Now that you understand the core concepts and constraints, here's how to generate the trajectory:

**For the initial state (window 0):**
1. Define the baseline state for attributes (both singular and collections), habits, and preferences in this domain
2. Write a summary describing the user's starting state

**For each subsequent time window (window 1, 2, 3, ...):**
1. **First, write the window_description**: Identify the salient external factors (from world background: seasons, events, circumstances) and the user's internal motivations/focus during this period. This is the **latent causal variable** that will drive coordinated changes across attributes, habits, and preferences.

2. **Then, generate the trajectory deltas**: Based on the window_description, determine how attributes, habits, and preferences evolve together in directionally coherent ways:
   - user_attributes_delta: what singular attributes get modified, collections get added, or collections get removed (delta format)
   - habits_delta: what habits are acquired, adjusted, or dropped (delta format)
   - preferences_delta: what preferences shift or refine (delta format)
   
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
   - Example for **Health & Self-care** domain: new fitness goal (motivation) leads to buys running shoes (attribute) leads to starts jogging routine (habit) leads to prefers cardio activities (preference)

4. **Essential items planned to evolve must start in initial_state**:
   - **CRITICAL**: If an item is **essential for this domain** AND you plan to **evolve it in later windows** (modify, adjust, shift), it MUST be initialized in initial_state with a baseline value.
   - Essential = things a realistic person in this situation would naturally already have.
   
   **An example of a bad case (essential item not initialized before evolution):**
   - **Finances & Material Living** domain: NO smartphone in initial_state, then adds first phone in window 3 (Unrealistic: most people already own a smartphone as an essential item)

5. **Short-term vs Long-term changes**:
   - Be aware of whether a change is driven by **short-term external factors** (e.g., seasonal changes, special events, temporary circumstances) or **long-term shifts** (e.g., lifestyle changes, permanent acquisitions)
   - **For short-term changes**, you MUST plan follow-up adjustments to revert or normalize the behavior once the external factor ends
   
   Examples of short-term changes requiring follow-up:
   - **Seasonal adjustment**: If a user starts a daily hydration habit in winter to combat dry air, this should be adjusted back (reduced or dropped) when summer arrives
   - **Event-driven habit**: If a user acquires a habit of watching Olympic events every evening during the Olympics, this habit MUST be dropped once the Olympics end
   - **Temporary circumstance**: If a user switches from outdoor to indoor exercise during winter, they should switch back when spring arrives
   
   **How to implement**:
   - When you introduce a change motivated by a short-term factor, note this in the "reason" field
   - In a subsequent window (when the factor ends or reverses), include a corresponding delta that reverts, adjusts, or drops the temporary change
   - Ensure the timeline is realistic (e.g., Olympics last approximately 2 weeks in summer; winter lasts approximately 3 months; heatwaves last days to weeks)

6. **Special considerations for preferences**:
   - **Preferences are inherently long-term** and should rarely change due to short-term factors
   - If external factors (season, event, temporary constraint) affect user behavior, model this as **habit changes (drop + acquire)**, not preference shift
   - Reserve preference changes for windows where there's evidence of **sustained experience or fundamental reassessment**
   - Typical frequency: preferences may change every 2-4 windows at most, while habits and attributes can change more frequently

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
       "Blue Buffalo Life Protection (mid-range dry dog food focused on digestion)"
       "Canon EF 50mm f/1.8 (entry-level prime lens for portraits)"
       "iRobot Roomba 694 (budget robot vacuum for small apartments)"

2. **No vague descriptions**:
   - Facts about the user must be stated definitively, not as guesses using words like "likely" or "probably."

---
### Output Format (JSON only, Dict format)

**CRITICAL FORMATTING REQUIREMENTS:**
- Every window object MUST include "window_description" and "summary" fields
- The initial_state object MUST include a "summary" field
- These fields are NOT optional - your output will be considered incomplete without them

Return strictly valid JSON with this schema (**Dict format, NOT List format**):
{
  "life_domain": "{{ life_domain }}",
  "initial_state": {
    "user_attributes_state": {
      "singular": {
        "<attribute_name>": "<concrete value string>"
      },
      "collections": {
        "<collection_name>": [
          "<concrete description string>"
        ]
      }
    },
    "habits_state": {
      "<habit_name>": {
        "schedule": {
          "frequency_type": "daily | weekly | biweekly | monthly_by_date | monthly_nth_weekday",
          "...": "other required schedule fields"
        },
        "timing": {
          "start_time": "HH:MM",
          "end_time": "HH:MM"
        },
        "location": "<3–10 words>",
        "priority": "critical | high | medium | low"
      }
    },
    "preferences_state": {
      "<preference_name>": {
        "statement": "<10–30 word concrete preference statement>",
        "signals": [
          "<observable signal 1 (within window)>",
          "<observable signal 2 (within window)>"
        ]
      }
    },
    "summary": "<2–4 sentence summary of initial state>"
  },
  "time_windows": [
    {
      "window_id": "w1",
      "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "window_description": "<1–3 sentences describing external + internal drivers>",
      "user_attributes_delta": {
        "changes": [
          {
            "change_type": "<modify | add | remove>",
            "attribute_type": "<singular | collections>",
            "attribute_name": "<singular attribute name | collection name>",
            "delta": "<new concrete value | new items to add | items to remove>",
            "reason": "<short reason for the change>"
          },
          ...
        ]
      },
      "habits_delta": {
        "changes": [
          {
            "change_type": "<acquire | adjust | drop>",
            "habit_name": "<habit name>",
            "delta": {
              "schedule": {
                "frequency_type": "daily | weekly | biweekly | monthly_by_date | monthly_nth_weekday",
                "...": "other required schedule fields, no need for daily, weekly need .."
              },
              "timing": {
                "start_time": "HH:MM",
                "end_time": "HH:MM"
              },
              "location": "<3–10 words>",
              "priority": "critical | high | medium | low"
            },
            "reason": "<short reason>"
          },
          ...
        ]
      },
      "preferences_delta": {
        "changes": [
          {
            "change_type": "shift | refine",
            "preference_name": "<existing preference name>",
            "delta": {
              "statement": "<10–30 word preference statement>",
              "signals": [
                "<observable signal 1 (within window)>",
                "<observable signal 2 (within window)>"
              ]
            },
            "reason": "<short reason>"
          }
        ]
      },
      "summary": "<2–4 sentence synthesis explicitly linked to window_description>"
    }
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
    return llm_client.generate_json(prompt)


# =============================================================================
# World Background Generation
# =============================================================================

@dataclass
class WorldBackgroundRequest:
    """Request for generating a personalized world background."""
    user_basic_profile: Dict[str, Any]
    domain_name: str
    domain_scope_definition: str


@dataclass
class WorldBackgroundResult:
    """Result from world background generation."""
    world_backgrounds: List[Dict[str, Any]]  # Per-window backgrounds
    combined_background: str  # Unified narrative for all windows
    usage: Dict[str, int]
    prompt: str
    raw_text: str


def render_world_background_prompt(request: WorldBackgroundRequest) -> str:
    """Render the prompt for world background generation."""
    import json
    return WORLD_BACKGROUND_GENERATION_PROMPT.render(
        user_basic_profile_json=json.dumps(request.user_basic_profile, indent=2, ensure_ascii=False),
        domain_name=request.domain_name,
        domain_scope_definition=request.domain_scope_definition,
    )


def generate_world_background(
    llm_client: GeminiJSONClient,
    request: WorldBackgroundRequest,
) -> WorldBackgroundResult:
    """
    Generate a personalized world background for a specific user and domain.

    The world background is tailored to:
    1. The user's geographic location and cultural context
    2. The specific life domain being simulated
    3. The predefined 5 time windows (2023-10-01 to 2024-12-31)

    Args:
        llm_client: The LLM client for generation
        request: WorldBackgroundRequest containing user profile and domain info

    Returns:
        WorldBackgroundResult with per-window backgrounds and combined narrative
    """
    prompt = render_world_background_prompt(request)
    result = llm_client.generate_json(prompt)

    return WorldBackgroundResult(
        world_backgrounds=result.data.get("world_backgrounds", []),
        combined_background=result.data.get("combined_background", ""),
        usage=result.usage,
        prompt=prompt,
        raw_text=result.raw_text,
    )
