QUESTION7_PROMPT_TEMPLATE = """You are an automatic question writer based on an atomic fact (“atom”) and a supporting context (“context”).

I will provide you with two inputs:
- atom: a single atomic fact object containing fields such as atom_id, domain, type, anchor, time_window, key, value, path, etc.
- context: a relevant excerpt from the original source material (e.g., a book or document), used as source material for writing questions

Your task is to generate questions whose answers are uniquely and precisely locked to the given atom, while using the context as the source of the question statements.

You must strictly follow ALL rules below:

1) The context is only question-writing material. You are NOT required to use all of it, but the wording and factual basis of each question must come from the context.
2) The answer to each question must be locked to the atom:
   - By default, the answer should equal atom.value.
   - If atom.key is a more appropriate unique anchor, the answer must still resolve to the concrete value associated with that key.
   - Do NOT generalize or allow multiple possible values.
3) You must use the context to construct the question so that the answer is uniquely determined.
   - Avoid broad or vague questions that could have multiple valid answers in the original book but are not uniquely covered by the provided context.
4) Task background (important for correctness):
   - I locate answers automatically from structured notes (atoms), then retrieve related excerpts (context) to help you write questions.
   - The answering side will read the original book/source, NOT my notes.
   - Therefore, your questions must appear to come naturally from the original source material, and must not rely on information outside the given context to be uniquely answered.
5) You must generate EXACTLY 7 questions:
   - 3 multiple-choice questions (single choice, 4 options A/B/C/D, only one correct)
   - 2 fill-in-the-blank questions (short, exact answers)
   - 2 short-answer questions (1–3 sentences, tightly focused on the atom)
   All questions must be narrowly scoped and tightly bound to the answer.
6) Output format:
   - Output MUST be a JSON array and ONLY a JSON array (no extra text).
   - Each element must be an object with the following fields:
     - draft1: your reasoning BEFORE writing the question (how you ensure uniqueness and alignment with the atom)
     - question: the question text
     - draft2: your reasoning BEFORE writing the final answer (why the answer is uniquely correct)
     - answer: the standard answer
   - draft1 and draft2 may be written in English or Chinese, but must be concise and focused on uniqueness and atom alignment.

Additional strict rules:
- Do NOT mention “atom”, “context”, “notes”, “path”, or any internal system concepts in the questions.
- Do NOT invent facts not present in the context.
- If the context contains similar entities or values, you MUST disambiguate using time, object name, or explicit constraints from the context.
- If the context is insufficient to uniquely lock the answer, you must refine the question (not add new facts) until it becomes uniquely answerable.

CRITICAL CONSTRAINT ON QUESTION NATURALNESS:

- You MUST NOT write questions that reference schema-level concepts such as:
  "operation entry", "op field", "time_window", "preference_name",
  "operations list", or any internal structural terminology.

- All questions must be phrased as if they were written directly
  from the original source text (book / article / document),
  using natural language descriptions of actions, changes, or decisions.

- The atom is ONLY an answer anchor, NOT a topic to be mentioned.
  You must first translate the atom.value into a human-readable semantic intent,
  then express that intent using wording grounded in the context.

- Answers must be uniquely correct by semantic meaning,
  not by schema label matching.
  A reader unfamiliar with the schema but familiar with the source text
  should be able to answer correctly.

- If the only way to make the answer unique is to mention schema terms,
  the question is INVALID and must be rewritten.

ADDITIONAL TASK BACKGROUND — TIME SEMANTICS (IMPORTANT):

- The questions are generated for an agent being evaluated on its ability
  to answer questions based on a user's application logs over time.

- The timeline covers the full year of 2024.
  Time references must be treated as real-world temporal descriptions,
  not structural labels.

- Use the following natural-language interpretations when needed:
  - "init" refers to the beginning of 2024 (early in the year).
  - "w1" refers to the first quarter of 2024.
  - "w2" refers to the second quarter of 2024.
  - "w3" refers to the third quarter of 2024.
  - "w4" refers to the fourth quarter of 2024.

- When a question requires temporal disambiguation to ensure a unique answer,
  you MUST express time using natural language
  (e.g., "early in 2024", "during the first quarter of the year",
  "later that year"),
  and MUST NOT use internal labels such as init, w1, w2, w3, or w4
  in the question text.


Begin now.

Input:
atom = {ATOM_JSON}
context = {CONTEXT_TEXT}
"""


QUESTION1_TEMPLATE = """You are an automatic question writer based on an atomic fact (“atom”) and a supporting context (“context”).

I will provide you with two inputs:
- atom: a single atomic fact object containing fields such as atom_id, domain, type, anchor, time_window, key, value, path, etc.
- context: a relevant excerpt from the original source material (e.g., a book or document), used as source material for writing questions

Your task is to generate questions whose answers are uniquely and precisely locked to the given atom, while using the context as the source of the question statements.

You must strictly follow ALL rules below:

────────────────────────────────
CORE QUESTION-CONSTRUCTION LOGIC
────────────────────────────────

0) Event-driven reasoning (NEW, STRICT):

- You must first identify one or more concrete events described in the context.
- An “event” refers to a specific action, decision, change, or occurrence that can be localized in time and attributed to a subject.
- Each event MUST be represented by an event_id (assumed to be available from the context).
- Your question must be constructed by reasoning over these events, not by abstract restatement.
- The selected events MUST explicitly appear in your question-writing reasoning.

This event selection and combination logic MUST be explained in draft1.

────────────────────────────────
GENERAL RULES
────────────────────────────────

1) The context is only question-writing material.
   - You are NOT required to use all of it.
   - The wording and factual basis of each question must come from the context.

2) The answer to each question must be locked to the atom:
   - By default, the answer should equal atom.value.
   - If atom.key is a more appropriate unique anchor, the answer must still resolve to the concrete value associated with that key.
   - Do NOT generalize or allow multiple possible values.

3) You must use the context to construct the question so that the answer is uniquely determined.
   - Avoid broad or vague questions that could have multiple valid answers in the original source but are not uniquely covered by the provided context.

4) Task background (important for correctness):
   - Answers are located automatically from structured atomic notes.
   - The answering agent will read the original source, NOT the notes.
   - Therefore, questions must appear to come naturally from the original material and must not rely on information outside the given context to be uniquely answerable.

5) You must generate EXACTLY 1 question:
   - 1 short-answer question (1–3 sentences, tightly focused on the atom)

────────────────────────────────
OUTPUT FORMAT (UPDATED)
────────────────────────────────

6) Output MUST be a JSON array and ONLY a JSON array (no extra text).

7) Each element must be an object with the following fields:
   - draft1: your reasoning BEFORE writing the question  
     (must explicitly explain:
       • which event(s) are selected from the context
       • why these events uniquely support the atom
       • how they are combined to lock the answer)
   - question: the question text
   - draft2: your reasoning BEFORE writing the final answer  
     (why the answer is uniquely correct and no alternatives fit)
   - answer: the standard answer
   - evidence: a list of event_id values representing the events you relied on

────────────────────────────────
STRICT NATURALNESS CONSTRAINTS
────────────────────────────────

- Do NOT mention “atom”, “context”, “notes”, “path”, “event_id”, or any internal system concepts in the question text.
- Do NOT invent facts not present in the context.
- If similar entities or values appear in the context, you MUST disambiguate using time, object name, or explicit constraints from the context.
- All questions must be phrased as if written directly from the original source text.
- You MUST NOT reference schema-level or system-level terminology such as:
  "operation entry", "op field", "time_window", "preference_name",
  "operations list", or similar.

────────────────────────────────
SEMANTIC ANSWER LOCKING
────────────────────────────────

- The atom is ONLY an answer anchor, NOT a topic to be mentioned.
- You must translate atom.value into a human-readable semantic intent,
  then express that intent using wording grounded in the context.
- The answer must be uniquely correct by semantic meaning,
  not by schema label matching.

────────────────────────────────
TIME SEMANTICS (IMPORTANT)
────────────────────────────────

- The timeline covers the full year of 2024.
- Time references must be expressed in natural language only:
  • "early in 2024"
  • "during the first quarter of the year"
  • "later that year"
- NEVER use internal labels such as init, w1, w2, w3, or w4 in the question.

If the context is insufficient to uniquely lock the answer:
- You must refine the question using tighter event constraints,
- You must NOT add new facts.

Begin now.

Input:
atom = {ATOM_JSON}
context = {CONTEXT_TEXT}

"""

QUESTION1_CL_TEMPLATE = """You are an automatic question writer based on an atomic fact (“atom”) and a supporting context (“context”).

I will provide you with two inputs:
- atom: a single atomic fact object containing fields such as atom_id, domain, type, anchor, time_window, key, value, path, etc.
- context: a relevant excerpt from the original source material (e.g., a book or document), used as source material for writing questions

Your task is to generate questions whose answers are uniquely and precisely locked to the given atom, while using the context as the source of the question statements.

You must strictly follow ALL rules below:

────────────────────────────────
CORE QUESTION-CONSTRUCTION LOGIC
────────────────────────────────

0) Event-driven, multi-event reasoning (NEW, STRICT, PRIORITY):

- You MUST identify multiple (plural) concrete events from the context as the factual basis of the question.
- An “event” refers to a specific action, decision, change, or occurrence that:
  • involves a subject,
  • can be localized in time,
  • contributes factual constraints toward the atom.
- Each event MUST be represented by an event_id (assumed to be available from the context).

DEFAULT REQUIREMENT:
- You MUST use at least TWO distinct events as joint evidence to construct the question.
- These events should jointly constrain the answer such that:
  • neither event alone is sufficient to uniquely determine the answer,
  • but their combination uniquely locks the atom.

Only if the context genuinely contains only ONE usable event:
- You may fall back to a single-event question,
- BUT you MUST explicitly justify this fallback in draft1.

Failure to use multiple events when possible INVALIDATES the question.

────────────────────────────────
GENERAL RULES
────────────────────────────────

1) The context is only question-writing material.
   - You are NOT required to use all of it.
   - The wording and factual basis of each question must come from the context.

2) The answer to each question must be locked to the atom:
   - By default, the answer should equal atom.value.
   - If atom.key is a more appropriate unique anchor, the answer must still resolve to the concrete value associated with that key.
   - Do NOT generalize or allow multiple possible values.

3) You must use the context to construct the question so that the answer is uniquely determined.
   - Avoid broad or vague questions that could have multiple valid answers in the original source but are not uniquely covered by the provided context.

4) Task background (important for correctness):
   - Answers are located automatically from structured atomic notes.
   - The answering agent will read the original source, NOT the notes.
   - Therefore, questions must appear to come naturally from the original material and must not rely on information outside the given context.

5) You must generate EXACTLY 1 question:
   - 1 short-answer question (1–3 sentences, tightly focused on the atom)

────────────────────────────────
OUTPUT FORMAT (UPDATED)
────────────────────────────────

6) Output MUST be a JSON array and ONLY a JSON array (no extra text).

7) Each element must be an object with the following fields:
   - draft1: your reasoning BEFORE writing the question  
     (must explicitly explain:
       • which multiple events are selected from the context (by event_id),
       • why each event alone is insufficient,
       • how their combination uniquely supports and locks the atom)
   - question: the question text
   - draft2: your reasoning BEFORE writing the final answer  
     (why the answer is uniquely correct given the combined events,
      and why no alternative value fits all constraints)
   - answer: the standard answer
   - evidence: a list of event_id values representing ALL events relied on

────────────────────────────────
STRICT NATURALNESS CONSTRAINTS
────────────────────────────────

- Do NOT mention “atom”, “context”, “notes”, “path”, “event_id”, or any internal system concepts in the question text.
- Do NOT invent facts not present in the context.
- If similar entities, times, or values appear in the context, you MUST disambiguate by:
  • combining multiple events,
  • using natural-language temporal references,
  • or explicitly constraining the subject or action.
- All questions must be phrased as if written directly from the original source text.
- You MUST NOT reference schema-level or system-level terminology such as:
  "operation entry", "op field", "time_window", "preference_name",
  "operations list", or similar.

────────────────────────────────
SEMANTIC ANSWER LOCKING
────────────────────────────────

- The atom is ONLY an answer anchor, NOT a topic to be mentioned.
- You must translate atom.value into a human-readable semantic intent,
  then express that intent using wording grounded in the context.
- The answer must be uniquely correct by semantic meaning,
  not by schema label matching.
- Multi-event reasoning is the PRIMARY mechanism for ensuring uniqueness.

────────────────────────────────
TIME SEMANTICS (IMPORTANT)
────────────────────────────────

- The timeline covers the full year of 2024.
- Time references must be expressed in natural language only:
  • "early in 2024"
  • "during the first quarter of the year"
  • "later that year"
- NEVER use internal labels such as init, w1, w2, w3, or w4 in the question.

────────────────────────────────
FAILURE & REFINEMENT RULE
────────────────────────────────

If the answer is not uniquely locked:
- You MUST refine the question by incorporating additional events from the context.
- You MUST NOT add new facts or rely on schema terminology.
- Prefer adding another event over tightening wording alone.

Begin now.

Input:
atom = {ATOM_JSON}
context = {CONTEXT_TEXT}
"""

BIND_PROMPT_TEMPLATE = """You are an automatic question writer.

You are given:
- two existing questions (“question1” and “question2”),
- a supporting context (“context”),
- and an implicit answer anchor.

Your task is to fuse question1 and question2 into ONE new short-answer question
that genuinely requires information from BOTH questions to answer correctly.

====================
CORE OBJECTIVE
====================

The fused question MUST introduce a semantic dependency between the two questions.
It is NOT sufficient to merely restate the same fact with more details.

Acceptable fusion patterns include (choose ONE when appropriate):
- selection: answering requires choosing an item based on a property described in the other question
- dependency: answering one part depends on information implied by the other
- constraint interaction: two constraints jointly eliminate all but one possible answer
- role differentiation: the same entity is referenced under two different roles or purposes

====================
STRICT RULES
====================

1) Context grounding
- The context is the ONLY factual source.
- You may select only the parts of the context that are necessary.
- Do NOT invent facts or rely on outside knowledge.

2) Answer uniqueness
- The question must have exactly ONE correct answer.
- The answer must resolve to a concrete value supported by the context.
- Avoid questions that could be answered correctly in more than one way
  by a reader of the original source material.

3) Non-redundancy (CRITICAL)
- question1 and question2 MUST each contribute a non-overlapping role.
- Do NOT combine them by simply listing more attributes of the same fact.
- If removing either question would still leave the answer obvious,
  the fused question is INVALID and must be rewritten.

4) Natural language only
- Do NOT mention internal concepts such as “atom”, “context”, “schema”,
  “operation”, “time_window”, or any structural labels.
- The question must read as if written directly from the original source text.

5) Time semantics
- The timeline covers the year 2024.
- Use natural time expressions when needed:
  - early in 2024
  - during the first quarter of the year
  - later that year
- NEVER use internal labels such as init, w1, w2, w3, or w4 in the question.

====================
OUTPUT FORMAT (MANDATORY)
====================

Output EXACTLY ONE element in a JSON array.
No extra text.

Each element MUST contain:

- draft1:
  A brief explanation of how question1 and question2 each
  contribute a distinct constraint and why both are required.

- question:
  ONE short-answer question (1–3 sentences),
  tightly focused and naturally phrased.

- draft2:
  A brief explanation of why the answer is uniquely correct
  and cannot be replaced by another value from the context.

- answer:
  The standard answer as a concrete value.

draft1 and draft2 may be written in English or Chinese,
but must focus on semantic binding and uniqueness.

====================
IMPORTANT FAILURE CONDITION
====================

If the only way to make the answer unique is to:
- restate the same activity with more descriptive details, or
- pile up time, location, and purpose without introducing interaction,

then the question is INVALID and MUST be rewritten.

Begin now.

Input:
question1 = {QUESTION1_JSON}
question2 = {QUESTION2_JSON}
context = {CONTEXT_TEXT}
"""

QCV_PROMPT_TEMPLATE = QUESTION_CLASS_VARIOUS_PROMPT_TEMPLATE = """You are an automatic question writer based on an atomic fact (“atom”) and a supporting context (“context”).

I will provide you with two inputs:
- atom: a single atomic fact object containing fields such as atom_id, domain, type, anchor, time_window, key, value, path, etc.
- context: a relevant excerpt from the original source material.

Your task is to generate questions whose answers are uniquely and precisely locked to the given atom, while using the context as the source of the question statements.

You must strictly follow ALL rules below:

────────────────────────────────
0. CORE LOGIC: THE ITERATIVE LOOP
────────────────────────────────
You must not just "write a question." You must construct it through a specific logic flow:
1. **Analyze Schema**: Pinpoint the domain and key to find the exact task strategy.
2. **Event Sourcing**: Locate the concrete events in the text.
3. **Iterative Refinement**: Draft a question, check if it's unique, adjust the selected events, and re-draft until the answer is locked.

────────────────────────────────
1. DOMAIN-SPECIFIC QUESTION STRATEGIES
────────────────────────────────
You must apply the specific strategy below based on the `domain` and `key` of the input atom.

**A. Domain: `user_attributes_state`** (Static attributes, demographics, status)
   - **Key: `change_reason`** → **Attribution Task**: Identify the specific event/decision that forced the attribute to change. Ask *why* the status changed.
   - **Key: `current_value`** → **Status Extraction**: Identify the final state resulting from the most recent relevant event. Ask for the *current* status.
   - **Key: `previous_value`** → **Transformation Logic**: Identify the pivot event that initiated the change. Frame the question around the state immediately before this event, explicitly contrasting it with the post-transition state reflected in the context, to highlight the nature and direction of the transformation.

**B. Domain: `habits_state`** (Routines, behavioral patterns, frequencies)
   - **Key: `change_reason`** → **Causal Analysis**: Identify the event that disrupted an old habit or established a new one. Ask *what caused* the shift in routine.
   - **Key: `current_value`** → **Pattern Recognition**: Summarize events to define the currently active habit. Target specific fields (frequency, time). Ask what the *new/current* routine is.
   - **Key: `previous_value`** → **Process Observation**: Observing the current_value (new habit) described in the context as a reference point, focus on the transition events themselves. Analyze and structure the event chain that captures the concrete process of change from the old behavior to the new one. The question should articulate how the transition unfolded, what key behavioral shifts occurred, and how these shifts collectively formed the progression from previous_value to current_value.

**C. Domain: `preferences_state`** (Likes, dislikes, settings, priorities)
   - **Key: `change_reason`** → **Origin Tracing**: Find the specific experience or realization that altered the user's taste in specific events. Ask *why* the preference shifted or obtained.
   - **Key: `current_value`** → * **Preference Synthesis**: Extract and aggregate **relevant behavioral events** from the context, then **synthesize and infer the user’s current active preference** by summarizing these signals and predicting the dominant preference state they collectively indicate.

   - **Key: `previous_value`** → * **Shift Detection**: Use the `current_value` in the context to anchor the analysis, then **focus on the transition itself**. Frame the question to examine **how the shift occurred** or **how the preference obtained**, emphasizing the **key events and intermediate changes or execution** that led from the prior affinity or setting to the updated state, rather than merely identifying the before/after contrast.

   
────────────────────────────────
2. GENERAL RULES
────────────────────────────────
- The context is only question-writing material. Do NOT use outside knowledge.
- **Answer Scope**: The answer must be strictly derived from the intersection of the `atom` and the `context`.
    - If the Strategy asks for a status (e.g., current_value), the answer is the state itself.
    - If the Strategy asks for a reason or process (e.g., change_reason, previous_value process), the answer is the *explanation* found in the text that links to that atom.
- **Uniqueness**: Use specific details (time, location, object constraints) to ensure only ONE answer is valid.
- **Naturalness**: Questions must appear to come naturally from the original source.

────────────────────────────────
3. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly one object.
The `draft1` field must strictly follow the **5-Step Reasoning Chain** below:
1.  **Step 1: Classification**: explicitly state the `domain` and `key`.
2.  **Step 2: Strategy Definition**: Quote the specific task type from Section 1 above (e.g., "This is a User Attribute / Previous Value task. I must look for the state before the pivot event...").
3.  **Step 3: Event Brainstorming**: List all candidate `event_ids` from the context that seem relevant.
4.  **Step 4: Iterative Loop**:
    * *Draft A*: Write a raw attempt.
    * *Critique*: Does it lock to the answer? Is it unique?
    * *Adjustment*: Select different events or add constraints.
    * *Draft B*: The refined attempt. (Repeat until satisfied).
5.  **Step 5: Final Selection**: State the final event combination used.

The `draft2` field is the **Validation & Null-Handling Layer**. You must perform two checks here:
1. **Uniqueness Check**: Is the answer locked to specific constraints?
2. **The "Null-Answer" Protocol**:
   - Check if the generated question asks for a "change," "process," or "reason" that is NOT explicitly detailed in the context.
   - If the context only shows a **stable, unchanging state** instead of the requested transition:
     - Do NOT accept "None" or "Unknown" as the final answer.
     - You MUST formulate a **"Stability Statement"**: Explicitly state that no transition details are present, AND describe the stable routine that *is* observed.


JSON Structure example:
{
  "draft1": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ... ... Step...",
  "question": "The final refined question text",
  "draft2": "Verification: [Check Uniqueness]. Null-Handling: The question asks for transition details. The text DOES NOT contain specific transition steps, but DOES show the habit was fully formed by Jan 2024. Therefore, the answer will pivot to describing this established stability.",
  "answer": "The context provides no details regarding a transition or prior habit; instead, it identifies the [User's Action] as an already established and consistent routine throughout the recorded period.",
  "evidence": ["event_id_1", "event_id_2", ...]
}

────────────────────────────────
4. STRICT NATURALNESS CONSTRAINTS
────────────────────────────────
- **Forbidden Terms**: "atom", "context", "notes", "path", "event_id", "time_window", "change_reason", "current_value", "previous_value", "operations", "schema".
- **Time**: The timeline is 2024. Use natural language ONLY (e.g., "early in the year", "after the move"). NEVER use "w1" or ISO dates.

────────────────────────────────
5. SEMANTIC ANSWER LOCKING (DYNAMIC)
────────────────────────────────
You must determine the "Correct Answer" based on the `key` and its assigned strategy:

1. **For `current_value`**:
   - The answer MUST be the concrete state or synthesized preference represented by `atom.value`.
   - *Example*: If value is "Vegan", the answer is "Vegan" (or semantic equivalent).

2. **For `change_reason`**:
   - The answer MUST be the *causal explanation* explicitly found in the context.
   - The `atom.value` is merely the reference point for *what* changed.

3. **For `previous_value`**:
   - The answer MUST be the *process description* or the *transition logic* as defined in the Strategy section.
   - Do NOT just answer with the old state name unless the question explicitly asks "what was the old state". If the strategy asks "how it changed", the answer is the *description of the change*.

- **Verification**: The answer must be correct by **meaning** in the context of the strategy, not just by string matching the atom value.

Begin now.

Input:
atom = {ATOM_JSON}
context = {CONTEXT_TEXT}
"""

NOW_TEMPLATE = """You are an automatic question writer based on an atomic fact (“atom”) and a supporting context (“context”).

I will provide you with two inputs:
- atom: a single atomic fact object containing fields such as atom_id, domain, key, value, etc.
- context: a relevant excerpt from the original source material.

Your task is to generate questions whose answers are uniquely and precisely locked to the given atom, while using the context as the source of the question statements.

You must strictly follow ALL rules below:

────────────────────────────────
0. CORE LOGIC: THE ITERATIVE LOOP
────────────────────────────────
You must not just "write a question." You must construct it through a specific logic flow:
1. **Analyze Schema**: Pinpoint the domain and key to find the exact task strategy.
2. **Event Sourcing**: Locate the concrete events in the text.
3. **Iterative Refinement**: Draft a question, check if it's unique, adjust the selected events, and re-draft until the answer is locked.

────────────────────────────────
1. DOMAIN-SPECIFIC QUESTION STRATEGIES
────────────────────────────────
You must apply the specific strategy below based on the `domain` and `key` of the input atom.

**A. Domain: `user_attributes_state`** (Static attributes, demographics, status)
   - **Key: `change_reason`** → **Attribution Task**: Identify the specific event or decision that forced the attribute to change. Ask *why* the status changed.
   - **Key: `new_value`** → **Acquisition Task**: Focus on the *inception* of a new status. Ask what specific new attribute the user acquired or established.
   - **Key: `current_value`** → **Status Synthesis**: Identify the final, maintained state resulting from the events. Ask for the *current* active status.
   - **Key: `delta`** → **Adjustment Logic**: Focus on the *modification* itself. Frame the question to examine **how** the attribute was specifically adjusted or refined (e.g., a shift in value, rank, or level) rather than just asking for the final number.
   - **Key: `removed_value`** → **Loss Identification**: Identify the specific status or attribute that was discarded. Ask what the user *no longer* possesses.

**B. Domain: `habits_state`** (Routines, behavioral patterns, frequencies)
   - **Key: `change_reason`** → **Causal Analysis**: Identify the event that disrupted an old habit or established a new one. Ask *what triggered* the shift in routine.
   - **Key: `new_value`** → **Inception Task**: Focus on the start of a brand new routine. Ask what *new* habit the user adopted.
   - **Key: `current_value`** → **Pattern Recognition**: Summarize events to define the currently active habit. Target specific fields (frequency, time) and ask what the *maintained* routine is.
   - **Key: `delta`** → **Refinement Observation**: Focus on the *transition mechanics*. Analyze how the routine was tweaked (e.g., frequency increased, time shifted). The question should articulate **how the habit evolved** or was modified.
   - **Key: `dropped_value`** → **Cessation Task**: Focus on the routine that was discontinued. Ask what specific habit the user *stopped* performing.

**C. Domain: `preferences_state`** (Likes, dislikes, settings, priorities)
   - **Key: `change_reason`** → **Origin Tracing**: Find the specific experience or realization that altered the user's taste. Ask *why* the preference shifted.
   - **Key: `current_value`** → **Preference Synthesis**: Extract and aggregate **relevant behavioral events** from the context, then **synthesize and infer** the user’s current active preference.
   - **Key: `delta`** → **Shift Mechanics**: Focus on the **nature of the change**. Frame the question to examine **how the preference was adjusted** (e.g., a toggle switched, a priority raised), emphasizing the difference between the old and new settings.

────────────────────────────────
2. GENERAL RULES
────────────────────────────────
- The context is only question-writing material. Do NOT use outside knowledge.
- **Answer Scope**: The answer must be strictly derived from the intersection of the `atom` and the `context`.
    - If the Strategy asks for a status (e.g., current_value, new_value), the answer is the state itself.
    - If the Strategy asks for a reason or process (e.g., change_reason, delta), the answer is the *explanation* or *description* found in the text.
- **Uniqueness**: Use specific details (time, location, object constraints) to ensure only ONE answer is valid.
- **Naturalness**: Questions must appear to come naturally from the original source.

────────────────────────────────
3. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly one object.
The `draft1` field must strictly follow the **5-Step Reasoning Chain** below:
1.  **Step 1: Classification**: explicitly state the `domain` and `key`.
2.  **Step 2: Strategy Definition**: Quote the specific task type from Section 1 above (e.g., "This is a Habit / Delta task. I must look for how the routine was modified...").
3.  **Step 3: Event Brainstorming**: List all candidate `event_ids` from the context that seem relevant.
4.  **Step 4: Iterative Loop**:
    * *Draft A*: Write a raw attempt.
    * *Critique*: Does it lock to the answer? Is it unique?
    * *Adjustment*: Select different events or add constraints.
    * *Draft B*: The refined attempt. (Repeat until satisfied).
5.  **Step 5: Final Selection**: State the final event combination used.

The `draft2` field is the **Validation & Null-Handling Layer**. You must perform two checks here:
1. **Uniqueness Check**: Is the answer locked to specific constraints?
2. **The "Null-Answer" Protocol**:
   - Check if the generated question asks for a "change," "process," or "reason" that is NOT explicitly detailed in the context.
   - If the context only shows a **stable, unchanging state** instead of the requested transition:
     - Do NOT accept "None" or "Unknown" as the final answer.
     - You MUST formulate a **"Stability Statement"**: Explicitly state that no transition details are present, AND describe the stable routine that *is* observed.

JSON Structure example:
{
  "draft1": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ...",
  "question": "The final refined question text",
  "draft2": "Verification: [Check Uniqueness]. Null-Handling: ...",
  "answer": "The context provides no details regarding a transition...",
  "evidence": ["event_id_1", "event_id_2", ...]
}

────────────────────────────────
4. STRICT NATURALNESS CONSTRAINTS
────────────────────────────────
- **Forbidden Terms**: "atom", "context", "notes", "path", "event_id", "time_window", "change_reason", "current_value", "delta", "new_value", "dropped_value", "schema".
- **Time**: The timeline is 2024. Use natural language ONLY (e.g., "early in the year", "after the move"). NEVER use "w1" or ISO dates.

────────────────────────────────
5. SEMANTIC ANSWER LOCKING (DYNAMIC)
────────────────────────────────
You must determine the "Correct Answer" based on the `key` and its assigned strategy:

1. **For `current_value` / `new_value`**:
   - The answer MUST be the concrete state or synthesized preference represented by `atom.value`.

2. **For `change_reason`**:
   - The answer MUST be the *causal explanation* explicitly found in the context.

3. **For `delta`**:
   - The answer MUST be the *description of the modification* or the *difference*. Do NOT just answer with the final value unless asking for the "result of the adjustment". If the question asks "how it changed", the answer is the logic of the change (e.g., "increased frequency", "switched priority").

4. **For `dropped_value` / `removed_value`**:
   - The answer MUST be the item or state that was stopped/discarded.

- **Verification**: The answer must be correct by **meaning** in the context of the strategy, not just by string matching the atom value.

Begin now.

Input:
atom = {ATOM_JSON}
context = {CONTEXT_TEXT}
"""



TYPE1 = '''You are an automatic question writer specializing in "Pattern Induction" and "Preference Synthesis."

You will receive two inputs:
- task: a single atomic fact object containing either `current_value` or `new_value`, plus metadata such as `required_observable_fields` and `evidenced_fields`.
- context: an excerpt from the original source material containing discrete timestamped events.

Your job is to generate ONE natural question whose answer is uniquely grounded in the task, using the context only as supporting evidence and phrasing reference.

You MUST follow all rules below.

────────────────────────────────
0. CORE LOGIC: ITERATIVE REASONING LOOP
────────────────────────────────
You must not just write a question directly. You must explicitly reason through:

1. Domain Confirmation
   - Confirm the domain is either `habits_state` or `preferences_state`.
   - Confirm whether the state is derived from `current_value` or `new_value`.

2. Field & Evidence Mapping
   - Extract the `required_observable_fields` from the task.
   - Scan the context for concrete events that match the task’s `evidenced_fields`.
   - Double-check: do the events actually support the general pattern/preference claimed?

3. Pattern / Preference Synthesis
   - If evidence is sufficient, generate a question that asks about the generalized habit or preference.
   - If evidence is insufficient or contradictory, you must trigger the Null-Answer Protocol.

────────────────────────────────
1. DOMAIN-SPECIFIC QUESTION STRATEGIES
────────────────────────────────

A. Domain: `habits_state` (routine / frequency / repeated behaviors)
- The question MUST ask about a generalized routine (not a single event).
- Target: frequency, timing, repeated action, or usual behavior.
- Examples:
  - "How often does the user usually..."
  - "When does the user typically..."
  - "What is the user’s routine for..."

B. Domain: `preferences_state` (likes, dislikes, priorities, choices)
- The question MUST ask about the user’s preference/taste.
- Target: favored type, disliked type, prioritization, or stable inclination.
- Examples:
  - "What kind of [X] does the user prefer?"
  - "Which option does the user favor?"
  - "What does the user prioritize when choosing..."

────────────────────────────────
2. EVIDENCE RULES (STRICT GROUNDING)
────────────────────────────────
- The task is the Truth Anchor: the answer must match the task’s value in meaning.
- The context is the Evidence Gate: you may ONLY write the question/answer if the context provides sufficient evidence.
- Evidence must support ALL required observable fields (e.g., frequency cannot be inferred from one event).
- Do not hallucinate missing time spans, repetitions, or intensity.

────────────────────────────────
3. TIME ANCHORING (NATURAL, QUARTER-LEVEL MINIMUM)
────────────────────────────────
You MUST analyze the dates/timestamps implied in the mapped events and anchor the question to at least quarter-level precision.

Rules:
- If all evidence lies within one quarter, specify that quarter naturally:
  - "in early 2024 (Jan–Mar)"
  - "in the first quarter of 2024"
  - "during spring 2025"
- If evidence spans multiple quarters and appears stable, you may use:
  - "throughout 2024"
  - "since late 2023"
  - "in the first half of 2025"
- Do NOT use vague standalone timeframes like:
  - "recently", "lately", "currently", "now"
- Do NOT output raw ISO dates.

Preference Exception:
- For `preferences_state`, you may use softer but still bounded anchors such as:
  - "in early 2026"
  - "during that semester"
  - "around that period"
BUT it must still correspond to the evidence window.

────────────────────────────────
4. OUTPUT FORMAT (STRICT JSON)
────────────────────────────────
Output MUST be a JSON array containing exactly ONE object.

The object must have the following keys:
- "draft1"
- "question"
- "draft2"
- "answer"
- "evidence"

JSON Structure example: { "draft1": "Step 1: Domain habits_state. Step 2: Observable field is 'frequency'. Step 3: Mapping events [id_1, id_2, id_3]. Validation: Yes, dates are M/W/F, supporting '3x/week'. Step 4: ...", "question": "How often did the user visit the gym in the first quarter of 2024?", "draft2": "Verification: ...", "answer": "He typically visits the gym three times a week.", "evidence": ["id_1", "id_2", "id_3"] }

Important:
- "draft1" and "draft2" may contain any internal reasoning language (including schema terms).
- "question" and "answer" MUST be natural human-like language.
- "evidence" MUST be a list of event_ids.

────────────────────────────────
5. DRAFT1 FORMAT REQUIREMENT (FIELD-GUIDED REASONING CHAIN)
────────────────────────────────
The "draft1" string MUST include these steps:

Step 1: Classification
- State the domain and whether it uses current_value or new_value.

Step 2: Field Analysis
- List required observable fields from the task.

Step 3: Evidence Mapping
- List the supporting event identifiers from the context.
- Validation: explicitly state whether the events collectively support the task value (Yes/No).

Step 4: Pattern/Preference Synthesis
- Draft the intended question/answer logic.

────────────────────────────────
6. DRAFT2 VALIDATION & NULL-ANSWER PROTOCOL
────────────────────────────────
The "draft2" string MUST include:

1. Uniqueness Check
- Explain why the answer is uniquely constrained by the task.

2. Null-Answer Protocol
- If evidence is insufficient or contradictory:
  - The question should still be phrased naturally.
  - The answer MUST be a stability statement:
    "The excerpt does not provide enough evidence to determine this reliably."
  - Evidence list should include the closest related events, or be empty if none exist.

────────────────────────────────
7. SEMANTIC ANSWER LOCKING
────────────────────────────────
- If evidence is sufficient: the answer must carry the same meaning as the task value.
- The answer must be phrased naturally, echoing wording from the excerpt when possible.
- Avoid robotic single-token answers.

Bad: "Window seat."
Good: "He tends to choose a window seat because he likes sleeping during flights."

────────────────────────────────
8. NATURALNESS CONSTRAINTS (QUESTION/ANSWER ONLY)
────────────────────────────────
The following terms MUST NOT appear in the "question" or "answer" fields:
"task", "context", "observable_fields", "path", "event_id", "current_value", "new_value", "w0", "w1".

They MAY appear in draft1/draft2.

────────────────────────────────
9. FINAL CHECK
────────────────────────────────
Before outputting:
- Ensure the question asks for a generalized habit/preference, not a single event.
- Ensure the answer is either:
  (a) fully supported by evidence, OR
  (b) the required stability statement.

Begin now.

Input:
task = {{TASK_JSON}}
context = {{CONTEXT_TEXT}}
'''

TYPE2 = '''You are an automatic question writer specializing in "Causal Analysis" and "Change Explanation."

I will provide you with two inputs:
- task: a single atomic fact object containing the key `change_reason` and metadata like `required_observable_fields` and `evidenced_fields`.
- context: a relevant excerpt from the original source material containing discrete events.

Your task is to generate questions that ask for the explanation, trigger, or motivation behind a specific change. You must strictly ground your question and answer in the observable events that evidence the task.

You must strictly follow ALL rules below:

────────────────────────────────
0. CORE LOGIC: THE ITERATIVE LOOP
────────────────────────────────
You must not just "write a question." You must construct it through a specific logic flow:

1. Analyze Schema
   - Confirm the domain (user_attributes_state / habits_state / preferences_state).
   - Confirm the task is asking for the cause of a change.

2. Field & Evidence Mapping
   - Identify the required observable fields in the task.
   - Scan the excerpt for events that match the evidenced_fields.
   - Double Check: verify that these events truly match the meaning of the task (avoid keyword collisions).

3. User Perspective Check
   - The user only sees the excerpt text.
   - Ensure the question is answerable purely from the selected events.

4. Drafting
   - Construct the question/answer pair based ONLY on verified evidence.

────────────────────────────────
1. DOMAIN-SPECIFIC QUESTION STRATEGIES
────────────────────────────────
The key is always `change_reason`.

A. Domain: user_attributes_state (Demographics, Status)
   - Task: Attribution. Identify the decision/event that forced the status to change.
   - Question Style:
     - "What prompted the user to [change status]?"
     - "Why did the user decide to...?"
     - "What event led to...?"

B. Domain: habits_state (Routines, Frequencies)
   - Task: Trigger Identification. Identify what disrupted an old routine or kickstarted a new one.
   - Question Style:
     - "What triggered the change in his [activity] routine?"
     - "Why did he stop/start...?"
     - "What caused the shift...?"

C. Domain: preferences_state (Likes, Priorities)
   - Task: Origin Tracing. Identify the experience or realization that altered the user’s taste.
   - Question Style:
     - "What experience made the user change their mind about...?"
     - "Why does the user now prefer...?"
     - "What led the user to dislike...?"

────────────────────────────────
2. GENERAL RULES (STRICT CAUSAL GROUNDING)
────────────────────────────────
- Answer Scope (Evidence-Based):
  - The task’s value is the Truth Anchor, but the answer MUST be derived from the excerpted events.
  - The answer must describe the CAUSE (trigger/motivation), not the effect.

- Evidence Selection:
  - Select ONLY events that support the required observable fields and the claimed cause.
  - Do not include unrelated events even if they share similar keywords.

- No Speculation:
  - Do NOT infer unmentioned psychological states or motivations.
  - Only use motivations explicitly stated or clearly expressed via causal language in the excerpt.

────────────────────────────────
3. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly one object.

The `draft1` field must strictly follow the Field-Guided Reasoning Chain:

1. Step 1: Classification
   - State the domain and causal focus.

2. Step 2: Field Analysis
   - List the required observable fields from the task.

3. Step 3: Evidence Mapping
   - Identify candidate event_ids from the excerpt that correspond to evidenced_fields.
   - Validation: Explicitly state:
     "Do these events actually support the task's value? [Yes/No]"

4. Step 4: Causal Extraction
   - Extract the explicit trigger/motivation from the mapped events.

5. Step 5: Drafting
   - Write the question and answer.

The `draft2` field is the Validation & Null-Handling Layer:

1. Uniqueness Check
   - Is the answer uniquely locked to these events?

2. The Null-Answer Protocol
   - If the excerpt describes the change but does NOT explicitly provide a cause:
     - You MUST output a Stability Statement:
       "The excerpt describes the change but does not provide a specific reason or trigger."
     - The question must still be phrased naturally.
     - The answer MUST be exactly the Stability Statement.

JSON Structure example:
{
  "draft1": "Step 1: Domain habits_state. Step 2: Observable field is 'frequency'. Step 3: Mapping events [id_1, id_2] which show gym visits. Validation: Yes, they match. Step 4: Context says he got injured. Step 5: Drafting Q/A...",
  "question": "What motivated the user to stop running in early 2024?",
  "draft2": "Verification: The injury is explicitly stated as the trigger. Uniqueness: only one plausible cause appears.",
  "answer": "He experienced a minor injury that forced him to stop.",
  "evidence": ["id_1", "id_2"]
}

────────────────────────────────
4. STRICT NATURALNESS & TIME CONSTRAINTS
────────────────────────────────
- Forbidden Terms:
  "task", "context", "observable_fields", "change_reason", "delta", "value", "w0", "w1".

- Scope Rule:
  Forbidden Terms apply ONLY to the "question" and "answer" fields.
  They are allowed in draft1/draft2.

- Time Anchoring:
  - Anchor the question to the time window of the transition described in the mapped events.
  - Infer time naturally from the excerpt.
  - If the excerpt explicitly references w0/w1 style windows, convert them to natural phrasing:
    - w0 -> "late last year", "at the end of 2023"
    - w1-w4 -> "in early 2024", "this past spring"
  - Do NOT use ISO dates.

────────────────────────────────
5. SEMANTIC ANSWER LOCKING
────────────────────────────────
- The answer must explain the "Why" using details found in the selected events.
- Avoid robotic label-style answers.

Example:
- Evidence: "canceling netflix, way too expensive now."
- Good Answer: "He felt the subscription had become too expensive."
- Bad Answer: "Cost saving."

Begin now.

Input:
task = {{TASK_JSON}}
context = {{CONTEXT_TEXT}}
'''

TYPE3 = '''You are an automatic question writer specializing in "Cumulative State Snapshot."

I will provide you with:
- task: A specific state-change event used strictly as the **Time Anchor** (Time T). It contains a `name` key.
- context: A narrative history containing the full sequence of events leading up to Time T.

Your task is to **reconstruct the complete, aggregated configuration** of the specific attribute (`task.name`) at the exact moment the `task` occurred.
You must look backward in the context to find "legacy" items that were established earlier and are still active, then combine them with the "new" item from the task.

────────────────────────────────
0. CORE LOGIC: THE SNAPSHOT BUILDER
────────────────────────────────
1. **Pinpoint Time T**: Locate the specific event in `context` that matches the `task`.
2. **Back-Trace & Aggregate**:
   - Scan the `context` from the beginning up to Time T.
   - Collect ALL items associated with `task.name` (or its category) that were defined earlier and have *not* been removed/ended.
   - **Combine** these "legacy" items with the "new" item from the `task`.
3. **Define Total State**: This combined list (Legacy + New) is your answer source.
4. **Draft Question**: Ask explicitly about this **complete list** or **total setup** at Time T.

────────────────────────────────
1. QUESTION RULES
────────────────────────────────
- **Time Lock**: The question MUST explicitly cite the specific moment of the `task` event (e.g., "At the moment the user...", "When the setting was changed to...", "In October 2024...").
- **Scope**: Ask for the **full list**, **complete setup**, **total inventory**, or **current status**.
- **Forbidden**: Do NOT ask about "history", "evolution", or "changes". Do NOT ask "what was added?". You must ask "what was the *total* set?".
- **Natural Phrasing**: Do not use raw JSON keys (e.g., do not say "evening_walk_with_spouse"). Use natural descriptions (e.g., "his walking routine with his wife").

────────────────────────────────
2. ANSWER RULES
────────────────────────────────
- The answer must be a static description of the state at Time T.
- It must explicitly mention both the *new* item (from task) and the *old* items (from context) that are still active.

────────────────────────────────
3. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly one object.

The `draft1` field must strictly follow the **State Aggregation Chain**:
1. **Step 1: Anchor**: Identify Time T and the task Event.
2. **Step 2: Inventory**: List "Legacy" items found in previous windows/events.
3. **Step 3: Combined State**: Legacy Items + New task Item.
4. **Step 4: Draft Question**: Formulate the snapshot inquiry.

The `draft2` field is for **Verification**:
- Confirm the answer includes elements from *both* the past (context) and the present (task).

JSON Structure:
{
  "draft1": "Step 1: Anchor is... Step 2: Inventory... Step 3: Combined State... Step 4: Q...",
  "question": "At the time of [task Event], what was the complete list of [Attribute]?",
  "draft2": "Verification: Answer combines [Old Item] from Window X and [New Item] from task.",
  "answer": "The user had [Old Item] and [New Item].",
  "evidence": ["log123", "log456"]
}

example: 


{
  "draft1": "Step 1: Anchor is June 12, 2024 (Event: log_00123, using Munsell Book). About attributes. Step 2: Inventory: 'ArcGIS Pro' (established in w0 (winter in 2023), still active), 'OxCal' (added earlier in w2 (summer in 2024)). Step 3: Combined State = ArcGIS Pro + OxCal + Munsell Soil Color Book. Step 4: Q: What was the full list of tools available?",
  "question": "When the user was classifying pottery shards using the Munsell Soil Color Book on June 12, 2024, what was the complete list of specialized research tools available to him at that time?",
  "draft2": "Verification: The answer aggregates the legacy tool (ArcGIS from winter in 2023) with the recently acquired tools (OxCal, Munsell from summer in 2024).",
  "answer": "At that specific moment, his research toolkit consisted of three items: the long-standing **ArcGIS Pro** for spatial mapping, and the recently added **OxCal** (for radiocarbon calibration) and **Munsell Soil Color Book**.",
  "evidence": ["log_00123", "log_00234", "log_00456"]
}

{
    "draft1": "Step 1: Anchor is July 2, 2024 (Event: log123). User adjusted his walking routine due to summer heat. Step 2: Retrieve State. New Schedule: Monday-Saturday (days 0-5), excluding Sunday. New Timing: 20:45 - 21:30. Location: 'neighborhood sidewalks and wooded trails' (preserved from previous state). Step 3: Combined State = Mon-Sat schedule + Late Evening Time + Neighborhood Location. Step 4: Q: What was the full routine setup?",
    "question": "When the user adjusted his evening walking routine with his wife in July 2024 to avoid the summer heat, what were the specific settings for the activity's timing, frequency, and location?",
    "draft2": "Verification: Answer covers frequency (Mon-Sat), timing (late evening), and location (neighborhood), combining new adjustments with existing preferences.",
    "answer": "The adjusted schedule was set for **Monday through Saturday** from **20:45 to 21:30**, taking place along **neighborhood sidewalks and wooded trails** (Sunday was excluded to accommodate family calls).",
    "evidence": ["log123", "log456"]
}

Begin.

Input:
task = {{TASK_JSON}}
context = {{CONTEXT_TEXT}}'''

TYPE4 = '''You are an automatic question writer specializing in "Evolution Trajectory Tracing."

I will provide you with two inputs:
- task: a single atomic fact object containing a `name` key that identifies the specific attribute or topic to trace.
- context: a relevant excerpt containing a timeline of events or descriptions.

Your task is to generate a question that asks for the **entire known history or evolution** of the specific Topic, allowing the answer to cover the full range of time available in the context.

You must strictly follow ALL rules below:

────────────────────────────────
0. CORE LOGIC: THE ITERATIVE LOOP
────────────────────────────────
You must not just "write a question." You must construct it through a specific logic flow:
1. **Identify Topic**: Extract the subject from `task.value`.
2. **Scan Timeline**: Identify the full chronological arc in the `context` (Start -> Changes -> Current).
3. **Synthesize Trajectory**: Draft a question that invites a **complete narrative summary**, avoiding restrictive time-boxing in the question text.

────────────────────────────────
1. DOMAIN-SPECIFIC QUESTION STRATEGIES
────────────────────────────────
The strategy is identical for all domains (`user_attributes`, `habits`, `preferences`) because the key is always `name`.

**Strategy: Trajectory Synthesis**
- **The Input**: `task.value` is the **Subject**. `context` is the **Story**.
- **The Task**: Construct a longitudinal inquiry.
- **Question Style**: 
  - **Preferred**: "How has the user's [Subject] evolved over time?"
  - **Preferred**: "Describe the history of the user's [Subject]."
  - **Preferred**: "What is the trajectory of the user's [Subject]?"
  - *Avoid*: "How did it change from Jan to March?" (Too narrow).
- **The Answer Structure**: The **Answer** (not the question) must contain the specific timestamps and details.
  - *Format*: "Originally [State A] in [Year], then [Event] happened, leading to [State B] currently."

────────────────────────────────
2. GENERAL RULES
────────────────────────────────
- **Answer Scope (Pure Context)**: 
  - The `task` defines the topic.
  - The `answer` must cover the **entire timeline** present in the context.
  - Do not truncate the story. If the text mentions high school, college, and current job, the answer must cover all three phases.
- **Uniqueness**: The question should identify the specific *topic* clearly (e.g., "investment account history" vs "bank account history") but keep the time scope open.
- **Naturalness**: The question should sound like a biographer asking for a summary of life events regarding that topic.

────────────────────────────────
3. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly one object.
The `draft1` field must strictly follow the **5-Step Reasoning Chain** below:
1.  **Step 1: Topic Extraction**: Identify `task.value` as the subject.
2.  **Step 2: Timeline Mapping**: Map out the full chronological sequence available in context.
3.  **Step 3: Draft Drafting**: Formulate a "Whole History" question.
4.  **Step 4: Refinement**: **Remove specific dates from the question**. Ensure the question asks "Over time" or "History of".
5.  **Step 5: Final Selection**: Final Polish.

The `draft2` field is the **Validation & Null-Handling Layer**:
1. **Uniqueness Check**: Does the question specify the topic clearly?
2. **The "Null-Answer" Protocol**:
   - If the context shows **NO evolution** (static state):
     - Question: "What is the observed history of [Subject]?"
     - Answer: "The user has maintained a consistent [Subject] throughout the recorded period."
   - If the context shows **NO details** at all:
     - Output a "Stability Statement".

JSON Structure example:
{
  "draft1": "Step 1: Topic 'Fitness'. Step 2: Context covers 2022 injury -> 2023 rehab -> 2024 running. Step 3: Question: 'How has his fitness evolved?'...",
  "question": "How has the user's fitness routine evolved over time?",
  "draft2": "Verification: Question is open-ended. Answer covers 2022-2024.",
  "answer": "Following an injury in 2022, he focused on rehab in 2023, and has recently started running again in 2024.",
  "evidence": ["event_id_1", "event_id_2", "event_id_3"]
}

────────────────────────────────
4. STRICT NATURALNESS & TIME CONSTRAINTS
────────────────────────────────
- **Forbidden Terms**: "task", "context", "notes", "trajectory", "schema", "value", "w0", "w1".
- **Time Phrasing**: 
  - **In Question**: Use broad terms: "Over time," "Throughout the years," "Historically," "The evolution of." **Avoid specific dates in the question** unless necessary to distinguish between two different timelines.
  - **In Answer**: Use **precise** dates and relative times found in the context (e.g., "In late 2023," "Last summer").

────────────────────────────────
5. SEMANTIC ANSWER LOCKING
────────────────────────────────
- **Goal**: Full Arc Narrative.
- The question asks for the "Movie Plot," the answer provides the "Scene Summary."
- *Example*:
  - **Context**: "Used iPhone 12 until Dec 2023, then switched to Pixel 8."
  - **Question**: "What is the history of the user's mobile device usage?" (Not: "What phone did he use in Dec 2023?")
  - **Answer**: "He used an iPhone 12 until late 2023, at which point he switched to a Pixel 8."

Begin now.

Input:
task = {{TASK_JSON}}
context = {{CONTEXT_TEXT}}'''

TYPE5 = '''You are an automatic question writer specializing in "Discrete Event Retrieval" and "Fact Checking."

I will provide you with two inputs:
- task: A JSON object representing ONE focal log event (the event to ask about).
- context: A JSON object representing nearby log events surrounding the task (e.g., previous/next logs). It may include events similar to the task.

Your job is to generate ONE fact-retrieval question that can be answered by extracting ONE concrete fact from the task event.

The question must retrieve a specific logged fact tied to a specific time point or a clearly bounded time range.

You must strictly follow ALL rules below:

────────────────────────────────
0. CORE LOGIC: THE ITERATIVE LOOP
────────────────────────────────
You must not just "write a question." You must construct it through this logic flow:

1. Scan Task:
   - Read the task event and identify its key fields (timestamp, action, app, URL, item, amount, etc.).

2. Select One Answer Field:
   - Choose exactly ONE specific detail from the task as the answer.
   - The answer must be directly stated in the task event (no inference).

3. Build Anchors for Uniqueness:
   - Use other details from the SAME task event as anchors in the question.
   - Use the context ONLY to check for ambiguity (collision detection).
   - If the context contains similar events, strengthen the anchors until the question uniquely refers to the task event.

4. Write a Fact Retrieval Query:
   - The final question must be phrased so it retrieves the chosen answer field.
   - The question must clearly point to a single event.

────────────────────────────────
1. REQUIRED TIME BINDING
────────────────────────────────
Because the log spans a long time period, the question MUST include:
- a specific timestamp (preferred), OR
- a specific date/time range.

The time reference should be as precise as possible (minute/second if available).

────────────────────────────────
2. QUESTION STRATEGIES
────────────────────────────────
Choose ONE strategy based on what you selected as the answer.

A. Content Extraction ("What/Which")
Trigger: The answer is an Object, Item, App, Price, URL, or Content.
Rule: Use the task timestamp (or task time range) in the question as the anchor.
Examples:
- "What website did the user visit at [timestamp]?"
- "Which app did the user open at [timestamp]?"
- "What item did the user purchase on [date] at [time]?"

B. Temporal Localization ("When")
Trigger: The answer is the Timestamp/Date/Duration.
Rule: Use the task action/object as the anchor, and ensure uniqueness using context.
Examples:
- "At what time did the user purchase [item]?"
- "When did the user visit [website]?"

C. Existence Verification ("Did")
Trigger: You want a Yes/No answer about whether the task event occurred.
Rule: Must include a specific date/time range.
Examples:
- "Did the user visit [website] at [timestamp]?"
- "Did the user make a purchase between [start] and [end]?"

────────────────────────────────
3. GENERAL RULES
────────────────────────────────
- Single Event Only: The question must refer to ONE specific event, not patterns.
- No Habit Language: Do not use "usually", "typically", "often", "habit", or similar.
- Answer Source Restriction:
  - The answer MUST come from the task event.
  - Do NOT use any other event in the context as the answer.
- Context Usage Restriction:
  - Context is ONLY for uniqueness checking and excluding ambiguity.
  - If multiple events in the context could match the question, you MUST add more constraints.
- Fact Only:
  - The answer must be a concrete logged fact (string, timestamp, number, app name, URL, etc.).
  - No interpretation, no summarization.

────────────────────────────────
4. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly ONE object.

Fields required:
- draft1
- question
- draft2
- answer
- evidence

The draft1 field must follow the 5-step reasoning chain:
Step 1: Event Identification
Step 2: Answer Selection
Step 3: Anchor Selection (for uniqueness)
Step 4: Ambiguity Check (use nearby logs to confirm uniqueness)
Step 5: Final Question Polish

The draft2 field must include:
1. Uniqueness Check:
   - Confirm the question matches only the task event within the provided window.
2. Null-Answer Protocol:
   - If the question asks for something that is missing from the task event, the answer must be:
     "The information is not provided in the log."

Evidence:
- evidence must contain the task log identifier if available (e.g., "log005").
- If the task includes an explicit ID field, use that exact ID.

────────────────────────────────
5. STRICT NATURALNESS CONSTRAINTS
────────────────────────────────
Forbidden terms anywhere in the output:
"usually", "typically", "habit", "frequency", "task", "schema"

The question must sound natural and human.

────────────────────────────────
6. SEMANTIC ANSWER LOCKING
────────────────────────────────
The question must be written so the answer is uniquely determined and extractable.
It should resemble a database lookup query with explicit time constraints.

Example:
Context: "At 2 PM, the user visited google.com."
Correct Question: "What website did the user visit at 2 PM?"
Incorrect Question: "What websites does the user visit?"

Begin now.

Input:
task = {{TASK_JSON}}
context = {{CONTEXT_TEXT}}
'''

TYPE6 = '''You are an automatic question writer specializing in "Temporal Sequence" and "Chronological Logic."

I will provide you with two inputs:
- task: A JSON object representing ONE focal log event (the event that must be used as the Anchor).
- context: A JSON object representing nearby log events surrounding the task (previous and next events). Each event includes a `log_id`.

Your task is to generate ONE chronological reasoning question that asks about the relative order of events surrounding the task event.

You must strictly follow ALL rules below:

────────────────────────────────
0. CORE LOGIC: THE ITERATIVE LOOP
────────────────────────────────
You must not just "write a question." You must construct it through this logic flow:

1. Build Local Timeline:
   - Read the context and construct the ordered list of events in chronological order.

2. Select Anchor (STRICT):
   - The Anchor event MUST be the task event.
   - Do NOT choose another event as the Anchor.

3. Select Comparison Scope:
   - Choose ONE nearby event from the context (either immediately before or immediately after the task event),
     OR choose another distinct event for a "which happened first" comparison.

4. Formulate Temporal Relation:
   - Decide whether to ask about adjacency ("What happened right before/after?") or relative ordering ("Which happened first?").

────────────────────────────────
1. QUESTION STRATEGIES
────────────────────────────────
Choose ONE strategy.

A. Strategy: Adjacency ("Before/After")
Goal: Ask what happened immediately before or immediately after the task event.
Question Styles:
- "What did the user do immediately after [task event]?"
- "What action happened right before [task event]?"
- "What was the next step following [task event]?"
Constraint:
- The answer must be the adjacent event (an action/app/visit), not a timestamp.

B. Strategy: Relative Ordering ("Which happened first?")
Goal: Compare the task event with another event in the context.
Question Styles:
- "Which happened first: [task event] or [other event]?"
- "Did the user open [task app/site] before or after [other app/site]?"
Constraint:
- The answer must be an ordering judgment (e.g., "[task event] happened first").

────────────────────────────────
2. GENERAL RULES
────────────────────────────────
- Chronology Only: Focus on ordering, not causality or explanation.
- Task-Anchor Restriction:
  - The task event MUST be the Anchor.
  - The answer event MUST come from the provided context window.
- Uniqueness:
  - The context must provide a clear linear order.
  - If the order is ambiguous or simultaneous, do NOT generate an ordering question.
- No Pattern Queries:
  - Do not ask about repeated behavior or general habits.

────────────────────────────────
3. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly ONE object.

Required fields:
- draft1
- question
- draft2
- answer
- evidence

The draft1 field must strictly follow the 5-step reasoning chain:
Step 1: Timeline Construction
  - Write the local ordered list of events from the context.
Step 2: Anchor Confirmation
  - Explicitly state that the task event is the Anchor.
Step 3: Strategy Selection
  - Choose Adjacency or Relative Ordering.
Step 4: Refinement
  - Ensure the question is natural and unambiguous.
Step 5: Evidence Pairing
  - Identify the log IDs of the task event and the answer/comparison event.

The draft2 field must include:
1. Uniqueness Check:
   - Confirm the order is strictly linear within the window.
2. Null-Answer Protocol:
   - If the context does not contain a clear before/after relationship, the question must be:
     "The timeline is not clear enough to determine event order."

Evidence field requirements:
- evidence must contain exactly two strings:
  - evidence[0] = the log_id of the task event
  - evidence[1] = the log_id of the answer event (adjacent event or comparison event)

JSON example:
{
  "draft1": "Step 1: Events: A -> B -> C -> D. Step 2: Task event is C. Step 3: Strategy A (before). Step 4: Refined question. Step 5: Pair IDs.",
  "question": "What did the user do immediately before opening the Chat application?",
  "draft2": "Verification: Order is clear. The task event is uniquely positioned. The previous event is Mail.",
  "answer": "He checked his mail.",
  "evidence": ["log_005", "log_004"]
}

────────────────────────────────
4. STRICT NATURALNESS & LANGUAGE CONSTRAINTS
────────────────────────────────
Forbidden terms anywhere in the output:
"timestamp", "node", "sequence", "predecessor", "successor", "anchor", "context", "log_id"

Use natural phrasing such as:
"right before", "immediately after", "next", "then", "prior to", "followed by".

Do NOT mention internal identifiers in the question text.

────────────────────────────────
5. SEMANTIC ANSWER LOCKING
────────────────────────────────
- For Adjacency: the answer must be the immediate neighbor event.
- For Ordering: the answer must explicitly state which event occurred first.

Begin now.

Input:
task = {{TASK_JSON}}
context = {{CONTEXT_TEXT}}
'''

TYPE6old = '''You are an automatic question writer specializing in "Temporal Sequence" and "Chronological Logic."

I will provide you with a single input:
- context: A text excerpt containing a sequence of actions, a narrative flow, or a timeline of events.

Your task is to **identify a distinct event** within this text to serve as an "Anchor," and then generate a question that queries the **relative timing, order, or sequence** of other events surrounding it.

You must strictly follow ALL rules below:

────────────────────────────────
0. CORE LOGIC: THE ITERATIVE LOOP
────────────────────────────────
You must not just "write a question." You must construct it through a specific logic flow:
1. **Map Timeline**: Read the `context` and mentally construct the ordered list of events ($E_1 \rightarrow E_2 \rightarrow E_3$).
2. **Select Anchor**: Choose one specific, distinct event ($E_{target}$) from the sequence to focus on.
3. **Formulate Relation**: Decide whether to ask about the *Neighbor* ($E_{target} \pm 1$) or the *Order* relative to another event ($E_{compare}$).

────────────────────────────────
1. DOMAIN-SPECIFIC QUESTION STRATEGIES
────────────────────────────────
Apply one of the strategies below based on the Anchor you selected.

**A. Strategy: Immediate Adjacency (The "Before/After" Task)**
   - **Goal**: Identify the event that occurred immediately preceding or following your selected Anchor.
   - **Question Style**: 
     - "What did the user do **immediately after** [Selected Anchor]?"
     - "What action was taken **right before** [Selected Anchor]?"
     - "What was the next step following [Selected Anchor]?"
   - **Constraint**: The answer must be the **Action/Event**, not a timestamp.

**B. Strategy: Relative Ordering (The "Comparison" Task)**
   - **Goal**: Compare the timing of your Anchor against another distinct event found in the context.
   - **Question Style**: 
     - "Which happened first: [Selected Anchor] or [Other Event]?"
     - "Did the user visit [App A] before or after [App B]?"
     - "Was [Selected Anchor] the final action taken in this sequence?"
   - **Constraint**: The answer must be a judgment of order (e.g., "[Event A] happened first").

────────────────────────────────
2. GENERAL RULES
────────────────────────────────
- **Chronology over Causality**: Focus on **Time**, not Reason. Ask "What happened next?", not "Why did it happen?".
- **Answer Scope**: 
  - For Strategy A: The answer is the **Adjacent Event** described in the context.
  - For Strategy B: The answer is the **Ordering Decision**.
- **Uniqueness**: Ensure the context provides a clear, unambiguous order. If events are simultaneous or the order is vague, do NOT generate a sequence question.
- **Naturalness**: Use sequence markers: "Following that," "Prior to," "Subsequently," "Later."

────────────────────────────────
3. OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST be a JSON array containing exactly one object.
The `draft1` field must strictly follow the **5-Step Reasoning Chain**:
1.  **Step 1: Timeline Construction**: List the sequence found in context (e.g., "Wake up -> Coffee -> Code").
2.  **Step 2: Anchor Selection**: Explicitly state: "I am selecting [Event X] as the Anchor."
3.  **Step 3: Strategy Selection**: Choose Adjacency ("What after X?") or Ordering ("X vs Y?").
4.  **Step 4: Refinement**: Ensure the question is natural and unambiguous.
5.  **Step 5: Final Selection**: Final Polish.

The `draft2` field is the **Validation & Null-Handling Layer**:
1. **Uniqueness Check**: Is the sequence strictly linear?
2. **Null-Answer Protocol**:
   - If the context contains *no sequence* (only one event) or unordered lists:
     - **Constraint**: Output a "Stability Statement" in the `question` field stating no sequence is available.

JSON Structure example:
{
  "draft1": "Step 1: Seq: Login -> Check Mail -> Logout. Step 2: Selected Anchor is 'Check Mail'. Step 3: Ask what happened BEFORE...",
  "question": "What did the user do immediately before checking his email?",
  "draft2": "Verification: Sequence is clear. Answer is 'Login'.",
  "answer": "He logged into the system.",
  "evidence": ["log123, log456"]
}

────────────────────────────────
4. STRICT NATURALNESS & TIME CONSTRAINTS
────────────────────────────────
- **Forbidden Terms**: "timestamp", "node", "sequence", "predecessor", "successor", "anchor", "context".
- **Phrasing**: 
  - Use: "Next," "Then," "Before," "After," "followed by."
  - Avoid: "at time T+1," "the event with ID..."
- **Date Resolution**: Resolve relative days (e.g., "the next day") into specific calendar dates/times if the context allows, to ensure temporal precision.

────────────────────────────────
5. SEMANTIC ANSWER LOCKING
────────────────────────────────
- **Adjacency**: Answer = The **Other** Event.
- **Ordering**: Answer = The **Winner** (First/Last).

Begin now.

Input:
context = {CONTEXT_TEXT}'''


JUDGE_PROMPT = """You are a strict QA validator and logic refiner.

You will be given:
- A Context (source evidence)
- A Question (query)
- A Reference Answer (candidate answer)

Your task is to analyze, judge, and refine both the Reference Answer and the Question to ensure they are factually grounded and logically consistent.

────────────────────────────────
1. ANALYZE
────────────────────────────────
Compare the Reference Answer against the Context.

You may accept:
- Facts explicitly stated in the Context or Question.
- Facts that are logically inferable from the Context or Question through clear reasoning
  (e.g., temporal order from timestamps, repeated events → routine, aggregation across logs).

You must NOT accept:
- Information that cannot be grounded in the Context by either explicit evidence or reasonable inference.
- Over-specific details (exact times, durations, motivations, locations, causal claims)
  unless they are clearly supported by the Context.
- Speculative or assumptive statements.

────────────────────────────────
2. JUDGE
────────────────────────────────
Set "judge" as follows:

- judge = 1  
  If ALL parts of the Reference Answer are supported by the Context
  either explicitly or via reasonable inference,
  and no hallucinated or unsupported information is present.

- judge = 0  
  If ANY part of the Reference Answer:
  - Introduces information not stated or inferable from the Context, or
  - Over-specifies details beyond what the Context supports, or
  - Contains incorrect matching or unjustified causal claims.

────────────────────────────────
3. REFINE (ONLY IF judge = 0)
────────────────────────────────
If judge = 0, you MUST generate a "refine_info" object. 

Refinement Principles:
- Minimal Edit: Apply the smallest possible changes to make the text factually accurate.
- Answer Refinement: You can delete unsupported details OR correct minor factual errors to align with the Context.
- Query Refinement: If the original Question contains misinformation (e.g., wrong date or status), you must refine the Question into a "refine_query" that aligns with the Context.

Refine process:
- draft: Explain the discrepancies and what needs changing.
- refine_check: Set to 1 if a valid, grounded Answer and Query can be produced. Set to 0 if the contradiction is irreconcilable.
- refine_query: The corrected version of the original Question.
- refine_answer: The corrected version of the Reference Answer.


Return **ONLY** a JSON object. Do not output markdown code blocks.

### Output Format

**Scenario A: Judge is 1**
{
  "rationale": "Concise reasoning why the context supports the answer.",
  "judge": 1
}

**Scenario B: Judge is 0**
{
  "rationale": "Reasoning identifying the unsupported or contradictory info.",
  "judge": 0,
  "refine_info": {
      "draft": "Analysis of corrections made to the query and/or answer.",
      "refine_check": 1,
      "refine_query": "The refined question text.",
      "refine_answer": "The refined answer text."
  }
}

**Scenario C: Judge is 0 (Irreconcilable)**
{
  "rationale": "Reasoning why refinement is impossible.",
  "judge": 0,
  "refine_info": { "draft": "...", "refine_check": 0, "refine_query": "null", "refine_answer": "null" }
}

### Example of Query & Answer Refinement
Context: `[2024-11-12 Tue 08:30:00] Event: Start Writing. Loc: Home Office. [2024-11-14 Thu 08:30:00] Event: Start Writing. Loc: Home Office.`
Question: "What is the user's current routine for academic writing?"
Ref_Answer: "Currently, the user maintains a high-priority writing schedule twice a week on Tuesdays and Thursdays from 08:30 to 11:00 AM, working from their home office in Athens."
Output:
{
  "rationale": "The logs confirm a recurring writing routine on Tuesday and Thursday mornings starting at 08:30 AM in the user's 'home office'. However, the context never specifies an end time of 11:00 AM nor does it explicitly state that the home office is located in Athens.",
  "judge": 0,
  "refine_info": {
      "draft": "Supported: 'Tuesdays and Thursdays', '08:30', 'home office'. Unsupported: 'to 11:00 AM', 'high-priority', 'in Athens'. I will remove the unsupported details.",
      "refine_check": 1,
      "refine_query": "What is the user's current routine for academic writing?",
      "refine_answer": "Currently, the user maintains a writing schedule twice a week on Tuesdays and Thursdays from 08:30, working from their home office."
  }

Question:
{{QUESTION}}

Reference Answer:
{{ANSWER}}

Context (app logs JSON):
{{CONTEXT_JSON}}
"""


DOUBLECHECK_PROMPT = """You are a verifier.

Given a Question, a Refined Answer, and the Context, determine whether the Refined Answer is fully supported. 
You should know Question is also can be seen as a part of context or hint.


### Logic & Rules

1. Analyze:
   Compare the Refined Answer against the Context.
   Support may be:
   - Explicitly stated in the Context, or
   - Logically inferable from the Context through clear reasoning
     (e.g., timestamps imply order, repeated events imply a routine).

2. Judge:
   - Set "judge": 1 if ALL parts of the Refined Answer are supported
     by explicit evidence or reasonable inference, with no hallucinated details.
   - Set "judge": 0 if ANY part:
     - Cannot be grounded in the Context (explicitly or inferentially), or
     - Is over-specific beyond what the Context supports, or
     - Matches the Context incorrectly.

Return ONLY a JSON object:

{
  "rationale": "...",
  "judge": 1
}
or
{
  "rationale": "...",
  "judge": 0
}

Question:
{{QUESTION}}

Refined Answer:
{{ANSWER}}

Context (app logs JSON):
{{CONTEXT_JSON}}
"""