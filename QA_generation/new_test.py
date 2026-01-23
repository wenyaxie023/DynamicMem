# test_fetch.py
import json
from pprint import pprint

from client import LLMClient
from config import QAConfig
from context_fetch_by_anchor import fetch_context_by_anchor

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
{{
  "draft1": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ... ... Step...",
  "question": "The final refined question text",
  "draft2": "Verification: [Check Uniqueness]. Null-Handling: The question asks for transition details. The text DOES NOT contain specific transition steps, but DOES show the habit was fully formed by Jan 2024. Therefore, the answer will pivot to describing this established stability.",
  "answer": "The context provides no details regarding a transition or prior habit; instead, it identifies the [User's Action] as an already established and consistent routine throughout the recorded period.",
  "evidence": ["event_id_1", "event_id_2", ...]
}}

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
{{
  "draft1": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ...",
  "question": "The final refined question text",
  "draft2": "Verification: [Check Uniqueness]. Null-Handling: ...",
  "answer": "The context provides no details regarding a transition...",
  "evidence": ["event_id_1", "event_id_2", ...]
}}

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


def ez_test(config: QAConfig | None = None):
    config = config or QAConfig()
    # ---------- load atoms ----------
    with open(config.real_atoms_path, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    # ---------- load schema ----------
    with open(config.bg_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # ---------- pick the 5th atom ----------
    # target_id = "9cd72c8fae1142409f74edd79d73353f"
    # target_atom = next(
    #     (atom for atom in atoms if atom.get("atom_id") == target_id),
    #     None
    # )
    print(f"Total atoms loaded: {len(atoms)}")
    target_atom = atoms[config.ez_target_index]
    print("\n========== TARGET ATOM ==========\n")
    pprint(target_atom)

    # ---------- fetch context ----------
    context = fetch_context_by_anchor(
        anchor=target_atom.get("anchor", ""),
        atoms=atoms,
        schema=schema,
    )

    print("\n========== FETCHED CONTEXT ==========\n")
    pprint(context)

    llm = LLMClient(
        provider=config.gen_provider,
        model_name=config.gen_model_name,
        max_workers=config.llm_max_workers,
    )

    print("\n========== LLM RESPONSE ==========\n")
    llm_response = llm.ask(
        prompt=QCV_PROMPT_TEMPLATE.format(
            ATOM_JSON=json.dumps(target_atom, ensure_ascii=True),
            CONTEXT_TEXT=json.dumps(context, ensure_ascii=True, indent=2),
        )
    )
    pprint(llm_response)


    print("\n========== END ==========\n")

def bind_test(config: QAConfig | None = None):
    config = config or QAConfig()
    # ---------- load atoms ----------
    with open(config.real_atoms_path, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    # ---------- load schema ----------
    with open(config.bg_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # ---------- pick the 5th atom ----------
    # target_id = "9cd72c8fae1142409f74edd79d73353f"
    # target_atom = next(
    #     (atom for atom in atoms if atom.get("atom_id") == target_id),
    #     None
    # )

    target_atom1 = atoms[config.bind_target_indices[0]]
    target_atom2 = atoms[config.bind_target_indices[1]]
    print("\n========== TARGET ATOM ==========\n")
    pprint(target_atom1)

    # ---------- fetch context ----------
    context1 = fetch_context_by_anchor(
        atom=target_atom1,
        schema=schema,
    )

    context2 = fetch_context_by_anchor(
        atom=target_atom2,
        schema=schema,
    )
    print("\n========== FETCHED CONTEXT ==========\n")
    pprint(context1)
    pprint(context2)

    llm = LLMClient(
        provider=config.gen_provider,
        model_name=config.gen_model_name,
        max_workers=config.llm_max_workers,
    )

    print("\n========== LLM RESPONSE ==========\n")
    prompts = [
        QUESTION1_TEMPLATE.format(
            ATOM_JSON=json.dumps(target_atom1, ensure_ascii=True),
            CONTEXT_TEXT=json.dumps(context1, ensure_ascii=True, indent=2),
        ),
        QUESTION1_TEMPLATE.format(
            ATOM_JSON=json.dumps(target_atom2, ensure_ascii=True),
            CONTEXT_TEXT=json.dumps(context2, ensure_ascii=True, indent=2),
        ),
    ]
    futures = llm.ask_many_async(prompts)
    results = llm.collect(futures)
    llm_response1 = results[0]
    llm_response2 = results[1]
    if isinstance(llm_response1, Exception):
        raise llm_response1
    if isinstance(llm_response2, Exception):
        raise llm_response2
    pprint(llm_response1)
    pprint(llm_response2)
    llm_bind_response = llm.ask(
        prompt=BIND_PROMPT_TEMPLATE.format(
            QUESTION1_JSON=llm_response1,
            QUESTION2_JSON=llm_response2,
            CONTEXT_TEXT= {
                "context_1": context1,
                "context_2": context2,
            },
        )
    )
    pprint(llm_bind_response)
    print("\n========== END ==========\n")



if __name__ == "__main__":
    ez_test()
