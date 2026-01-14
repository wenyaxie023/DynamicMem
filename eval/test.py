# test_fetch.py
import json
from pprint import pprint

from client import LLMClient
from config import BG_PATH, LLM_MAX_WORKERS, REAL_ATOMS_PATH
from context_fetch import fetch_context

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
5) You must generate EXACTLY 1 questions:
   - 1 short-answer question (1–3 sentences, tightly focused on the atom)
   The questions must be narrowly scoped and tightly bound to the answer.
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

def ez_test():
    # ---------- load atoms ----------
    with open(REAL_ATOMS_PATH, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    # ---------- load schema ----------
    with open(BG_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # ---------- pick the 5th atom ----------
    # target_id = "9cd72c8fae1142409f74edd79d73353f"
    # target_atom = next(
    #     (atom for atom in atoms if atom.get("atom_id") == target_id),
    #     None
    # )

    target_atom = atoms[300]
    print("\n========== TARGET ATOM ==========\n")
    pprint(target_atom)

    # ---------- fetch context ----------
    context = fetch_context(
        atom=target_atom,
        schema=schema,
    )

    print("\n========== FETCHED CONTEXT ==========\n")
    pprint(context)

    llm = LLMClient(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        max_workers=LLM_MAX_WORKERS,
    )

    print("\n========== LLM RESPONSE ==========\n")
    llm_response = llm.ask(
        prompt=QUESTION1_TEMPLATE.format(
            ATOM_JSON=json.dumps(target_atom, ensure_ascii=True),
            CONTEXT_TEXT=json.dumps(context, ensure_ascii=True, indent=2),
        )
    )
    pprint(llm_response)


    print("\n========== END ==========\n")

def bind_test():
    # ---------- load atoms ----------
    with open(REAL_ATOMS_PATH, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    # ---------- load schema ----------
    with open(BG_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # ---------- pick the 5th atom ----------
    # target_id = "9cd72c8fae1142409f74edd79d73353f"
    # target_atom = next(
    #     (atom for atom in atoms if atom.get("atom_id") == target_id),
    #     None
    # )

    target_atom1 = atoms[300]
    target_atom2 = atoms[500]
    print("\n========== TARGET ATOM ==========\n")
    pprint(target_atom1)

    # ---------- fetch context ----------
    context1 = fetch_context(
        atom=target_atom1,
        schema=schema,
    )

    context2 = fetch_context(
        atom=target_atom2,
        schema=schema,
    )
    print("\n========== FETCHED CONTEXT ==========\n")
    pprint(context1)
    pprint(context2)

    llm = LLMClient(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        max_workers=LLM_MAX_WORKERS,
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
    bind_test()
