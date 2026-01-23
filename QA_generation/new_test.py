# test_fetch.py
import json
from pprint import pprint

from client import LLMClient
from config import QAConfig
from context_fetch_by_anchor import fetch_context_by_anchor
from prompts import (
    BIND_PROMPT_TEMPLATE,
    NOW_TEMPLATE,
    QCV_PROMPT_TEMPLATE,
    QUESTION1_CL_TEMPLATE,
    QUESTION1_TEMPLATE,
    QUESTION7_PROMPT_TEMPLATE,
)

def ez_test(config: QAConfig | None = None):
    config = config or QAConfig()
    # ---------- load atoms ----------
    with open(config.real_atoms_path, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    # ---------- load schema ----------
    with open(config.schema_path, "r", encoding="utf-8") as f:
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
    with open(config.schema_path, "r", encoding="utf-8") as f:
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

