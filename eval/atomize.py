# atomize.py
import json
import uuid
from typing import Any, Dict, List

from atom import Atom
from config import BG_PATH, REAL_ATOMS_PATH


def new_atom_id() -> str:
    return uuid.uuid4().hex


def emit_atom(
    atoms: List[Atom],
    domain: str,
    type_: str,
    anchor: str,
    time_window: str,
    key: str,
    value: Any,
    path: str | None = None,
):
    atoms.append(
        Atom(
            atom_id=new_atom_id(),
            domain=domain,
            type=type_,
            anchor=anchor,
            time_window=time_window,
            key=key,
            value=value,
            path=path,
        )
    )


def walk_object(
    atoms: List[Atom],
    domain: str,
    type_: str,
    anchor: str,
    time_window: str,
    base_path: str,
    obj: Any,
):
    """
    把一个 dict / list / scalar 展平为原子：
    - dict: key 是提问点
    - list: 整个 list 作为 value（不拆 index）
    - scalar: 直接记录
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            emit_atom(
                atoms,
                domain,
                type_,
                anchor,
                time_window,
                key=k,
                value=v,
                path=f"{base_path}.{k}",
            )
    else:
        # list / scalar 都整体存
        emit_atom(
            atoms,
            domain,
            type_,
            anchor,
            time_window,
            key=base_path.split(".")[-1],
            value=obj,
            path=base_path,
        )


def atomize_schema(schema: Dict[str, Any]) -> List[Atom]:
    atoms: List[Atom] = []

    for domain, domain_obj in schema.items():

        # ========== initial ==========
        init = domain_obj["initial_state"]

        # ---- singular ----
        singular = init["user_attributes_state"]["singular"]
        for k, v in singular.items():
            emit_atom(
                atoms,
                domain,
                "singular",
                anchor=k,
                time_window="init",
                key=k,
                value=v,
                path=f"{domain}.initial_state.user_attributes_state.singular.{k}",
            )

        # ---- collections ----
        collections = init["user_attributes_state"]["collections"]
        for cname, items in collections.items():
            emit_atom(
                atoms,
                domain,
                "collection",
                anchor=cname,
                time_window="init",
                key=cname,
                value=items,
                path=f"{domain}.initial_state.user_attributes_state.collections.{cname}",
            )

        # ---- habits ----
        habits = init["habits_state"]["initial"]
        for hname, hobj in habits.items():
            for k, v in hobj.items():
                emit_atom(
                    atoms,
                    domain,
                    "habit",
                    anchor=hname,
                    time_window="init",
                    key=k,
                    value=v,
                    path=f"{domain}.initial_state.habits_state.initial.{hname}.{k}",
                )

        # ---- preferences ----
        prefs = init["preferences_state"]["initial"]
        for pname, pobj in prefs.items():
            for k, v in pobj.items():
                emit_atom(
                    atoms,
                    domain,
                    "preferences",
                    anchor=pname,
                    time_window="init",
                    key=k,
                    value=v,
                    path=f"{domain}.initial_state.preferences_state.initial.{pname}.{k}",
                )

        # ========== time windows ==========
        for w in domain_obj.get("time_windows", []):
            wid = w["window_id"]

            # ---- operation_attribute ----
            for op in w.get("user_attributes_delta", {}).get("operations", []):
                for k, v in op.items():
                    if v is None:
                        continue
                    emit_atom(
                        atoms,
                        domain,
                        "operation_attribute",
                        anchor=op.get("attribute_name") or op.get("collection_name") or "singular",
                        time_window=wid,
                        key=k,
                        value=v,
                        path=f"{domain}.time_windows.{wid}.operation_attribute.{k}",
                    )

            # ---- operation_habit ----
            for op in w.get("habits_delta", {}).get("operations", []):
                hname = op.get("habit_name")
                for k, v in op.items():
                    if v is None:
                        continue
                    emit_atom(
                        atoms,
                        domain,
                        "operation_habit",
                        anchor=hname,
                        time_window=wid,
                        key=k,
                        value=v,
                        path=f"{domain}.time_windows.{wid}.operation_habit.{hname}.{k}",
                    )

            # ---- operation_preference ----
            for op in w.get("preferences_delta", {}).get("operations", []):
                pname = op.get("preference_name")
                for k, v in op.items():
                    if v is None:
                        continue
                    emit_atom(
                        atoms,
                        domain,
                        "operation_preference",
                        anchor=pname,
                        time_window=wid,
                        key=k,
                        value=v,
                        path=f"{domain}.time_windows.{wid}.operation_preference.{pname}.{k}",
                    )

    return atoms


if __name__ == "__main__":
    with open(BG_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    atoms = atomize_schema(schema)
    with open(REAL_ATOMS_PATH, "w", encoding="utf-8") as f:
        json.dump([a.__dict__ for a in atoms], f, indent=2, ensure_ascii=False)
    print(f"Emitted {len(atoms)} atoms.")
