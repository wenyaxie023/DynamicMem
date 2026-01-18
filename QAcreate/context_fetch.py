# fetch_context.py
from typing import Any, Dict, List, Optional, Tuple


def _infer_from_path(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not path:
        return None, None, None
    segments = [seg for seg in path.split(".") if seg]
    if not segments:
        return None, None, None

    if "time_windows" in segments:
        idx = segments.index("time_windows")
        time_window = segments[idx + 1] if idx + 1 < len(segments) else None
        delta = segments[idx + 2] if idx + 2 < len(segments) else None
        type_map = {
            "user_attributes_delta": "operation_attribute",
            "habits_delta": "operation_habit",
            "preferences_delta": "operation_preference",
        }
        return type_map.get(delta), None, time_window

    if "initial_state" in segments:
        if "user_attributes_state" in segments:
            if "singular" in segments:
                idx = segments.index("singular")
                anchor = segments[idx + 1] if idx + 1 < len(segments) else None
                return "singular", anchor, "init"
            if "collections" in segments:
                idx = segments.index("collections")
                anchor = segments[idx + 1] if idx + 1 < len(segments) else None
                return "collection", anchor, "init"

        if "habits_state" in segments:
            idx = segments.index("habits_state")
            if idx + 2 < len(segments) and segments[idx + 1] == "initial":
                return "habit", segments[idx + 2], "init"

        if "preferences_state" in segments:
            idx = segments.index("preferences_state")
            if idx + 2 < len(segments) and segments[idx + 1] == "initial":
                return "preferences", segments[idx + 2], "init"

    return None, None, None


def fetch_context(
    atom: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:

    domain = atom["domain"]
    a_type = atom.get("type")
    anchor = atom.get("anchor")
    time_window = atom.get("time_window")

    if not a_type or not anchor or not time_window:
        inferred_type, inferred_anchor, inferred_tw = _infer_from_path(atom.get("path", ""))
        if not a_type:
            a_type = inferred_type
        if not anchor:
            anchor = inferred_anchor
        if not time_window:
            time_window = inferred_tw or "init"

    if not a_type:
        raise ValueError("Atom type is missing and could not be inferred from path.")

    domain_obj = schema[domain]
    init = domain_obj["initial_state"]

    # =========================
    # Step 1: locate base slice
    # =========================

    collections = init["user_attributes_state"]["collections"]

    if a_type == "singular":
        base_value = init["user_attributes_state"]["singular"]

    elif a_type == "collection":
        base_value = collections.get(anchor)

    elif a_type == "operation_attribute":
        if anchor in collections:
            base_value = collections.get(anchor)
        else:
            base_value = init["user_attributes_state"]["singular"]

    elif a_type == "habit" or a_type == "operation_habit":
        base_value = init["habits_state"]["initial"].get(anchor)

    elif a_type == "preferences" or a_type == "operation_preference":
        base_value = init["preferences_state"]["initial"].get(anchor)

    else:
        raise ValueError(f"Unknown atom type: {a_type}")

    if base_value is None:
        base_value = {}

    base = {
        "time_window": "init",
        "value": base_value,
    }

    # =========================
    # Step 2: collect operations
    # =========================

    operations: List[Dict[str, Any]] = []

    for w in domain_obj.get("time_windows", []):
        wid = w["window_id"]

        if wid > time_window:
            continue

        # ---- attribute ops ----
        for op in w.get("user_attributes_delta", {}).get("operations", []):
            attr_type = op.get("attribute_type")

            if a_type in ("singular", "operation_attribute") and attr_type == "singular":
                operations.append({"time_window": wid, "op": op})

            if a_type in ("collection", "operation_attribute") and attr_type == "collections":
                if op.get("collection_name") == anchor:
                    operations.append({"time_window": wid, "op": op})

        # ---- habit ops ----
        for op in w.get("habits_delta", {}).get("operations", []):
            if a_type in ("habit", "operation_habit") and op.get("habit_name") == anchor:
                operations.append(
                    {"time_window": wid, "op": op}
                )

        # ---- preference ops ----
        for op in w.get("preferences_delta", {}).get("operations", []):
            if (
                a_type in ("preferences", "operation_preference")
                and op.get("preference_name") == anchor
            ):
                operations.append(
                    {"time_window": wid, "op": op}
                )

    # =========================
    # Step 3: return context
    # =========================

    return {
        "domain": domain,
        "type": a_type,
        "anchor": anchor,
        "query_time_window": time_window,
        "base": base,
        "operations": operations,
    }
