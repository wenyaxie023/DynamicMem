# fetch_context.py
from typing import Any, Dict, List


def fetch_context(
    atom: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:

    domain = atom["domain"]
    a_type = atom["type"]
    anchor = atom["anchor"]
    time_window = atom["time_window"]

    domain_obj = schema[domain]
    init = domain_obj["initial_state"]

    # =========================
    # Step 1: locate base slice
    # =========================

    if a_type == "singular" or a_type == "operation_attribute":
        base_value = init["user_attributes_state"]["singular"]

    elif a_type == "collection":
        base_value = init["user_attributes_state"]["collections"].get(anchor)

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
            if a_type in ("singular", "operation_attribute"):
                if op.get("attribute_type") == "singular":
                    operations.append(
                        {"time_window": wid, "op": op}
                    )

            if a_type == "collection":
                if (
                    op.get("attribute_type") == "collections"
                    and op.get("collection_name") == anchor
                ):
                    operations.append(
                        {"time_window": wid, "op": op}
                    )

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
