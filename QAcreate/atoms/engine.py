import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from .model import Atom
import yaml



@dataclass(frozen=True)
class AtomSpec:
    rule_id: str
    match: str
    emit: Dict[str, Any]


@dataclass(frozen=True)
class Node:
    path_segments: List[str]
    value: Any
    key: str
    parent_key: Optional[str]
    root: Dict[str, Any]
    domain_root: Any


def stable_atom_id(rule_id: str, path: str) -> str:
    raw = f"{rule_id}|{path}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_ir(path: str) -> List[AtomSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    atoms = data.get("atoms", []) or []
    specs: List[AtomSpec] = []
    for item in atoms:
        specs.append(
            AtomSpec(
                rule_id=item["id"],
                match=item["match"],
                emit=item.get("emit", {}) or {},
            )
        )
    return specs


def build_context(dicts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for d in dicts:
        if isinstance(d, dict):
            ctx.update(d)
    return ctx


def list_segment(parent_key: Optional[str], item: Any, index: int) -> str:
    if parent_key == "time_windows" and isinstance(item, dict):
        window_id = item.get("window_id")
        if window_id:
            return str(window_id)
    return str(index)


def walk_nodes(
    value: Any,
    path: List[str],
    ancestors: List[Dict[str, Any]],
    domain_root: Any,
) -> Iterable[Node]:
    key = path[-1] if path else ""
    parent_key = path[-2] if len(path) >= 2 else None
    dicts_for_context = list(ancestors)
    if isinstance(value, dict):
        dicts_for_context.append(value)
    root = build_context(dicts_for_context)
    yield Node(path, value, key, parent_key, root, domain_root)

    if isinstance(value, dict):
        new_ancestors = ancestors + [value]
        for k, v in value.items():
            yield from walk_nodes(v, path + [str(k)], new_ancestors, domain_root)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            seg = list_segment(parent_key, item, i)
            yield from walk_nodes(item, path + [seg], ancestors, domain_root)


def match_path(pattern: str, path_segments: List[str]) -> bool:
    parts = pattern.split(".")
    if len(parts) != len(path_segments):
        return False
    for p, s in zip(parts, path_segments):
        if p == "*":
            continue
        if p == "<domain>":
            continue
        if p != s:
            return False
    return True


def _parse_coalesce_args(expr: str) -> List[str]:
    inner = expr[len("$coalesce(") : -1]
    args: List[str] = []
    buf = []
    in_quote = None
    for ch in inner:
        if ch in ("'", '"'):
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
            buf.append(ch)
            continue
        if ch == "," and in_quote is None:
            args.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _get_from_root(root: Dict[str, Any], path: str) -> Any:
    cur: Any = root
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _split_relative(path: str) -> List[str]:
    parts = path.split("/")
    return [part for part in parts if part != ""]


def _resolve_relative_segments(path_segments: List[str], rel: str) -> Optional[List[str]]:
    parts = _split_relative(rel)
    if not parts:
        return path_segments
    idx = len(path_segments) - 1
    i = 0
    while i < len(parts) and parts[i] in (".", ".."):
        if parts[i] == "..":
            idx -= 1
            if idx < 0:
                return None
        i += 1
    target = path_segments[: idx + 1]
    target.extend(parts[i:])
    return target


def _get_by_path(root_value: Any, path_segments: List[str]) -> Any:
    if not path_segments:
        return None
    cur = root_value
    for seg in path_segments[1:]:
        if isinstance(cur, dict):
            cur = cur.get(seg)
        elif isinstance(cur, list):
            if seg.isdigit():
                idx = int(seg)
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
            else:
                match = None
                for item in cur:
                    if isinstance(item, dict) and item.get("window_id") == seg:
                        match = item
                        break
                cur = match
        else:
            return None
        if cur is None:
            return None
    return cur


def _maybe_eval_relative(expr: str, node: Node) -> Optional[Tuple[bool, Any]]:
    value_mode = False
    rel = expr
    if expr == "*" or expr.startswith("*.") or expr.startswith("*.."):
        value_mode = True
        rel = expr[1:] or "."
    elif expr.startswith("."):
        rel = expr
    else:
        return None
    target = _resolve_relative_segments(node.path_segments, rel)
    if not target:
        return True, None
    if value_mode:
        return True, _get_by_path(node.domain_root, target)
    return True, target[-1] if target else None


def eval_expr(expr: Any, node: Node, domain: str) -> Any:
    if not isinstance(expr, str) or not expr.startswith("$"):
        if isinstance(expr, str):
            rel_eval = _maybe_eval_relative(expr, node)
            if rel_eval is not None:
                matched, value = rel_eval
                if matched:
                    return value
        return expr
    if expr == "$key":
        return node.key
    if expr == "$parent_key":
        return node.parent_key
    if expr == "$value":
        return node.value
    if expr == "$domain":
        return domain
    if expr.startswith("$root."):
        return _get_from_root(node.root, expr[len("$root.") :])
    if expr.startswith("$coalesce(") and expr.endswith(")"):
        for arg in _parse_coalesce_args(expr):
            if arg.startswith("$"):
                val = eval_expr(arg, node, domain)
            elif arg.startswith("#"):
                val = eval_template(arg, node, domain)
            elif arg.startswith(".") or arg.startswith("..") or arg.startswith("*"):
                val = eval_expr(arg, node, domain)
            elif arg.startswith("root."):
                val = eval_expr(f"$root.{arg[len('root.'):]}", node, domain)
            else:
                val = _unquote(arg)
            if val is not None:
                return val
        return None
    raise ValueError(f"Unsupported expression: {expr}")


def eval_template(value: Any, node: Node, domain: str) -> Any:
    if isinstance(value, str):
        if value.startswith("#"):
            inner = value[1:]
            if not inner:
                return ""
            if inner.startswith("$"):
                return eval_expr(inner, node, domain)
            rel_eval = _maybe_eval_relative(inner, node)
            if rel_eval is not None:
                matched, rel_value = rel_eval
                if matched:
                    return rel_value
            return inner
        if value.startswith("$"):
            return eval_expr(value, node, domain)
        return value
    if isinstance(value, list):
        return [eval_template(item, node, domain) for item in value]
    if isinstance(value, dict):
        return {k: eval_template(v, node, domain) for k, v in value.items()}
    return value


def apply_emit(spec: AtomSpec, node: Node, domain: str) -> Atom:
    emit = {k: eval_template(v, node, domain) for k, v in spec.emit.items()}
    anchor = emit.get("anchor")
    key = emit.get("key")
    value = emit.get("value")
    time_window = emit.get("time_window")
    domain_value = emit.get("domain") if "domain" in emit else domain
    path = ".".join(node.path_segments)
    atom_id = stable_atom_id(spec.rule_id, path)
    extra = {
        k: v
        for k, v in emit.items()
        if k not in {"anchor", "key", "value", "time_window", "domain"}
    }
    return Atom(
        atom_id=atom_id,
        domain=domain_value,
        anchor=anchor,
        time_window=time_window,
        key=key,
        value=value,
        path=path,
        extra=extra,
    )


def atom_to_dict(atom: Atom) -> Dict[str, Any]:
    data = asdict(atom)
    extra = data.pop("extra", {}) or {}
    for key, value in extra.items():
        if key not in data:
            data[key] = value
    return data


def atomize(schema: Dict[str, Any], specs: List[AtomSpec]) -> List[Atom]:
    atoms: List[Atom] = []
    for domain, domain_obj in schema.items():
        for node in walk_nodes(domain_obj, [str(domain)], [], domain_obj):
            for spec in specs:
                if match_path(spec.match, node.path_segments):
                    atoms.append(apply_emit(spec, node, str(domain)))
    return atoms


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Atom engine: ir + schema -> atoms")
    parser.add_argument(
        "--schema",
        default="schema.json",
        help="Path to schema.json",
    )
    parser.add_argument(
        "--ir",
        default="atom_ir.yaml",
        help="Path to atom_ir.yaml",
    )
    parser.add_argument(
        "--out",
        default="atoms.json",
        help="Output path for atoms json",
    )
    args = parser.parse_args()

    schema = load_schema(args.schema)
    specs = load_ir(args.ir)
    atoms = atomize(schema, specs)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([atom_to_dict(a) for a in atoms], f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
