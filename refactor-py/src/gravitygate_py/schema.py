"""Reduce tool schemas to the upstream's supported subset."""

import copy
import json

HINTS = {
    "format": "Format",
    "pattern": "Pattern",
    "minLength": "Min length",
    "maxLength": "Max length",
    "minimum": "Minimum",
    "maximum": "Maximum",
    "exclusiveMinimum": "Exclusive minimum",
    "exclusiveMaximum": "Exclusive maximum",
    "multipleOf": "Multiple of",
    "minItems": "Min items",
    "maxItems": "Max items",
    "uniqueItems": "Items must be unique",
    "minProperties": "Min properties",
    "maxProperties": "Max properties",
    "default": "Default",
    "examples": "Examples",
}
ALLOWED = {"type", "description", "properties", "required", "items", "enum", "nullable"}


def render(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(render(item) for item in value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def score(value) -> int:
    if not isinstance(value, dict):
        return 0
    if value.get("properties"):
        return 3
    kind = value.get("type", "string")
    return {"object": 3, "array": 2, "null": 0}.get(kind, 1) if isinstance(kind, str) else 1


def sanitize(value, depth: int = 0):
    if not isinstance(value, dict):
        return None if depth == 0 else {"type": "STRING"}
    if depth > 32:
        return {"type": "STRING"}
    src = copy.deepcopy(value)
    hints = []
    if "$ref" in src:
        hints.append("See: " + str(src.pop("$ref")).rsplit("/", 1)[-1])
    props = {}
    for member in src.pop("allOf", []):
        if isinstance(member, dict):
            props.update(member.get("properties", {}))
            for key, item in member.items():
                src.setdefault(key, item)
    if props:
        props.update(value.get("properties", {}))
        src["properties"] = props
    options = src.pop("anyOf", src.pop("oneOf", []))
    if options:
        best = max(reversed(options), key=score)
        if isinstance(best, dict):
            for key, item in best.items():
                src.setdefault(key, item)
        names = []
        for item in options:
            if isinstance(item, dict) and isinstance(item.get("type"), str):
                name = item["type"]
                if "enum" in item:
                    name += "(" + ",".join(render(v) for v in item["enum"]) + ")"
                names.append(name)
        if names:
            hints.append("Accepts: " + " | ".join(names))
    if "const" in src:
        src.setdefault("enum", [src.pop("const")])
    nullable = src.get("nullable", False)
    if isinstance(src.get("type"), list):
        types = src["type"]
        nullable |= "null" in types
        non_null = [t for t in types if t != "null"]
        src["type"] = (
            max(reversed(non_null), key=lambda t: score({"type": t})) if non_null else "string"
        )
        if len(non_null) > 1:
            hints.append("One of: " + ", ".join(non_null))
    for keyword, label in HINTS.items():
        if keyword in src:
            text = render(src.pop(keyword))
            if text:
                hints.append(label if keyword == "uniqueItems" else f"{label}: {text}")
    out = {k: v for k, v in src.items() if k in ALLOWED}
    kind = str(out.get("type", "string")).upper()
    out["type"] = "STRING" if kind == "NULL" else kind
    if kind == "OBJECT":
        out["properties"] = {
            k: sanitize(v, depth + 1) for k, v in out.get("properties", {}).items()
        }
        if not out["properties"]:
            out["properties"] = {
                "reason": {"type": "STRING", "description": "Reason for calling this tool"}
            }
        out.pop("items", None)
        required = [k for k in out.get("required", []) if k in out["properties"]]
        if required:
            out["required"] = required
        else:
            out.pop("required", None)
    elif kind == "ARRAY":
        items = out.get("items", {})
        if isinstance(items, list):
            items = max(reversed(items), key=score) if items else {}
        out["items"] = sanitize(items, depth + 1)
        out.pop("properties", None)
        out.pop("required", None)
    else:
        for key in ("properties", "required", "items"):
            out.pop(key, None)
    if nullable:
        out["nullable"] = True
    if hints:
        hint = ". ".join(hints)
        out["description"] = f"{out['description']} ({hint})" if out.get("description") else hint
    return out
