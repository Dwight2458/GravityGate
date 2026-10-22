"""Model metadata captured by the Rust implementation; no invented routes."""

from dataclasses import dataclass

TIERS = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class Spec:
    id: str
    display_name: str
    family: str = "gemini"
    context_limit: int = 1_048_576
    output_limit: int = 65_536
    routes: tuple[tuple[str, int], ...] = ()
    supports_tools: bool = True
    input_modalities: tuple[str, ...] = ("text", "image", "pdf")
    output_modalities: tuple[str, ...] = ("text",)


def flash(version: str, routes: tuple[tuple[str, int], ...]) -> Spec:
    return Spec(f"gemini-{version}-flash", f"Gemini {version} Flash", routes=routes)


MODELS = (
    Spec(
        "gemini-3.1-pro",
        "Gemini 3.1 Pro",
        output_limit=65_535,
        routes=(
            ("gemini-3.1-pro-low", 1001),
            ("gemini-3.1-pro-low", 1001),
            ("gemini-pro-agent", 10001),
        ),
    ),
    flash(
        "3.5",
        (
            ("gemini-3.5-flash-extra-low", 1000),
            ("gemini-3.5-flash-low", 4000),
            ("gemini-3-flash-agent", 10000),
        ),
    ),
    flash(
        "3.6",
        tuple(
            (f"gemini-3.6-flash-{t}", b)
            for t, b in zip(TIERS[1:], (1000, 4000, 10000), strict=True)
        ),
    ),
    flash(
        "3.7",
        tuple(
            (f"gemini-3.7-flash-{t}", b) for t, b in zip(TIERS[1:], (1000, 4000, -1), strict=True)
        ),
    ),
    flash(
        "3.8",
        tuple(
            (f"gemini-3.8-flash-{t}", b) for t, b in zip(TIERS[1:], (1000, 4000, -1), strict=True)
        ),
    ),
    Spec(
        "claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "claude",
        250_000,
        64_000,
        tuple(("claude-sonnet-4-6", b) for b in (8192, 16384, 32768)),
    ),
    Spec(
        "claude-opus-4-6-thinking",
        "Claude Opus 4.6 Thinking",
        "claude",
        250_000,
        64_000,
        tuple(("claude-opus-4-6-thinking", b) for b in (8192, 16384, 32768)),
    ),
    Spec(
        "gpt-oss-120b-medium",
        "GPT-OSS 120B",
        "gpt-oss",
        131_072,
        32_768,
        tuple(("gpt-oss-120b-medium", b) for b in (4096, 8192, 16384)),
    ),
    Spec(
        "gemini-3.1-flash-image",
        "Gemini 3.1 Flash Image",
        context_limit=66_000,
        output_limit=33_000,
        supports_tools=False,
        input_modalities=("text", "image"),
        output_modalities=("text", "image"),
    ),
)
CATALOGUE = {spec.id: spec for spec in MODELS}


def family_for(name: str) -> str:
    return "claude" if "claude" in name else "gpt-oss" if name.startswith("gpt-") else "gemini"


def base_for_wire(name: str) -> Spec | None:
    return next((s for s in MODELS if name == s.id or any(w == name for w, _ in s.routes)), None)


@dataclass(frozen=True)
class Resolved:
    wire_model: str
    family: str
    tier: str
    budget: int | None
    thinking: bool
    spec: Spec | None

    @property
    def quota_key(self) -> str:
        spec = base_for_wire(self.wire_model)
        if not spec and self.wire_model.endswith("-tiered"):
            spec = CATALOGUE.get(self.wire_model.removesuffix("-tiered"))
        base = spec.id if spec else self.wire_model
        pool = "non-gemini" if self.family == "gpt-oss" else self.family
        return f"{pool}:{base}"

    @property
    def output_limit(self) -> int:
        return self.spec.output_limit if self.spec else 32_768


def resolve(request: dict, default_tier: str = "medium") -> Resolved:
    name = request.get("model", "").strip().lower().removeprefix("antigravity-")
    if not name:
        raise ValueError("model name is empty")
    thinking = request.get("thinking") or {}
    budget = thinking.get("budget_tokens")
    tier = request.get("reasoning_effort", default_tier)
    tier = tier if tier in TIERS else default_tier
    if isinstance(budget, int) and budget > 0:
        tier = "low" if budget <= 4000 else "medium" if budget <= 16384 else "high"
    spec = CATALOGUE.get(name)
    if not spec:
        for suffix in TIERS:
            if name.endswith("-" + suffix):
                tier = suffix
                spec = CATALOGUE.get(name[: -len(suffix) - 1])
                break
    enabled = thinking.get("type") not in {"disabled", "off", "none"} and thinking.get(
        "enabled", True
    )
    if not spec:
        return Resolved(name, family_for(name), tier, None, enabled, None)
    if not spec.routes:
        return Resolved(spec.id, spec.family, tier, None, False, spec)
    wire, budget = spec.routes[max(0, TIERS.index(tier) - 1)]
    return Resolved(wire, spec.family, tier, budget if enabled else None, enabled, spec)


def model_list(live: dict | None = None) -> list[dict]:
    entries = {}
    for spec in MODELS:
        names = [spec.id]
        if spec.routes:
            base_wire = resolve({"model": spec.id}).wire_model
            names += [
                spec.id + "-" + tier
                for tier in TIERS[1:]
                if resolve({"model": spec.id + "-" + tier}).wire_model not in {spec.id, base_wire}
            ]
        for name in names:
            entries[name] = {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": {"claude": "anthropic", "gpt-oss": "openai"}.get(spec.family, "google"),
                "display_name": spec.display_name,
                "supports_reasoning": bool(spec.routes),
                "supports_tools": spec.supports_tools,
                "context_length": spec.context_limit,
                "max_output_tokens": spec.output_limit,
                "input_modalities": list(spec.input_modalities),
                "output_modalities": list(spec.output_modalities),
            }
    for name, value in (live or {}).items():
        value = value if isinstance(value, dict) else {}
        if name.startswith(("gemini-", "claude-", "gpt-")):
            entries.setdefault(
                name,
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": {"claude": "anthropic", "gpt-oss": "openai"}.get(
                        family_for(name), "google"
                    ),
                    "display_name": value.get("displayName", name),
                },
            )
    return list(entries.values())
