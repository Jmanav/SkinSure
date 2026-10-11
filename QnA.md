# SkinSure — Q&A Log

A running log of questions asked about this codebase and their answers, kept for future reference.

---

### Q: Why do we inherit from `BaseModel` instead of just using a plain class with attributes?

`BaseModel` (Pydantic) gives you, for free, things a plain class doesn't:
- **Validation at construction time** — bad data (e.g. a string where a float is expected) raises immediately at the boundary, not silently later.
- **Type coercion** — e.g. `"0.82"` → `0.82`.
- **Free serialization** — `.model_dump()` / `.model_dump_json()`.
- **Parsing from untrusted input** — `Predictions(**raw_dict)` validates the whole shape in one call.
- **Custom cross-field rules** via `@model_validator`, which a plain class has no hook for.

For SkinSure specifically, this matters because the schemas enforce clinical safety boundaries (Fitzpatrick IV–VI gate, urgent → immediate_referral) that must be impossible to accidentally skip — not just convention.

---

### Q: What does `@model_validator(mode="after")` do? (`api/schemas/predictions.py` — `ClinicalSummary.validate_urgency`)

It validates a field **in relation to other fields**, after the object is fully constructed.

- `mode="after"` — all individual fields are already validated/coerced; you get `self` as a fully-populated instance and can run cross-field logic.
- (`mode="before"` runs on the raw input dict before field validation, for reshaping input.)
- Raising inside it (e.g. `ValueError`) becomes a `ValidationError`, so construction fails outright — you never get an invalid instance.
- Must `return self` at the end.

Here it prevents `severity="urgent"` from ever coexisting with `immediate_referral=False` — a second line of defense alongside the LLM system prompt telling it the same rule.

**Update (2026-10-10):** an earlier version of this validator compared `confidence_level` to a 0.80 threshold (CLAUDE.md said 0.85) and had a wrong error message. It was rewritten to enforce `urgent → immediate_referral`, as `CLAUDE.md` and `docs/architecture.md` specify; `confidence_level` is now a `high`/`moderate`/`low` enum.

---

### Q: What does `@field_validator` do, and why does it need `@classmethod`? (`api/schemas/patient.py` — `PatientMetadata.validate_fitzpatrick`)

`@field_validator` validates a **single field** in isolation (vs. `@model_validator`, which validates across the whole object).

- `@field_validator("fitzpatrick_type")` runs whenever that field is set; `v` arrives already coerced to `FitzpatrickType` by the enum.
- Enforces the domain rule "only Fitzpatrick IV–VI in scope" — something the `Enum` alone can't express (the enum only rejects invalid values like `"VII"`).

**Why `@classmethod`:** Pydantic calls field validators during construction, before an instance exists — there's no `self` yet. The expected signature is `(cls, v)`, and only a classmethod can be called against the class itself rather than needing an already-built instance. `cls` isn't used in the validator's body here, but Pydantic's calling convention still requires the method be shaped this way.

---

### Q: What are YAML files?

YAML ("YAML Ain't Markup Language") is a plain-text structured-data format, like JSON but easier to hand-write — indentation instead of braces/quotes/commas. Any JSON is valid YAML, but the indented style is far more readable for hand-edited config.

In this repo: `configs/agents.yaml` and `configs/models.yaml` hold config meant to be tweaked without touching Python (model choice, token limits, temperature, ensemble weights). `src/utils/config.py`'s `get_settings()` loads `agents.yaml` into a `Settings` (`BaseSettings`) object — same validation idea as `BaseModel`, but `BaseSettings` also knows how to read YAML/env vars directly.

Note: as of 2026-10-10 only `agents.yaml` is wired up via `get_settings()`; `models.yaml` (checkpoint paths and ensemble weights) is now read by `get_model_config()` / `get_checkpoint_paths()`.
