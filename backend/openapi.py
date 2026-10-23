"""OpenAPI export, verification, and drift detection.

Three jobs, in increasing order of how much they save you later.

**Export.** Write ``openapi.json`` for ``openapi-typescript`` so the frontend
generates its types instead of hand-writing them, and therefore cannot drift
from the contract.

**Verification.** Assert the document is actually usable by a code generator:
every operation has an ``operationId``, every ``$ref`` resolves, every path
documents its error responses, every enum is closed.

**Drift detection.** A fingerprint over a *structural signature* of the
document, snapshotted to disk and compared on every run.

The signature is the part worth explaining. Hashing the raw OpenAPI JSON would
be simpler and useless: every reworded description would fail the build, the
team would learn to run ``--update`` reflexively, and the check would stop
catching anything real. So the signature strips prose — descriptions, summaries,
titles, examples — and keeps only what a client can break against: paths,
methods, status codes, schema references, required fields, property types and
enum members. Rewriting a docstring is then free; removing a required field is
not.

The snapshot stores the whole signature rather than just its hash, because a
developer facing a failed check needs to see *what* moved, not merely that
something did.

Usage::

    python -m backend.openapi --export build/openapi.json
    python -m backend.openapi --verify
    python -m backend.openapi --check-drift
    python -m backend.openapi --update-snapshot     # accept intended changes
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(__file__).resolve().parent / "openapi.snapshot.json"

# Keys that carry prose or presentation and must not influence the fingerprint.
_PROSE_KEYS = frozenset(
    {"description", "summary", "title", "examples", "example", "x-mock-mode"}
)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def build_schema() -> dict[str, Any]:
    """Generate the OpenAPI document from a freshly built app.

    Settings are pinned so the document does not vary with the developer's
    environment — an ``api_prefix`` set locally must not change the committed
    snapshot.
    """
    from backend.app import create_app
    from backend.config import Settings

    app = create_app(
        Settings(
            api_prefix="",
            docs_enabled=True,
            stub_scoring=True,
            log_json=False,
            log_level="CRITICAL",
        )
    )
    return app.openapi()


# --------------------------------------------------------------------------
# Structural signature
# --------------------------------------------------------------------------


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------
# Error response post-processing
# --------------------------------------------------------------------------


def strip_phantom_validation_errors(schema: dict[str, Any]) -> int:
    """Remove FastAPI's automatic 422 ``HTTPValidationError`` responses.

    They are documented on every operation with a body or a typed parameter, and
    this API never emits that shape: the ``RequestValidationError`` handler
    installed at S1 answers 400 with an ``ErrorResponse`` instead. A documented
    response the server cannot produce is worse than an undocumented one — an
    integrator writes a branch for it that never runs, and types the branch that
    does run as ``unknown``.

    Only auto-generated ones are removed. ``/results`` declares a real 422
    (``decode_failed``), which is recognised by its ``ErrorResponse`` schema and
    left alone.
    """
    removed = 0
    for methods in schema.get("paths", {}).values():
        for op in methods.values():
            if not isinstance(op, dict):
                continue
            response = op.get("responses", {}).get("422")
            if not isinstance(response, dict):
                continue
            body = response.get("content", {}).get("application/json", {}).get("schema", {})
            if _ref_name(body.get("$ref", "")) == "HTTPValidationError":
                del op["responses"]["422"]
                removed += 1

    if removed:
        # Repeat until stable. The orphans form a chain — HTTPValidationError
        # references ValidationError — so a single pass removes only the head
        # and leaves the tail looking used by the thing that just died.
        components = schema.get("components", {}).get("schemas", {})
        while True:
            still_used = {_ref_name(r) for r in _walk_refs(schema)}
            orphans = [
                name
                for name in ("HTTPValidationError", "ValidationError")
                if name in components and name not in still_used
            ]
            if not orphans:
                break
            for name in orphans:
                del components[name]
    return removed


def strip_streaming_type_hints(schema: dict[str, Any]) -> int:
    """Drop the ``"type": "string"`` FastAPI places beside a ``$ref`` in a
    non-JSON success response.

    A route with a custom ``response_class`` (here ``text/event-stream``) and a
    documented ``model`` gets ``{"type": "string", "$ref": ...}``: the class's
    default schema merged with the model's. The two contradict each other — the
    referenced model is an object — and a generator honouring ``type`` would
    type every frame as a bare string. The ``$ref`` is the documented truth.
    """
    fixed = 0
    for methods in schema.get("paths", {}).values():
        for op in methods.values():
            if not isinstance(op, dict):
                continue
            for code, response in op.get("responses", {}).items():
                if not str(code).startswith("2") or not isinstance(response, dict):
                    continue
                for media, body in response.get("content", {}).items():
                    node = body.get("schema", {})
                    if media != "application/json" and "$ref" in node and "type" in node:
                        del node["type"]
                        fixed += 1
    return fixed


def promote_error_responses(schema: dict[str, Any]) -> dict[str, list[str]]:
    """Hoist error responses into ``components.responses`` and reference them.

    Every operation that can return ``job_not_found`` should point at one
    definition of it, not carry its own copy. Beyond size, duplication is how a
    message gets updated in four places and missed in a fifth.

    The markers ``x-error-codes`` and ``x-component-name`` are placed by
    ``backend.errors.error_responses`` and stripped here, so they never reach the
    published document.

    Returns component name → error codes, for the tests and the audit.
    """
    responses_root = schema.setdefault("components", {}).setdefault("responses", {})
    catalogue: dict[str, list[str]] = {}

    for methods in schema.get("paths", {}).values():
        for op in methods.values():
            if not isinstance(op, dict):
                continue
            for status, response in list(op.get("responses", {}).items()):
                if not isinstance(response, dict):
                    continue
                name = response.pop("x-component-name", None)
                codes = response.pop("x-error-codes", None)
                if not name or not codes:
                    continue

                if name not in responses_root:
                    responses_root[name] = response
                    catalogue[name] = list(codes)
                elif catalogue.get(name) != list(codes):
                    # Two different code sets resolving to one name would make
                    # the shared component wrong for one of them. Fail loudly
                    # rather than silently documenting the wrong errors.
                    raise ValueError(
                        f"component {name!r} claimed by two different code sets: "
                        f"{catalogue.get(name)} and {list(codes)}"
                    )
                op["responses"][status] = {"$ref": f"#/components/responses/{name}"}

    return catalogue


def error_component_catalogue(schema: dict[str, Any]) -> dict[str, list[str]]:
    """Component name → the error codes it documents. Read from the document."""
    out: dict[str, list[str]] = {}
    for name, response in schema.get("components", {}).get("responses", {}).items():
        examples = (
            response.get("content", {}).get("application/json", {}).get("examples", {})
        )
        out[name] = sorted(examples)
    return out


def documented_error_codes(schema: dict[str, Any]) -> set[str]:
    """Every error code the document claims, across all operations."""
    catalogue = error_component_catalogue(schema)
    seen: set[str] = set()
    for methods in schema.get("paths", {}).values():
        for op in methods.values():
            if not isinstance(op, dict):
                continue
            for response in op.get("responses", {}).values():
                if isinstance(response, dict) and "$ref" in response:
                    seen.update(catalogue.get(_ref_name(response["$ref"]), []))
    return seen


def _type_of(node: dict[str, Any]) -> str:
    """Compact, stable, human-readable type descriptor for one schema node.

    Readability matters: these strings land in the snapshot diff, and
    ``number|null -> number`` is a sentence a reviewer can act on.
    """
    if not isinstance(node, dict):
        return "unknown"
    if "$ref" in node:
        return f"${_ref_name(node['$ref'])}"
    if "enum" in node:
        return "enum[" + ",".join(sorted(str(v) for v in node["enum"])) + "]"
    if "const" in node:
        return f"const[{node['const']}]"
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in node:
            parts = sorted(_type_of(m) for m in node[combinator])
            return "|".join(parts)
    node_type = node.get("type")
    if node_type == "array":
        return f"array<{_type_of(node.get('items', {}))}>"
    if node_type == "object":
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            return f"map<{_type_of(extra)}>"
        return "object"
    if node_type is None:
        return "any"
    fmt = node.get("format")
    return f"{node_type}({fmt})" if fmt else str(node_type)


def _component_signature(schema: dict[str, Any]) -> dict[str, Any]:
    sig: dict[str, Any] = {}
    if "enum" in schema:
        sig["enum"] = sorted(str(v) for v in schema["enum"])
        return sig
    sig["required"] = sorted(schema.get("required", []))
    props = schema.get("properties", {})
    sig["properties"] = {name: _type_of(node) for name, node in sorted(props.items())}
    if schema.get("additionalProperties") is False:
        sig["additionalProperties"] = False
    return sig


def _operation_signature(op: dict[str, Any]) -> dict[str, Any]:
    responses: dict[str, str] = {}
    for code, body in sorted(op.get("responses", {}).items()):
        content = body.get("content", {})
        payload = content.get("application/json", {}).get("schema")
        if payload:
            responses[str(code)] = _type_of(payload)
        elif content:
            # A non-JSON body (the event stream) is recorded with its media
            # type, so swapping the frame model is drift like any other.
            media = sorted(content)[0]
            other = content[media].get("schema")
            responses[str(code)] = f"{media}:{_type_of(other)}" if other else "none"
        else:
            responses[str(code)] = "none"

    params = sorted(
        f"{p.get('in')}:{p.get('name')}:{'required' if p.get('required') else 'optional'}"
        for p in op.get("parameters", [])
    )

    body_schema = (
        op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
    )

    return {
        "operation_id": op.get("operationId"),
        "request_body": _type_of(body_schema) if body_schema else None,
        "parameters": params,
        "responses": responses,
    }


def structural_signature(schema: dict[str, Any]) -> dict[str, Any]:
    """Everything a client can break against; nothing a writer can change."""
    paths: dict[str, Any] = {}
    for path, methods in sorted(schema.get("paths", {}).items()):
        for method, op in sorted(methods.items()):
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            paths[f"{method.upper()} {path}"] = _operation_signature(op)

    components = {
        name: _component_signature(body)
        for name, body in sorted(schema.get("components", {}).get("schemas", {}).items())
    }

    return {
        "openapi": schema.get("openapi"),
        "contract_version": schema.get("info", {}).get("x-contract-version"),
        "paths": paths,
        "components": components,
    }


def fingerprint(signature: dict[str, Any]) -> str:
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _walk_refs(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_walk_refs(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_refs(item))
    return found


def verify(schema: dict[str, Any]) -> list[str]:
    """Return the reasons this document is unfit for code generation.

    Empty list means fit.
    """
    problems: list[str] = []
    components = schema.get("components", {}).get("schemas", {})

    # Every $ref resolves. A dangling ref makes a generator emit `any`, which
    # silently removes the type safety the whole approach exists to provide.
    #
    # Two component roots now: schemas for models, responses for the shared
    # error responses hoisted out of the operations.
    roots = {
        "#/components/schemas/": components,
        "#/components/responses/": schema.get("components", {}).get("responses", {}),
    }
    for ref in set(_walk_refs(schema)):
        for prefix, root in roots.items():
            if ref.startswith(prefix):
                if _ref_name(ref) not in root:
                    problems.append(f"dangling $ref: {ref}")
                break
        else:
            problems.append(f"unexpected $ref target: {ref}")

    for path, methods in schema.get("paths", {}).items():
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            where = f"{method.upper()} {path}"

            # operationId becomes the generated client's function name.
            if not op.get("operationId"):
                problems.append(f"{where}: missing operationId")

            responses = op.get("responses", {})
            if not any(str(c).startswith("2") for c in responses):
                problems.append(f"{where}: no 2xx response documented")

            # Every operation must document at least one failure. An endpoint
            # that documents only its happy path teaches integrators that it
            # cannot fail.
            if not any(str(c)[0] in "45" for c in responses):
                problems.append(f"{where}: no 4xx/5xx response documented")

    # Enums must be closed, or a generator emits a bare string and the
    # exhaustiveness checking on the client side disappears.
    for name, body in components.items():
        if name.endswith(("Verdict", "Status", "State", "Kind", "Code", "Reason")):
            has_enum = "enum" in body or any(
                "enum" in v for v in body.get("properties", {}).values() if isinstance(v, dict)
            )
            if not has_enum and "properties" not in body:
                problems.append(f"component {name}: expected a closed enum")

    return problems


# --------------------------------------------------------------------------
# Snapshot and drift
# --------------------------------------------------------------------------


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_snapshot(signature: dict[str, Any], path: Path = SNAPSHOT_PATH) -> str:
    fp = fingerprint(signature)
    path.write_text(
        json.dumps({"fingerprint": fp, "signature": signature}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fp


def diff_signatures(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Human-readable drift report.

    A bare "the hash changed" tells a developer nothing actionable, so this
    names the endpoint, the component and the field.
    """
    changes: list[str] = []

    if old.get("contract_version") != new.get("contract_version"):
        changes.append(
            f"contract_version: {old.get('contract_version')} -> {new.get('contract_version')}"
        )

    old_paths, new_paths = old.get("paths", {}), new.get("paths", {})
    for gone in sorted(set(old_paths) - set(new_paths)):
        changes.append(f"REMOVED endpoint: {gone}")
    for added in sorted(set(new_paths) - set(old_paths)):
        changes.append(f"added endpoint: {added}")
    for key in sorted(set(old_paths) & set(new_paths)):
        o, n = old_paths[key], new_paths[key]
        if o.get("operation_id") != n.get("operation_id"):
            changes.append(
                f"{key}: operationId {o.get('operation_id')} -> {n.get('operation_id')}"
            )
        if o.get("request_body") != n.get("request_body"):
            changes.append(f"{key}: request body {o.get('request_body')} -> {n.get('request_body')}")
        if o.get("parameters") != n.get("parameters"):
            changes.append(f"{key}: parameters {o.get('parameters')} -> {n.get('parameters')}")
        for code in sorted(set(o.get("responses", {})) - set(n.get("responses", {}))):
            changes.append(f"{key}: REMOVED response {code}")
        for code in sorted(set(n.get("responses", {})) - set(o.get("responses", {}))):
            changes.append(f"{key}: added response {code}")
        for code in sorted(set(o.get("responses", {})) & set(n.get("responses", {}))):
            if o["responses"][code] != n["responses"][code]:
                changes.append(
                    f"{key}: response {code} {o['responses'][code]} -> {n['responses'][code]}"
                )

    old_c, new_c = old.get("components", {}), new.get("components", {})
    for gone in sorted(set(old_c) - set(new_c)):
        changes.append(f"REMOVED component: {gone}")
    for added in sorted(set(new_c) - set(old_c)):
        changes.append(f"added component: {added}")
    for name in sorted(set(old_c) & set(new_c)):
        o, n = old_c[name], new_c[name]
        if o.get("enum") != n.get("enum"):
            changes.append(f"{name}: enum {o.get('enum')} -> {n.get('enum')}")
        for gone in sorted(set(o.get("required", [])) - set(n.get("required", []))):
            changes.append(f"{name}.{gone}: no longer required")
        for added in sorted(set(n.get("required", [])) - set(o.get("required", []))):
            changes.append(f"{name}.{added}: NOW REQUIRED (breaking for existing clients)")
        op, np_ = o.get("properties", {}), n.get("properties", {})
        for gone in sorted(set(op) - set(np_)):
            changes.append(f"{name}.{gone}: REMOVED field")
        for added in sorted(set(np_) - set(op)):
            changes.append(f"{name}.{added}: added field ({np_[added]})")
        for field in sorted(set(op) & set(np_)):
            if op[field] != np_[field]:
                changes.append(f"{name}.{field}: type {op[field]} -> {np_[field]}")

    return changes


def check_drift(path: Path = SNAPSHOT_PATH) -> tuple[bool, list[str]]:
    """(unchanged, changes). A missing or corrupt snapshot counts as drift."""
    signature = structural_signature(build_schema())
    snapshot = load_snapshot(path)
    if snapshot is None:
        return False, ["no snapshot on disk; run --update-snapshot to create one"]

    # The stored fingerprint and the stored signature must agree. They can
    # diverge through a hand edit or a bad merge, and the consequence is worse
    # than it looks: the comparison below would still pass on the fingerprint,
    # while the signature that produces the human-readable diff has silently
    # become fiction. A snapshot that cannot explain itself is not a snapshot.
    stored_sig = snapshot.get("signature")
    stored_fp = snapshot.get("fingerprint")
    if stored_sig is None or stored_fp is None:
        return False, ["snapshot is malformed: expected both 'fingerprint' and 'signature'"]
    if fingerprint(stored_sig) != stored_fp:
        return False, [
            "snapshot is internally inconsistent: its fingerprint does not match "
            "its own signature (hand-edited or a bad merge). "
            "Regenerate with --update-snapshot after reviewing the diff."
        ]

    if fingerprint(signature) == stored_fp:
        return True, []
    return False, diff_signatures(stored_sig, signature)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    schema = build_schema()

    if "--verify" in argv:
        problems = verify(schema)
        if problems:
            print("OpenAPI verification FAILED:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        paths = len(schema.get("paths", {}))
        comps = len(schema.get("components", {}).get("schemas", {}))
        print(f"OpenAPI verified: {paths} paths, {comps} components")
        return 0

    if "--update-snapshot" in argv:
        fp = write_snapshot(structural_signature(schema))
        print(f"snapshot written to {SNAPSHOT_PATH}\nfingerprint {fp}")
        return 0

    if "--check-drift" in argv:
        unchanged, changes = check_drift()
        if unchanged:
            print("no contract drift")
            return 0
        print("CONTRACT DRIFT DETECTED:", file=sys.stderr)
        for c in changes:
            print(f"  - {c}", file=sys.stderr)
        print(
            "\nIf these changes are intended, re-run with --update-snapshot "
            "and commit the result.",
            file=sys.stderr,
        )
        return 1

    out = Path(argv[argv.index("--export") + 1]) if "--export" in argv else Path("openapi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
