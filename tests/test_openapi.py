"""OpenAPI verification and contract-drift detection.

Two distinct concerns:

**Verification** — is this document usable by a code generator right now?
Dangling refs, missing operationIds and undocumented error responses all
degrade a generated client silently rather than loudly.

**Stability** — has the contract structure changed since it was last agreed?
The snapshot compares a prose-stripped structural signature, so rewording a
description is free and removing a required field is not. When it fails, it
prints what moved.

To accept an intended change::

    python -m backend.openapi --update-snapshot
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import openapi as oa  # noqa: E402

SCHEMA = oa.build_schema()
SIGNATURE = oa.structural_signature(SCHEMA)

# Paths FastAPI serves for its own sake, which are not part of the API contract.
_INFRA_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


def _registered_operations() -> set[tuple[str, str]]:
    """(method, path) for every route the app actually serves.

    Derived from the app rather than listed here. A hardcoded set has to be
    edited every time a route is added, which means it is either updated
    mechanically — teaching nobody anything — or forgotten, at which point it
    asserts the wrong thing confidently.
    """
    from backend.app import create_app
    from backend.config import Settings
    from jobqueue.memory import InMemoryQueue

    app = create_app(
        Settings(
            api_prefix="",
            docs_enabled=True,
            log_json=False,
            log_level="CRITICAL",
            storage_root=Path(tempfile.mkdtemp(prefix="dfp-openapi-")),
        ),
        queue=InMemoryQueue(),
    )
    found: set[tuple[str, str]] = set()

    def walk(routes, depth: int = 0) -> None:
        """Descend through included routers to the real endpoints.

        FastAPI (0.141) does not flatten ``include_router`` into ``app.routes``.
        It inserts ``_IncludedRouter`` wrappers whose own ``.routes`` is ``None``
        and whose real content hangs off ``original_router`` — which itself may
        hold further wrappers, since this app includes a router of routers.

        A single-level walk therefore finds only the ``/docs`` routes and
        concludes the app serves nothing. The assertion below exists because
        that failure is silent: an empty set makes "every route is documented"
        vacuously true, and the check would pass forever while testing nothing.
        """
        if depth > 8:  # pragma: no cover - structural guard
            raise AssertionError("route tree deeper than expected; walk may be looping")
        for route in routes:
            nested = getattr(route, "original_router", None) or getattr(route, "routes", None)
            if nested is not None:
                walk(getattr(nested, "routes", nested), depth + 1)
                continue
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if not path or not methods or path in _INFRA_PATHS:
                continue
            for method in methods:
                if method in ("HEAD", "OPTIONS"):
                    continue
                found.add((method.lower(), path))

    walk(app.routes)
    assert found, "route discovery found nothing; the walk is broken, not the app"
    return found


# --------------------------------------------------------------------------
# Document shape
# --------------------------------------------------------------------------


def test_every_registered_route_is_documented():
    """Every route the app serves must appear in the schema.

    Stronger than asserting a fixed list: an undocumented route is a route a
    generated client cannot call, and this catches it the moment it is added
    rather than whenever someone remembers to update a constant.
    """
    documented = {
        (method, path)
        for path, methods in SCHEMA["paths"].items()
        for method in methods
        if method in ("get", "post", "put", "patch", "delete")
    }
    missing = _registered_operations() - documented
    assert not missing, f"routes served but absent from the OpenAPI document: {sorted(missing)}"


def test_no_phantom_routes_are_documented():
    """And nothing documented that the app does not serve."""
    documented = {
        (method, path)
        for path, methods in SCHEMA["paths"].items()
        for method in methods
        if method in ("get", "post", "put", "patch", "delete")
    }
    phantom = documented - _registered_operations()
    assert not phantom, f"documented but not served: {sorted(phantom)}"


def test_document_declares_the_contract_version():
    from contracts import CONTRACT_VERSION

    assert SCHEMA["info"]["x-contract-version"] == CONTRACT_VERSION


def test_document_is_json_serialisable():
    json.dumps(SCHEMA)


def test_verification_passes():
    problems = oa.verify(SCHEMA)
    assert problems == [], "OpenAPI problems:\n  " + "\n  ".join(problems)


def test_every_ref_resolves():
    """Both component roots: schemas for models, responses for shared errors."""
    roots = {
        "#/components/schemas/": set(SCHEMA["components"]["schemas"]),
        "#/components/responses/": set(SCHEMA["components"].get("responses", {})),
    }
    for ref in set(oa._walk_refs(SCHEMA)):
        prefix = next((p for p in roots if ref.startswith(p)), None)
        assert prefix is not None, f"unexpected $ref target: {ref}"
        assert oa._ref_name(ref) in roots[prefix], f"dangling: {ref}"


def test_every_operation_has_an_operation_id():
    """operationId becomes the generated client's function name."""
    seen = set()
    for path, methods in SCHEMA["paths"].items():
        for method, op in methods.items():
            oid = op.get("operationId")
            assert oid, f"{method.upper()} {path} has no operationId"
            assert oid not in seen, f"duplicate operationId {oid}"
            seen.add(oid)


def test_every_operation_documents_a_failure():
    for path, methods in SCHEMA["paths"].items():
        for method, op in methods.items():
            codes = [str(c) for c in op["responses"]]
            assert any(c[0] in "45" for c in codes), f"{method.upper()} {path} documents no failure"


# --------------------------------------------------------------------------
# Error documentation (audit item D-2)
# --------------------------------------------------------------------------

# What each operation can actually raise, traced from the code and confirmed by
# probing the running app. Documenting anything else is a phantom; documenting
# less leaves an integrator guessing.
EXPECTED_ERRORS: dict[tuple[str, str], set[str]] = {
    ("post", "/uploads"): {
        "invalid_request", "file_too_large", "unsupported_media_type", "internal_error",
    },
    ("put", "/uploads/{upload_id}"): {
        "invalid_request", "file_empty", "checksum_mismatch",
        "upload_not_found", "upload_expired", "file_too_large", "internal_error",
    },
    ("post", "/analyze"): {
        "invalid_request", "upload_not_found", "upload_incomplete", "internal_error",
    },
    ("get", "/jobs/{job_id}"): {"invalid_request", "job_not_found", "internal_error"},
    ("get", "/results/{job_id}"): {
        "invalid_request", "job_not_found", "result_not_ready",
        "decode_failed", "internal_error",
    },
    ("get", "/health"): set(),  # readiness reports condition, never an error envelope
}


def _codes_for(method: str, path: str) -> set[str]:
    catalogue = oa.error_component_catalogue(SCHEMA)
    op = SCHEMA["paths"][path][method]
    found: set[str] = set()
    for response in op["responses"].values():
        if isinstance(response, dict) and "$ref" in response:
            found.update(catalogue.get(oa._ref_name(response["$ref"]), []))
    return found


def test_every_operation_documents_the_errors_it_can_raise():
    """Completeness. A missing code is a branch a client never writes."""
    problems = []
    for (method, path), expected in EXPECTED_ERRORS.items():
        actual = _codes_for(method, path)
        if missing := expected - actual:
            problems.append(f"{method.upper()} {path} omits {sorted(missing)}")
    assert not problems, "undocumented errors:\n  " + "\n  ".join(problems)


def test_no_phantom_error_codes_are_documented():
    """The converse. A documented error the server cannot produce sends an
    integrator down a branch that never executes."""
    problems = []
    for (method, path), expected in EXPECTED_ERRORS.items():
        actual = _codes_for(method, path)
        if extra := actual - expected:
            problems.append(f"{method.upper()} {path} invents {sorted(extra)}")
    assert not problems, "phantom errors:\n  " + "\n  ".join(problems)


def test_the_auto_generated_422_is_gone():
    """FastAPI documents 422 HTTPValidationError on every operation with a body.

    This API never emits it: the RequestValidationError handler answers 400 with
    an ErrorResponse. Leaving it documented types the one branch that never runs
    and leaves the one that does as unknown.
    """
    assert "HTTPValidationError" not in SCHEMA["components"]["schemas"]
    assert "ValidationError" not in SCHEMA["components"]["schemas"]
    for path, methods in SCHEMA["paths"].items():
        for method, op in methods.items():
            response = op.get("responses", {}).get("422")
            if response is None:
                continue
            assert "$ref" in response, f"{method.upper()} {path} has a non-component 422"
            assert oa._ref_name(response["$ref"]) == "DecodeFailed", (
                f"{method.upper()} {path} documents an unexpected 422"
            )


def test_error_responses_are_shared_components_not_inlined():
    """Component reuse. Four operations can return job_not_found; there must be
    one definition of it, not four copies to update inconsistently."""
    inlined = []
    for path, methods in SCHEMA["paths"].items():
        for method, op in methods.items():
            for status, response in op["responses"].items():
                if not str(status).startswith(("4", "5")) or "$ref" in response:
                    continue
                body = response.get("content", {}).get("application/json", {}).get("schema", {})
                if oa._ref_name(body.get("$ref", "")) == "ErrorResponse":
                    # An error envelope spelled out in full instead of hoisted.
                    # /health's 503 is deliberately exempt: it carries a
                    # HealthResponse, not an error envelope.
                    inlined.append(f"{method.upper()} {path} {status}")
    assert not inlined, "inline error responses:\n  " + "\n  ".join(inlined)


def test_shared_components_are_actually_shared():
    """Prove reuse rather than assuming it: at least one component must be
    referenced by more than one operation, or hoisting achieved nothing."""
    counts: dict[str, int] = {}
    for methods in SCHEMA["paths"].values():
        for op in methods.values():
            for response in op["responses"].values():
                if isinstance(response, dict) and "$ref" in response:
                    name = oa._ref_name(response["$ref"])
                    counts[name] = counts.get(name, 0) + 1
    reused = {n: c for n, c in counts.items() if c > 1}
    assert reused, "no error component is referenced twice"
    assert counts.get("InternalError", 0) >= 5


def test_every_error_component_has_a_schema_description_and_examples():
    components = SCHEMA["components"]["responses"]
    assert components, "no error components were hoisted"
    for name, response in components.items():
        assert response.get("description"), f"{name} has no description"
        content = response.get("content", {}).get("application/json", {})
        schema = content.get("schema", {})
        assert schema, f"{name} references no schema"
        assert oa._ref_name(schema.get("$ref", "")) == "ErrorResponse", (
            f"{name} does not reference the ErrorResponse component"
        )
        examples = content.get("examples", {})
        assert examples, f"{name} has no example payload"
        for label, example in examples.items():
            assert "value" in example, f"{name}[{label}] has no value"


def test_every_documented_example_validates_against_contracts():
    """An example that the server would reject is worse than none, because an
    integrator codes against it."""
    from contracts.errors import ErrorResponse

    checked = 0
    for name, response in SCHEMA["components"]["responses"].items():
        examples = response["content"]["application/json"]["examples"]
        for label, example in examples.items():
            ErrorResponse.model_validate(example["value"])
            assert example["value"]["error"]["code"] == label, (
                f"{name}[{label}] example carries code "
                f"{example['value']['error']['code']!r}"
            )
            checked += 1
    assert checked >= 12, f"only {checked} error examples documented"


def test_error_response_component_exists():
    """It was entirely absent before D-2 — every error branch generated untyped."""
    assert "ErrorResponse" in SCHEMA["components"]["schemas"]
    assert "ApiError" in SCHEMA["components"]["schemas"]


def test_documented_codes_are_a_subset_of_the_contract_enum():
    from contracts.enums import ErrorCode

    known = {c.value for c in ErrorCode}
    documented = oa.documented_error_codes(SCHEMA)
    assert documented <= known, f"unknown codes documented: {sorted(documented - known)}"


def test_health_does_not_document_an_error_envelope():
    """Readiness answers the question it was asked. Wrapping a failing probe in
    an error envelope would make a load balancer check look like a failure."""
    op = SCHEMA["paths"]["/health"]["get"]
    for status, response in op["responses"].items():
        assert "$ref" not in response, f"/health {status} points at an error component"
        schema = response["content"]["application/json"]["schema"]
        assert oa._ref_name(schema["$ref"]) == "HealthResponse"


def test_key_response_models_are_referenced():
    for name in (
        "EvidenceBundle",
        "JobStatus",
        "CreateUploadResponse",
        "AnalyzeAcceptedResponse",
        "HealthResponse",
    ):
        assert name in SCHEMA["components"]["schemas"], f"{name} missing from components"


def test_every_success_response_resolves_to_a_real_schema():
    """A 2xx that documents a body must point at something a generator can emit.

    A response whose schema is absent or dangling makes a generated client
    return ``any``, silently removing the type safety this whole approach
    exists to provide.
    """
    components = set(SCHEMA["components"]["schemas"])
    for path, methods in SCHEMA["paths"].items():
        for method, op in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            for code, body in op["responses"].items():
                if not str(code).startswith("2"):
                    continue
                content = body.get("content", {})
                if not content:
                    continue  # 204 and friends legitimately carry no body
                schema = content.get("application/json", {}).get("schema")
                assert schema, f"{method.upper()} {path} {code}: JSON response with no schema"
                for ref in oa._walk_refs(schema):
                    assert oa._ref_name(ref) in components, (
                        f"{method.upper()} {path} {code}: dangling ref {ref}"
                    )


def test_operation_ids_are_unique_across_the_document():
    """operationId becomes the generated client's function name; a duplicate
    silently overwrites one of the two methods."""
    seen: dict[str, str] = {}
    for path, methods in SCHEMA["paths"].items():
        for method, op in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            oid = op.get("operationId")
            assert oid, f"{method.upper()} {path} has no operationId"
            assert oid not in seen, (
                f"duplicate operationId {oid!r}: {seen[oid]} and {method.upper()} {path}"
            )
            seen[oid] = f"{method.upper()} {path}"


def test_examples_are_attached_to_components():
    bundle = SCHEMA["components"]["schemas"]["EvidenceBundle"]
    assert len(bundle.get("examples", [])) == 8
    verdicts = {e["verdict"] for e in bundle["examples"]}
    assert verdicts == {"authentic", "manipulated", "insufficient_evidence"}


def test_enums_are_closed_in_the_document():
    defs = SCHEMA["components"]["schemas"]
    assert set(defs["Verdict"]["enum"]) == {
        "authentic",
        "manipulated",
        "insufficient_evidence",
    }
    assert set(defs["BranchId"]["enum"]) == {"A", "B", "P"}


def test_results_route_documents_the_conflict_and_not_found_cases():
    responses = SCHEMA["paths"]["/results/{job_id}"]["get"]["responses"]
    for code in ("200", "404", "409"):
        assert code in responses, f"/results must document {code}"


# --------------------------------------------------------------------------
# Structural signature
# --------------------------------------------------------------------------


def test_signature_is_deterministic():
    assert oa.fingerprint(SIGNATURE) == oa.fingerprint(
        oa.structural_signature(oa.build_schema())
    )


def test_signature_ignores_prose_but_not_structure():
    """The property that keeps this check credible rather than noisy."""
    reworded = json.loads(json.dumps(SCHEMA))
    reworded["info"]["description"] = "completely different prose"
    reworded["paths"]["/health"]["get"]["summary"] = "Reworded summary"
    reworded["components"]["schemas"]["EvidenceBundle"]["description"] = "new words"
    assert oa.fingerprint(oa.structural_signature(reworded)) == oa.fingerprint(SIGNATURE)

    breaking = json.loads(json.dumps(SCHEMA))
    breaking["components"]["schemas"]["EvidenceBundle"]["required"].remove("verdict")
    assert oa.fingerprint(oa.structural_signature(breaking)) != oa.fingerprint(SIGNATURE)


def test_signature_detects_a_removed_endpoint():
    mutated = json.loads(json.dumps(SCHEMA))
    del mutated["paths"]["/results/{job_id}"]
    changes = oa.diff_signatures(SIGNATURE, oa.structural_signature(mutated))
    assert any("REMOVED endpoint" in c and "/results" in c for c in changes)


def test_signature_detects_a_type_change():
    mutated = json.loads(json.dumps(SCHEMA))
    mutated["components"]["schemas"]["EvidenceBundle"]["properties"]["p_fake"] = {
        "type": "number"
    }
    changes = oa.diff_signatures(SIGNATURE, oa.structural_signature(mutated))
    assert any("p_fake" in c and "type" in c for c in changes), changes


def test_signature_detects_a_newly_required_field_as_breaking():
    mutated = json.loads(json.dumps(SCHEMA))
    mutated["components"]["schemas"]["EvidenceBundle"]["required"].append("disagreement")
    changes = oa.diff_signatures(SIGNATURE, oa.structural_signature(mutated))
    assert any("NOW REQUIRED" in c for c in changes)


def test_signature_detects_a_removed_response_code():
    mutated = json.loads(json.dumps(SCHEMA))
    del mutated["paths"]["/results/{job_id}"]["get"]["responses"]["409"]
    changes = oa.diff_signatures(SIGNATURE, oa.structural_signature(mutated))
    assert any("REMOVED response 409" in c for c in changes)


def test_signature_detects_an_enum_change():
    mutated = json.loads(json.dumps(SCHEMA))
    mutated["components"]["schemas"]["BranchId"]["enum"].append("C")
    changes = oa.diff_signatures(SIGNATURE, oa.structural_signature(mutated))
    assert any("BranchId" in c and "enum" in c for c in changes)


def test_signature_is_independent_of_the_api_prefix():
    """The committed snapshot must not depend on a developer's local env."""
    import os

    previous = os.environ.get("DFP_API_PREFIX")
    os.environ["DFP_API_PREFIX"] = "api/v99"
    try:
        assert oa.fingerprint(oa.structural_signature(oa.build_schema())) == oa.fingerprint(
            SIGNATURE
        )
    finally:
        if previous is None:
            os.environ.pop("DFP_API_PREFIX", None)
        else:
            os.environ["DFP_API_PREFIX"] = previous


# --------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------


def test_snapshot_exists_and_is_committed():
    assert oa.SNAPSHOT_PATH.exists(), (
        "no OpenAPI snapshot on disk. Create it with "
        "`python -m backend.openapi --update-snapshot` and commit the file."
    )


def test_no_contract_drift():
    """Fails when the API structure changed without the snapshot being updated."""
    unchanged, changes = oa.check_drift()
    assert unchanged, (
        "OpenAPI contract drift detected:\n  "
        + "\n  ".join(changes)
        + "\n\nIf intended: python -m backend.openapi --update-snapshot"
    )


def test_snapshot_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snap.json"
        fp = oa.write_snapshot(SIGNATURE, path)
        loaded = oa.load_snapshot(path)
        assert loaded["fingerprint"] == fp
        assert loaded["signature"] == SIGNATURE


def test_snapshot_stores_the_signature_not_only_the_hash():
    """A bare hash mismatch tells a developer nothing actionable."""
    snapshot = oa.load_snapshot()
    assert "signature" in snapshot
    assert snapshot["signature"]["paths"], "signature must carry the path structure"
    assert snapshot["signature"]["components"], "signature must carry component structure"


def test_missing_snapshot_counts_as_drift():
    with tempfile.TemporaryDirectory() as tmp:
        unchanged, changes = oa.check_drift(Path(tmp) / "absent.json")
    assert not unchanged
    assert any("no snapshot" in c for c in changes)


def test_real_drift_is_reported_with_a_readable_diff():
    """End-to-end: a snapshot taken from a different API must fail the check."""
    mutated = json.loads(json.dumps(SCHEMA))
    mutated["components"]["schemas"]["EvidenceBundle"]["required"].remove("verdict")
    del mutated["paths"]["/results/{job_id}"]["get"]["responses"]["409"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snap.json"
        oa.write_snapshot(oa.structural_signature(mutated), path)
        unchanged, changes = oa.check_drift(path)

    assert not unchanged
    assert any("verdict" in c and "NOW REQUIRED" in c for c in changes), changes
    assert any("added response 409" in c for c in changes), changes


def test_hand_edited_snapshot_is_rejected_as_inconsistent():
    """A snapshot whose fingerprint disagrees with its own signature would let
    the fingerprint check pass while the diff it produces is fiction."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snap.json"
        oa.write_snapshot(SIGNATURE, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["signature"]["components"]["EvidenceBundle"]["required"].remove("verdict")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        unchanged, changes = oa.check_drift(path)

    assert not unchanged
    assert any("internally inconsistent" in c for c in changes), changes


def test_malformed_snapshot_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snap.json"
        path.write_text(json.dumps({"fingerprint": "abc"}), encoding="utf-8")
        unchanged, changes = oa.check_drift(path)
    assert not unchanged
    assert any("malformed" in c for c in changes)


def test_cli_verify_and_drift_modes_succeed():
    assert oa.main(["openapi.py", "--verify"]) == 0
    assert oa.main(["openapi.py", "--check-drift"]) == 0


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = sorted(
        (n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
