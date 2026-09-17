"""JSON Schema generation and example-payload tests.

Runs standalone (``python tests/test_schema.py``) and under pytest unchanged.

The load-bearing test here is ``test_every_example_validates``: a published
example payload that the server would reject is worse than no example at all,
because it sends integrators down a path that cannot work. This suite makes that
a build failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from contracts import CONTRACT_VERSION, EvidenceBundle, MediaKind, Verdict  # noqa: E402
from contracts import examples as ex  # noqa: E402
from contracts import schema as sch  # noqa: E402


# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------


def test_every_example_validates():
    """No published example may be one the server would reject."""
    problems = sch.verify_examples()
    assert problems == [], "invalid examples:\n  " + "\n  ".join(problems)


def test_every_root_model_has_at_least_one_example():
    missing = [m.__name__ for m in sch.ROOT_MODELS if not ex.examples_for(m)]
    assert missing == [], f"root models with no example payload: {missing}"


def test_examples_are_json_serialisable():
    for name, labelled in ex.all_examples().items():
        for label, payload in labelled.items():
            json.dumps(payload)  # raises TypeError on a stray datetime or UUID
            assert isinstance(payload, dict), f"{name}[{label}] is not an object"


def test_example_labels_are_unique_per_model():
    for model, entries in ex.EXAMPLES.items():
        labels = [label for label, _ in entries]
        assert len(set(labels)) == len(labels), f"{model.__name__} has duplicate labels"


def test_evidence_bundle_examples_cover_every_verdict():
    verdicts = {
        EvidenceBundle.model_validate(p).verdict
        for p in ex.examples_for(EvidenceBundle)
    }
    assert verdicts == {
        Verdict.AUTHENTIC,
        Verdict.MANIPULATED,
        Verdict.INSUFFICIENT_EVIDENCE,
    }


def test_evidence_bundle_examples_cover_both_media_kinds():
    kinds = {
        EvidenceBundle.model_validate(p).media.kind for p in ex.examples_for(EvidenceBundle)
    }
    assert kinds == {MediaKind.IMAGE, MediaKind.VIDEO}


def test_upload_response_examples_cover_hit_and_miss():
    labels = set(ex.labelled_examples_for(sch.CreateUploadResponse))
    assert labels == {"cache_hit", "cache_miss"}


def test_error_examples_cover_both_retry_classes():
    payloads = ex.labelled_examples_for(sch.ErrorResponse)
    assert payloads["deterministic"]["error"]["retryable"] is False
    assert payloads["transient"]["error"]["retryable"] is True


def test_examples_are_deterministic():
    assert ex.all_examples() == ex.all_examples()


# --------------------------------------------------------------------------
# Schema generation
# --------------------------------------------------------------------------


def test_all_root_models_produce_a_schema():
    schemas = sch.all_schemas()
    assert len(schemas) == len(sch.ROOT_MODELS)
    for name, s in schemas.items():
        assert s["title"] == name
        assert s["$id"].endswith(f"{name}.json")
        assert "properties" in s, f"{name} has no properties"


def test_schemas_embed_their_examples():
    s = sch.schema_for(EvidenceBundle)
    assert len(s["examples"]) == 8
    assert s["examples"][0]["verdict"] == "manipulated"


def test_schema_can_be_generated_without_examples():
    s = sch.schema_for(EvidenceBundle, with_examples=False)
    assert "examples" not in s


def test_schemas_are_json_serialisable():
    json.dumps(sch.bundle())


def test_bundle_carries_the_contract_version():
    b = sch.bundle()
    assert b["contract_version"] == CONTRACT_VERSION
    assert set(b["models"]) == {m.__name__ for m in sch.ROOT_MODELS}


def test_evidence_bundle_schema_marks_required_fields():
    s = sch.schema_for(EvidenceBundle)
    required = set(s["required"])
    for field in ("job_id", "media", "verdict", "threshold", "quality", "branches"):
        assert field in required, f"{field} should be required"


def test_nested_models_appear_in_defs():
    s = sch.schema_for(EvidenceBundle)
    defs = set(s.get("$defs", {}))
    for nested in ("MediaInfo", "BranchResult", "QualityReport", "ProvenanceReport"):
        assert nested in defs, f"{nested} missing from $defs"


def test_enums_are_emitted_as_closed_string_sets():
    defs = sch.schema_for(EvidenceBundle).get("$defs", {})
    assert set(defs["Verdict"]["enum"]) == {
        "authentic",
        "manipulated",
        "insufficient_evidence",
    }
    assert set(defs["BranchId"]["enum"]) == {"A", "B", "P"}


def test_field_descriptions_survive_into_the_schema():
    s = sch.schema_for(EvidenceBundle)
    assert "description" in s["properties"]["p_fake"]
    assert "full precision" in s["properties"]["p_fake"]["description"].lower()


def test_provenance_credit_bounds_are_in_the_schema():
    """The one-sided rule must be visible to a client reading the schema."""
    defs = sch.schema_for(EvidenceBundle)["$defs"]
    credit = defs["ProvenanceReport"]["properties"]["authenticity_credit"]
    assert credit["minimum"] == 0.0
    assert credit["maximum"] == 1.0


def test_write_all_emits_a_file_per_model_plus_bundle_and_index():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        written = sch.write_all(out)
        assert len(written) == len(sch.ROOT_MODELS) + 2
        assert (out / "contracts.bundle.json").exists()
        assert (out / "index.json").exists()
        assert (out / "EvidenceBundle.schema.json").exists()
        loaded = json.loads((out / "EvidenceBundle.schema.json").read_text(encoding="utf-8"))
        assert loaded["title"] == "EvidenceBundle"


def test_written_index_lists_every_model():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        sch.write_all(out)
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
        assert index["contract_version"] == CONTRACT_VERSION
        assert index["models"] == sorted(m.__name__ for m in sch.ROOT_MODELS)


# --------------------------------------------------------------------------
# Serialisation round trips through the example payloads
# --------------------------------------------------------------------------


def test_every_example_round_trips_byte_identically():
    for model, entries in ex.EXAMPLES.items():
        for label, build in entries:
            original = build()
            raw = original.model_dump_json()
            restored = model.model_validate_json(raw)
            assert restored == original, f"{model.__name__}[{label}] changed on round trip"
            assert restored.model_dump_json() == raw, f"{model.__name__}[{label}] re-serialised differently"


def test_examples_reject_unknown_fields_on_reload():
    payload = ex.examples_for(EvidenceBundle)[0] | {"unexpected": True}
    try:
        EvidenceBundle.model_validate(payload)
    except ValidationError as exc:
        assert "extra" in str(exc).lower()
    else:
        raise AssertionError("extra fields should be rejected")


def test_cli_check_mode_succeeds():
    assert sch.main(["schema.py", "--check"]) == 0


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - a runner reports everything
            failed.append((name, exc))
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
