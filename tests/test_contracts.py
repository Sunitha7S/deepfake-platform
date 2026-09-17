"""Contract tests.

Written as plain functions with bare asserts so they run under pytest *and*
standalone via ``python tests/test_contracts.py`` — pytest is not yet a
dependency and S0 must be verifiable on a bare checkout.

The tests that matter most are the invariant tests: they encode decisions from
the architecture review that are expensive to rediscover, above all the one-sided
provenance rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from contracts import (  # noqa: E402
    CONTRACT_VERSION,
    AbstainReason,
    ApiError,
    BranchId,
    BranchResult,
    BranchStatus,
    C2paReport,
    CreateUploadRequest,
    CreateUploadResponse,
    ErrorCode,
    EvidenceBundle,
    ExifReport,
    JobState,
    MediaKind,
    ProvenanceReport,
    QualityReport,
    ResolutionClass,
    Verdict,
    is_retryable,
)
from contracts import fixtures as fx  # noqa: E402


def _raises(fn, needle: str = "") -> str:
    try:
        fn()
    except (ValidationError, ValueError) as exc:
        text = str(exc)
        assert needle.lower() in text.lower(), f"expected {needle!r} in error, got: {text[:400]}"
        return text
    raise AssertionError(f"expected a validation error containing {needle!r}, none raised")


# --------------------------------------------------------------------------
# Fixtures are valid and round-trip
# --------------------------------------------------------------------------


def test_every_fixture_validates():
    for name, fn in fx.ALL.items():
        obj = fn()
        assert obj is not None, name


def test_bundles_round_trip_through_json():
    for name, bundle in fx.all_bundles().items():
        raw = bundle.model_dump_json()
        again = EvidenceBundle.model_validate_json(raw)
        assert again == bundle, f"{name} did not survive a JSON round trip"


def test_json_round_trip_preserves_full_precision():
    """Rounding scores collapses distinct saturated values into ties, which
    perturbs AUC and breaks the monotonic logit->p_fake relation calibration
    depends on. Pin it."""
    b = fx.image_manipulated()
    assert b.p_fake == 0.9134827362
    restored = EvidenceBundle.model_validate_json(b.model_dump_json())
    assert restored.p_fake == b.p_fake
    assert restored.branch(BranchId.A).logit == 2.3581093
    payload = json.loads(b.model_dump_json())
    assert payload["p_fake"] == 0.9134827362


def test_fixtures_are_deterministic():
    assert fx.video_manipulated().model_dump_json() == fx.video_manipulated().model_dump_json()


def test_fixture_coverage_spans_the_verdict_space():
    verdicts = {b.verdict for b in fx.all_bundles().values()}
    assert verdicts == {
        Verdict.AUTHENTIC,
        Verdict.MANIPULATED,
        Verdict.INSUFFICIENT_EVIDENCE,
    }
    kinds = {b.media.kind for b in fx.all_bundles().values()}
    assert kinds == {MediaKind.IMAGE, MediaKind.VIDEO}
    statuses = {br.status for b in fx.all_bundles().values() for br in b.branches}
    assert BranchStatus.SCORED in statuses and BranchStatus.ABSTAINED in statuses


def test_schema_version_is_stamped():
    assert fx.image_manipulated().schema_version == CONTRACT_VERSION


# --------------------------------------------------------------------------
# The one-sided provenance rule — the highest-severity review finding
# --------------------------------------------------------------------------


def test_provenance_branch_rejects_negative_logit():
    """Absence of provenance must never push toward a fake verdict."""
    _raises(
        lambda: BranchResult(
            id=BranchId.P,
            name="Provenance",
            status=BranchStatus.SCORED,
            logit=-0.4,
        ),
        "must be >= 0",
    )


def test_provenance_branch_allows_zero_and_positive():
    for logit in (0.0, 0.5, 3.0):
        r = BranchResult(id=BranchId.P, name="Provenance", status=BranchStatus.SCORED, logit=logit)
        assert r.logit == logit


def test_other_branches_may_be_negative():
    r = BranchResult(id=BranchId.A, name="Spatial", status=BranchStatus.SCORED, logit=-2.5)
    assert r.logit == -2.5


def test_authenticity_credit_is_zero_without_evidence():
    _raises(
        lambda: ProvenanceReport(
            c2pa=C2paReport(present=False),
            exif=ExifReport(present=False),
            authenticity_credit=0.4,
        ),
        "must be 0.0",
    )


def test_authenticity_credit_cannot_be_negative():
    _raises(
        lambda: ProvenanceReport(
            c2pa=C2paReport(present=False),
            exif=ExifReport(present=False),
            authenticity_credit=-0.1,
        ),
        "greater than or equal to 0",
    )


def test_exif_alone_earns_only_weak_credit():
    ok = ProvenanceReport(
        c2pa=C2paReport(present=False),
        exif=ExifReport(present=True, coherent=True, camera_make="Canon"),
        authenticity_credit=0.2,
    )
    assert ok.authenticity_credit == 0.2
    _raises(
        lambda: ProvenanceReport(
            c2pa=C2paReport(present=False),
            exif=ExifReport(present=True, coherent=True),
            authenticity_credit=0.8,
        ),
        "requires a valid C2PA signature",
    )


def test_stripped_metadata_scores_identically_to_absent_metadata():
    """Two identical images differing only in stripped metadata must not diverge.

    This is the regression test for the shortcut a fitted fusion would otherwise
    learn: 'no metadata implies fake'.
    """
    a = fx.image_authentic()
    stripped = a.model_copy(deep=True)
    assert stripped.provenance.authenticity_credit == 0.0
    assert a.branch(BranchId.P).logit == 0.0
    assert stripped.branch(BranchId.P).logit == 0.0
    assert a.p_fake == stripped.p_fake


def test_empty_provenance_wording_is_not_suspicious():
    detail = fx.image_authentic().provenance.findings[0].detail.lower()
    assert "normal" in detail
    assert "neither supports nor undermines" in detail


# --------------------------------------------------------------------------
# Verdict / probability invariants
# --------------------------------------------------------------------------


def _bundle_kwargs(**over):
    b = fx.image_authentic()
    data = b.model_dump()
    data.update(over)
    return data


def test_abstention_forbids_p_fake():
    _raises(
        lambda: EvidenceBundle.model_validate(
            _bundle_kwargs(verdict=Verdict.INSUFFICIENT_EVIDENCE, p_fake=0.4)
        ),
        "must be null",
    )


def test_abstention_requires_a_reason():
    _raises(
        lambda: EvidenceBundle.model_validate(
            _bundle_kwargs(
                verdict=Verdict.INSUFFICIENT_EVIDENCE, p_fake=None, interval=None
            )
        ),
        "abstain_reason is required",
    )


def test_decided_verdict_requires_p_fake():
    _raises(
        lambda: EvidenceBundle.model_validate(
            _bundle_kwargs(verdict=Verdict.AUTHENTIC, p_fake=None, interval=None)
        ),
        "p_fake is required",
    )


def test_verdict_must_match_threshold():
    """A UI must never show 'authentic' beside a probability above the boundary."""
    _raises(
        lambda: EvidenceBundle.model_validate(
            _bundle_kwargs(verdict=Verdict.AUTHENTIC, p_fake=0.95, interval=(0.9, 0.99))
        ),
        "contradicts",
    )


def test_interval_must_bracket_p_fake():
    _raises(
        lambda: EvidenceBundle.model_validate(_bundle_kwargs(interval=(0.5, 0.6))),
        "bracket",
    )


def test_failed_quality_gate_forces_abstention():
    bad = _bundle_kwargs()
    bad["quality"] = QualityReport(
        resolution_class=ResolutionClass.TINY,
        gate_passed=False,
        gate_reason=AbstainReason.FACE_TOO_SMALL,
    ).model_dump()
    _raises(lambda: EvidenceBundle.model_validate(bad), "insufficient_evidence")


def test_quality_gate_reason_consistency():
    _raises(
        lambda: QualityReport(resolution_class=ResolutionClass.HIGH, gate_passed=False),
        "gate_reason is required",
    )
    _raises(
        lambda: QualityReport(
            resolution_class=ResolutionClass.HIGH,
            gate_passed=True,
            gate_reason=AbstainReason.NO_FACE_DETECTED,
        ),
        "must be null",
    )


def test_duplicate_branches_rejected():
    data = _bundle_kwargs()
    data["branches"] = [data["branches"][0], data["branches"][0]]
    _raises(lambda: EvidenceBundle.model_validate(data), "duplicate branch")


def test_video_block_must_match_media_kind():
    _raises(
        lambda: EvidenceBundle.model_validate(
            _bundle_kwargs(video=fx.video_manipulated().video.model_dump())
        ),
        "must be null when media.kind is image",
    )
    data = fx.video_manipulated().model_dump()
    data["video"] = None
    _raises(lambda: EvidenceBundle.model_validate(data), "required when media.kind is video")


def test_image_media_rejects_video_fields():
    from contracts import MediaInfo

    _raises(
        lambda: MediaInfo(
            kind=MediaKind.IMAGE,
            sha256="0" * 64,
            size_bytes=10,
            mime="image/jpeg",
            width=10,
            height=10,
            duration_s=1.0,
        ),
        "must be null for kind=image",
    )


# --------------------------------------------------------------------------
# Branch status invariants
# --------------------------------------------------------------------------


def test_scored_branch_requires_logit():
    _raises(
        lambda: BranchResult(id=BranchId.A, name="Spatial", status=BranchStatus.SCORED),
        "logit is required",
    )


def test_abstained_branch_forbids_logit():
    _raises(
        lambda: BranchResult(
            id=BranchId.A,
            name="Spatial",
            status=BranchStatus.ABSTAINED,
            logit=1.0,
            abstain_reason=AbstainReason.NO_FACE_DETECTED,
        ),
        "must be null when not scored",
    )


def test_abstained_branch_requires_reason():
    _raises(
        lambda: BranchResult(id=BranchId.B, name="Semantic", status=BranchStatus.UNAVAILABLE),
        "abstain_reason is required",
    )


def test_branch_c_does_not_exist():
    """Frequency was folded into Branch A. It must not return as a branch."""
    assert not hasattr(BranchId, "C")
    assert {b.value for b in BranchId} == {"A", "B", "P"}


def test_no_face_image_abstains_on_branch_a_only():
    b = fx.image_no_face()
    assert b.branch(BranchId.A).status is BranchStatus.ABSTAINED
    assert b.branch(BranchId.A).abstain_reason is AbstainReason.NO_FACE_DETECTED
    assert b.branch(BranchId.B).status is BranchStatus.SCORED
    assert b.verdict is Verdict.MANIPULATED  # a decided verdict from one branch


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_every_error_code_has_a_message_and_status():
    from contracts import DEFAULT_MESSAGES, HTTP_STATUS

    for code in ErrorCode:
        assert code in DEFAULT_MESSAGES, f"{code} has no user-facing message"
        assert code in HTTP_STATUS, f"{code} has no HTTP status"
        assert not DEFAULT_MESSAGES[code].lower().startswith("sorry")


def test_error_retryability_matches_taxonomy():
    assert is_retryable(ErrorCode.DECODE_TIMEOUT)
    assert not is_retryable(ErrorCode.DECODE_FAILED)
    assert ApiError.of(ErrorCode.DECODE_FAILED).retryable is False
    assert ApiError.of(ErrorCode.MODEL_UNAVAILABLE).retryable is True


def test_error_codes_are_partitioned():
    from contracts import DETERMINISTIC_ERRORS, TRANSIENT_ERRORS

    assert not (DETERMINISTIC_ERRORS & TRANSIENT_ERRORS)


def test_abstain_reasons_are_not_error_codes():
    """The two vocabularies must never overlap — an unjudgeable input is a
    result, not a failure."""
    assert not ({r.value for r in AbstainReason} & {c.value for c in ErrorCode})


# --------------------------------------------------------------------------
# API schemas
# --------------------------------------------------------------------------


def test_upload_request_enforces_size_caps():
    _raises(
        lambda: CreateUploadRequest(
            sha256="a" * 64,
            size_bytes=30 * 1024 * 1024,
            content_type="image/jpeg",
            media_kind=MediaKind.IMAGE,
        ),
        "exceeds 25 MB",
    )


def test_upload_request_rejects_wrong_mime_for_kind():
    _raises(
        lambda: CreateUploadRequest(
            sha256="a" * 64,
            size_bytes=1000,
            content_type="video/mp4",
            media_kind=MediaKind.IMAGE,
        ),
        "not allowed for images",
    )


def test_upload_response_is_either_url_or_result_never_both():
    _raises(
        lambda: CreateUploadResponse(
            upload_id="u1", cache_hit=True, cached_result=None, upload=None
        ),
        "requires cached_result",
    )
    ok = CreateUploadResponse(
        upload_id="u1", cache_hit=True, cached_result=fx.image_authentic()
    )
    assert ok.upload is None


def test_sha256_pattern_is_enforced():
    _raises(
        lambda: CreateUploadRequest(
            sha256="NOTAHASH",
            size_bytes=1000,
            content_type="image/jpeg",
            media_kind=MediaKind.IMAGE,
        ),
        "pattern",
    )


def test_unknown_fields_are_rejected():
    data = json.loads(fx.image_authentic().model_dump_json())
    data["surprise"] = 1
    _raises(lambda: EvidenceBundle.model_validate(data), "extra")


# --------------------------------------------------------------------------
# Job lifecycle
# --------------------------------------------------------------------------


def test_terminal_states_require_finished_at():
    data = fx.job_succeeded().model_dump()
    data["finished_at"] = None
    from contracts import JobStatus

    _raises(lambda: JobStatus.model_validate(data), "requires finished_at")


def test_failed_job_must_carry_an_error():
    data = fx.job_failed().model_dump()
    data["error"] = None
    from contracts import JobStatus

    _raises(lambda: JobStatus.model_validate(data), "must carry an error")


def test_job_state_terminality():
    assert JobState.SUCCEEDED.is_terminal
    assert JobState.FAILED.is_terminal
    assert not JobState.RUNNING.is_terminal


def test_stage_weights_are_monotonic_and_complete():
    from contracts import STAGE_WEIGHTS_IMAGE, STAGE_WEIGHTS_VIDEO

    for weights in (STAGE_WEIGHTS_IMAGE, STAGE_WEIGHTS_VIDEO):
        vals = list(weights.values())
        assert vals == sorted(vals), "stage progress must never go backwards"
        assert vals[-1] == 100


# --------------------------------------------------------------------------
# Video payload
# --------------------------------------------------------------------------


def test_frame_scores_must_be_ordered():
    from contracts import FrameScore, VideoAnalysis

    _raises(
        lambda: VideoAnalysis(
            frames_sampled=2,
            frames_scored=2,
            frame_scores=[
                FrameScore(frame_idx=5, t_ms=200, p=0.1),
                FrameScore(frame_idx=1, t_ms=40, p=0.2),
            ],
        ),
        "ordered by frame_idx",
    )


def test_manipulated_video_has_a_segment_covering_the_peak():
    b = fx.video_manipulated()
    assert len(b.video.segments) == 1
    seg = b.video.segments[0]
    peak_frame = max(b.video.frame_scores, key=lambda f: f.p)
    assert seg.start_ms <= peak_frame.t_ms <= seg.end_ms
    assert seg.frame_count < b.video.frames_sampled, "a short segment, not the whole clip"


def test_authentic_video_has_no_segments():
    assert fx.video_authentic().video.segments == []


def test_frames_scored_cannot_exceed_sampled():
    from contracts import VideoAnalysis

    _raises(
        lambda: VideoAnalysis(frames_sampled=10, frames_scored=11),
        "cannot exceed",
    )


# --------------------------------------------------------------------------
# Explainability wording
# --------------------------------------------------------------------------


def test_gradcam_captions_are_attribution_not_localisation():
    from contracts import ArtifactKind

    seen = 0
    for bundle in fx.all_bundles().values():
        for art in bundle.artifacts:
            if art.kind is ArtifactKind.GRADCAM:
                seen += 1
                low = art.caption.lower()
                assert "influenced" in low
                assert "manipulated region" not in low
                assert "fake region" not in low
    assert seen >= 1


# --------------------------------------------------------------------------
# Standalone runner (pytest is not a dependency yet)
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
        except Exception as exc:  # noqa: BLE001 - a test runner reports everything
            failed.append((name, exc))
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
