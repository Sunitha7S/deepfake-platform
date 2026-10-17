"""Run every contract test suite without pytest.

pytest is not yet a dependency, and S0 must be verifiable on a bare checkout
with nothing but pydantic installed. This runner is what CI calls until the
dev extra lands.

    python tests/run_all.py

Exits non-zero if any suite fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_backend  # noqa: E402
import test_band_derivation  # noqa: E402
import test_benchmark_reproducibility  # noqa: E402
import test_calibration  # noqa: E402
import test_calibration_fingerprint  # noqa: E402
import test_calibration_witness  # noqa: E402
import test_evaluation_benchmark  # noqa: E402
import test_evaluation_dataset  # noqa: E402
import test_evaluation_identity  # noqa: E402
import test_evaluation_metrics  # noqa: E402
import test_evaluation_operating  # noqa: E402
import test_evaluation_protocol  # noqa: E402
import test_evaluation_svg  # noqa: E402
import test_face_selection_identity  # noqa: E402
import test_video_face_semantics  # noqa: E402
import test_face_size_policy  # noqa: E402
import test_crop_geometry  # noqa: E402
import test_crop_identity  # noqa: E402
import test_video_selection_identity  # noqa: E402
import test_startup_alignment  # noqa: E402
import test_contracts  # noqa: E402
import test_faces_consistency  # noqa: E402
import test_face_geometry_publication  # noqa: E402
import test_branch_b_semantics  # noqa: E402
import test_headline_and_face_semantics  # noqa: E402
import test_frame_count_reproducibility  # noqa: E402
import test_decode_identity_completeness  # noqa: E402
import test_image_decode_identity  # noqa: E402
import test_decode_determinism  # noqa: E402
import test_dependencies  # noqa: E402
import test_run_identity  # noqa: E402
import test_frame_extraction  # noqa: E402
import test_threshold_band  # noqa: E402
import test_timing_contract  # noqa: E402
import test_fusion_semantics  # noqa: E402
import test_inference  # noqa: E402
import test_inference_determinism  # noqa: E402
import test_lease_safety  # noqa: E402
import test_lock_sealing  # noqa: E402
import test_protocol_enforcement  # noqa: E402
import test_metrics  # noqa: E402
import test_model_version  # noqa: E402
import test_openapi  # noqa: E402
import test_postprocess  # noqa: E402
import test_preprocess  # noqa: E402
import test_queue  # noqa: E402
import test_result_sink_race  # noqa: E402
import test_schema  # noqa: E402
import test_split_lock  # noqa: E402
import test_storage  # noqa: E402
import test_video_branches  # noqa: E402
import test_worker_inference_integration  # noqa: E402
import test_worker_main  # noqa: E402
import test_workers  # noqa: E402

SUITES = [
    ("S0 contracts", test_contracts),
    ("S0 schema + examples", test_schema),
    ("S1 backend routes", test_backend),
    ("S1 openapi + drift", test_openapi),
    ("S2 storage", test_storage),
    ("S3 job queue", test_queue),
    ("S4 workers", test_workers),
    ("S6 calibration", test_calibration),
    ("S6 metrics", test_metrics),
    ("S6 preprocess", test_preprocess),
    ("S6 postprocess", test_postprocess),
    ("S6 inference", test_inference),
    ("S8-B0 fusion semantics", test_fusion_semantics),
    ("S8-B0 model version chain", test_model_version),
    ("S8-B0 video branch provenance", test_video_branches),
    ("S8-B0 lease safety", test_lease_safety),
    ("S8-B1 threshold band", test_threshold_band),
    ("S8-B1 timing contract", test_timing_contract),
    ("S8-B1 frame extraction", test_frame_extraction),
    ("S7 worker<->inference", test_worker_inference_integration),
    ("D-3 result sink race", test_result_sink_race),
    ("S8-B2 dataset identity + leakage", test_evaluation_dataset),
    ("S8-B2 metrics + calibration", test_evaluation_metrics),
    ("S8-B2 thresholds + risk-coverage", test_evaluation_operating),
    ("S8-B2 benchmark + versioning", test_evaluation_benchmark),
    ("S8-B3 split lock", test_split_lock),
    ("S8-B3 protocol + slice metadata", test_evaluation_protocol),
    ("S8-B3 svg rendering", test_evaluation_svg),
    ("S8-B4 calibration fingerprint", test_calibration_fingerprint),
    ("S8-B4 benchmark reproducibility", test_benchmark_reproducibility),
    ("S8-B5 dependency declaration", test_dependencies),
    ("S8-B5 lock identity sealing", test_lock_sealing),
    ("S8-B5 protocol enforcement", test_protocol_enforcement),
    ("S8-B6 calibration witness", test_calibration_witness),
    ("S8-B7 band derivation", test_band_derivation),
    ("S8-B8 decode determinism", test_decode_determinism),
    ("S8-B9 run identity", test_run_identity),
    ("S8-B10 inference determinism", test_inference_determinism),
    ("S8-B11 evaluation identity", test_evaluation_identity),
    ("S8-B12 face selection identity", test_face_selection_identity),
    ("S8-B13 video face semantics", test_video_face_semantics),
    ("S8-B14 face size policy", test_face_size_policy),
    ("S8-B15 crop geometry", test_crop_geometry),
    ("S8-B16 faces[] consistency", test_faces_consistency),
    ("S8-B17 branch B semantics", test_branch_b_semantics),
    ("S8-B18 headline + face semantics", test_headline_and_face_semantics),
    ("S8-B19 frame_count reproducibility", test_frame_count_reproducibility),
    ("S8-B20 decode identity completeness", test_decode_identity_completeness),
    ("S8-B22 crop identity", test_crop_identity),
    ("S8-B23 video selection identity (R29 + N4)", test_video_selection_identity),
    ("S8-B23 startup alignment (OR1)", test_startup_alignment),
    ("S8-B25 worker composition root (OR1-b)", test_worker_main),
    ("S8-B26 image decode identity (R30 + D8/D9)", test_image_decode_identity),
    ("S8-B27 face geometry publication (R16 + R26)", test_face_geometry_publication),
]


def main() -> int:
    total = failed = 0
    for name, module in SUITES:
        print(f"\n=== {name} " + "=" * (56 - len(name)))
        rc = module._run()
        count = sum(1 for n in dir(module) if n.startswith("test_"))
        total += count
        if rc:
            failed += 1

    print("\n" + "=" * 62)
    if failed:
        print(f"FAILED - {failed} of {len(SUITES)} suites had failures ({total} tests total)")
        return 1
    print(f"OK - {len(SUITES)} suites, {total} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
