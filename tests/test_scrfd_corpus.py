"""SCRFD Phase 4 prerequisite P3 — the evaluation-corpus builder and manifest
(``tests/scrfd_corpus.py``), tested on synthetic annotations and generated
images. No WIDER FACE data is needed or used."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tests import scrfd_corpus as corpus  # noqa: E402

# name, then faces as (x, y, w, h, blur, expression, illumination, invalid, occlusion, pose)
_IMAGES: dict[str, list[tuple[int, ...]]] = {
    "0--Parade/a.jpg": [(10, 10, 40, 48, 0, 0, 0, 0, 0, 0)],
    "0--Parade/b.jpg": [(10, 10, 90, 100, 0, 0, 1, 0, 0, 0), (200, 10, 70, 80, 0, 0, 0, 0, 0, 1)],
    "1--Handshaking/c.jpg": [(5, 5, 150, 160, 2, 0, 0, 0, 0, 0)],
    "1--Handshaking/d.jpg": [(5, 5, 300, 320, 0, 0, 0, 0, 0, 0)],
    "2--Demonstration/tiny.jpg": [(5, 5, 8, 9, 0, 0, 0, 0, 0, 0), (40, 5, 12, 12, 0, 0, 0, 0, 0, 0)],
    "2--Demonstration/invalid.jpg": [(5, 5, 120, 120, 0, 0, 0, 1, 0, 0)],
    "3--Riot/empty.jpg": [],
    "3--Riot/e.jpg": [(1, 1, 60, 70, 0, 0, 0, 0, 0, 0), (100, 1, 35, 40, 0, 0, 0, 0, 0, 0)],
}


def _annotation_text() -> str:
    out = []
    for name, faces in _IMAGES.items():
        out.append(name)
        out.append(str(len(faces)))
        if faces:
            out.extend(" ".join(str(v) for v in face) + " " for face in faces)
        else:
            out.append("0 0 0 0 0 0 0 0 0 0 ")
    return "\n".join(out) + "\n"


def _write_dataset() -> tuple[Path, Path]:
    root = Path(tempfile.mkdtemp(prefix="dfp-corpus-"))
    images = root / "images"
    for i, name in enumerate(_IMAGES):
        path = images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.random.default_rng(i).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), img)
    annotations = root / "gt.txt"
    annotations.write_text(_annotation_text(), encoding="utf-8")
    return images, annotations


def test_the_parser_reads_every_entry_including_zero_face_ones():
    _, gt = _write_dataset()
    parsed = corpus.parse_wider_annotations(gt)
    assert set(parsed) == set(_IMAGES)
    assert parsed["3--Riot/empty.jpg"] == []
    assert len(parsed["0--Parade/b.jpg"]) == 2
    b0 = parsed["0--Parade/b.jpg"][0]
    assert (b0.w, b0.h, b0.illumination) == (90, 100, 1)


def test_selection_respects_the_face_size_floor_and_validity():
    _, gt = _write_dataset()
    chosen = corpus.select_corpus(corpus.parse_wider_annotations(gt), n=100, min_face=32)
    assert "2--Demonstration/tiny.jpg" not in chosen, "every face is below the floor"
    assert "2--Demonstration/invalid.jpg" not in chosen, "its only face is marked invalid"
    assert "3--Riot/empty.jpg" not in chosen
    assert set(chosen) == {
        "0--Parade/a.jpg", "0--Parade/b.jpg", "1--Handshaking/c.jpg",
        "1--Handshaking/d.jpg", "3--Riot/e.jpg",
    }


def test_selection_is_deterministic_and_seeded():
    annotations = {
        f"x/{i}.jpg": [corpus.Face(0, 0, 40 + 3 * i, 50 + 3 * i, 0, 0, 0, 0, 0, 0)]
        for i in range(120)
    }
    a = corpus.select_corpus(annotations, n=30, seed=0)
    b = corpus.select_corpus(annotations, n=30, seed=0)
    c = corpus.select_corpus(annotations, n=30, seed=1)
    assert a == b and len(a) == 30
    assert a != c


def test_selection_draws_across_size_strata_rather_than_the_largest_pool():
    """Small faces near the floor must not be crowded out by an abundance of
    large ones — that is the regime R13's mechanism lives in."""
    annotations = {f"big/{i}.jpg": [corpus.Face(0, 0, 300, 300, 0, 0, 0, 0, 0, 0)] for i in range(200)}
    annotations.update(
        {f"small/{i}.jpg": [corpus.Face(0, 0, 40, 44, 0, 0, 0, 0, 0, 0)] for i in range(10)}
    )
    chosen = corpus.select_corpus(annotations, n=20, seed=0)
    assert sum(name.startswith("small/") for name in chosen) == 10


def test_coverage_counts_each_dimension():
    _, gt = _write_dataset()
    parsed = corpus.parse_wider_annotations(gt)
    chosen = corpus.select_corpus(parsed, n=100)
    report = corpus.coverage(parsed, chosen)
    assert report["images"] == 5
    assert report["multi_face"] == 2 and report["single_face"] == 3
    assert report["extreme_illumination"] == 1
    assert report["atypical_pose"] == 1
    assert report["heavy_blur"] == 1
    assert sum(report["size_bucket"].values()) == 5


def test_materialized_corpus_is_flat_and_is_what_the_study_loads():
    images, gt = _write_dataset()
    chosen = corpus.select_corpus(corpus.parse_wider_annotations(gt), n=100)
    out = Path(tempfile.mkdtemp(prefix="dfp-corpus-out-"))
    corpus.materialize(chosen, images, out)
    corpus.write_manifest(out, provenance={"source": "synthetic"})
    assert sorted(p.name for p in corpus.corpus_files(out)) == sorted(
        corpus.flat_name(n) for n in chosen
    )

    import scrfd_detector_study as study  # noqa: PLC0415 - sibling study module

    assert len(study._load_photos(str(out))) == len(chosen), "MANIFEST.json must not be loaded"


def test_the_manifest_digest_is_stable_and_moves_on_any_byte():
    images, gt = _write_dataset()
    chosen = corpus.select_corpus(corpus.parse_wider_annotations(gt), n=100)
    out = Path(tempfile.mkdtemp(prefix="dfp-corpus-out-"))
    corpus.materialize(chosen, images, out)
    first = corpus.write_manifest(out, provenance={"source": "synthetic"})
    assert corpus.manifest_digest(out) == first["corpus_digest"]
    assert first["file_count"] == len(chosen)

    victim = corpus.corpus_files(out)[0]
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0x01
    victim.write_bytes(bytes(data))
    assert corpus.manifest_digest(out) != first["corpus_digest"]


def test_the_manifest_digest_ignores_non_image_files():
    images, gt = _write_dataset()
    chosen = corpus.select_corpus(corpus.parse_wider_annotations(gt), n=100)
    out = Path(tempfile.mkdtemp(prefix="dfp-corpus-out-"))
    corpus.materialize(chosen, images, out)
    before = corpus.manifest_digest(out)
    (out / "notes.txt").write_text("not part of the corpus", encoding="utf-8")
    corpus.write_manifest(out, provenance={})
    assert corpus.manifest_digest(out) == before


def test_the_cli_refuses_a_corpus_below_the_criterion_floor():
    images, gt = _write_dataset()
    out = Path(tempfile.mkdtemp(prefix="dfp-corpus-out-")) / "c"
    rc = corpus.main(["--images", str(images), "--annotations", str(gt), "--out", str(out)])
    assert rc == 1, "five qualifying images must not become an R13 corpus"
    assert not out.exists()


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:500]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
