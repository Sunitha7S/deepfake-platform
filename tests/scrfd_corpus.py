"""SCRFD Phase 4 prerequisite P3 — build the R13 evaluation corpus and its
reproducibility manifest. Standalone tooling beside ``scrfd_detector_study.py``;
not a ``run_all.py`` suite (``tests/test_scrfd_corpus.py`` covers it).

    python tests/scrfd_corpus.py --images <WIDER_val/images> \\
        --annotations <wider_face_val_bbx_gt.txt> --out <corpus_dir>

Reads WIDER FACE's published ground-truth format, selects a deterministic,
stratified subset, copies it flat into ``--out`` (the layout
``scrfd_detector_study._load_photos`` reads) and writes ``MANIFEST.json``: every
file's SHA-256 plus one digest over the sorted list, so "the corpus R13 was
measured on" is a checkable identity rather than "a" 300-image folder
(``docs/SCRFD_PHASE4_PREREQUISITES_PLAN.md`` §B.5).

**Selection, tied to R13's mechanism** (``docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md``
§3.2 — amplification is a box-*size* effect near a detection boundary). An
image qualifies when its largest valid face has a short edge of at least
``min_face`` px — the measurement tracks the top-ranked face only, and WIDER's
crowd scenes are otherwise dominated by faces far below any detector's floor.
Qualifying images are stratified on (largest-face size bucket) x (one face vs
several) and drawn round-robin across strata, so small faces near the floor
are represented as deliberately as large ones. Illumination, pose and blur
coverage is reported, not forced — forcing four attributes at once would
leave some strata empty.

**Licence.** WIDER FACE is CC BY-NC-ND (``docs/SCRFD_PHASE4_PREREQUISITES_PLAN.md``
§B.3). Local, non-commercial measurement is within it; publishing the copied
images, noised variants or crops is not (NoDerivatives). Keep ``--out``
outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "MANIFEST.json"
SIZE_BUCKETS = ((32, 64), (64, 128), (128, 256), (256, 10**9))
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    w: int
    h: int
    blur: int
    expression: int
    illumination: int
    invalid: int
    occlusion: int
    pose: int

    @property
    def short_edge(self) -> int:
        return min(self.w, self.h)


def parse_wider_annotations(path: Path) -> dict[str, list[Face]]:
    """``wider_face_*_bbx_gt.txt``: a relative path, a face count, then one
    ``x y w h blur expression illumination invalid occlusion pose`` line per
    face. A zero-face entry still carries one all-zero line, which is skipped."""
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    out: dict[str, list[Face]] = {}
    i = 0
    while i < len(lines):
        name, count = lines[i], int(lines[i + 1])
        i += 2
        faces = []
        for row in lines[i : i + max(count, 1)]:
            values = [int(v) for v in row.split()[:10]]
            if count and values[2] > 0 and values[3] > 0:
                faces.append(Face(*values))
        i += max(count, 1)
        out[name] = faces
    return out


def _bucket(edge: int) -> int | None:
    for idx, (lo, hi) in enumerate(SIZE_BUCKETS):
        if lo <= edge < hi:
            return idx
    return None


def _valid(faces: list[Face]) -> list[Face]:
    return [f for f in faces if not f.invalid]


def select_corpus(
    annotations: dict[str, list[Face]], *, n: int = 300, seed: int = 0, min_face: int = 32
) -> list[str]:
    """Deterministic for a given ``(annotations, n, seed, min_face)``."""
    strata: dict[tuple[int, bool], list[str]] = {}
    for name in sorted(annotations):
        faces = _valid(annotations[name])
        if not faces:
            continue
        largest = max(f.short_edge for f in faces)
        if largest < min_face:
            continue
        bucket = _bucket(largest)
        if bucket is None:
            continue
        multi = sum(1 for f in faces if f.short_edge >= min_face) > 1
        strata.setdefault((bucket, multi), []).append(name)

    rng = random.Random(seed)
    queues = []
    for key in sorted(strata):
        names = list(strata[key])
        rng.shuffle(names)
        queues.append(names)
    chosen: list[str] = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop())
    return sorted(chosen)


def coverage(annotations: dict[str, list[Face]], names: list[str], *, min_face: int = 32) -> dict:
    """Counts per diversity dimension, for the evidence record."""
    report = {
        "images": len(names),
        "size_bucket": {f"{lo}-{hi if hi < 10**9 else 'inf'}": 0 for lo, hi in SIZE_BUCKETS},
        "multi_face": 0,
        "single_face": 0,
        "extreme_illumination": 0,
        "atypical_pose": 0,
        "heavy_blur": 0,
    }
    labels = list(report["size_bucket"])
    for name in names:
        faces = [f for f in _valid(annotations[name]) if f.short_edge >= min_face]
        top = max(faces, key=lambda f: f.short_edge)
        report["size_bucket"][labels[_bucket(top.short_edge)]] += 1
        report["multi_face" if len(faces) > 1 else "single_face"] += 1
        report["extreme_illumination"] += int(any(f.illumination == 1 for f in faces))
        report["atypical_pose"] += int(any(f.pose == 1 for f in faces))
        report["heavy_blur"] += int(any(f.blur == 2 for f in faces))
    return report


def flat_name(relative: str) -> str:
    """``0--Parade/0_Parade_x.jpg`` -> ``0--Parade__0_Parade_x.jpg`` (unique, flat)."""
    return relative.replace("\\", "/").replace("/", "__")


def materialize(names: list[str], images_root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        src = Path(images_root) / name
        dst = out_dir / flat_name(name)
        shutil.copyfile(src, dst)
        written.append(dst)
    return written


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_files(corpus_dir: Path) -> list[Path]:
    """Exactly the files ``scrfd_detector_study._load_photos`` would read."""
    return sorted(
        p for p in Path(corpus_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def manifest_entries(corpus_dir: Path) -> list[tuple[str, str]]:
    return [(p.name, _sha256(p)) for p in corpus_files(corpus_dir)]


def manifest_digest(corpus_dir: Path) -> str:
    """One SHA-256 over the sorted ``name<TAB>sha256`` lines."""
    body = "".join(f"{name}\t{sha}\n" for name, sha in manifest_entries(corpus_dir))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_manifest(corpus_dir: Path, *, provenance: dict) -> dict:
    entries = manifest_entries(corpus_dir)
    manifest = {
        "corpus_digest": manifest_digest(corpus_dir),
        "file_count": len(entries),
        "files": [{"name": n, "sha256": s} for n, s in entries],
        "provenance": provenance,
    }
    (Path(corpus_dir) / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", required=True, type=Path, help="WIDER_val/images directory")
    parser.add_argument("--annotations", required=True, type=Path, help="wider_face_val_bbx_gt.txt")
    parser.add_argument("--out", required=True, type=Path, help="corpus directory to create")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-face", type=int, default=32)
    parser.add_argument("--source", default="WIDER FACE validation split (CC BY-NC-ND 4.0)")
    args = parser.parse_args(argv)

    annotations = parse_wider_annotations(args.annotations)
    names = select_corpus(annotations, n=args.n, seed=args.seed, min_face=args.min_face)
    if len(names) < 200:
        print(f"only {len(names)} qualifying images; R13 requires >= 200", file=sys.stderr)
        return 1
    materialize(names, args.images, args.out)
    manifest = write_manifest(
        args.out,
        provenance={
            "source": args.source,
            "annotations_sha256": _sha256(args.annotations),
            "selection": {"n": args.n, "seed": args.seed, "min_face": args.min_face},
            "coverage": coverage(annotations, names, min_face=args.min_face),
        },
    )
    print(json.dumps({k: manifest[k] for k in ("corpus_digest", "file_count")}, indent=2))
    print(json.dumps(manifest["provenance"]["coverage"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
