"""R7: what the code imports and what the project declares must agree.

THE DEFECT, AND WHY IT SURVIVED EIGHT STAGES
--------------------------------------------
``pyproject.toml`` declared ``pydantic`` and nothing else. ``numpy`` was imported
at module scope in twelve files across ``inference`` and ``evaluation``;
``fastapi``, ``opencv-python``, ``starlette`` and ``pydantic-settings`` were
likewise undeclared. Every test passed throughout, because the machine they ran
on had all of them installed for unrelated reasons. The declaration only matters
on a machine that does not exist yet, which is precisely why nothing caught it.

``LEGACY_PYPROJECT`` below is the real pre-fix ``[project]`` table.
``test_the_legacy_declaration_fails_this_audit`` runs the current audit against
it and asserts it fails, naming all five. Without that, this suite would be a
set of assertions that today's file matches today's imports — true by
construction on the day it was written and untested thereafter.

THE THREE CLASSES ARE NOT BOOKKEEPING
-------------------------------------
Runtime-that-should-be-optional puts a multi-gigabyte CUDA wheel in every
deployment. Optional-that-should-be-runtime is an ImportError on somebody else's
machine. Both are asserted here, in both directions.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_audit import (  # noqa: E402
    ROOT,
    RUNTIME_PACKAGES,
    build_matrix,
    declared_dependencies,
    distribution_for,
    scan,
)

MATRIX = build_matrix()
DECLARED = declared_dependencies()


LEGACY_PYPROJECT = """
[project]
name = "deepfake-platform"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.9,<3",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
"""
"""The declaration as it stood from S0 through S8 Batch 4, verbatim."""


def write_temp(text: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix="dfp-pyproject-")) / "pyproject.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The legacy declaration fails the audit
# --------------------------------------------------------------------------


def test_the_legacy_declaration_fails_this_audit():
    """The premise: the audit must reject the file it was written to catch."""
    legacy = declared_dependencies(write_temp(LEGACY_PYPROJECT))
    needed = {distribution_for(m) for m in MATRIX.runtime}
    missing = needed - legacy.runtime
    assert missing == {
        "numpy",
        "opencv-python",
        "fastapi",
        "starlette",
        "pydantic-settings",
    }, f"the legacy declaration was missing {sorted(missing)}"


def test_the_legacy_declaration_also_misclassified_the_test_only_dependency():
    """psutil was in no extra at all, so a dev install could not run the suite."""
    legacy = declared_dependencies(write_temp(LEGACY_PYPROJECT))
    assert "psutil" in MATRIX.test_only
    assert "psutil" not in legacy.optional_all
    assert "psutil" in DECLARED.extras["dev"], "psutil is still undeclared"


# --------------------------------------------------------------------------
# Runtime imports are declared
# --------------------------------------------------------------------------


def test_every_module_scope_import_is_a_declared_runtime_dependency():
    """The headline check. Fails if an import lands without a declaration."""
    undeclared = sorted(
        {distribution_for(m) for m in MATRIX.runtime} - DECLARED.runtime
    )
    assert not undeclared, (
        f"imported at module scope by a shipping package but not declared in "
        f"[project].dependencies: {undeclared}. Run `python "
        f"tests/import_audit.py` for the matrix."
    )


def test_every_declared_runtime_dependency_is_actually_imported():
    """The other direction. A dependency nobody imports is one nobody can drop.

    Orphan declarations are how a lock file accumulates a transitive tree that
    outlives the code that needed it, and the cost is paid at install time by
    everyone, forever.
    """
    imported = {distribution_for(m) for m in MATRIX.runtime}
    orphans = sorted(DECLARED.runtime - imported)
    assert not orphans, (
        f"declared in [project].dependencies but imported nowhere: {orphans}"
    )


def test_numpy_is_declared():
    """Named explicitly because it is the one the audit found.

    `np.quantile` sits directly under `threshold_at_fpr`, so an install without
    numpy does not degrade — it cannot choose a threshold at all.
    """
    assert "numpy" in DECLARED.runtime
    assert "numpy" in MATRIX.runtime
    packages = {s.package for s in MATRIX.runtime["numpy"]}
    assert {"inference", "evaluation"} <= packages


def test_starlette_is_declared_even_though_fastapi_provides_it():
    """A direct import must be a direct declaration."""
    assert "starlette" in DECLARED.runtime
    sites = MATRIX.runtime["starlette"]
    assert any("backend/errors.py" in s.path for s in sites), (
        "starlette is no longer imported directly; the explicit declaration can "
        "be reconsidered"
    )


# --------------------------------------------------------------------------
# Optional stays optional
# --------------------------------------------------------------------------


def test_the_model_backends_are_optional_not_runtime():
    """Promoting either to `dependencies` would undo a design decision."""
    for module in ("torch", "onnxruntime"):
        assert module in MATRIX.optional, (
            f"{module} is now imported at module scope; a CPU deployment would "
            "be made to install it"
        )
        assert distribution_for(module) not in DECLARED.runtime
        assert distribution_for(module) in DECLARED.optional_all, (
            f"{module} is imported but declared in no extra"
        )


def test_the_optional_imports_are_genuinely_lazy():
    """Read from the AST, not from a comment claiming they are guarded."""
    for module in ("torch", "onnxruntime"):
        sites = MATRIX.optional[module]
        assert sites and all(s.lazy for s in sites), (
            f"{module} has a module-scope import site: "
            f"{[f'{s.path}:{s.line}' for s in sites if not s.lazy]}"
        )


def test_the_api_process_does_not_import_torch_at_module_scope():
    """Batch 0 invariant #3, restated as a dependency fact.

    `backend` importing anything from `inference` runs `inference/__init__` and
    drags cv2 and numpy into the API process. The dependency audit sees the same
    boundary from the other side.
    """
    backend_sites = [s for s in scan(("backend",)) if not s.lazy]
    forbidden = {"torch", "onnxruntime", "cv2"}
    offenders = [f"{s.path}:{s.line} ({s.module})" for s in backend_sites if s.module in forbidden]
    assert not offenders, f"the API process imports {offenders}"


# --------------------------------------------------------------------------
# Test-only stays out of runtime
# --------------------------------------------------------------------------


def test_test_only_dependencies_are_not_runtime_dependencies():
    for module in MATRIX.test_only:
        assert distribution_for(module) not in DECLARED.runtime, (
            f"{module} is imported only by tests but declared as a runtime "
            "dependency; it would ship in a production image"
        )


def test_test_only_dependencies_are_declared_in_the_dev_extra():
    dev = DECLARED.extras.get("dev", set())
    missing = sorted({distribution_for(m) for m in MATRIX.test_only} - dev)
    assert not missing, f"imported by tests but in no dev extra: {missing}"


def test_pytest_is_declared_but_deliberately_not_imported():
    """The one legitimate orphan, and the reason the orphan check is scoped.

    Every suite is written as plain-assert functions with a `__main__` runner
    precisely so the tests run without pytest installed — it is not installed on
    this machine. So pytest is declared in `dev` and imported nowhere, which
    would be an orphan if the orphan check covered extras. It covers
    `[project].dependencies` only, and this test pins why.
    """
    assert "pytest" in DECLARED.extras["dev"]
    assert "pytest" not in MATRIX.all_modules


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------


def test_every_shipping_package_is_listed_for_setuptools():
    """This said `["contracts"]` from S0 to Batch 4 — an install shipped no code.

    Tests run from the source tree, so nothing exercised it and nothing failed.
    """
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = set(data["tool"]["setuptools"]["packages"])
    assert set(RUNTIME_PACKAGES) == listed, (
        f"packages listed for setuptools {sorted(listed)} do not match the "
        f"shipping packages {sorted(RUNTIME_PACKAGES)}"
    )


def test_no_top_level_datasets_package_exists():
    """The shadowing rule, asserted where the packaging list is asserted.

    `queue/` shadowed the stdlib module and cost a day; `datasets/` would shadow
    HuggingFace the same way. Dataset code is `evaluation.dataset`.
    """
    assert not (ROOT / "datasets").exists(), (
        "a top-level datasets/ package would shadow the HuggingFace library"
    )
    assert not (ROOT / "queue").exists(), "queue/ shadows the stdlib module"


# --------------------------------------------------------------------------
# The matrix itself
# --------------------------------------------------------------------------


def test_the_matrix_classifies_every_import_exactly_once():
    classes = (set(MATRIX.runtime), set(MATRIX.optional), set(MATRIX.test_only))
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            assert not (a & b), f"module classified twice: {sorted(a & b)}"


def test_the_matrix_renders_for_a_human():
    text = MATRIX.render()
    assert "numpy" in text and "RUNTIME" in text
    assert "torch" in text and "lazy at" in text


def test_a_module_scope_import_anywhere_makes_a_dependency_runtime():
    """One unguarded site is enough, whatever the other sites do."""
    sites = MATRIX.runtime["numpy"]
    assert any(not s.lazy for s in sites)


# --------------------------------------------------------------------------
# S8-B24 (N5): dependency lockfile
# --------------------------------------------------------------------------
#
# Batch 24 (N5) added `[tool.uv]` to `pyproject.toml` and a new `uv.lock`,
# pinning `opencv-python`/`numpy` to the exact versions the shipped `.d.`/`.f.`
# identities were computed against (`docs/B24_N5_DESIGN.md` §4.1). None of
# this is read by `import_audit.py` (it parses `[project]` only) or by any
# `fingerprint()` call — the tests below make that claim self-verifying
# rather than asserted once and left to rot.


def test_a_lockfile_exists_and_matches_pyproject():
    """PA1: the lock is present and has not drifted from pyproject.toml."""
    assert (ROOT / "uv.lock").exists(), "no uv.lock — run `uv lock`"
    result = subprocess.run(
        ["uv", "lock", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"uv.lock is out of date with pyproject.toml:\n{result.stdout}\n{result.stderr}"
    )


def test_the_lockfile_pins_the_versions_the_shipped_identity_was_computed_against():
    """PA2: opencv-python/numpy resolve to exactly what .d.=c9d4025c assumes
    (R30, Batch 26 — was a97ccb68 through Batch 25; the pinned *versions* are
    unchanged, only the decode-identity value they feed is different now that
    it also covers the image codec libraries)."""
    import tomllib

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p["version"] for p in lock["package"]}
    assert packages.get("opencv-python") == "4.8.1.78", (
        f"opencv-python locked to {packages.get('opencv-python')!r}, not the "
        "version the shipped .d.=c9d4025c was computed against"
    )
    assert packages.get("numpy") == "1.26.4", (
        f"numpy locked to {packages.get('numpy')!r}, not the version the "
        "shipped identity fingerprints assume"
    )


def test_the_installed_environment_matches_the_lock():
    """PA3: whatever environment is running this suite matches the lock's pins."""
    import importlib.metadata as metadata

    assert metadata.version("opencv-python") == "4.8.1.78"
    assert metadata.version("numpy") == "1.26.4"


def test_cv2_version_matches_what_the_shipped_decode_identity_assumes():
    """PA4: ties the lock to the live decode_identity() input, not just to metadata."""
    import cv2

    assert cv2.__version__ == "4.8.1"


def test_the_override_pins_are_documented():
    """PA5: a future reader of [tool.uv] finds the reason, not a bare pin."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.uv]" in text
    section = text[text.index("[tool.uv]") :]
    assert "override-dependencies" in section
    assert "B24_N5_DESIGN.md" in section or "B24_R2_DESIGN.md" in section, (
        "the override-dependencies pin has no pointer to the design doc that "
        "justifies it"
    )


def test_the_ci_workflow_installs_from_the_lock_with_locked_flag():
    """PA6: CI must fail on drift, not silently rewrite the lock."""
    workflow = ROOT / ".github" / "workflows" / "tests.yml"
    assert workflow.exists(), "no CI workflow — see docs/B24_N5_DESIGN.md §3.2"
    text = workflow.read_text(encoding="utf-8")
    assert "uv sync" in text
    assert "--locked" in text or "--frozen" in text, (
        "the CI install step must use --locked/--frozen so drift fails the "
        "job instead of silently updating uv.lock"
    )


def test_the_python_version_pin_matches_the_shipped_identity_box():
    """Not in the original design's PA list. G1's first execution picked
    CPython 3.14 by default (no .python-version existed) and failed to build
    numpy==1.26.4 — no wheel exists for it past cp312. `.python-version` is
    what makes the pin above actually installable; this pins that fact.
    """
    pin = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert pin == "3.12", (
        f".python-version is {pin!r}; numpy==1.26.4 has no wheel past cp312 — "
        "changing this without re-verifying G1 will silently break `uv sync`"
    )


def test_decode_identity_is_unchanged():
    """NM1. Pinned to N5's shipped value through Batch 25. R30 (Batch 26)
    deliberately moves this — image codec libraries joined the identity — so
    the pin moves with it: a97ccb68 -> c9d4025c. See
    docs/R30_BUNDLE_DESIGN.md. N5 itself (the lockfile) still contributes
    nothing to this value changing; the *inputs* it pins (opencv-python
    4.8.1.78, numpy 1.26.4) are exactly what c9d4025c was computed against,
    same as a97ccb68 was."""
    from inference.decode_env import decode_identity

    assert decode_identity() == "c9d4025c"


def test_stub_engine_version_is_unchanged():
    """NM2."""
    from backend.config import STUB_ENGINE_VERSION

    assert STUB_ENGINE_VERSION == (
        "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.c9d4025c"
    )


def test_model_identity_is_unchanged():
    """NM3."""
    from backend.config import STUB_ENGINE_VERSION
    from contracts.version import model_identity_of

    assert model_identity_of(STUB_ENGINE_VERSION) == (
        "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c"
    )


def test_no_contract_schema_witness_or_split_lock_version_bump():
    """NM4. Named explicitly to defeat the LOCK_VERSION/dependency-lock
    terminology collision (docs/B24_N5_DESIGN.md §1.7) — LOCK_VERSION here is
    the corpus split-lock schema, unrelated to uv.lock.
    """
    from contracts.enums import CONTRACT_VERSION
    from evaluation.benchmark import RUN_SCHEMA_VERSION
    from evaluation.calibration_protocol import CALIBRATION_WITNESS_VERSION
    from evaluation.split_lock import LOCK_VERSION

    assert CONTRACT_VERSION == "1.0.0"
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert CALIBRATION_WITNESS_VERSION == "1.3.0"
    assert LOCK_VERSION == "1.1.0"


def test_the_tool_uv_section_is_invisible_to_the_dependency_audit():
    """NM5: import_audit.py parses [project] with tomllib; [tool.uv] is a
    sibling table it never reads. `DECLARED.runtime` is checked against the
    exact pre-N5 baseline captured before [tool.uv]/uv.lock existed, not
    merely asserted post hoc. `dev` gained `httpx2` mid-batch (the
    clean-environment gap this batch also fixed — see
    REPRODUCIBILITY_AUDIT.md); that addition is a real, intended dependency
    declaration, not an artifact of [tool.uv], so it is asserted here too
    rather than making this test coincidentally pass either way.
    """
    assert DECLARED.runtime == {
        "pydantic",
        "numpy",
        "opencv-python",
        "fastapi",
        "starlette",
        "pydantic-settings",
    }
    assert DECLARED.extras == {
        "torch": {"torch"},
        "onnx": {"onnxruntime"},
        "dev": {"psutil", "pytest", "pillow", "ruff", "httpx2"},
    }


def test_schema_out_regen_is_lock_independent():
    """NM6 (schema_out half — the OpenAPI half is already covered by
    test_openapi.py::test_no_contract_drift, which reads no lockfile either).
    contracts.schema.write_all() is generated from Pydantic model
    definitions, never from an installed package version or uv.lock.
    """
    from contracts.schema import write_all

    tmp = Path(tempfile.mkdtemp(prefix="dfp-schema-out-"))
    written = write_all(tmp)
    assert written, "write_all produced no files"
    for path in written:
        committed = ROOT / "schema_out" / path.name
        assert committed.exists(), f"schema_out/{path.name} is not committed"
        assert path.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8"), (
            f"schema_out/{path.name} has drifted from what contracts.schema generates"
        )


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
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
