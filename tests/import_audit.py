"""Derive the dependency matrix from the source, never from a list somebody kept.

WHY THIS IS COMPUTED RATHER THAN WRITTEN DOWN
---------------------------------------------
``pyproject.toml`` declared ``pydantic`` and nothing else while ``numpy`` was
imported at module scope in twelve files across ``inference`` and ``evaluation``.
Nobody noticed for eight stages, and nobody was careless: the box these were
written on had numpy installed for other reasons, so every test passed, every
import resolved, and the only machine on which the declaration mattered was one
that did not exist yet. A hand-maintained dependency list fails exactly this way
— silently, and only for the next person.

So this module parses the AST of every source file and reports what is actually
imported. It is a helper rather than a test so the same code answers the question
two ways: ``python tests/import_audit.py`` prints the matrix for a human, and
``test_dependencies.py`` asserts against it.

THE THREE CLASSES, AND WHY THE DISTINCTION IS LOAD-BEARING
----------------------------------------------------------
**Runtime** — imported at module scope by a package that ships. Importing the
package fails without it, so it belongs in ``[project].dependencies``.

**Optional** — imported inside a function or a ``try``. ``torch`` and
``onnxruntime`` are imported inside ``ModelRegistry._build_torch_backend`` and
``_build_onnx_backend``, deliberately: a CPU deployment serving ONNX must not be
made to install a 2 GB CUDA wheel, and the API process must not import torch at
all. Declaring these as runtime would undo that design decision in a config
file, which is the kind of regression no test would catch.

**Test-only** — imported only under ``tests/``. ``psutil`` measures peak RSS in
the frame-extraction suite and has no business in a production image.

A dependency in the wrong class is not a cosmetic error. Runtime-that-should-be-
optional bloats every deployment; optional-that-should-be-runtime is an
ImportError on a machine that is not this one.

IMPORT NAME IS NOT DISTRIBUTION NAME
------------------------------------
``import cv2`` installs ``opencv-python``; ``import pydantic_settings`` installs
``pydantic-settings``. The mapping cannot be derived from the import — it is
metadata the package author chose — so :data:`DISTRIBUTION_NAMES` records the
ones this repository uses, and anything absent from it falls back to the import
name with underscores normalised, which is right for the large majority.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNTIME_PACKAGES = ("backend", "contracts", "evaluation", "inference", "jobqueue", "storage", "workers")
"""Packages that ship. ``tests`` is scanned too, but classified separately."""

TEST_PACKAGES = ("tests",)

DISTRIBUTION_NAMES = {
    "cv2": "opencv-python",
    "pydantic_settings": "pydantic-settings",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
}
"""Import name -> distribution name, where the two differ.

Only entries this repository actually needs. Guessing at a package it does not
import would be the same species of error as the hand-maintained list this
module replaces.
"""


def distribution_for(import_name: str) -> str:
    return DISTRIBUTION_NAMES.get(import_name, import_name.replace("_", "-"))


@dataclass(frozen=True)
class ImportSite:
    """One import of one third-party module, and how it is guarded."""

    module: str
    package: str
    path: str
    line: int
    lazy: bool
    """True when the import is inside a function, a class body, or a ``try``.

    The distinction between "this package needs it to import at all" and "this
    code path needs it if you take that path", which is the difference between a
    runtime dependency and an optional one.
    """

    @property
    def distribution(self) -> str:
        return distribution_for(self.module)


def _local_module_names() -> set[str]:
    """Everything importable from within this repository.

    Both the top-level packages and the bare module names under ``tests/``,
    because ``run_all.py`` imports its suites as top-level names after putting
    the repo root on ``sys.path``. Without the second half, every ``test_*``
    module would be reported as an undeclared third-party dependency — which is
    what the first draft of this did.
    """
    names = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    names |= {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")}
    names |= {p.stem for p in (ROOT / "tests").glob("*.py")}
    return names


def _lazy_parents(tree: ast.AST) -> dict[int, bool]:
    """Map each node id to whether it sits inside a function or a ``try``."""
    lazy: dict[int, bool] = {}

    def walk(node: ast.AST, inside: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_inside = inside or isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Try, ast.ClassDef)
            )
            lazy[id(child)] = child_inside
            walk(child, child_inside)

    lazy[id(tree)] = False
    walk(tree, False)
    return lazy


def scan(packages: tuple[str, ...]) -> list[ImportSite]:
    """Every third-party import site in ``packages``, in source order."""
    local = _local_module_names()
    stdlib = set(sys.stdlib_module_names)
    sites: list[ImportSite] = []

    for package in packages:
        base = ROOT / package
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            lazy = _lazy_parents(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative import, necessarily local
                        continue
                    modules = [node.module] if node.module else []
                else:
                    continue
                for module in modules:
                    top = module.split(".")[0]
                    if top in stdlib or top in local:
                        continue
                    sites.append(
                        ImportSite(
                            module=top,
                            package=package,
                            path=str(path.relative_to(ROOT)).replace("\\", "/"),
                            line=node.lineno,
                            lazy=lazy.get(id(node), False),
                        )
                    )
    return sites


@dataclass(frozen=True)
class Matrix:
    """The dependency matrix: what is imported, from where, and how."""

    runtime: dict[str, list[ImportSite]]
    """Module-scope in a shipping package. Must be a declared runtime dep."""
    optional: dict[str, list[ImportSite]]
    """Only ever imported lazily in a shipping package."""
    test_only: dict[str, list[ImportSite]]
    """Imported only under ``tests/``."""

    @property
    def all_modules(self) -> set[str]:
        return set(self.runtime) | set(self.optional) | set(self.test_only)

    def render(self) -> str:
        lines = ["package -> imported by -> class", "=" * 72]
        for label, group in (
            ("RUNTIME", self.runtime),
            ("OPTIONAL", self.optional),
            ("TEST-ONLY", self.test_only),
        ):
            for module in sorted(group):
                sites = group[module]
                packages = sorted({s.package for s in sites})
                files = len({s.path for s in sites})
                lines.append(
                    f"{distribution_for(module):<24} {', '.join(packages):<34} "
                    f"{label:<10} {files} file(s)"
                )
                if label == "OPTIONAL":
                    for site in sites:
                        lines.append(f"{'':<24}   lazy at {site.path}:{site.line}")
        return "\n".join(lines)


def build_matrix() -> Matrix:
    """Classify every third-party import by where and how it is used."""
    runtime_sites = scan(RUNTIME_PACKAGES)
    test_sites = scan(TEST_PACKAGES)

    by_module: dict[str, list[ImportSite]] = {}
    for site in runtime_sites:
        by_module.setdefault(site.module, []).append(site)

    runtime: dict[str, list[ImportSite]] = {}
    optional: dict[str, list[ImportSite]] = {}
    for module, sites in by_module.items():
        # One module-scope import anywhere makes it a runtime dependency: that
        # package cannot be imported without it, whatever the other sites do.
        if any(not s.lazy for s in sites):
            runtime[module] = sites
        else:
            optional[module] = sites

    test_only: dict[str, list[ImportSite]] = {}
    for site in test_sites:
        if site.module in runtime or site.module in optional:
            continue
        test_only.setdefault(site.module, []).append(site)

    return Matrix(runtime=runtime, optional=optional, test_only=test_only)


# --------------------------------------------------------------------------
# What pyproject declares
# --------------------------------------------------------------------------


def _requirement_name(spec: str) -> str:
    """``"numpy>=1.26,<2"`` -> ``"numpy"``. Normalised for comparison."""
    name = spec.strip()
    for delimiter in ("[", ">", "<", "=", "!", "~", ";", " "):
        name = name.split(delimiter, 1)[0]
    return name.strip().lower().replace("_", "-")


@dataclass(frozen=True)
class Declared:
    runtime: set[str]
    extras: dict[str, set[str]]

    @property
    def optional_all(self) -> set[str]:
        return {name for names in self.extras.values() for name in names}


def declared_dependencies(pyproject: Path | None = None) -> Declared:
    data = tomllib.loads(
        (pyproject or ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = data.get("project", {})
    return Declared(
        runtime={_requirement_name(d) for d in project.get("dependencies", [])},
        extras={
            extra: {_requirement_name(d) for d in deps}
            for extra, deps in project.get("optional-dependencies", {}).items()
        },
    )


if __name__ == "__main__":
    matrix = build_matrix()
    declared = declared_dependencies()
    print(matrix.render())
    print("\ndeclared runtime :", sorted(declared.runtime))
    for extra, names in sorted(declared.extras.items()):
        print(f"declared [{extra}] :", sorted(names))
    missing = {distribution_for(m) for m in matrix.runtime} - declared.runtime
    print("\nUNDECLARED RUNTIME:", sorted(missing) or "none")
