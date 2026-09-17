"""JSON Schema generation.

Two consumers, one source:

* **FastAPI / OpenAPI** — the backend derives its OpenAPI document from these
  same pydantic models, so the published API docs and this export agree by
  construction.
* **The frontend** — ``schema_out/`` feeds a TypeScript generator, which is why
  the frontend never hand-writes a type and therefore cannot drift from the
  contract.

Schemas are emitted with ``examples`` injected from ``contracts.examples`` rather
than pinned into each model's ``model_config``. Keeping them out of the model
modules avoids a circular import with fixtures, and means a documented payload is
always a payload that actually validates.

Usage::

    python -m contracts.schema                 # write to ./schema_out
    python -m contracts.schema --out build/    # choose a directory
    python -m contracts.schema --check         # verify examples validate, write nothing
    python -m contracts.schema --stdout EvidenceBundle
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from contracts.api import (
    AnalyzeAcceptedResponse,
    AnalyzeRequest,
    CreateUploadRequest,
    CreateUploadResponse,
    HealthResponse,
)
from contracts.enums import CONTRACT_VERSION
from contracts.errors import ErrorResponse
from contracts.evidence import EvidenceBundle
from contracts.examples import EXAMPLES, examples_for
from contracts.job import JobSpec, JobStatus, ProgressEvent

ROOT_MODELS: tuple[type[BaseModel], ...] = (
    EvidenceBundle,
    CreateUploadRequest,
    CreateUploadResponse,
    AnalyzeRequest,
    AnalyzeAcceptedResponse,
    JobStatus,
    JobSpec,
    ProgressEvent,
    HealthResponse,
    ErrorResponse,
)
"""Models that cross the wire as a whole payload. Nested models appear in
``$defs`` and do not need their own file."""


def schema_for(model: type[BaseModel], *, with_examples: bool = True) -> dict[str, Any]:
    """JSON Schema for one model, with registered examples attached."""
    schema = model.model_json_schema(mode="serialization")
    schema["title"] = model.__name__
    schema["$id"] = f"https://deepfake-platform/schema/{CONTRACT_VERSION}/{model.__name__}.json"
    if with_examples:
        found = examples_for(model)
        if found:
            schema["examples"] = found
    return schema


def all_schemas(*, with_examples: bool = True) -> dict[str, dict[str, Any]]:
    return {m.__name__: schema_for(m, with_examples=with_examples) for m in ROOT_MODELS}


def bundle() -> dict[str, Any]:
    """One combined document. This is the file a TypeScript generator reads."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "contract_version": CONTRACT_VERSION,
        "models": all_schemas(),
    }


def verify_examples() -> list[str]:
    """Re-validate every registered example against its own model.

    Guards the failure mode that makes API documentation actively harmful: a
    published example payload the server would reject. Runs in CI.
    """
    problems: list[str] = []
    for model, entries in EXAMPLES.items():
        for label, build in entries:
            try:
                payload = build().model_dump(mode="json")
            except (ValidationError, ValueError) as exc:
                problems.append(f"{model.__name__}[{label}]: failed to build — {exc}")
                continue
            try:
                model.model_validate(payload)
            except ValidationError as exc:
                problems.append(
                    f"{model.__name__}[{label}]: example does not validate — "
                    f"{str(exc).splitlines()[0]}"
                )
    return problems


def write_all(out_dir: Path) -> list[Path]:
    """One file per root model, plus the combined bundle and an index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in all_schemas().items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        written.append(path)

    combined = out_dir / "contracts.bundle.json"
    combined.write_text(json.dumps(bundle(), indent=2) + "\n", encoding="utf-8")
    written.append(combined)

    index = out_dir / "index.json"
    index.write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "models": sorted(m.__name__ for m in ROOT_MODELS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(index)
    return written


def main(argv: list[str]) -> int:
    problems = verify_examples()
    if problems:
        print("example validation FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if "--check" in argv:
        print(f"all examples validate ({sum(len(v) for v in EXAMPLES.values())} across "
              f"{len(EXAMPLES)} models)")
        return 0

    if "--stdout" in argv:
        rest = [a for a in argv[1:] if not a.startswith("-")]
        name = rest[0] if rest else "EvidenceBundle"
        match = next((m for m in ROOT_MODELS if m.__name__ == name), None)
        if match is None:
            print(
                f"unknown model {name!r}; choose from "
                f"{sorted(m.__name__ for m in ROOT_MODELS)}",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(schema_for(match), indent=2))
        return 0

    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("schema_out")
    written = write_all(out)
    print(f"wrote {len(written)} schema files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
