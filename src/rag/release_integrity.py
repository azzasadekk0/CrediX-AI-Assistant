"""Lightweight integrity checks for the runtime RAG release bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ReleaseValidationError(ValueError):
    """Raised when a RAG release bundle is not safe to use at runtime."""


def _load_release_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / "release_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseValidationError(f"RAG release manifest not found: {manifest_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"RAG release manifest cannot be read: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ReleaseValidationError("RAG release manifest must contain a JSON object")
    return manifest


def validate_runtime_bundle(index_dir: Path) -> dict[str, Any]:
    """Fail closed unless ``index_dir`` is an intact production release.

    Runtime retrieval must never silently use a candidate, ineligible, or
    modified bundle. This validates the release identity and declared artifact
    bytes; the more expensive corpus/smoke checks remain in the release-time
    validator.
    """
    root = Path(index_dir).resolve()
    if not root.is_dir():
        raise ReleaseValidationError(f"RAG release directory not found: {root}")
    manifest = _load_release_manifest(root)

    status = manifest.get("release_status")
    legacy_status = manifest.get("status")
    if status is None:
        status = legacy_status
    if legacy_status is not None and status != legacy_status:
        raise ReleaseValidationError("RAG release manifest has conflicting status fields")
    if status != "production_released":
        raise ReleaseValidationError(f"RAG runtime requires production_released status, found {status!r}")
    if manifest.get("production_release_eligible") is not True:
        raise ReleaseValidationError("RAG release is not marked production eligible")
    if manifest.get("pending_verification_count") != 0:
        raise ReleaseValidationError("RAG release has pending verification work")
    if manifest.get("release_blockers") not in ([], None):
        raise ReleaseValidationError("RAG release manifest contains release blockers")

    artifacts = manifest.get("artifacts_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ReleaseValidationError("RAG release manifest has no artifact hashes")

    hash_results: dict[str, dict[str, Any]] = {}
    for filename, expected in artifacts.items():
        if not isinstance(filename, str) or not filename:
            raise ReleaseValidationError("RAG release manifest contains an invalid artifact name")
        artifact = (root / filename).resolve()
        if root not in artifact.parents:
            raise ReleaseValidationError(f"RAG artifact escapes release directory: {filename}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ReleaseValidationError(f"RAG artifact has an invalid SHA-256 hash: {filename}")
        try:
            int(expected, 16)
        except ValueError as exc:
            raise ReleaseValidationError(f"RAG artifact has an invalid SHA-256 hash: {filename}") from exc
        try:
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except (FileNotFoundError, OSError) as exc:
            raise ReleaseValidationError(f"RAG release artifact not found: {filename}") from exc
        matches = actual == expected.lower()
        hash_results[filename] = {"expected": expected, "actual": actual, "matches": matches}
        if not matches:
            raise ReleaseValidationError(f"RAG release artifact hash mismatch: {filename}")

    return {
        "index_dir": str(root),
        "release_status": status,
        "production_release_eligible": True,
        "artifact_hashes": hash_results,
    }
