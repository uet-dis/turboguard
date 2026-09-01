"""Constrained projection for feature-space adversarial examples."""

from __future__ import annotations

from typing import Optional

import numpy as np

from turboguard.attacks.constraints import ConstraintSet


def project_features(
    reference: np.ndarray,
    candidate: np.ndarray,
    constraints: ConstraintSet,
    eps: Optional[float] = None,
) -> np.ndarray:
    """Project candidate features while preserving immutable columns."""
    reference = np.asarray(reference, dtype=np.float32)
    projected = np.asarray(candidate, dtype=np.float32).copy()
    if projected.shape != reference.shape:
        raise ValueError("reference and candidate must have identical shapes")
    mutable = constraints.mutable_indices
    if eps is not None:
        projected[:, mutable] = np.clip(
            projected[:, mutable], reference[:, mutable] - eps, reference[:, mutable] + eps
        )
    for idx, spec in enumerate(constraints.features):
        if spec.preserved:
            projected[:, idx] = reference[:, idx]
        if spec.lower is not None:
            projected[:, idx] = np.maximum(projected[:, idx], spec.lower)
        if spec.upper is not None:
            projected[:, idx] = np.minimum(projected[:, idx], spec.upper)
        if spec.integer or spec.kind == "integer":
            projected[:, idx] = np.round(projected[:, idx])
    if constraints.recompute_derived is not None:
        projected = constraints.recompute_derived(projected)
    return projected.astype(np.float32)


def projection_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    constraints: ConstraintSet,
) -> dict[str, float | int]:
    """Summarize feasibility and perturbation distortion."""
    checks = constraints.validate(candidate, reference)
    delta = np.asarray(candidate) - np.asarray(reference)
    return {
        "n": int(len(candidate)),
        "valid_rate": float(checks["valid"].mean() * 100) if len(candidate) else 0.0,
        "bounds_valid_rate": float(checks["bounds_valid"].mean() * 100) if len(candidate) else 0.0,
        "integer_valid_rate": float(checks["integer_valid"].mean() * 100) if len(candidate) else 0.0,
        "immutable_valid_rate": float(checks["immutable_valid"].mean() * 100) if len(candidate) else 0.0,
        "preserved_valid_rate": float(checks["preserved_valid"].mean() * 100) if len(candidate) else 0.0,
        "dependency_valid_rate": float(checks["dependency_valid"].mean() * 100) if len(candidate) else 0.0,
        "l0_median": float(np.median(np.count_nonzero(delta, axis=1))) if len(candidate) else 0.0,
        "l1_median": float(np.median(np.abs(delta).sum(axis=1))) if len(candidate) else 0.0,
        "l2_median": float(np.median(np.linalg.norm(delta, axis=1))) if len(candidate) else 0.0,
        "linf_median": float(np.median(np.abs(delta).max(axis=1))) if len(candidate) else 0.0,
    }
