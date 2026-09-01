"""Feature capability and dependency constraints for flow attacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    """Constraint metadata for one feature column."""

    name: str
    kind: str = "mutable"  # immutable, mutable, integer, categorical, derived, uncertain
    lower: Optional[float] = None
    upper: Optional[float] = None
    integer: bool = False
    rationale: str = ""

    @property
    def preserved(self) -> bool:
        """Return whether attacks must preserve this aggregate feature."""
        return self.kind in {
            "immutable",
            "categorical",
            "derived",
            "uncertain",
        }


@dataclass(frozen=True)
class DependencySpec:
    """Named row-level consistency check for aggregate flow features."""

    name: str
    check: Callable[[np.ndarray], np.ndarray]


@dataclass
class ConstraintSet:
    """Dataset-specific constraints applied in original feature space."""

    features: list[FeatureSpec]
    dependency_checks: list[DependencySpec | Callable[[np.ndarray], np.ndarray]] = field(
        default_factory=list
    )
    recompute_derived: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @property
    def mutable_indices(self) -> np.ndarray:
        return np.asarray(
            [i for i, spec in enumerate(self.features) if spec.kind in {"mutable", "integer"}],
            dtype=np.int64,
        )

    def validate(self, X: np.ndarray, reference: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        valid = np.isfinite(X).all(axis=1)
        bounds_valid = valid.copy()
        integer_valid = valid.copy()
        immutable_valid = valid.copy()
        for idx, spec in enumerate(self.features):
            if spec.lower is not None:
                bounds_valid &= X[:, idx] >= spec.lower
            if spec.upper is not None:
                bounds_valid &= X[:, idx] <= spec.upper
            if spec.integer or spec.kind == "integer":
                # Inverse transforming float32 MinMax-scaled values can incur
                # up to half-unit rounding error for large CICFlowMeter counts.
                integer_valid &= np.isclose(
                    X[:, idx], np.round(X[:, idx]), atol=0.51
                )
            if reference is not None and spec.preserved:
                immutable_valid &= np.isclose(
                    X[:, idx], reference[:, idx], atol=1e-4, rtol=1e-6
                )
        dependency_valid = valid.copy()
        dependency_results: dict[str, np.ndarray] = {}
        for index, dependency in enumerate(self.dependency_checks):
            if isinstance(dependency, DependencySpec):
                name, check = dependency.name, dependency.check
            else:
                name, check = f"dependency_{index}", dependency
            result = np.asarray(check(X), dtype=bool)
            dependency_results[name] = result
            dependency_valid &= result
        return {
            "valid": valid & bounds_valid & integer_valid & immutable_valid & dependency_valid,
            "finite_valid": valid,
            "bounds_valid": bounds_valid,
            "integer_valid": integer_valid,
            "immutable_valid": immutable_valid,
            "preserved_valid": immutable_valid,
            "dependency_valid": dependency_valid,
            "dependency_results": dependency_results,
        }


def nonnegative_dependency(columns: Mapping[str, int], names: list[str]):
    """Build a dependency check requiring selected columns to be non-negative."""
    indices = [columns[name] for name in names]
    return lambda X: np.all(X[:, indices] >= 0, axis=1)


_DIRECT_MUTABLE = {
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
}

_INTEGER_FEATURES = {
    "Dst Port",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Fwd IAT Tot",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Len",
    "Bwd Header Len",
    "Pkt Len Min",
    "Pkt Len Max",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Max",
    "Active Min",
    "Idle Max",
    "Idle Min",
}

_BINARY_FLAGS = {
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "ECE Flag Cnt",
}


def cicflowmeter_constraints(feature_names: list[str]) -> ConstraintSet:
    """Build the conservative CIC-IDS2017/2018 aggregate-flow policy.

    The saved datasets contain flow aggregates, not packet traces. Only five
    direct primitives are considered mutable. All summary statistics are
    preserved because changing them independently without regenerating the
    underlying packet sequence is not physically justified.
    """
    features: list[FeatureSpec] = []
    for name in feature_names:
        if name == "Dst Port":
            kind = "immutable"
            rationale = "Existing-flow destination/service identifier."
        elif name in _DIRECT_MUTABLE:
            kind = "mutable"
            rationale = "Direct packet-count, byte-count, or duration primitive."
        elif name == "Init Fwd Win Byts":
            kind = "uncertain"
            rationale = "Transport-negotiated field; preserved without packet traces."
        else:
            kind = "derived"
            rationale = "CICFlowMeter aggregate requiring packet-level recomputation."
        lower = 0.0
        upper = 65535.0 if name == "Dst Port" else None
        if name in _BINARY_FLAGS:
            upper = 1.0
        features.append(
            FeatureSpec(
                name=name,
                kind=kind,
                lower=lower,
                upper=upper,
                integer=name in _INTEGER_FEATURES,
                rationale=rationale,
            )
        )

    columns = {name: index for index, name in enumerate(feature_names)}
    dependencies: list[DependencySpec] = []

    def ordering(prefix: str, minimum: str, mean: str, maximum: str) -> None:
        if not all(name in columns for name in (minimum, mean, maximum)):
            return
        lo, avg, hi = columns[minimum], columns[mean], columns[maximum]
        dependencies.append(
            DependencySpec(
                f"{prefix}_min_le_mean_le_max",
                lambda X, lo=lo, avg=avg, hi=hi: (
                    (X[:, lo] <= X[:, avg] + 1e-3)
                    & (X[:, avg] <= X[:, hi] + 1e-3)
                ),
            )
        )

    ordering(
        "fwd_packet_length",
        "Fwd Pkt Len Min",
        "Fwd Pkt Len Mean",
        "Fwd Pkt Len Max",
    )
    ordering(
        "bwd_packet_length",
        "Bwd Pkt Len Min",
        "Bwd Pkt Len Mean",
        "Bwd Pkt Len Max",
    )
    ordering("packet_length", "Pkt Len Min", "Pkt Len Mean", "Pkt Len Max")
    ordering("flow_iat", "Flow IAT Min", "Flow IAT Mean", "Flow IAT Max")
    ordering("fwd_iat", "Fwd IAT Min", "Fwd IAT Mean", "Fwd IAT Max")
    ordering("bwd_iat", "Bwd IAT Min", "Bwd IAT Mean", "Bwd IAT Max")
    ordering("active", "Active Min", "Active Mean", "Active Max")
    ordering("idle", "Idle Min", "Idle Mean", "Idle Max")

    def less_equal(name: str, left: str, right: str) -> None:
        if left not in columns or right not in columns:
            return
        lhs, rhs = columns[left], columns[right]
        dependencies.append(
            DependencySpec(
                name,
                lambda X, lhs=lhs, rhs=rhs: X[:, lhs] <= X[:, rhs] + 0.51,
            )
        )

    less_equal("active_data_packets_le_fwd_packets", "Fwd Act Data Pkts", "Tot Fwd Pkts")

    def approximately_equal(name: str, left: str, right: str) -> None:
        if left not in columns or right not in columns:
            return
        lhs, rhs = columns[left], columns[right]
        dependencies.append(
            DependencySpec(
                name,
                lambda X, lhs=lhs, rhs=rhs: np.isclose(
                    X[:, lhs], X[:, rhs], atol=0.51, rtol=1e-6
                ),
            )
        )

    approximately_equal(
        "subflow_fwd_packets_eq_total", "Subflow Fwd Pkts", "Tot Fwd Pkts"
    )
    approximately_equal(
        "subflow_bwd_packets_eq_total", "Subflow Bwd Pkts", "Tot Bwd Pkts"
    )
    approximately_equal(
        "subflow_fwd_bytes_eq_total", "Subflow Fwd Byts", "TotLen Fwd Pkts"
    )
    approximately_equal(
        "subflow_bwd_bytes_eq_total", "Subflow Bwd Byts", "TotLen Bwd Pkts"
    )

    def relative_consistency(
        name: str,
        actual: str,
        expected: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        if actual not in columns:
            return
        actual_index = columns[actual]

        def check(X: np.ndarray) -> np.ndarray:
            observed = X[:, actual_index]
            target = expected(X)
            tolerance = np.maximum(
                1e-2, 0.01 * np.maximum(np.abs(observed), np.abs(target))
            )
            return np.abs(observed - target) <= tolerance

        dependencies.append(DependencySpec(name, check))

    fwd_packets = columns["Tot Fwd Pkts"]
    bwd_packets = columns["Tot Bwd Pkts"]
    fwd_bytes = columns["TotLen Fwd Pkts"]
    bwd_bytes = columns["TotLen Bwd Pkts"]
    duration = columns["Flow Duration"]

    def safe_ratio(
        numerator: Callable[[np.ndarray], np.ndarray],
        denominator: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        def ratio(X: np.ndarray) -> np.ndarray:
            top = numerator(X)
            bottom = denominator(X)
            return np.divide(
                top,
                bottom,
                out=np.zeros_like(top, dtype=np.float64),
                where=bottom > 0,
            )

        return ratio

    relative_consistency(
        "fwd_packet_mean_eq_bytes_per_packet",
        "Fwd Pkt Len Mean",
        safe_ratio(lambda X: X[:, fwd_bytes], lambda X: X[:, fwd_packets]),
    )
    relative_consistency(
        "bwd_packet_mean_eq_bytes_per_packet",
        "Bwd Pkt Len Mean",
        safe_ratio(lambda X: X[:, bwd_bytes], lambda X: X[:, bwd_packets]),
    )
    def duration_seconds(X: np.ndarray) -> np.ndarray:
        return X[:, duration] / 1_000_000.0

    relative_consistency(
        "flow_bytes_rate_consistent",
        "Flow Byts/s",
        safe_ratio(
            lambda X: X[:, fwd_bytes] + X[:, bwd_bytes], duration_seconds
        ),
    )
    relative_consistency(
        "flow_packets_rate_consistent",
        "Flow Pkts/s",
        safe_ratio(
            lambda X: X[:, fwd_packets] + X[:, bwd_packets], duration_seconds
        ),
    )
    relative_consistency(
        "fwd_packets_rate_consistent",
        "Fwd Pkts/s",
        safe_ratio(lambda X: X[:, fwd_packets], duration_seconds),
    )
    relative_consistency(
        "bwd_packets_rate_consistent",
        "Bwd Pkts/s",
        safe_ratio(lambda X: X[:, bwd_packets], duration_seconds),
    )

    return ConstraintSet(features=features, dependency_checks=dependencies)
