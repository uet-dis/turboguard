"""Closed Isolation Forest score contract used by production and evidence.

The finite runtime witnesses used elsewhere in Phase 1 are deliberately not
the proof of comparator semantics.  This module exposes one declarative
contract, four branch-free private kernels, and a fixed-path attestation that
binds their source AST to the live loaded code objects.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np


_RAW_LT_UFUNC: Final = np.less
_NEGATE_UFUNC: Final = np.negative
_NEXTAFTER_UFUNC: Final = np.nextafter
_CANONICAL_GE_UFUNC: Final = np.greater_equal
_ARRAY_FUNCTION: Final = np.array
_ASARRAY_FUNCTION: Final = np.asarray
_ASCONTIGUOUSARRAY_FUNCTION: Final = np.ascontiguousarray
_ARRAY_EQUAL_FUNCTION: Final = np.array_equal
_ISFINITE_UFUNC: Final = np.isfinite
_ISNAN_UFUNC: Final = np.isnan
_ISNEGINF_FUNCTION: Final = np.isneginf
_DTYPE_TYPE: Final = np.dtype
_FLOATING_TYPE: Final = np.floating
_NDARRAY_TYPE: Final = np.ndarray
_UFUNC_TYPE: Final = np.ufunc
_BOOL_TYPE: Final = np.bool_
_FLOAT32_TYPE: Final = np.float32
_FLOAT64_TYPE: Final = np.float64
_UINT32_TYPE: Final = np.uint32
_UINT64_TYPE: Final = np.uint64
_FLOAT32_DTYPE: Final = _DTYPE_TYPE(_FLOAT32_TYPE)
_FLOAT64_DTYPE: Final = _DTYPE_TYPE(_FLOAT64_TYPE)


CONTRACT_ID: Final = "RAW_LT_CANONICAL_NEGATE_V1"
RAW_COMPARATOR_ENUM: Final = "RAW_LT"
ORIENTATION_TRANSFORM_ENUM: Final = "NEGATE_SAME_DTYPE"
CANONICAL_COMPARATOR_ENUM: Final = "CANONICAL_GE_NEXTAFTER_POS_INF"
NONFINITE_POLICY: Final = (
    "REJECT_NAN_AND_POS_NEG_INFINITY_INPUTS;"
    "ALLOW_EXACT_DERIVED_POS_INFINITY_CANONICAL_THRESHOLD"
)
RAW_SCORE_SEMANTICS: Final = (
    "normality_score; lower_values_are_more_anomalous"
)
CANONICAL_SCORE_SEMANTICS: Final = (
    "anomaly_score; higher_values_are_more_anomalous"
)
ALLOWED_COMPARISON_DTYPES: Final = ("float32", "float64")

TRUSTED_MODULE: Final = "turboguard.core.if_score_contract"
TRUSTED_SOURCE_RELATIVE_PATH: Final = (
    "src/turboguard/core/if_score_contract.py"
)
TRUST_MANIFEST_RELATIVE_PATH: Final = (
    "src/turboguard/core/if_score_implementation_manifest.json"
)
TRUST_POLICY_RELATIVE_PATH: Final = "plan/phase1/artifact_policy.json"
TRUST_POLICY_ID: Final = "turboguard-phase1-artifact-policy-v2"
TRUST_POLICY_FIELD: Final = "implementation_trust_contract"
ATTESTATION_SCHEMA_VERSION: Final = "phase1-if-contract-attestation-v3"
CODE_FINGERPRINT_SCHEMA_VERSION: Final = (
    "phase1-python-code-fingerprint-v1"
)

_KERNEL_ARGUMENTS: Final = {
    "_raw_lt_kernel": ("raw_score", "raw_tau"),
    "_negate_kernel": ("raw_score",),
    "_canonical_tau_kernel": ("raw_tau",),
    "_canonical_ge_kernel": ("anomaly_score", "canonical_tau"),
}
_KERNEL_BODY_TEMPLATES: Final = {
    "_raw_lt_kernel": (
        "def template(raw_score, raw_tau):\n"
        "    return _RAW_LT_UFUNC(raw_score, raw_tau)\n"
    ),
    "_negate_kernel": (
        "def template(raw_score):\n"
        "    return _NEGATE_UFUNC(raw_score)\n"
    ),
    "_canonical_tau_kernel": (
        "def template(raw_tau):\n"
        "    return _NEXTAFTER_UFUNC(\n"
        "        _negate_kernel(raw_tau),\n"
        "        _ASARRAY_FUNCTION(float('inf'), dtype=raw_tau.dtype),\n"
        "        dtype=raw_tau.dtype,\n"
        "    )\n"
    ),
    "_canonical_ge_kernel": (
        "def template(anomaly_score, canonical_tau):\n"
        "    return _CANONICAL_GE_UFUNC(anomaly_score, canonical_tau)\n"
    ),
}
_ATTESTED_FUNCTION_QUALNAMES: Final = (
    "_raw_lt_kernel",
    "_negate_kernel",
    "_canonical_tau_kernel",
    "_canonical_ge_kernel",
    "_coerce_raw_scores",
    "_coerce_raw_threshold",
    "_immutable_array",
    "_bit_hex_array",
    "float_bit_pattern_hex",
    "apply_if_score_contract",
    "contract_spec",
    "_verify_live_numpy_ufuncs",
    "_verify_live_numpy_dependencies",
    "_require_deeply_frozen_json",
    "attest_if_score_contract",
)

_TRUSTED_MODULE_SPECS: Final = (
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE),
    ("src/turboguard/core/turboguard.py", "turboguard.core.turboguard"),
)
_TRUSTED_SYMBOL_SPECS: Final = (
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_raw_lt_kernel", "production_comparator_kernel"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_negate_kernel", "orientation_transform_kernel"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_canonical_tau_kernel", "strict_boundary_transform_kernel"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_canonical_ge_kernel", "canonical_comparator_kernel"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_coerce_raw_scores", "raw_score_validator"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_coerce_raw_threshold", "raw_threshold_validator"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_immutable_array", "immutable_output_helper"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_bit_hex_array", "bit_pattern_helper"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "float_bit_pattern_hex", "public_bit_pattern_encoder"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "apply_if_score_contract", "public_score_contract"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "contract_spec", "declarative_score_contract"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_verify_live_numpy_ufuncs", "numpy_ufunc_attester"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_verify_live_numpy_dependencies", "numpy_dependency_attester"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "_require_deeply_frozen_json", "runtime_immutability_attester"),
    (TRUSTED_SOURCE_RELATIVE_PATH, TRUSTED_MODULE, "attest_if_score_contract", "fixed_path_implementation_attester"),
    ("src/turboguard/core/turboguard.py", "turboguard.core.turboguard", "isolation_forest_reject", "production_raw_decision_adapter"),
    ("src/turboguard/core/turboguard.py", "turboguard.core.turboguard", "isolation_forest_anomaly_score", "production_anomaly_score_adapter"),
    ("src/turboguard/core/turboguard.py", "turboguard.core.turboguard", "TurboGuard.decision_details", "production_cascade_consumer"),
    ("src/turboguard/core/turboguard.py", "turboguard.core.turboguard", "TurboGuard.profile_decision", "production_profile_cascade_consumer"),
)
_PRODUCTION_COMPARATOR_SYMBOL_ID: Final = (
    "src/turboguard/core/if_score_contract.py::"
    "turboguard.core.if_score_contract::_raw_lt_kernel"
)


class IFScoreContractError(ValueError):
    """Raised when values cannot satisfy the closed score contract."""


class IFScoreAttestationError(RuntimeError):
    """Raised when source, AST, or live-code identity is not exact."""


@dataclass(frozen=True)
class IFScoreContractResult:
    """Exact typed result of ``RAW_LT_CANONICAL_NEGATE_V1``."""

    contract_id: str
    comparison_dtype: str
    if_score_raw: np.ndarray
    tau_if_raw: np.ndarray
    canonical_anomaly_score: np.ndarray
    tau_if_canonical: np.ndarray
    raw_if_reject: np.ndarray
    canonical_if_reject: np.ndarray
    if_score_raw_bits: np.ndarray
    tau_if_raw_bits: str
    canonical_anomaly_score_bits: np.ndarray
    tau_if_canonical_bits: str


def _raw_lt_kernel(raw_score: np.ndarray, raw_tau: np.ndarray) -> np.ndarray:
    return _RAW_LT_UFUNC(raw_score, raw_tau)


def _negate_kernel(raw_score: np.ndarray) -> np.ndarray:
    return _NEGATE_UFUNC(raw_score)


def _canonical_tau_kernel(raw_tau: np.ndarray) -> np.ndarray:
    return _NEXTAFTER_UFUNC(
        _negate_kernel(raw_tau),
        _ASARRAY_FUNCTION(float("inf"), dtype=raw_tau.dtype),
        dtype=raw_tau.dtype,
    )


def _canonical_ge_kernel(
    anomaly_score: np.ndarray, canonical_tau: np.ndarray
) -> np.ndarray:
    return _CANONICAL_GE_UFUNC(anomaly_score, canonical_tau)


def _coerce_raw_scores(if_score_raw: Any) -> np.ndarray:
    raw = _ARRAY_FUNCTION(if_score_raw, copy=True, order="C", subok=False)
    if raw.dtype not in (_FLOAT32_DTYPE, _FLOAT64_DTYPE):
        raise IFScoreContractError(
            "IF scores must use native float32 or float64 exactly; "
            f"observed {raw.dtype}"
        )
    if not _ISFINITE_UFUNC(raw).all():
        raise IFScoreContractError("IF scores contain NaN or infinity")
    return raw


def _coerce_raw_threshold(tau_if_raw: Any, dtype: np.dtype[Any]) -> np.ndarray:
    if type(tau_if_raw) is float:
        candidate = tau_if_raw
    elif isinstance(tau_if_raw, _FLOATING_TYPE):
        if tau_if_raw.dtype != dtype:
            raise IFScoreContractError(
                "NumPy raw IF threshold dtype must exactly match score dtype"
            )
        candidate = tau_if_raw
    elif type(tau_if_raw) is _NDARRAY_TYPE:
        if tau_if_raw.shape != ():
            raise IFScoreContractError("raw IF threshold must be exactly scalar")
        if tau_if_raw.dtype != dtype:
            raise IFScoreContractError(
                "NumPy raw IF threshold dtype must exactly match score dtype"
            )
        candidate = tau_if_raw
    else:
        raise IFScoreContractError(
            "raw IF threshold must be a Python float or exact-dtype NumPy scalar"
        )
    try:
        threshold = _ASARRAY_FUNCTION(candidate, dtype=dtype)
    except (FloatingPointError, TypeError, ValueError) as exc:
        raise IFScoreContractError(
            "raw IF threshold cannot be represented in the comparison dtype"
        ) from exc
    if threshold.dtype != dtype or threshold.shape != ():
        raise IFScoreContractError("raw IF threshold dtype or shape changed")
    if not bool(_ISFINITE_UFUNC(threshold)):
        raise IFScoreContractError("raw IF threshold is NaN or infinity")
    return _ARRAY_FUNCTION(threshold, dtype=dtype, copy=True)


def _immutable_array(values: np.ndarray) -> np.ndarray:
    result = _ARRAY_FUNCTION(values, copy=True, order="C", subok=False)
    result.setflags(write=False)
    return result


def _bit_hex_array(values: np.ndarray) -> np.ndarray:
    if values.dtype == _FLOAT32_DTYPE:
        unsigned_dtype = _DTYPE_TYPE(_UINT32_TYPE)
        width = 8
    elif values.dtype == _FLOAT64_DTYPE:
        unsigned_dtype = _DTYPE_TYPE(_UINT64_TYPE)
        width = 16
    else:
        raise IFScoreContractError(
            f"bit encoding does not support dtype {values.dtype}"
        )
    unsigned = _ASCONTIGUOUSARRAY_FUNCTION(values).view(unsigned_dtype)
    encoded = _ASARRAY_FUNCTION(
        [f"0x{int(value):0{width}x}" for value in unsigned.reshape(-1)],
        dtype=f"<U{width + 2}",
    ).reshape(values.shape)
    encoded.setflags(write=False)
    return encoded


def float_bit_pattern_hex(values: Any) -> np.ndarray:
    """Return authoritative IEEE bit-pattern hex for finite float32/float64."""
    return _bit_hex_array(_coerce_raw_scores(values))


def apply_if_score_contract(
    if_score_raw: Any, tau_if_raw: Any
) -> IFScoreContractResult:
    """Apply the one closed raw/canonical contract without dtype promotion."""
    raw = _coerce_raw_scores(if_score_raw)
    raw_tau = _coerce_raw_threshold(tau_if_raw, raw.dtype)
    try:
        anomaly = _ASARRAY_FUNCTION(_negate_kernel(raw))
        canonical_tau = _ASARRAY_FUNCTION(_canonical_tau_kernel(raw_tau))
    except FloatingPointError as exc:
        raise IFScoreContractError(
            "canonical score or threshold cannot remain finite"
        ) from exc
    if anomaly.dtype != raw.dtype or anomaly.shape != raw.shape:
        raise IFScoreContractError("canonical anomaly score changed dtype or shape")
    if canonical_tau.dtype != raw.dtype or canonical_tau.shape != ():
        raise IFScoreContractError("canonical IF threshold changed dtype or shape")
    if (
        not _ISFINITE_UFUNC(anomaly).all()
        or bool(_ISNAN_UFUNC(canonical_tau))
        or bool(_ISNEGINF_FUNCTION(canonical_tau))
    ):
        raise IFScoreContractError(
            "canonical anomaly score/threshold violates the derived-value policy"
        )
    raw_reject = _ASARRAY_FUNCTION(_raw_lt_kernel(raw, raw_tau), dtype=_BOOL_TYPE)
    canonical_reject = _ASARRAY_FUNCTION(
        _canonical_ge_kernel(anomaly, canonical_tau), dtype=_BOOL_TYPE
    )
    if raw_reject.shape != raw.shape or canonical_reject.shape != raw.shape:
        raise IFScoreContractError("IF decision shape differs from score shape")
    if not _ARRAY_EQUAL_FUNCTION(raw_reject, canonical_reject):
        raise IFScoreContractError(
            "raw and canonical IF decisions are not exactly equivalent"
        )
    raw_bits = _bit_hex_array(raw)
    raw_tau_bits = str(_bit_hex_array(raw_tau).item())
    anomaly_bits = _bit_hex_array(anomaly)
    canonical_tau_bits = str(_bit_hex_array(canonical_tau).item())
    return IFScoreContractResult(
        contract_id=CONTRACT_ID,
        comparison_dtype=raw.dtype.name,
        if_score_raw=_immutable_array(raw),
        tau_if_raw=_immutable_array(raw_tau),
        canonical_anomaly_score=_immutable_array(anomaly),
        tau_if_canonical=_immutable_array(canonical_tau),
        raw_if_reject=_immutable_array(raw_reject),
        canonical_if_reject=_immutable_array(canonical_reject),
        if_score_raw_bits=raw_bits,
        tau_if_raw_bits=raw_tau_bits,
        canonical_anomaly_score_bits=anomaly_bits,
        tau_if_canonical_bits=canonical_tau_bits,
    )


def contract_spec() -> dict[str, Any]:
    """Return the fixed declarative semantics without implementation selectors."""
    return {
        "contract_id": CONTRACT_ID,
        "raw_comparator_enum": RAW_COMPARATOR_ENUM,
        "orientation_transform_enum": ORIENTATION_TRANSFORM_ENUM,
        "canonical_comparator_enum": CANONICAL_COMPARATOR_ENUM,
        "raw_score_semantics": RAW_SCORE_SEMANTICS,
        "canonical_score_semantics": CANONICAL_SCORE_SEMANTICS,
        "allowed_comparison_dtypes": list(ALLOWED_COMPARISON_DTYPES),
        "nonfinite_policy": NONFINITE_POLICY,
        "threshold_conversion": (
            "nextafter_D(negate_D(tau_if_raw), positive_infinity_D)"
        ),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _qualified_ast_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _normalized_ast_sha256(node: ast.AST) -> str:
    normalized = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _constant_record(value: Any, source_path: Path) -> Any:
    if value is None:
        return {"type": "none"}
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if type(value) is complex:
        return {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if type(value) is tuple:
        return {
            "type": "tuple",
            "items": [_constant_record(item, source_path) for item in value],
        }
    if type(value) is frozenset:
        items = [_constant_record(item, source_path) for item in value]
        items.sort(key=_canonical_json_bytes)
        return {"type": "frozenset", "items": items}
    if isinstance(value, types.CodeType):
        return {"type": "code", "value": _code_record(value, source_path)}
    raise IFScoreAttestationError(
        "unsupported code constant type: " f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _code_record(code: types.CodeType, source_path: Path) -> dict[str, Any]:
    try:
        code_path = Path(code.co_filename).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IFScoreAttestationError(
            f"code object source path is unavailable: {code.co_qualname}"
        ) from exc
    if code_path != source_path:
        raise IFScoreAttestationError(
            f"code object escaped trusted source: {code.co_qualname}"
        )
    repository_root = _trusted_source_path().parents[3]
    try:
        relative_source = source_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise IFScoreAttestationError(
            f"code object source escaped repository: {code.co_qualname}"
        ) from exc
    if relative_source not in {path for path, _module in _TRUSTED_MODULE_SPECS}:
        raise IFScoreAttestationError(
            f"code object source is not a fixed trusted module: {code.co_qualname}"
        )
    return {
        "schema_version": CODE_FINGERPRINT_SCHEMA_VERSION,
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "python_cache_tag": sys.implementation.cache_tag,
        "name": code.co_name,
        "qualname": code.co_qualname,
        "filename": relative_source,
        "firstlineno": code.co_firstlineno,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_hex": code.co_code.hex(),
        "constants": [
            _constant_record(value, source_path) for value in code.co_consts
        ],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "linetable_hex": code.co_linetable.hex(),
        "exceptiontable_hex": code.co_exceptiontable.hex(),
    }


def _code_sha256(code: types.CodeType, source_path: Path) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_code_record(code, source_path))
    ).hexdigest()


def _all_code_objects(code: types.CodeType) -> list[types.CodeType]:
    result = [code]
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            result.extend(_all_code_objects(value))
    return result


def _trusted_source_path() -> Path:
    if (
        TRUSTED_MODULE != "turboguard.core.if_score_contract"
        or TRUSTED_SOURCE_RELATIVE_PATH
        != "src/turboguard/core/if_score_contract.py"
    ):
        raise IFScoreAttestationError("fixed contract identity constant was rebound")
    relative = Path("src/turboguard/core/if_score_contract.py")
    lexical = Path(__file__).absolute()
    if tuple(lexical.parts[-len(relative.parts) :]) != relative.parts:
        raise IFScoreAttestationError(
            "contract module is not loaded from its fixed repository path"
        )
    repository_root = lexical.parents[len(relative.parts) - 1]
    expected = repository_root / relative
    if lexical != expected:
        raise IFScoreAttestationError("contract source lexical path differs")
    cursor = repository_root
    for part in relative.parts:
        cursor = cursor / part
        try:
            mode = os.lstat(cursor).st_mode
        except OSError as exc:
            raise IFScoreAttestationError(
                f"contract source component is unavailable: {cursor}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise IFScoreAttestationError(
                f"contract source component is a symlink: {cursor}"
            )
    return expected


def _read_trusted_source(source_path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source_path, flags)
    except OSError as exc:
        raise IFScoreAttestationError("cannot open trusted contract source") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IFScoreAttestationError("contract source is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise IFScoreAttestationError("contract source changed during snapshot")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise IFScoreAttestationError("contract source snapshot is incomplete")
    current = os.lstat(source_path)
    if current.st_dev != before.st_dev or current.st_ino != before.st_ino:
        raise IFScoreAttestationError("contract source path changed after snapshot")
    return payload, before


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IFScoreAttestationError(
                f"duplicate literal key in trust JSON object: {key}"
            )
        result[key] = value
    return result


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return types.MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise IFScoreAttestationError("trust manifest contains unsupported JSON value")


def _require_deeply_frozen_json(value: Any, *, label: str) -> None:
    """Reject a same-content rebind to any mutable JSON container."""
    if type(value) is types.MappingProxyType:
        for key, item in value.items():
            if type(key) is not str:
                raise IFScoreAttestationError(
                    f"{label} contains a non-string mapping key"
                )
            _require_deeply_frozen_json(item, label=label)
        return
    if type(value) is tuple:
        for item in value:
            _require_deeply_frozen_json(item, label=label)
        return
    if value is None or type(value) in {bool, int, float, str}:
        return
    raise IFScoreAttestationError(
        f"{label} is not recursively runtime read-only"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, types.MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value


def _read_fixed_trust_file(path: Path, label: str) -> bytes:
    repository_root = _trusted_source_path().parents[3]
    try:
        relative = path.absolute().relative_to(repository_root)
    except ValueError as exc:
        raise IFScoreAttestationError(f"{label} escaped repository root") from exc
    if (
        not relative.parts
        or ".." in relative.parts
        or path.absolute() != repository_root / relative
    ):
        raise IFScoreAttestationError(f"{label} is not one fixed lexical path")
    cursor = repository_root
    for part in relative.parts:
        cursor = cursor / part
        try:
            observed_mode = os.lstat(cursor).st_mode
        except OSError as exc:
            raise IFScoreAttestationError(f"{label} component is unavailable") from exc
        if stat.S_ISLNK(observed_mode):
            raise IFScoreAttestationError(f"{label} component is a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IFScoreAttestationError(f"cannot open fixed {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise IFScoreAttestationError(
                f"fixed {label} is not one unaliased regular file"
            )
        blocks: list[bytes] = []
        offset = 0
        while offset < int(before.st_size):
            block = os.pread(
                descriptor, min(1024 * 1024, int(before.st_size) - offset), offset
            )
            if not block:
                raise IFScoreAttestationError(f"fixed {label} ended during read")
            blocks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    signatures = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_nlink),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    ), (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_nlink),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if signatures[0] != signatures[1] or offset != int(after.st_size):
        raise IFScoreAttestationError(f"fixed {label} changed during read")
    try:
        current = os.lstat(path)
    except OSError as exc:
        raise IFScoreAttestationError(f"fixed {label} disappeared after read") from exc
    if (
        int(current.st_dev) != int(after.st_dev)
        or int(current.st_ino) != int(after.st_ino)
        or int(current.st_size) != int(after.st_size)
        or int(current.st_nlink) != 1
        or int(current.st_mtime_ns) != int(after.st_mtime_ns)
        or int(current.st_ctime_ns) != int(after.st_ctime_ns)
    ):
        raise IFScoreAttestationError(f"fixed {label} path changed after read")
    return b"".join(blocks)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _implementation_symbol_id(module_path: str, module: str, qualname: str) -> str:
    return f"{module_path}::{module}::{qualname}"


def _load_external_implementation_manifest() -> tuple[Any, Any, Any]:
    if (
        TRUST_MANIFEST_RELATIVE_PATH
        != "src/turboguard/core/if_score_implementation_manifest.json"
        or TRUST_POLICY_RELATIVE_PATH != "plan/phase1/artifact_policy.json"
        or TRUST_POLICY_ID != "turboguard-phase1-artifact-policy-v2"
        or TRUST_POLICY_FIELD != "implementation_trust_contract"
        or _PRODUCTION_COMPARATOR_SYMBOL_ID
        != (
            "src/turboguard/core/if_score_contract.py::"
            "turboguard.core.if_score_contract::_raw_lt_kernel"
        )
    ):
        raise IFScoreAttestationError("fixed external trust identity was rebound")
    source_path = _trusted_source_path()
    repository_root = source_path.parents[3]
    manifest_path = repository_root / TRUST_MANIFEST_RELATIVE_PATH
    policy_path = repository_root / TRUST_POLICY_RELATIVE_PATH
    manifest_bytes = _read_fixed_trust_file(
        manifest_path, "IF implementation manifest"
    )
    policy_bytes = _read_fixed_trust_file(policy_path, "Phase 1 artifact policy")
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
        policy = json.loads(
            policy_bytes.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IFScoreAttestationError("external trust JSON cannot be decoded") from exc
    if type(manifest) is not dict or type(policy) is not dict:
        raise IFScoreAttestationError("external trust roots must be JSON objects")
    policy_contract = policy.get(TRUST_POLICY_FIELD)
    expected_policy_keys = {
        "schema_version",
        "manifest_path",
        "manifest_sha256",
        "trusted_module",
        "trusted_source_path",
        "production_comparator_symbol_id",
        "checkpoint_00_binding",
    }
    if (
        policy.get("policy_id") != TRUST_POLICY_ID
        or type(policy_contract) is not dict
        or set(policy_contract) != expected_policy_keys
        or policy_contract.get("schema_version")
        != "phase1-if-implementation-policy-binding-v1"
        or policy_contract.get("manifest_path") != TRUST_MANIFEST_RELATIVE_PATH
        or policy_contract.get("manifest_sha256")
        != hashlib.sha256(manifest_bytes).hexdigest()
        or policy_contract.get("trusted_module") != TRUSTED_MODULE
        or policy_contract.get("trusted_source_path")
        != TRUSTED_SOURCE_RELATIVE_PATH
    ):
        raise IFScoreAttestationError(
            "artifact policy does not bind the fixed implementation manifest"
        )
    checkpoint_binding = policy_contract.get("checkpoint_00_binding")
    scope = policy.get("checkpoint_00_contract", {}).get("source_scope", {})
    policy_rows = policy.get("artifacts")
    if type(policy_rows) is not list:
        raise IFScoreAttestationError("artifact policy rows are malformed")
    policy_artifacts = [
        row
        for row in policy_rows
        if type(row) is dict and row.get("artifact_id") == "p1_artifact_policy"
    ]
    if (
        checkpoint_binding
        != {
            "required": True,
            "inventory_unit": "regular_file_sha256",
            "manifest_source_scope_root": "src",
            "policy_source_scope_root": "plan",
            "policy_artifact_id": "p1_artifact_policy",
            "policy_path": TRUST_POLICY_RELATIVE_PATH,
            "digest_record_path": "pre_edit/source_sha256.json",
            "verification": "strict_checkpoint_00_and_phase_gate",
        }
        or scope.get("regular_files_only") is not True
        or "src" not in scope.get("directories", [])
        or "plan" not in scope.get("directories", [])
        or len(policy_artifacts) != 1
        or policy_artifacts[0].get("path") != TRUST_POLICY_RELATIVE_PATH
        or policy_artifacts[0].get("disposition") != "allowed"
        or policy_artifacts[0].get("lifecycle") != "required_now"
        or any(
            Path(TRUST_MANIFEST_RELATIVE_PATH).match(pattern)
            for pattern in scope.get("excluded_generated_patterns", [])
        )
    ):
        raise IFScoreAttestationError(
            "Task-00 source scope does not bind the implementation manifest"
        )
    expected_manifest_keys = {
        "schema_version",
        "manifest_id",
        "hash_algorithms",
        "trusted_modules",
        "production_comparator_symbol_id",
        "policy_binding",
        "entries",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version")
        != "phase1-if-implementation-manifest-v1"
        or type(manifest.get("entries")) is not list
        or not manifest["entries"]
    ):
        raise IFScoreAttestationError("implementation manifest schema differs")
    if manifest.get("hash_algorithms") != {
        "whole_file_sha256": "sha256(raw_regular_file_bytes)",
        "normalized_ast_sha256": (
            "sha256(utf8(ast.dump(function_node,annotate_fields=true,"
            "include_attributes=false)))"
        ),
        "code_object_sha256": (
            "sha256(canonical_json(phase1-python-code-fingerprint-v1))"
        ),
    }:
        raise IFScoreAttestationError("implementation hash algorithms differ")
    expected_policy_binding = {
        "policy_path": TRUST_POLICY_RELATIVE_PATH,
        "policy_id": TRUST_POLICY_ID,
        "policy_field": TRUST_POLICY_FIELD,
        "task00_policy_digest_required": True,
        "digest_record_path": "pre_edit/source_sha256.json",
    }
    if manifest.get("policy_binding") != expected_policy_binding:
        raise IFScoreAttestationError("implementation manifest policy binding differs")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected_manifest_id = (
        "if-implementation-manifest-sha256:"
        + hashlib.sha256(_canonical_json_bytes(semantic)).hexdigest()
    )
    if manifest.get("manifest_id") != expected_manifest_id:
        raise IFScoreAttestationError("implementation manifest ID differs")
    entry_keys = {
        "symbol_id",
        "module_path",
        "module",
        "qualname",
        "implementation_role",
        "whole_file_sha256",
        "normalized_ast_sha256",
        "code_object_sha256",
    }
    expected_by_id = {
        _implementation_symbol_id(module_path, module_name, qualname): {
            "module_path": module_path,
            "module": module_name,
            "qualname": qualname,
            "implementation_role": role,
        }
        for module_path, module_name, qualname, role in _TRUSTED_SYMBOL_SPECS
    }
    if len(expected_by_id) != len(_TRUSTED_SYMBOL_SPECS):
        raise IFScoreAttestationError("fixed trusted symbol identities are duplicated")
    by_id: dict[str, dict[str, Any]] = {}
    observed_ids: list[str] = []
    comparator_ids: list[str] = []
    for entry in manifest["entries"]:
        if type(entry) is not dict or set(entry) != entry_keys:
            raise IFScoreAttestationError("implementation manifest entry schema differs")
        module_path = entry.get("module_path")
        module_name = entry.get("module")
        qualname = entry.get("qualname")
        symbol_id = entry.get("symbol_id")
        if (
            type(module_path) is not str
            or type(module_name) is not str
            or type(qualname) is not str
            or symbol_id
            != _implementation_symbol_id(module_path, module_name, qualname)
            or symbol_id in by_id
            or symbol_id not in expected_by_id
            or {
                "module_path": module_path,
                "module": module_name,
                "qualname": qualname,
                "implementation_role": entry.get("implementation_role"),
            }
            != expected_by_id.get(symbol_id)
            or not all(
                _is_sha256(entry.get(field))
                for field in (
                    "whole_file_sha256",
                    "normalized_ast_sha256",
                    "code_object_sha256",
                )
            )
        ):
            raise IFScoreAttestationError(
                "implementation manifest identity/hash is malformed or duplicated"
            )
        by_id[symbol_id] = entry
        observed_ids.append(symbol_id)
        if entry.get("implementation_role") == "production_comparator_kernel":
            comparator_ids.append(symbol_id)
    if (
        set(by_id) != set(expected_by_id)
        or observed_ids != sorted(expected_by_id)
    ):
        raise IFScoreAttestationError(
            "implementation manifest fixed symbol coverage differs"
        )
    production_id = manifest.get("production_comparator_symbol_id")
    if (
        comparator_ids != [production_id]
        or production_id != policy_contract.get("production_comparator_symbol_id")
        or production_id != _PRODUCTION_COMPARATOR_SYMBOL_ID
        or manifest.get("trusted_modules")
        != [
            {"module_path": path, "module": module}
            for path, module in _TRUSTED_MODULE_SPECS
        ]
    ):
        raise IFScoreAttestationError(
            "implementation manifest does not select one fixed comparator kernel"
        )
    frozen_manifest = _freeze_json(manifest)
    frozen_by_id = types.MappingProxyType(
        {symbol_id: _freeze_json(entry) for symbol_id, entry in by_id.items()}
    )
    trust_binding = _freeze_json(
        {
            "manifest_path": TRUST_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "policy_path": TRUST_POLICY_RELATIVE_PATH,
            "policy_id": TRUST_POLICY_ID,
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "task00_policy_digest_binding": checkpoint_binding,
        }
    )
    return frozen_manifest, frozen_by_id, trust_binding


def _top_level_function_nodes(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    result: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in result:
                raise IFScoreAttestationError(
                    f"duplicate contract function definition: {node.name}"
                )
            if isinstance(node, ast.AsyncFunctionDef):
                raise IFScoreAttestationError(
                    f"contract function cannot be async: {node.name}"
                )
            result[node.name] = node
    return result


def _verify_kernel_ast(name: str, node: ast.FunctionDef) -> None:
    expected_arguments = _KERNEL_ARGUMENTS[name]
    observed_arguments = tuple(argument.arg for argument in node.args.args)
    if (
        observed_arguments != expected_arguments
        or node.args.posonlyargs
        or node.args.kwonlyargs
        or node.args.vararg is not None
        or node.args.kwarg is not None
        or node.args.defaults
        or node.args.kw_defaults
        or node.decorator_list
    ):
        raise IFScoreAttestationError(
            f"kernel signature is not exact: {name}"
        )
    expected_node = ast.parse(_KERNEL_BODY_TEMPLATES[name]).body[0]
    if not isinstance(expected_node, ast.FunctionDef):
        raise IFScoreAttestationError("internal kernel AST template is invalid")
    observed_body = ast.dump(
        ast.Module(body=node.body, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    expected_body = ast.dump(
        ast.Module(body=expected_node.body, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    if observed_body != expected_body:
        raise IFScoreAttestationError(
            f"kernel executable AST is outside the exact allowlist: {name}"
        )
    banned = (ast.If, ast.IfExp, ast.BoolOp, ast.Lambda, ast.Match)
    if any(isinstance(child, banned) for child in ast.walk(node)):
        raise IFScoreAttestationError(
            f"kernel contains branching or dynamic composition: {name}"
        )


def _verify_unique_kernel_ufuncs(tree: ast.Module) -> None:
    expected = {
        "_RAW_LT_UFUNC": 1,
        "_NEGATE_UFUNC": 1,
        "_NEXTAFTER_UFUNC": 1,
        "_CANONICAL_GE_UFUNC": 1,
    }
    observed = {name: 0 for name in expected}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            qualified = _qualified_ast_name(node.func)
            if qualified in observed:
                observed[qualified] += 1
    if observed != expected:
        raise IFScoreAttestationError(
            "contract semantic ufuncs are not unique private-kernel calls: "
            f"{observed}"
        )


def _verify_live_numpy_ufuncs() -> None:
    """Bind semantic ufuncs to the non-heap NumPy C type."""
    real_type = (0).__class__.__class__
    ufunc_type = _UFUNC_TYPE
    expected = {
        "_RAW_LT_UFUNC": ("less", 2),
        "_NEGATE_UFUNC": ("negative", 1),
        "_NEXTAFTER_UFUNC": ("nextafter", 2),
        "_CANONICAL_GE_UFUNC": ("greater_equal", 2),
    }
    if (
        real_type(ufunc_type) is not real_type
        or real_type.__getattribute__(ufunc_type, "__module__") != "numpy"
        or real_type.__getattribute__(ufunc_type, "__qualname__") != "ufunc"
        or (
            real_type.__getattribute__(ufunc_type, "__basicsize__"),
            real_type.__getattribute__(ufunc_type, "__itemsize__"),
            real_type.__getattribute__(ufunc_type, "__flags__"),
        )
        != (240, 0, 22912)
    ):
        raise IFScoreAttestationError("trusted NumPy ufunc C type differs")
    module = sys.modules[TRUSTED_MODULE]
    for global_name, (numpy_name, nin) in expected.items():
        candidate = getattr(module, global_name, None)
        if (
            real_type(candidate) is not ufunc_type
            or np.__dict__.get(numpy_name) is not candidate
            or candidate.__name__ != numpy_name
            or candidate.nin != nin
            or candidate.nout != 1
        ):
            raise IFScoreAttestationError(
                f"live NumPy ufunc identity differs: {global_name}"
            )


def _verify_live_numpy_dependencies() -> None:
    """Bind every mutable NumPy name that can change contract semantics."""
    real_type = (0).__class__.__class__
    expected = {
        "_ARRAY_FUNCTION": (
            "array", "builtins", "builtin_function_or_method", "array",
            (56, 0, 22914), None,
        ),
        "_ASARRAY_FUNCTION": (
            "asarray", "builtins", "builtin_function_or_method", "asarray",
            (56, 0, 22914), None,
        ),
        "_ASCONTIGUOUSARRAY_FUNCTION": (
            "ascontiguousarray", "builtins", "builtin_function_or_method",
            "ascontiguousarray", (56, 0, 22914), None,
        ),
        "_ARRAY_EQUAL_FUNCTION": (
            "array_equal", "numpy", "_ArrayFunctionDispatcher", "array_equal",
            (64, 0, 137472), None,
        ),
        "_ISFINITE_UFUNC": (
            "isfinite", "numpy", "ufunc", "isfinite", (240, 0, 22912), None,
        ),
        "_ISNAN_UFUNC": (
            "isnan", "numpy", "ufunc", "isnan", (240, 0, 22912), None,
        ),
        "_ISNEGINF_FUNCTION": (
            "isneginf", "numpy", "_ArrayFunctionDispatcher", "isneginf",
            (64, 0, 137472), None,
        ),
        "_DTYPE_TYPE": (
            "dtype", "numpy", "_DTypeMeta", "dtype",
            (992, 40, 2155895040), (88, 0, 5376),
        ),
        "_FLOATING_TYPE": (
            "floating", "builtins", "type", "floating",
            (928, 40, 2155896066), (16, 0, 5376),
        ),
        "_NDARRAY_TYPE": (
            "ndarray", "builtins", "type", "ndarray",
            (928, 40, 2155896066), (96, 0, 5376),
        ),
        "_UFUNC_TYPE": (
            "ufunc", "builtins", "type", "ufunc",
            (928, 40, 2155896066), (240, 0, 22912),
        ),
        "_BOOL_TYPE": (
            "bool_", "builtins", "type", "bool",
            (928, 40, 2155896066), (24, 0, 5376),
        ),
        "_FLOAT32_TYPE": (
            "float32", "builtins", "type", "float32",
            (928, 40, 2155896066), (24, 0, 5376),
        ),
        "_FLOAT64_TYPE": (
            "float64", "builtins", "type", "float64",
            (928, 40, 2155896066), (24, 0, 5376),
        ),
        "_UINT32_TYPE": (
            "uint32", "builtins", "type", "uint32",
            (928, 40, 2155896066), (24, 0, 5376),
        ),
        "_UINT64_TYPE": (
            "uint64", "builtins", "type", "uint64",
            (928, 40, 2155896066), (24, 0, 5376),
        ),
    }
    module = sys.modules[TRUSTED_MODULE]
    for global_name, signature in expected.items():
        (
            numpy_name,
            type_module,
            type_qualname,
            callable_name,
            expected_type_layout,
            expected_object_layout,
        ) = signature
        candidate = getattr(module, global_name, None)
        candidate_type = real_type(candidate)
        candidate_type_layout = (
            real_type.__getattribute__(candidate_type, "__basicsize__"),
            real_type.__getattribute__(candidate_type, "__itemsize__"),
            real_type.__getattribute__(candidate_type, "__flags__"),
        )
        object_layout = None
        if expected_object_layout is not None:
            object_layout = (
                real_type.__getattribute__(candidate, "__basicsize__"),
                real_type.__getattribute__(candidate, "__itemsize__"),
                real_type.__getattribute__(candidate, "__flags__"),
            )
        if (
            np.__dict__.get(numpy_name) is not candidate
            or real_type.__getattribute__(candidate_type, "__module__")
            != type_module
            or real_type.__getattribute__(candidate_type, "__qualname__")
            != type_qualname
            or getattr(candidate, "__name__", None) != callable_name
            or getattr(candidate, "__module__", None) != "numpy"
            or candidate_type_layout != expected_type_layout
            or object_layout != expected_object_layout
        ):
            raise IFScoreAttestationError(
                f"live NumPy dependency identity differs: {global_name}"
            )
    if (
        _FLOAT32_DTYPE != _DTYPE_TYPE(_FLOAT32_TYPE)
        or _FLOAT64_DTYPE != _DTYPE_TYPE(_FLOAT64_TYPE)
    ):
        raise IFScoreAttestationError("fixed comparison dtype identity differs")


def _fixed_module_source_path(relative_path: str, module_name: str) -> Path:
    expected_pair = (relative_path, module_name)
    if expected_pair not in _TRUSTED_MODULE_SPECS:
        raise IFScoreAttestationError("module identity is not in the fixed trust set")
    repository_root = _trusted_source_path().parents[3]
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise IFScoreAttestationError("fixed trusted module path is malformed")
    expected = repository_root / relative
    cursor = repository_root
    for part in relative.parts:
        cursor = cursor / part
        try:
            mode = os.lstat(cursor).st_mode
        except OSError as exc:
            raise IFScoreAttestationError(
                f"fixed trusted module is unavailable: {relative_path}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise IFScoreAttestationError(
                f"fixed trusted module path contains a symlink: {relative_path}"
            )
    if not stat.S_ISREG(os.lstat(expected).st_mode):
        raise IFScoreAttestationError(
            f"fixed trusted module is not a regular file: {relative_path}"
        )
    return expected


def _function_nodes_by_qualname(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    result: dict[str, ast.FunctionDef] = {}

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.AsyncFunctionDef):
                qualname = f"{prefix}.{node.name}" if prefix else node.name
                raise IFScoreAttestationError(
                    f"trusted callable cannot be async: {qualname}"
                )
            if isinstance(node, ast.FunctionDef):
                qualname = f"{prefix}.{node.name}" if prefix else node.name
                if qualname in result:
                    raise IFScoreAttestationError(
                        f"duplicate trusted callable definition: {qualname}"
                    )
                result[qualname] = node
            elif isinstance(node, ast.ClassDef):
                qualname = f"{prefix}.{node.name}" if prefix else node.name
                visit(node.body, qualname)

    visit(tree.body)
    return result


def _resolve_fixed_live_callable(module: types.ModuleType, qualname: str) -> Any:
    current: Any = module
    for component in qualname.split("."):
        current = getattr(current, component, None)
        if current is None:
            raise IFScoreAttestationError(
                f"fixed live callable is missing: {module.__name__}.{qualname}"
            )
    return current


def attest_if_score_contract() -> dict[str, Any]:
    """Recompute every fixed-path manifest hash without caller selectors."""
    if (
        __name__ != TRUSTED_MODULE
        or sys.modules.get(TRUSTED_MODULE) is not sys.modules.get(__name__)
    ):
        raise IFScoreAttestationError("contract module identity differs")
    fresh_manifest, fresh_by_id, fresh_binding = (
        _load_external_implementation_manifest()
    )
    forbidden_runtime_caches = (
        "_APPROVED_IMPLEMENTATION_MANIFEST",
        "_APPROVED_IMPLEMENTATION_BY_ID",
        "_APPROVED_IMPLEMENTATION_TRUST_BINDING",
        "_APPROVED_IMPLEMENTATION_RUNTIME_ROOTS",
        "_APPROVED_IMPLEMENTATION_MANIFEST_ANCHOR",
        "_APPROVED_IMPLEMENTATION_BY_ID_ANCHOR",
        "_APPROVED_IMPLEMENTATION_TRUST_BINDING_ANCHOR",
    )
    if any(name in globals() for name in forbidden_runtime_caches):
        raise IFScoreAttestationError(
            "mutable or rebindable runtime implementation trust cache is forbidden"
        )
    for label, fresh_value in (
        ("fixed-path implementation manifest", fresh_manifest),
        ("fixed-path implementation identity index", fresh_by_id),
        ("fixed-path implementation policy binding", fresh_binding),
    ):
        _require_deeply_frozen_json(fresh_value, label=label)

    contract_source_path = _trusted_source_path()
    contract_source_bytes, _contract_source_stat = _read_trusted_source(
        contract_source_path
    )
    try:
        contract_tree = ast.parse(
            contract_source_bytes.decode("utf-8"),
            filename=str(contract_source_path),
            type_comments=True,
        )
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise IFScoreAttestationError("contract source cannot be parsed") from exc
    contract_nodes = _function_nodes_by_qualname(contract_tree)
    if set(_ATTESTED_FUNCTION_QUALNAMES) - set(contract_nodes):
        raise IFScoreAttestationError("contract source omits an attested function")
    for name in _KERNEL_ARGUMENTS:
        _verify_kernel_ast(name, contract_nodes[name])
    _verify_unique_kernel_ufuncs(contract_tree)
    _verify_live_numpy_ufuncs()
    _verify_live_numpy_dependencies()

    function_records: dict[str, dict[str, Any]] = {}
    module_records: dict[str, dict[str, Any]] = {}
    for module_path, module_name in _TRUSTED_MODULE_SPECS:
        source_path = _fixed_module_source_path(module_path, module_name)
        source_bytes = _read_fixed_trust_file(
            source_path, f"trusted implementation source {module_name}"
        )
        try:
            source_text = source_bytes.decode("utf-8")
            source_tree = ast.parse(
                source_text,
                filename=str(source_path),
                type_comments=True,
            )
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise IFScoreAttestationError(
                f"trusted implementation source cannot be parsed: {module_name}"
            ) from exc
        function_nodes = _function_nodes_by_qualname(source_tree)
        source_code = compile(
            source_text,
            str(source_path),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        compiled_by_qualname: dict[str, list[types.CodeType]] = {}
        for code in _all_code_objects(source_code):
            compiled_by_qualname.setdefault(code.co_qualname, []).append(code)
        module = importlib.import_module(module_name)
        if (
            sys.modules.get(module_name) is not module
            or getattr(module, "__file__", None) is None
            or Path(str(module.__file__)).resolve(strict=True) != source_path
        ):
            raise IFScoreAttestationError(
                f"trusted module is not loaded from its fixed path: {module_name}"
            )
        whole_file_sha256 = hashlib.sha256(source_bytes).hexdigest()
        module_records[module_name] = {
            "module_path": module_path,
            "whole_file_sha256": whole_file_sha256,
            "size_bytes": len(source_bytes),
        }
        fixed_specs = [
            spec for spec in _TRUSTED_SYMBOL_SPECS if spec[1] == module_name
        ]
        for fixed_path, fixed_module, qualname, role in fixed_specs:
            symbol_id = _implementation_symbol_id(
                fixed_path, fixed_module, qualname
            )
            entry = fresh_by_id.get(symbol_id)
            node = function_nodes.get(qualname)
            live = _resolve_fixed_live_callable(module, qualname)
            compiled_matches = compiled_by_qualname.get(qualname, [])
            if (
                entry is None
                or node is None
                or len(compiled_matches) != 1
                or not isinstance(live, types.FunctionType)
                or live.__module__ != module_name
                or live.__qualname__ != qualname
                or live.__globals__ is not module.__dict__
                or live.__closure__ is not None
            ):
                raise IFScoreAttestationError(
                    f"fixed live/source callable identity differs: {symbol_id}"
                )
            if module_name == TRUSTED_MODULE and (
                node.args.defaults
                or any(default is not None for default in node.args.kw_defaults)
                or live.__defaults__ is not None
                or live.__kwdefaults__ is not None
            ):
                raise IFScoreAttestationError(
                    f"contract attester function defaults differ: {symbol_id}"
                )
            normalized_ast_sha256 = _normalized_ast_sha256(node)
            independently_normalized_ast_sha256 = hashlib.sha256(
                ast.dump(
                    node,
                    annotate_fields=True,
                    include_attributes=False,
                ).encode("utf-8")
            ).hexdigest()
            loaded_code_sha256 = _code_sha256(live.__code__, source_path)
            compiled_code_sha256 = _code_sha256(
                compiled_matches[0], source_path
            )
            independently_loaded_code_sha256 = hashlib.sha256(
                _canonical_json_bytes(_code_record(live.__code__, source_path))
            ).hexdigest()
            independently_compiled_code_sha256 = hashlib.sha256(
                _canonical_json_bytes(
                    _code_record(compiled_matches[0], source_path)
                )
            ).hexdigest()
            if (
                entry.get("whole_file_sha256") != whole_file_sha256
                or entry.get("normalized_ast_sha256")
                != normalized_ast_sha256
                or normalized_ast_sha256
                != independently_normalized_ast_sha256
                or entry.get("code_object_sha256") != loaded_code_sha256
                or loaded_code_sha256 != compiled_code_sha256
                or loaded_code_sha256 != independently_loaded_code_sha256
                or compiled_code_sha256
                != independently_compiled_code_sha256
                or entry.get("implementation_role") != role
            ):
                raise IFScoreAttestationError(
                    f"fixed-path implementation hash differs: {symbol_id}"
                )
            function_records[symbol_id] = {
                "symbol_id": symbol_id,
                "module_path": module_path,
                "module": module_name,
                "qualname": qualname,
                "implementation_role": role,
                "whole_file_sha256": whole_file_sha256,
                "normalized_ast_sha256": normalized_ast_sha256,
                "code_object_sha256": loaded_code_sha256,
                "compiled_source_code_sha256": compiled_code_sha256,
            }

    module = sys.modules[TRUSTED_MODULE]
    for kernel_name in _KERNEL_ARGUMENTS:
        kernel = getattr(module, kernel_name)
        if kernel.__globals__.get("np") is not np:
            raise IFScoreAttestationError(
                f"kernel NumPy global binding differs: {kernel_name}"
            )
    semantic = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "contract": contract_spec(),
        "implementation_manifest_id": fresh_manifest["manifest_id"],
        "implementation_manifest_sha256": fresh_binding["manifest_sha256"],
        "external_policy_binding": _thaw_json(fresh_binding),
        "production_comparator_symbol_id": _PRODUCTION_COMPARATOR_SYMBOL_ID,
        "trusted_modules": module_records,
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "python_cache_tag": sys.implementation.cache_tag,
        "numpy_version": np.__version__,
        "byteorder": sys.byteorder,
        "functions": function_records,
        "all_manifest_hashes_recomputed_from_fixed_paths": True,
        "external_task00_policy_digest_binding_verified": True,
        "runtime_manifest_mapping_read_only": True,
        "ephemeral_fixed_path_manifest_authority": True,
        "one_production_comparator_kernel_verified": True,
        "kernel_semantic_ast_allowlist_verified": True,
        "unique_semantic_ufunc_calls_verified": True,
        "live_numpy_ufunc_identities_verified": True,
        "live_numpy_dependency_identities_verified": True,
        "static_numpy_c_type_layouts_verified": True,
        "attested_function_defaults_absent_verified": True,
        "loaded_code_matches_compiled_source": True,
    }
    return {
        **semantic,
        "attestation_id": (
            "if-contract-attestation-sha256:"
            + hashlib.sha256(_canonical_json_bytes(semantic)).hexdigest()
        ),
    }


class _TrustManifestReadOnlyModule(types.ModuleType):
    """Reject insertion of a mutable module-level implementation trust cache."""

    def __setattr__(self, name: str, value: Any) -> None:
        protected = (
            "__class__",
            "TRUSTED_MODULE",
            "TRUSTED_SOURCE_RELATIVE_PATH",
            "TRUST_MANIFEST_RELATIVE_PATH",
            "TRUST_POLICY_RELATIVE_PATH",
            "TRUST_POLICY_ID",
            "TRUST_POLICY_FIELD",
            "_TRUSTED_MODULE_SPECS",
            "_TRUSTED_SYMBOL_SPECS",
            "_PRODUCTION_COMPARATOR_SYMBOL_ID",
            "_APPROVED_IMPLEMENTATION_MANIFEST",
            "_APPROVED_IMPLEMENTATION_BY_ID",
            "_APPROVED_IMPLEMENTATION_TRUST_BINDING",
            "_APPROVED_IMPLEMENTATION_RUNTIME_ROOTS",
            "_APPROVED_IMPLEMENTATION_MANIFEST_ANCHOR",
            "_APPROVED_IMPLEMENTATION_BY_ID_ANCHOR",
            "_APPROVED_IMPLEMENTATION_TRUST_BINDING_ANCHOR",
        )
        if name in protected:
            raise AttributeError(f"fixed implementation trust binding is read-only: {name}")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        protected = (
            "__class__",
            "TRUSTED_MODULE",
            "TRUSTED_SOURCE_RELATIVE_PATH",
            "TRUST_MANIFEST_RELATIVE_PATH",
            "TRUST_POLICY_RELATIVE_PATH",
            "TRUST_POLICY_ID",
            "TRUST_POLICY_FIELD",
            "_TRUSTED_MODULE_SPECS",
            "_TRUSTED_SYMBOL_SPECS",
            "_PRODUCTION_COMPARATOR_SYMBOL_ID",
            "_APPROVED_IMPLEMENTATION_MANIFEST",
            "_APPROVED_IMPLEMENTATION_BY_ID",
            "_APPROVED_IMPLEMENTATION_TRUST_BINDING",
            "_APPROVED_IMPLEMENTATION_RUNTIME_ROOTS",
            "_APPROVED_IMPLEMENTATION_MANIFEST_ANCHOR",
            "_APPROVED_IMPLEMENTATION_BY_ID_ANCHOR",
            "_APPROVED_IMPLEMENTATION_TRUST_BINDING_ANCHOR",
        )
        if name in protected:
            raise AttributeError(f"fixed implementation trust binding is read-only: {name}")
        super().__delattr__(name)


sys.modules[__name__].__class__ = _TrustManifestReadOnlyModule


__all__ = [
    "ALLOWED_COMPARISON_DTYPES",
    "ATTESTATION_SCHEMA_VERSION",
    "CANONICAL_COMPARATOR_ENUM",
    "CANONICAL_SCORE_SEMANTICS",
    "CODE_FINGERPRINT_SCHEMA_VERSION",
    "CONTRACT_ID",
    "IFScoreAttestationError",
    "IFScoreContractError",
    "IFScoreContractResult",
    "NONFINITE_POLICY",
    "ORIENTATION_TRANSFORM_ENUM",
    "RAW_COMPARATOR_ENUM",
    "RAW_SCORE_SEMANTICS",
    "TRUSTED_MODULE",
    "TRUSTED_SOURCE_RELATIVE_PATH",
    "apply_if_score_contract",
    "attest_if_score_contract",
    "contract_spec",
    "float_bit_pattern_hex",
]
