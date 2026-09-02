"""Deterministic builders for D56/D57 real-model manifests.

Architecture interpretation is explicit: callers provide tensor roles and module
metadata. The builder derives storage aliasing from the live checkpoint rather
than trusting manually declared storage groups.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from .contracts import validate_transformer_extraction
from .storage_inventory import analyze_storage_slices, torch_like_storage_slice


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for artifact hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def contiguous_gqa_map(q_heads: int, kv_heads: int) -> list[int]:
    """Return the standard contiguous GQA mapping when divisibility is exact.

    This is a helper, not an architecture inference rule. Callers must only use
    it when the checkpoint architecture actually uses contiguous grouped heads.
    """
    if not isinstance(q_heads, int) or not isinstance(kv_heads, int) or q_heads <= 0 or kv_heads <= 0:
        raise ValueError("head counts must be positive integers")
    if q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads for contiguous GQA mapping")
    group = q_heads // kv_heads
    return [kv for kv in range(kv_heads) for _ in range(group)]


def _shape_list(tensor: Any) -> list[int]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ValueError("tensor must expose shape")
    out = [int(d) for d in shape]
    if not out or any(d <= 0 for d in out):
        raise ValueError("tensor shape must be positive and nonempty")
    return out


def build_transformer_extraction_manifest(
    *,
    model_identity: Mapping[str, Any],
    named_tensors: Mapping[str, Any],
    tensor_roles: Mapping[str, str],
    attention_modules: Sequence[Mapping[str, Any]] = (),
    mlp_modules: Sequence[Mapping[str, Any]] = (),
    normalization_modules: Sequence[Mapping[str, Any]] = (),
    external_fixed_state: Sequence[str] = (),
    derived_primitives: Sequence[Mapping[str, Any]] = (),
    extraction_policy: Mapping[str, Any] | None = None,
    baseline_excluded: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a D57-style manifest from explicit architecture metadata.

    `named_tensors` should normally be the model parameters/buffers selected by
    an architecture adapter. Runtime storage pointers are used only to prove
    aliasing. Public storage-group IDs are deterministic and expose no pointer.
    """
    names = tuple(sorted(named_tensors))
    if not names:
        raise ValueError("named_tensors must be nonempty")
    if set(tensor_roles) != set(names):
        missing = sorted(set(names) - set(tensor_roles))
        extra = sorted(set(tensor_roles) - set(names))
        raise ValueError(f"tensor_roles must match named_tensors; missing={missing}, extra={extra}")
    excluded = set(baseline_excluded)
    unknown_excluded = sorted(excluded - set(names))
    if unknown_excluded:
        raise ValueError(f"unknown baseline_excluded tensors: {unknown_excluded}")

    slices = [torch_like_storage_slice(name, named_tensors[name]) for name in names]
    inventory = analyze_storage_slices(slices, reject_partial_overlap=True)

    tensor_inventory = []
    for name in names:
        tensor = named_tensors[name]
        tensor_inventory.append({
            "tensor_id": name,
            "shape": _shape_list(tensor),
            "dtype": str(tensor.dtype),
            "storage_group": inventory.exact_alias_group[name],
            "baseline_included": name not in excluded,
            "role": tensor_roles[name],
        })

    policy = deepcopy(dict(extraction_policy or {}))
    policy.setdefault("storage_alias_evidence", "derived_from_live_checkpoint_storage_ranges")
    manifest = {
        "model_identity": deepcopy(dict(model_identity)),
        "tensor_inventory": tensor_inventory,
        "modules": {
            "attention": [deepcopy(dict(x)) for x in attention_modules],
            "mlp": [deepcopy(dict(x)) for x in mlp_modules],
            "normalization": [deepcopy(dict(x)) for x in normalization_modules],
        },
        "external_fixed_state": list(external_fixed_state),
        "derived_primitives": [deepcopy(dict(x)) for x in derived_primitives],
        "extraction_policy": policy,
    }
    validation = validate_transformer_extraction(manifest)
    if not validation.valid:
        raise ValueError("invalid generated extraction manifest: " + "; ".join(validation.errors))
    manifest["model_identity"]["unique_baseline_scalar_count_N"] = validation.unique_baseline_scalar_count
    return manifest


def build_pilot_contract(
    *,
    model_id: str,
    checkpoint_hash: str,
    extraction_manifest: Mapping[str, Any],
    operator_manifest: Mapping[str, Any],
    paid_atom_manifest: Mapping[str, Any],
    quality_contract: Mapping[str, Any],
    replay_contract: Mapping[str, Any],
    metric_id: str,
    quality_target: float,
    decoder_protocol_id: str = "fqc",
    decoder_protocol_version: str = "1",
    evaluation_dtype: str = "float64",
    randomness_policy: str = "deterministic",
    external_free_items: Sequence[str] = (),
    package_version: str = "fqc-real-pilot-v1",
) -> dict[str, Any]:
    """Build the deterministic D56 pilot contract from concrete artifacts."""
    if not checkpoint_hash.startswith("sha256:"):
        raise ValueError("checkpoint_hash must use sha256:<hex> form")
    manifest_checkpoint = extraction_manifest.get("model_identity", {}).get("checkpoint_sha256")
    if manifest_checkpoint is not None and manifest_checkpoint != checkpoint_hash:
        raise ValueError("checkpoint hash mismatch between extraction manifest and pilot contract")
    validation = validate_transformer_extraction(extraction_manifest)
    if not validation.valid:
        raise ValueError("invalid extraction manifest: " + "; ".join(validation.errors))
    N = validation.unique_baseline_scalar_count
    if N <= 0:
        raise ValueError("extraction manifest has no paid baseline scalars")
    return {
        "package_version": package_version,
        "model": {
            "model_id": model_id,
            "checkpoint_hash": checkpoint_hash,
            "unique_paid_scalar_count_N": N,
        },
        "baseline": {
            "bits_per_scalar": 16,
            "baseline_bits": 16 * N,
            "unique_storage_rule": "count unique serialized scalar storage once",
            "external_free_items": list(external_free_items),
        },
        "target": {
            "compression_factor": 64,
            "B64_integer_bits": N // 4,
            "metric_id": metric_id,
            "quality_target": quality_target,
        },
        "decoder_protocol": {
            "protocol_id": decoder_protocol_id,
            "version": decoder_protocol_version,
        },
        "numeric_contract": {
            "evaluation_dtype": evaluation_dtype,
            "randomness_policy": randomness_policy,
        },
        "artifact_hashes": {
            "tensor_inventory": sha256_json(extraction_manifest.get("tensor_inventory", [])),
            "operator_manifest": sha256_json(operator_manifest),
            "paid_atom_manifest": sha256_json(paid_atom_manifest),
            "quality_contract": sha256_json(quality_contract),
            "replay_contract": sha256_json(replay_contract),
        },
    }
