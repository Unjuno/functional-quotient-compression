"""Explicit architecture-adapter plans for real-model FQC extraction.

The core never infers tensor semantics from names. An adapter plan records the
public tensor IDs, checkpoint keys, roles, module metadata, and the config facts
used to justify them. The plan is hashed and embedded in the generated manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .manifest_builder import build_transformer_extraction_manifest, sha256_json


@dataclass(frozen=True)
class TensorBinding:
    public_id: str
    checkpoint_key: str
    role: str
    baseline_included: bool = True


@dataclass(frozen=True)
class AdapterPlan:
    adapter_id: str
    adapter_version: str
    model_identity: Mapping[str, Any]
    tensor_bindings: Sequence[TensorBinding]
    attention_modules: Sequence[Mapping[str, Any]] = ()
    mlp_modules: Sequence[Mapping[str, Any]] = ()
    normalization_modules: Sequence[Mapping[str, Any]] = ()
    external_fixed_state: Sequence[str] = ()
    derived_primitives: Sequence[Mapping[str, Any]] = ()
    extraction_policy: Mapping[str, Any] | None = None
    config_evidence: Mapping[str, Any] | None = None


def _sorted_dict_records(records: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    out=[deepcopy(dict(x)) for x in records]
    return sorted(out, key=lambda x: str(x.get(key,'')))


def adapter_plan_payload(plan: AdapterPlan) -> dict[str, Any]:
    """Return a deterministic JSON-serializable description of the adapter plan."""
    bindings=[{
        'public_id':b.public_id,
        'checkpoint_key':b.checkpoint_key,
        'role':b.role,
        'baseline_included':b.baseline_included,
    } for b in plan.tensor_bindings]
    bindings=sorted(bindings,key=lambda x:(x['public_id'],x['checkpoint_key']))
    return {
        'adapter_id':plan.adapter_id,
        'adapter_version':plan.adapter_version,
        'model_identity':deepcopy(dict(plan.model_identity)),
        'tensor_bindings':bindings,
        'attention_modules':_sorted_dict_records(plan.attention_modules,'module_id'),
        'mlp_modules':_sorted_dict_records(plan.mlp_modules,'module_id'),
        'normalization_modules':_sorted_dict_records(plan.normalization_modules,'module_id'),
        'external_fixed_state':sorted(plan.external_fixed_state),
        'derived_primitives':_sorted_dict_records(plan.derived_primitives,'primitive_id'),
        'extraction_policy':deepcopy(dict(plan.extraction_policy or {})),
        'config_evidence':deepcopy(dict(plan.config_evidence or {})),
    }


def materialize_adapter_plan(plan: AdapterPlan, checkpoint_tensors: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize an explicit adapter plan against live checkpoint tensors."""
    if not plan.adapter_id or not plan.adapter_version:
        raise ValueError('adapter_id and adapter_version are required')
    bindings=tuple(plan.tensor_bindings)
    public_ids=[b.public_id for b in bindings]
    if any(not x for x in public_ids) or len(set(public_ids))!=len(public_ids):
        raise ValueError('tensor binding public_id values must be unique and nonempty')
    if any(not b.checkpoint_key for b in bindings):
        raise ValueError('checkpoint_key values must be nonempty')
    missing=sorted({b.checkpoint_key for b in bindings if b.checkpoint_key not in checkpoint_tensors})
    if missing:
        raise ValueError(f'missing checkpoint tensors: {missing}')

    named={b.public_id:checkpoint_tensors[b.checkpoint_key] for b in bindings}
    roles={b.public_id:b.role for b in bindings}
    excluded=[b.public_id for b in bindings if not b.baseline_included]
    payload=adapter_plan_payload(plan)
    plan_hash=sha256_json(payload)

    identity=deepcopy(dict(plan.model_identity))
    identity['adapter']={
        'adapter_id':plan.adapter_id,
        'adapter_version':plan.adapter_version,
        'adapter_plan_sha256':plan_hash,
        'config_evidence':deepcopy(dict(plan.config_evidence or {})),
    }
    policy=deepcopy(dict(plan.extraction_policy or {}))
    policy['tensor_binding_provenance']=[
        {'tensor_id':b.public_id,'checkpoint_key':b.checkpoint_key}
        for b in sorted(bindings,key=lambda x:x.public_id)
    ]

    return build_transformer_extraction_manifest(
        model_identity=identity,
        named_tensors=named,
        tensor_roles=roles,
        attention_modules=plan.attention_modules,
        mlp_modules=plan.mlp_modules,
        normalization_modules=plan.normalization_modules,
        external_fixed_state=plan.external_fixed_state,
        derived_primitives=plan.derived_primitives,
        extraction_policy=policy,
        baseline_excluded=excluded,
    )
