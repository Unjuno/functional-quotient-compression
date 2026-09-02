"""Safetensors header inventory for real-checkpoint accounting.

This module reads the serialized checkpoint structure without loading tensor
values. It is useful before a runtime model load because it can verify file
integrity, complete adapter coverage, shapes, dtypes, and the exact serialized
scalar inventory using only the safetensors header.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .adapter_plan import AdapterPlan, adapter_plan_payload
from .contracts import validate_transformer_extraction
from .manifest_builder import sha256_json


_DTYPE_BYTES={
    'BOOL':1,'I8':1,'U8':1,
    'I16':2,'U16':2,'F16':2,'BF16':2,
    'I32':4,'U32':4,'F32':4,
    'I64':8,'U64':8,'F64':8,
    'F8_E4M3':1,'F8_E5M2':1,'F8_E4M3FN':1,'F8_E5M2FNUZ':1,
}


@dataclass(frozen=True)
class SafetensorEntry:
    key: str
    dtype: str
    shape: tuple[int,...]
    data_start: int
    data_end: int

    @property
    def scalar_count(self) -> int:
        n=1
        for d in self.shape: n*=d
        return n

    @property
    def data_bytes(self) -> int:
        return self.data_end-self.data_start


@dataclass(frozen=True)
class SafetensorsHeader:
    path: str
    file_size_bytes: int
    header_size_bytes: int
    entries: Mapping[str,SafetensorEntry]
    metadata: Mapping[str,Any]

    @property
    def scalar_count(self) -> int:
        return sum(x.scalar_count for x in self.entries.values())


@dataclass(frozen=True)
class SerializedPreflight:
    extraction_manifest: Mapping[str,Any]
    serialized_scalar_count: int
    serialized_tensor_count: int
    dtype_scalar_counts: Mapping[str,int]
    unaccounted_checkpoint_keys: tuple[str,...]
    missing_checkpoint_keys: tuple[str,...]
    config_expected_scalar_count: int | None
    passed: bool


def sha256_file(path: str | Path, chunk_bytes: int=8*1024*1024) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk_bytes)
            if not b: break
            h.update(b)
    return h.hexdigest()


def read_safetensors_header(path: str | Path) -> SafetensorsHeader:
    p=Path(path)
    file_size=p.stat().st_size
    with p.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8: raise ValueError('file too short for safetensors header length')
        header_len=int.from_bytes(raw,'little',signed=False)
        if header_len<=0 or 8+header_len>file_size:
            raise ValueError('invalid safetensors header length')
        header_raw=f.read(header_len)
    try:
        obj=json.loads(header_raw.decode('utf-8'))
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise ValueError('invalid safetensors JSON header') from exc
    if not isinstance(obj,dict): raise ValueError('safetensors header must be an object')
    metadata=obj.get('__metadata__',{})
    if metadata is None: metadata={}
    if not isinstance(metadata,dict): raise ValueError('__metadata__ must be an object')
    entries={}
    intervals=[]
    for key,value in obj.items():
        if key=='__metadata__': continue
        if not isinstance(key,str) or not key or not isinstance(value,dict):
            raise ValueError('invalid safetensors tensor entry')
        dtype=value.get('dtype'); shape=value.get('shape'); offsets=value.get('data_offsets')
        if dtype not in _DTYPE_BYTES: raise ValueError(f'{key}: unsupported dtype {dtype}')
        if not isinstance(shape,list) or any(not isinstance(d,int) or isinstance(d,bool) or d<0 for d in shape):
            raise ValueError(f'{key}: invalid shape')
        if not isinstance(offsets,list) or len(offsets)!=2 or any(not isinstance(x,int) or isinstance(x,bool) for x in offsets):
            raise ValueError(f'{key}: invalid data_offsets')
        start,end=offsets
        if start<0 or end<start: raise ValueError(f'{key}: invalid data range')
        entry=SafetensorEntry(key,dtype,tuple(shape),start,end)
        expected=entry.scalar_count*_DTYPE_BYTES[dtype]
        if entry.data_bytes!=expected:
            raise ValueError(f'{key}: data byte count {entry.data_bytes} != expected {expected}')
        entries[key]=entry; intervals.append((start,end,key))
    intervals.sort()
    last_end=0
    for start,end,key in intervals:
        if start<last_end: raise ValueError(f'{key}: overlapping tensor data ranges')
        last_end=max(last_end,end)
    if 8+header_len+last_end!=file_size:
        raise ValueError('safetensors header/data ranges do not account for the full file size')
    return SafetensorsHeader(str(p),file_size,header_len,entries,metadata)


def materialize_adapter_plan_from_safetensors_header(
    plan: AdapterPlan,
    header: SafetensorsHeader,
    *,
    require_complete_checkpoint_coverage: bool=True,
) -> SerializedPreflight:
    """Build a D57 manifest from serialized checkpoint descriptors.

    The checkpoint file itself is treated as the storage authority: one serialized
    tensor key is one paid storage object. Multiple public roles may bind the same
    key (e.g. tied embedding/lm-head) and are then counted once.
    """
    bindings=tuple(plan.tensor_bindings)
    bound_keys={b.checkpoint_key for b in bindings}
    header_keys=set(header.entries)
    missing=tuple(sorted(bound_keys-header_keys))
    extra=tuple(sorted(header_keys-bound_keys))
    if missing: raise ValueError('adapter references missing serialized checkpoint keys: '+', '.join(missing))
    if require_complete_checkpoint_coverage and extra:
        raise ValueError('serialized checkpoint contains unaccounted tensor keys: '+', '.join(extra))

    unique_keys=sorted(bound_keys)
    group_by_key={key:f'sg{idx:06d}' for idx,key in enumerate(unique_keys)}
    tensor_inventory=[]
    for b in sorted(bindings,key=lambda x:x.public_id):
        entry=header.entries[b.checkpoint_key]
        tensor_inventory.append({
            'tensor_id':b.public_id,
            'shape':list(entry.shape),
            'dtype':entry.dtype,
            'storage_group':group_by_key[b.checkpoint_key],
            'baseline_included':b.baseline_included,
            'role':b.role,
        })
    payload=adapter_plan_payload(plan)
    identity=deepcopy(dict(plan.model_identity))
    identity['adapter']={
        'adapter_id':plan.adapter_id,
        'adapter_version':plan.adapter_version,
        'adapter_plan_sha256':sha256_json(payload),
        'config_evidence':deepcopy(dict(plan.config_evidence or {})),
    }
    policy=deepcopy(dict(plan.extraction_policy or {}))
    policy['storage_alias_evidence']='serialized_safetensors_tensor_key_identity'
    policy['tensor_binding_provenance']=[
        {'tensor_id':b.public_id,'checkpoint_key':b.checkpoint_key}
        for b in sorted(bindings,key=lambda x:x.public_id)
    ]
    manifest={
        'model_identity':identity,
        'tensor_inventory':tensor_inventory,
        'modules':{
            'attention':[deepcopy(dict(x)) for x in plan.attention_modules],
            'mlp':[deepcopy(dict(x)) for x in plan.mlp_modules],
            'normalization':[deepcopy(dict(x)) for x in plan.normalization_modules],
        },
        'external_fixed_state':list(plan.external_fixed_state),
        'derived_primitives':[deepcopy(dict(x)) for x in plan.derived_primitives],
        'extraction_policy':policy,
    }
    validation=validate_transformer_extraction(manifest)
    if not validation.valid:
        raise ValueError('serialized manifest validation failed: '+'; '.join(validation.errors))
    manifest['model_identity']['unique_baseline_scalar_count_N']=validation.unique_baseline_scalar_count
    dtype_counts={}
    for entry in header.entries.values(): dtype_counts[entry.dtype]=dtype_counts.get(entry.dtype,0)+entry.scalar_count
    expected=(plan.config_evidence or {}).get('config_expected_unique_scalar_count')
    passed=(not missing and (not extra or not require_complete_checkpoint_coverage) and
            (expected is None or expected==validation.unique_baseline_scalar_count==header.scalar_count))
    return SerializedPreflight(
        manifest,header.scalar_count,len(header.entries),dict(sorted(dtype_counts.items())),
        extra,missing,expected,passed,
    )
