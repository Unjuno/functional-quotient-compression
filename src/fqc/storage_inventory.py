"""Exact baseline storage accounting helpers for real-model extraction.

D57's `storage_group` is safe for exact aliases. This module derives alias
identity from concrete storage ranges and explicitly detects partial overlap,
which cannot be represented by the simple equal-size group convention without
additional range accounting.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Mapping, Any

@dataclass(frozen=True)
class StorageSlice:
    tensor_id: str
    storage_key: str
    offset: int
    length: int
    dtype: str
    element_size: int
    @property
    def end(self) -> int:
        return self.offset+self.length

@dataclass(frozen=True)
class StorageInventory:
    unique_scalar_count: int
    exact_alias_group: Mapping[str,str]
    partial_overlaps: tuple[tuple[str,str],...]


def _validate(s: StorageSlice):
    if not s.tensor_id or not s.storage_key: raise ValueError('tensor_id and storage_key are required')
    if s.offset<0 or s.length<0 or s.element_size<=0: raise ValueError('invalid storage range')


def analyze_storage_slices(slices: Iterable[StorageSlice], *, reject_partial_overlap: bool=True) -> StorageInventory:
    xs=tuple(slices)
    for s in xs: _validate(s)
    by_storage=defaultdict(list)
    for s in xs: by_storage[s.storage_key].append(s)
    groups={}; partial=[]; total=0
    for key,items in by_storage.items():
        dtypes={(s.dtype,s.element_size) for s in items}
        if len(dtypes)!=1:
            raise ValueError(f'storage {key}: mixed dtype/element-size views are unsupported')
        interval_groups=defaultdict(list)
        for s in items: interval_groups[(s.offset,s.end)].append(s)
        for (a,b),members in interval_groups.items():
            gid=f'{key}:{a}:{b}'
            for s in members: groups[s.tensor_id]=gid
        intervals=sorted((s.offset,s.end,s.tensor_id) for s in items)
        for i in range(len(intervals)):
            a0,a1,aid=intervals[i]
            for j in range(i+1,len(intervals)):
                b0,b1,bid=intervals[j]
                if b0>=a1: break
                if (a0,a1)!=(b0,b1): partial.append(tuple(sorted((aid,bid))))
        merged=[]
        for a,b,_ in intervals:
            if not merged or a>merged[-1][1]: merged.append([a,b])
            else: merged[-1][1]=max(merged[-1][1],b)
        total += sum(b-a for a,b in merged)
    partial=tuple(sorted(set(partial)))
    if partial and reject_partial_overlap:
        raise ValueError('partial storage overlap requires explicit range accounting: '+', '.join(f'{a}<->{b}' for a,b in partial))
    return StorageInventory(total,groups,partial)


def torch_like_storage_slice(tensor_id: str, tensor: Any) -> StorageSlice:
    """Extract a contiguous storage range from a PyTorch-like tensor object.

    No torch dependency is imported; the object must expose the usual tensor
    methods/attributes. Noncontiguous views are rejected because their exact
    scalar coverage is not one interval.
    """
    if not bool(tensor.is_contiguous()):
        raise ValueError(f'{tensor_id}: noncontiguous tensor requires explicit index accounting')
    storage=tensor.untyped_storage()
    element_size=int(tensor.element_size())
    offset=int(tensor.storage_offset())
    length=int(tensor.numel())
    dtype=str(tensor.dtype)
    device=str(tensor.device)
    key=f'{device}:{int(storage.data_ptr())}:{int(storage.nbytes())}'
    if (offset+length)*element_size > int(storage.nbytes()):
        raise ValueError(f'{tensor_id}: tensor range exceeds underlying storage')
    return StorageSlice(tensor_id,key,offset,length,dtype,element_size)
