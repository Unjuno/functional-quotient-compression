import pytest
from fqc.storage_inventory import StorageSlice, analyze_storage_slices, torch_like_storage_slice


def test_exact_tied_alias_is_counted_once():
    xs=[StorageSlice('embed','S',0,40,'bf16',2),StorageSlice('lm_head','S',0,40,'bf16',2),StorageSlice('other','T',0,16,'bf16',2)]
    inv=analyze_storage_slices(xs)
    assert inv.unique_scalar_count==56
    assert inv.exact_alias_group['embed']==inv.exact_alias_group['lm_head']
    assert not inv.partial_overlaps


def test_partial_overlap_is_not_silently_treated_as_tying():
    xs=[StorageSlice('a','S',0,10,'f32',4),StorageSlice('b','S',5,10,'f32',4)]
    with pytest.raises(ValueError,match='partial storage overlap'):
        analyze_storage_slices(xs)
    inv=analyze_storage_slices(xs,reject_partial_overlap=False)
    assert inv.unique_scalar_count==15
    assert inv.partial_overlaps==(('a','b'),)


def test_mixed_dtype_views_of_same_storage_are_rejected():
    xs=[StorageSlice('a','S',0,10,'f32',4),StorageSlice('b','S',0,20,'bf16',2)]
    with pytest.raises(ValueError,match='mixed dtype'):
        analyze_storage_slices(xs)


class _Storage:
    def data_ptr(self): return 1234
    def nbytes(self): return 200
class _Tensor:
    dtype='bf16'; device='cpu'
    def is_contiguous(self): return True
    def untyped_storage(self): return _Storage()
    def element_size(self): return 2
    def storage_offset(self): return 10
    def numel(self): return 40

def test_torch_like_slice_uses_real_storage_range_fields():
    s=torch_like_storage_slice('x',_Tensor())
    assert (s.offset,s.length,s.element_size)==(10,40,2)
    assert s.storage_key=='cpu:1234:200'


def test_public_alias_group_ids_do_not_depend_on_runtime_storage_pointer():
    a=[StorageSlice('embed','pointer-A',0,40,'bf16',2),StorageSlice('lm_head','pointer-A',0,40,'bf16',2)]
    b=[StorageSlice('embed','pointer-B',0,40,'bf16',2),StorageSlice('lm_head','pointer-B',0,40,'bf16',2)]
    ia=analyze_storage_slices(a); ib=analyze_storage_slices(b)
    assert ia.exact_alias_group==ib.exact_alias_group
    assert ia.exact_alias_group['embed']=='sg000000'
