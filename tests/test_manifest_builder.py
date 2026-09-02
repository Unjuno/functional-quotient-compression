import pytest

from fqc.manifest_builder import (
    build_pilot_contract,
    build_transformer_extraction_manifest,
    canonical_json_bytes,
    contiguous_gqa_map,
    sha256_json,
)
from fqc.contracts import validate_pilot_contract, validate_transformer_extraction


class _Storage:
    def __init__(self, ptr, nbytes=256): self._ptr=ptr; self._nbytes=nbytes
    def data_ptr(self): return self._ptr
    def nbytes(self): return self._nbytes


class _Tensor:
    device='cpu'
    def __init__(self, shape, dtype='bf16', ptr=1, offset=0, element_size=2, nbytes=256):
        self.shape=shape; self.dtype=dtype; self._storage=_Storage(ptr,nbytes)
        self._offset=offset; self._element_size=element_size
    def is_contiguous(self): return True
    def untyped_storage(self): return self._storage
    def element_size(self): return self._element_size
    def storage_offset(self): return self._offset
    def numel(self):
        n=1
        for d in self.shape: n*=d
        return n


def _fixture(ptr=100):
    tensors={
        'embed':_Tensor((10,4),ptr=ptr,nbytes=80),
        'lm_head':_Tensor((10,4),ptr=ptr,nbytes=80),
        'WQ':_Tensor((4,4),ptr=ptr+1,nbytes=32),
        'WK':_Tensor((4,2),ptr=ptr+2,nbytes=16),
        'WV':_Tensor((4,2),ptr=ptr+3,nbytes=16),
        'WO':_Tensor((4,4),ptr=ptr+4,nbytes=32),
    }
    roles={name:name for name in tensors}
    attention=[{
        'module_id':'L0.attn','q_heads':2,'kv_heads':1,'head_dim':2,
        'q_to_kv_map':[0,0],
        'projection_tensor_ids':{'WQ':'WQ','WK':'WK','WV':'WV','WO':'WO'},
        'position_operator':'rope','rope':{'rotary_dim':2},'qk_norm':'none',
        'exact_reduction_policy':'rope_profile_operator_family',
    }]
    return tensors,roles,attention


def _manifest(ptr=100):
    tensors,roles,attention=_fixture(ptr)
    return build_transformer_extraction_manifest(
        model_identity={'architecture_family':'test','checkpoint_sha256':'sha256:abc'},
        named_tensors=tensors,tensor_roles=roles,attention_modules=attention,
    )


def test_generated_manifest_derives_tied_storage_and_N():
    m=_manifest()
    result=validate_transformer_extraction(m)
    assert result.valid
    assert result.unique_baseline_scalar_count==88
    groups={t['tensor_id']:t['storage_group'] for t in m['tensor_inventory']}
    assert groups['embed']==groups['lm_head']
    assert m['model_identity']['unique_baseline_scalar_count_N']==88
    assert m['extraction_policy']['storage_alias_evidence']=='derived_from_live_checkpoint_storage_ranges'


def test_generated_manifest_is_pointer_independent():
    a=_manifest(100); b=_manifest(900)
    assert canonical_json_bytes(a)==canonical_json_bytes(b)
    assert sha256_json(a)==sha256_json(b)


def test_builder_rejects_partial_overlap():
    tensors,roles,attention=_fixture()
    tensors['embed']=_Tensor((10,),ptr=42,offset=0,nbytes=40)
    tensors['lm_head']=_Tensor((10,),ptr=42,offset=5,nbytes=40)
    with pytest.raises(ValueError,match='partial storage overlap'):
        build_transformer_extraction_manifest(
            model_identity={'architecture_family':'test'},named_tensors=tensors,
            tensor_roles=roles,attention_modules=attention,
        )


def test_pilot_contract_uses_generated_baseline_and_artifact_hashes():
    extraction=_manifest()
    atoms={'atoms':[{'atom_id':'payload','class':'PAID','bits':20,'dependencies':[]}], 'required_decode_outputs':['payload']}
    c=build_pilot_contract(
        model_id='m',checkpoint_hash='sha256:abc',extraction_manifest=extraction,
        operator_manifest={'ops':[]},paid_atom_manifest=atoms,
        quality_contract={'metric':'x'},replay_contract={'seed':0},metric_id='metric',quality_target=1.0,
    )
    assert c['model']['unique_paid_scalar_count_N']==88
    assert c['baseline']['baseline_bits']==1408
    assert c['target']['B64_integer_bits']==22
    assert all(v.startswith('sha256:') for v in c['artifact_hashes'].values())
    result=validate_pilot_contract(c,atoms)
    assert result.valid
    assert result.required_paid_atom_bits==20


def test_pilot_builder_rejects_checkpoint_hash_mismatch():
    with pytest.raises(ValueError,match='checkpoint hash mismatch'):
        build_pilot_contract(
            model_id='m',checkpoint_hash='sha256:different',extraction_manifest=_manifest(),
            operator_manifest={},paid_atom_manifest={'atoms':[],'required_decode_outputs':[]},
            quality_contract={},replay_contract={},metric_id='m',quality_target=1.0,
        )


def test_contiguous_gqa_helper_is_explicit_and_exact():
    assert contiguous_gqa_map(8,2)==[0,0,0,0,1,1,1,1]
    with pytest.raises(ValueError): contiguous_gqa_map(7,2)


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json_bytes({'x':float('nan')})
