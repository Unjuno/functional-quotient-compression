import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import experiment as ma

def test_world_generation_is_deterministic_and_shape_valid():
    a=ma.make_world(112001,True);b=ma.make_world(112001,True)
    assert (a[1]==b[1]).all() and (a[2]==b[2]).all()
    assert a[1].shape==(6,12,8)

def test_audit_seeds_frozen_and_chronological():
    d=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert d['fresh']['worlds_or_seeds']==[112101,112102,112103]
    assert d['fresh']['locked_before_access'] is True
    assert '0..7' in d['fresh']['split'] and '10..11' in d['fresh']['split']

def test_mirror_is_same_function_family_as_tucker_control():
    assert sum(p.numel() for p in ma.TemporalScorer('mirror').parameters())==sum(p.numel() for p in ma.TemporalScorer('tucker').parameters())
    assert not any('left' in k or 'right' in k for k in ma.TemporalScorer('mirror').state_dict())

def test_actual_payload_is_safetensors_and_hashable(tmp_path):
    model=ma.TemporalScorer('mirror');path=tmp_path/'payload.safetensors'
    n,digest=ma.payload(model,path)
    assert n==path.stat().st_size and len(digest)==64
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest

def test_draw33_rejection_sampling_replay():
    d=json.loads((ROOT/'source'/'draw33_exclusions.json').read_text())
    pool=d['pool_ids'];seed=bytes.fromhex(d['seed_hex']);pool_hash=bytes.fromhex(d['pool_sha256'])
    assert hashlib.sha256('\n'.join(pool).encode()).digest()==pool_hash
    v=int.from_bytes(hashlib.sha256(seed+pool_hash+d['rejection_counter'].to_bytes(4,'big')).digest(),'big')
    limit=(1<<256)-((1<<256)%len(pool))
    assert v<limit and v%len(pool)==d['selection_index_zero_based']
    assert pool[v%len(pool)]==d['selected_id']=='MA-1120'

def test_payload_contains_only_scoring_state(tmp_path):
    from safetensors.torch import load_file
    model=ma.TemporalScorer('mirror');path=tmp_path/'mirror.safetensors'
    size,digest=ma.payload(model,path);stored=load_file(str(path))
    assert set(stored)==set(model.state_dict())
    assert size==path.stat().st_size and hashlib.sha256(path.read_bytes()).hexdigest()==digest
