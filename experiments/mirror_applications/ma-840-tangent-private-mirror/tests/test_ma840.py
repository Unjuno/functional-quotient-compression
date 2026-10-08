import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import experiment as ma

def test_draw35_rejection_sampling_replay():
    d=json.loads((ROOT/'source'/'draw35_exclusions.json').read_text());pool=d['pool_ids'];seed=bytes.fromhex(d['seed_hex']);ph=bytes.fromhex(d['pool_sha256'])
    assert hashlib.sha256('\n'.join(pool).encode()).digest()==ph
    h=hashlib.sha256(seed+ph+d['rejection_counter'].to_bytes(4,'big')).digest();v=int.from_bytes(h,'big');lim=(1<<256)-((1<<256)%len(pool))
    assert v<lim and v%len(pool)==d['selection_index_zero_based'] and pool[v%len(pool)]==d['selected_id']=='MA-840'

def test_fresh_worlds_have_rank2_dev_basis_and_strata():
    for seed in [84011,84012,84013]:
        W,B,dev,tasks=ma.world(seed)
        assert B.shape==(2,128) and np.linalg.matrix_rank(B)==2
        assert len(dev)==4 and sum(not t['hard'] for t in tasks)==4 and sum(t['hard'] for t in tasks)==4

def test_fresh_rows_are_complete_and_stratified():
    rows=json.loads((ROOT/'source'/'audit_results.json').read_text())
    assert len(rows)==72
    for seed in [84011,84012,84013]:
        for stratum in ['low','high']:
            assert len([r for r in rows if r['world']==seed and r['stratum']==stratum])==12

def test_low_residual_tangent_passes_but_private_residual_fails_high_gate():
    rows=json.loads((ROOT/'source'/'audit_results.json').read_text())
    for seed in [84011,84012,84013]:
        r=[x for x in rows if x['world']==seed and x['stratum']=='low' and x['method']=='tangent']
        assert np.mean([x['query_mse'] for x in r])<1e-2
        hi=[x for x in rows if x['world']==seed and x['stratum']=='high']
        tang=np.mean([x['query_mse'] for x in hi if x['method']=='tangent'])
        hybrid=np.mean([x['query_mse'] for x in hi if x['method']=='hybrid'])
        assert hybrid>=.5*tang

def test_full_update_actual_payload_and_hash(tmp_path):
    p=tmp_path/'full.safetensors';state=[{'delta':torch.zeros(8,16)} for _ in range(8)]
    n,digest=ma.ser('full',(np.zeros((8,16),dtype='float32'),np.zeros((2,128),dtype='float32')),state,p)
    assert p.stat().st_size==n and len(digest)==64
