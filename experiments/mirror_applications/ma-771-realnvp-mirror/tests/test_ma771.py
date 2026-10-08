import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import experiment as ma

def test_realnvp_view_inverse_and_logdet():
    torch.manual_seed(771)
    x=torch.randn(128,2);m=torch.tensor([.27,-.41]);a=torch.tensor(.7);b=torch.tensor(.4)
    y=ma.view(x,a,b,m)
    assert (ma.view_inv(y,a,b,m)-x).abs().max()<1e-6
    assert (ma.view_logdet(x,a,b,m)+ma.view_logdet(y,a,b,-m)).abs().max()<1e-6

def test_draw34_rejection_sampling_replay():
    d=json.loads((ROOT/'source'/'draw34_exclusions.json').read_text());pool=d['pool_ids'];seed=bytes.fromhex(d['seed_hex']);ph=bytes.fromhex(d['pool_sha256'])
    assert hashlib.sha256('\n'.join(pool).encode()).digest()==ph
    h=hashlib.sha256(seed+ph+d['rejection_counter'].to_bytes(4,'big')).digest();v=int.from_bytes(h,'big')
    limit=(1<<256)-((1<<256)%len(pool))
    assert v<limit and v%len(pool)==d['selection_index_zero_based']
    assert pool[v%len(pool)]==d['selected_id']=='MA-771'

def test_fresh_seeds_locked_and_distinct_from_development():
    d=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert d['fresh']['locked_before_access'] is True
    assert d['fresh']['worlds_or_seeds']==[77111,77112,77113]
    assert not set(d['fresh']['worlds_or_seeds'])&set(d['development']['worlds_or_seeds'])

def test_audit_results_meet_registered_mechanism_gate():
    d=json.loads((ROOT/'source'/'audit_results.json').read_text())
    assert len(d)==90
    mirrors=[r for r in d if r['method']=='mirror']
    assert len(mirrors)==18 and max(r['query_mse'] for r in mirrors)<1e-3
    assert max(r['inverse_cycle_max'] for r in mirrors)<1e-6
    assert max(r['logdet_inverse_error'] for r in mirrors)<1e-6
    assert len({r['serialized_bytes'] for r in mirrors})==1

def test_payload_bytes_are_actual_safetensors(tmp_path):
    from safetensors.torch import load_file
    code=torch.zeros(2);states=[{'code':code.clone()} for _ in range(6)]
    path=tmp_path/'views.safetensors';size,digest=ma.payload('mirror',(.7,.4),states,path)
    assert size==path.stat().st_size and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert len(load_file(str(path)))==8
