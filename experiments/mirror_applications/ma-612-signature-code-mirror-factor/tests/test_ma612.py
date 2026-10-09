import sys,io
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import payload,decode,PHASE,FREQ,PAIRS

def test_factorized_codes_reconstruct_all_pairings():
    x=torch.linspace(-2,2,17);ref=torch.stack([torch.sin(FREQ[s]*x+PHASE[p]) for s,p in PAIRS])
    for m in ('unfactorized','factor_coeff','mirror'):
        y,r=decode(payload(m,1),x);assert torch.allclose(y,ref,atol=1e-6,rtol=1e-6);assert len(r)==64

def test_payload_roundtrip_is_deterministic():
    a=payload('mirror',2);d=torch.load(io.BytesIO(a),weights_only=False);b=payload('mirror',2)
    assert a==b and d['state']['pair_refs'].shape==(64,2)
