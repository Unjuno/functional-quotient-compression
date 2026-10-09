from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_basis_is_fit_before_private_values():
    w=run.world(47801);_,_,_,_,flags=run.encode_basis(w)
    assert int(flags.sum())==16
    assert torch.equal(flags,w['private_expected'])

def test_fallback_recovers_private_and_compresses_shared_values():
    w=run.world(47801);s,_,_,flags,_=run.prepare('mirror_fallback',w)
    assert int(flags.sum())==16
    assert torch.max(torch.abs(s['recon']-w['values']))<1e-5
    assert s['private_values'].shape==(16,run.D)

def test_same_native_pca_fallback_payload():
    w=run.world(47801);a,_,_,_,_=run.prepare('mirror_fallback',w);b,_,_,_,_=run.prepare('native_fallback',w)
    assert run.serialize('mirror_fallback',w,a)==run.serialize('native_fallback',w,b)
