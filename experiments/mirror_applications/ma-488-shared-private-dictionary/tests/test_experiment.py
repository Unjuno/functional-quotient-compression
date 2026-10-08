from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_heterogeneity_sizes_are_fixed():
 assert [round(run.N*p) for p in run.RATES]==[0,24,48,96]

def test_private_residual_gate_recovers_heterogeneous_heldout_rows():
 w=run.world(48801,.125);mirror,_=run.fit(w,'mirror_shared_private');shared,_=run.fit(w,'shared_only')
 assert mirror['private_indices'].numel()==24
 assert run.reconstruct('mirror_shared_private',mirror).sub(w['matrices']).abs().max().item()<1e-6
 assert (run.reconstruct('shared_only',shared)-w['matrices']).norm()>0.1

def test_native_shared_private_is_exact_alias():
 w=run.world(48801,.25);m,_=run.fit(w,'mirror_shared_private');n,_=run.fit(w,'native_shared_private')
 assert all(torch.equal(m[k],n[k]) for k in m)
