from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_heldout_grid_is_disjoint_and_axis_connected():
 m=run.mask();assert int(m.sum())==24 and int((~m).sum())==8
 assert bool(m.any(0).all()) and bool(m.any(1).all()) and bool((~m).sum(0).min()>0)

def test_true_rank_two_layer_task_code_reconstructs_all_pairs():
 w=run.world(63,0.0);recon=torch.einsum('lkr,tkr->ltk',w['a_true'],w['b_true'])
 assert torch.allclose(recon,w['codes_true'],atol=1e-7,rtol=1e-6)

def test_mirror_and_native_bilinear_serialization_alias():
 obj={'left':torch.randn(run.D,run.K),'right':torch.randn(run.D,run.K),'layer_factors':torch.randn(run.L,run.K,run.R),'task_factors':torch.randn(run.T,run.K,run.R)}
 assert run.pack('factor_mirror',obj)==run.pack('native_bilinear',obj)
