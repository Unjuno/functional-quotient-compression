from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_rank_eight_replays_aligned_heldout_behaviors():
 w=run.world(85,0.0);obj,_=run.objects('shared_mirror',w,8)
 assert run.score('shared_mirror',obj,w)['heldout_probe_relative_rmse']<1e-4

def test_native_pca_alias_and_codes_causally_change_hidden_outputs():
 w=run.world(86,0.0);obj,_=run.objects('shared_mirror',w,8)
 assert run.pack('shared_mirror',obj,8)==run.pack('native_pca',obj,8)
 assert float(obj['codes'][24].abs().max())>0


def test_rank_eight_actual_payload_under_half_explicit_vectors():
 w=run.world(87,0.0);m,_=run.objects('shared_mirror',w,8);e,_=run.objects('explicit_vector',w,8)
 assert len(run.pack('shared_mirror',m,8))<=0.65*len(run.pack('explicit_vector',e,8))
