from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_orthogonal_codes_have_known_pairwise_distance():
 c,_,_=run.codebook('random_orthogonal',49801);d=torch.cdist(c,c);d.fill_diagonal_(99);assert abs(float(d.min())-2**.5)<1e-5

def test_raw_id_and_nearest_code_decode_clean_addresses():
 c=run.codebook('raw_id',49801)[0];ids=torch.arange(run.E);base=((ids[:,None]>>torch.arange(3))&1).float()*2-1;z=torch.zeros(run.E,run.E);z[:,:3]=base/(3**.5);assert torch.equal(run.decode('raw_id',z,None),ids)
 c,_,_=run.codebook('random_orthogonal',49801);assert torch.equal(run.decode('random_orthogonal',c,c),ids)

def test_native_metric_control_uses_same_regularized_state():
 c,_,_=run.codebook('mirror_distance_reg8',49801);n,_,_=run.codebook('native_metric8',49801);assert torch.equal(c,n)
