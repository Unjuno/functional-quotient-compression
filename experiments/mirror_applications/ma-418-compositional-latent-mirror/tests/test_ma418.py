import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import SharedDecoder,make_world,O,S,D,FactorModel

def test_decoder_and_factor_shapes():
 dec=SharedDecoder();x=torch.randn(12,2);z=torch.randn(12,16)
 assert dec(x,z).shape==(12,)
 model=FactorModel();ids=torch.tensor([[0,1,2],[7,7,3]])
 assert model.latent(ids).shape==(2,16)

def test_unseen_split_retains_each_factor_level():
 pairs,z,tr,ho=make_world(41801)
 assert len(tr)+len(ho)==O*S*D
 for k in range(3):assert pairs[tr,k].unique().numel()==pairs[:,k].unique().numel()
