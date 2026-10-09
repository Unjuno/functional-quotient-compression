import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import ellipse_sdf,geometry_codes,Decoder,mirror_eval,SHAPES

def test_ellipse_sdf_sign_and_surface():
 p=torch.tensor([[0.,0.,0.,0.,0.]])
 x=torch.tensor([[[0.,0.],[1.,0.],[1.2,0.]]])
 d=ellipse_sdf(x,p)
 assert d[0,0]<0 and abs(float(d[0,1]))<1e-5 and d[0,2]>0

def test_geometry_latent_dimension():
 p=geometry_codes(torch.Generator().manual_seed(1),SHAPES)
 assert p.shape==(SHAPES,5)

def test_native_warp_same_as_mirror_interface():
 torch.manual_seed(2);m=Decoder(2);codes=torch.randn(3,5)*.1;x=torch.randn(3,20,2)
 a=mirror_eval(m,codes,x);b=mirror_eval(m,codes,x.clone())
 assert torch.max(torch.abs(a-b))==0
