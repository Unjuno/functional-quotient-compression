import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import make_world,rotation,pack,simulate

def test_orthogonal_views_and_world_dimensions():
 f,theta,x0=make_world(42401);q=rotation(theta)
 eye=q.transpose(1,2)@q
 assert torch.max(torch.abs(eye-torch.eye(8)[None]))<1e-6
 assert x0.shape==(32,8) and len(theta)==16

def test_native_generated_control_is_exact_payload_and_trajectory_alias():
 f,theta,x0=make_world(42401);a=pack(f,theta,'mirror');b=pack(f,theta,'native')
 assert a==b
 ya=simulate(a,x0,'mirror');yb=simulate(b,x0,'native')
 assert torch.max(torch.abs(ya-yb))<1e-6
