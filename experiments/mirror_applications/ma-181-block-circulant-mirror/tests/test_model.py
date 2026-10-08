import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import METHODS,BlockViews,block_matrix,ROLES,D,BLOCK,shift_codes
from run import world,targets

def test_every_inference_payload_restores_exact_outputs():
 x=torch.randn(32,D);r=torch.arange(32)%ROLES
 for method in METHODS:
  torch.manual_seed(18101);m=BlockViews(method,19001);m.eval();data=m.serialize();restored=BlockViews.from_serialized(data);restored.eval()
  assert torch.equal(m(x,r),restored(x,r));assert len(data)==m.serialized_payload_bytes()

def test_shifted_block_circulants_preserve_structure_and_code_range():
 k=torch.randn(D//BLOCK,D//BLOCK,BLOCK);s=shift_codes(18191);w=block_matrix(k,s[0])
 for ob in range(D//BLOCK):
  for ib in range(D//BLOCK):
   q=w[ob*BLOCK:(ob+1)*BLOCK,ib*BLOCK:(ib+1)*BLOCK]
   for row in range(BLOCK):assert torch.allclose(q[row],torch.roll(q[0],row))
 assert int(s.min())>=0 and int(s.max())<BLOCK

def test_block_shift_view_reconstructs_teacher_from_one_physical_bank():
 w=world(18101,'aligned_block_shift');m=BlockViews('block_shift',w['address_seed']);m.eval()
 with torch.no_grad():m.kernels.copy_(w['kernels']);m.bias.copy_(w['biases'][0])
 x=torch.randn(24,D);r=torch.arange(24)%ROLES
 assert torch.max(torch.abs(m(x,r)-targets(x,r,w)))<1e-6
