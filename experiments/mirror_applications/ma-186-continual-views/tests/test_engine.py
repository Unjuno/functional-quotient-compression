import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import METHODS,Learner,world

def test_all_methods_serialize_and_restore_identical_predictions():
 w=world(18601,'aligned');base=w['teacher_w'][0];bias=w['teacher_b'][0];x=torch.randn(24,16)
 for method in METHODS:
  m=Learner(method,base,bias,18611)
  if method!='hard_tie':m.acquire(1,w,.003,updates=3,batch=8)
  payload=m.serialize_inference();restored=Learner.from_inference(payload)
  for t in (0,1):assert torch.equal(m.forward(x,t),restored.forward(x,t))
  assert len(payload)==m.inference_bytes()

def test_mirror_old_task_code_is_not_changed_when_new_skill_is_added():
 w=world(18602,'aligned');m=Learner('mirror_code',w['teacher_w'][0],w['teacher_b'][0],18613)
 m.acquire(1,w,.003,updates=3,batch=8);old=m.codes[1].detach().clone();m.acquire(2,w,.003,updates=3,batch=8)
 assert torch.equal(old,m.codes[1])

def test_resume_checkpoint_charges_optimizer_state():
 w=world(18603,'aligned');m=Learner('lora_rank2',w['teacher_w'][0],w['teacher_b'][0],18614);m.acquire(1,w,.003,updates=3,batch=8)
 assert m.resume_bytes()>m.inference_bytes()

def test_incremental_inference_bytes_use_same_codec_baseline():
 w=world(18604,'aligned');m=Learner('mirror_code',w['teacher_w'][0],w['teacher_b'][0],18615)
 base=m.base_payload_bytes();m.acquire(1,w,.003,updates=3,batch=8)
 assert m.inference_bytes()>=base
 assert m.resume_bytes()>=m.resume_base_bytes()
