#!/usr/bin/env python3
"""Pinned Chronos adapter forward and forecast smoke; no benchmark targets."""
from pathlib import Path
import torch
import torch.nn.functional as F
from chronos import ChronosPipeline

MODEL='/home/agent/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/snapshots/29d808298f1a62493e7b9a5e08529d0d930fa189'
torch.set_num_threads(1)
pipe=ChronosPipeline.from_pretrained(MODEL,device_map='cpu',torch_dtype=torch.float32)
model=pipe.model.model; layer=model.lm_head
A=torch.nn.Parameter(torch.randn(2,256)*1e-3); B=torch.nn.Parameter(torch.randn(4096,2)*1e-3); theta=torch.nn.Parameter(torch.zeros(()))

def hook(_module,args,out):
 h=args[0]; z=F.linear(h,A); c=torch.cos(theta); s=torch.sin(theta)
 zr=torch.stack((c*z[...,0]-s*z[...,1],s*z[...,0]+c*z[...,1]),dim=-1)
 return out+F.linear(zr,B)
h=layer.register_forward_hook(hook)
ctx=torch.linspace(45,65,288).reshape(1,-1); y=torch.linspace(62,68,64).reshape(1,-1)
ids,mask,state=pipe.tokenizer.context_input_transform(ctx); labels,_=pipe.tokenizer.label_input_transform(y,state)
out=model(input_ids=ids,attention_mask=mask,labels=labels)
loss=F.cross_entropy(out.logits[:,:12].reshape(-1,4096),labels[:,:12].reshape(-1))
loss.backward()
assert torch.isfinite(loss) and A.grad is not None and B.grad is not None and theta.grad is not None
pred=pipe.predict(ctx,prediction_length=12,num_samples=4)
assert pred.shape==(1,4,12) and torch.isfinite(pred).all()
h.remove()
print({'model_parameters':sum(p.numel() for p in pipe.model.parameters()),'prediction_shape':list(pred.shape),'teacher_forced_loss':float(loss.detach()),'trainable_code_parameters':int(A.numel()+B.numel()+theta.numel())})
