"""Synthetic-shape smoke test for official Moirai physical context/horizon settings."""
import torch
from huggingface_hub import hf_hub_download
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

REPO='Salesforce/moirai-1.1-R-small'; REV='0c24ab99db2c1a70ea2a0fc03bf113329772ac64'; CACHE='/tmp/ma1057-hf'
module=MoiraiModule.from_pretrained(REPO,revision=REV,cache_dir=CACHE).eval()
print('model loaded',sum(p.numel() for p in module.parameters()))
for ctx,h in [(288,12),(288,288),(96,4),(96,96)]:
    pred=MoiraiForecast(prediction_length=h,target_dim=1,feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,context_length=ctx,module=module,patch_size=16,num_samples=32).eval()
    x=torch.zeros((1,ctx,1),dtype=torch.float32); observed=torch.ones_like(x,dtype=torch.bool)
    padding=torch.zeros(x.shape[:2],dtype=torch.bool)
    torch.manual_seed(105700+ctx+h)
    with torch.inference_mode(): out=pred(x,observed,padding,num_samples=32)
    print({'context':ctx,'horizon':h,'shape':list(out.shape),'finite':bool(torch.isfinite(out).all())})
