"""Isolated CPU update microbenchmark, not an equal-time training experiment."""
import json,time,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from core import decode_model

def main():
    torch.set_num_threads(1)
    root=Path('results/fresh66001');data=torch.load(root/'stream.pt',weights_only=True)
    x,y=data['tokens'][500],data['targets'][500];rows=[]
    for kind in ['dense','moe','lora','shared','hybrid','selected']:
        base=decode_model((root/kind/'step1600.bin').read_bytes());groups=[]
        for rep in range(5):
            m=copy.deepcopy(base);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4)
            elapsed=0.
            for i in range(50):
                start=time.perf_counter();opt.zero_grad(set_to_none=True)
                F.cross_entropy(m(x),y).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
                if i>=10:elapsed+=time.perf_counter()-start
            groups.append(elapsed/40)
        c=m.config;d=c['d'];t=5
        # Forward MAC proxy: linear, attention QK/AV, router and selected matrix multiplies.
        mac=2*(t*4*d*d+2*t*t*d)+t*2*d*(c['ff']+c['final_ff'])+t*d*c['vocab']
        if c['kind'] in ['shared','hybrid']:mac+=t*d*c['atoms']+t*min(c['topk'],c['atoms'])*2*d*c['arank']
        if c['kind'] in ['moe','lora','hybrid']:
            width=c['expert_ff'] if c['kind']=='moe' else c['rank']
            mac+=t*d*c['experts']+t*min(c['topk'],c['experts'])*2*d*width
        rows.append({'method':kind,'seconds_per_update_groups':groups,'median_seconds':float(np.median(groups)),
                     'min_seconds':min(groups),'max_seconds':max(groups),'forward_mac_proxy_per_example':mac})
    result={'torch':torch.__version__,'threads':1,'precision':'FP32','batch':96,'allocated_tokens_per_sequence':5,
            'clock_locked':False,'warmup_updates':10,'timed_updates_per_group':40,'groups':5,
            'notes':['after other training workers exited','not end-to-end equal-time learning','MAC proxy excludes activation, norm, softmax, topk, indexing, backward and optimizer'],
            'rows':rows}
    Path('BENCHMARK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
