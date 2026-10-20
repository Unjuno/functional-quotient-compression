"""Isolated round-robin CPU forward benchmark for MA-1166 Phase E; batch sizes 1/128."""
import json, statistics, sys
from pathlib import Path
import torch
from train import ROOT,MODES,make_model,benchmark

def run():
    torch.set_num_threads(1)
    source=ROOT/'checkpoints'
    data=[]
    for mode in MODES:
        f=source/f'aligned_461_{mode}.pt'
        m=make_model(mode,461)
        chk=torch.load(f,map_location='cpu',weights_only=True)
        m.load_state_dict(chk['state_dict'])
        m.eval()
        data.append((mode,m))
    trials=[]
    for k in range(3):
        # reverse traversal on every second repetition limits fixed position bias.
        for mode,m in (data if k%2==0 else data[::-1]):
            bs1=benchmark(m,1,120)
            bs128=benchmark(m,128,35)
            vals=dict(repetition=k,mode=mode,b1_us=bs1['median_us'],b128_us=bs128['median_us'],
                      physical_buffer_bytes=sum(b.numel()*b.element_size() for b in m.buffers()),
                      physical_learned_bytes=sum(p.numel()*p.element_size() for p in m.parameters()))
            trials.append(vals)
            print('AUDIT',k,mode,round(vals['b1_us'],2),round(vals['b128_us'],2),flush=True)
    final=[]
    for mode,_ in data:
        r=[x for x in trials if x['mode']==mode]
        final.append(dict(mode=mode,b1_us=statistics.median(x['b1_us'] for x in r),b128_us=statistics.median(x['b128_us'] for x in r),
                          buffers_B=r[0]['physical_buffer_bytes'],learned_B=r[0]['physical_learned_bytes'],
                          b1_trials=[x['b1_us'] for x in r],b128_trials=[x['b128_us'] for x in r]))
    (ROOT/'results'/'ISOLATED_BENCHMARK.json').write_text(json.dumps(dict(cpu_threads=1,seed=461,world='aligned',reps_b1=120,reps_b128=35,rounds=3,trials=trials,summary=final),indent=2)+'\n')
    return final

if __name__=='__main__':run()
