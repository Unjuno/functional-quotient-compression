"""Measured B=1/B=64 CPU latency for nine outputs, and explicit MAC lower bound.
Unfused eager PyTorch; no GPU/quantization or end-to-end Transformer claims.
"""
from pathlib import Path
import json,csv,statistics,time
import torch
from model import DualSwiGLU
ROOT=Path(__file__).resolve().parents[1]

def bench(fn,x,warm=40,count=250,groups=5):
    readings=[]
    with torch.no_grad():
        for _ in range(groups):
            for j in range(warm):fn(x)
            tic=time.perf_counter_ns()
            for j in range(count):fn(x)
            readings.append((time.perf_counter_ns()-tic)/1e6/count)
    return {'median_ms':statistics.median(readings),'min_ms':min(readings),'max_ms':max(readings)}

def main():
    torch.set_num_threads(1)
    methods=['tied','mirror1','mirror4','mirror12','gain12','pair_mirror4','factor_lora1','independent']
    out=[]
    for b in [1,64,256]:
        x=torch.randn(b,8)
        for method in methods:
            name=f'aligned_full_w73111_s0_{method}_lr0.018'
            record=json.loads((ROOT/'results/fresh/runs'/f'{name}.json').read_text())
            obj=torch.load(ROOT/'results/fresh'/record['model_path'],weights_only=True)
            cfg=obj['config'];model=DualSwiGLU(cfg['method'],cfg['d'],cfg['h'],cfg['K'],cfg['q'])
            model.load_state_dict(obj['state']);model.eval()
            t=bench(model,x)
            out.append({'batch':b,'method':method,'logical_outputs_per_input':9,'bytes':record['serialized_bytes'],**t})
        print('measured batch',b,flush=True)
    with (ROOT/'results/CPU_INFERENCE_BENCH.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print('MACs approx (large dense matrix multiplies only): mirror 2112, independent 5184 per input; transforms excluded')
    for r in out:
        if r['batch']==64:print(r)
if __name__=='__main__':main()
