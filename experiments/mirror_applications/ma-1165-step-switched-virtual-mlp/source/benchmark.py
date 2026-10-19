"""Isolated CPU FP32 eager batch-1 forward timings; virtual weights regenerated."""
from __future__ import annotations
import csv,json,statistics,time,platform
from pathlib import Path
import torch
from model import VirtualLoop

if __name__=='__main__':
    torch.set_num_threads(1)
    root=Path(__file__).parents[1]
    csv_path=root/'results'/'RESULTS_CORE.csv'
    rows=[]
    with csv_path.open() as f:table=list(csv.DictReader(f))
    for loops in (2,4):
        for mode in ('tied','same','switch','linear','gain','untied'):
            seed=31
            ck=root/'checkpoints'/f'independent_{mode}_L{loops}_s{seed}.pt'
            state=torch.load(ck,map_location='cpu',weights_only=True)
            model=VirtualLoop(mode,loops,seed=seed+1200)
            model.load_state_dict(state['model']);model.eval()
            x=torch.randn(1,8,generator=torch.Generator().manual_seed(7113))
            with torch.no_grad():
                for _ in range(20): model(x)
                samples=[]
                for _ in range(15):
                    t=time.perf_counter()
                    for j in range(40):model(x)
                    samples.append((time.perf_counter()-t)*1e6/40)
            rows.append({'mode':mode,'loops':loops,'p50_us':statistics.median(samples),
                         'min_us':min(samples),'max_us':max(samples),
                         'inference_bytes':model.payload_bytes(),
                         'hardware':platform.processor() or 'see /proc/cpuinfo',
                         'torch':torch.__version__,'threads':torch.get_num_threads(),'dtype':'FP32','batch':1,'cpu_clock_locked':False})
    with (root/'results'/'BENCHMARK.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('benchmark completed',len(rows))
