"""MA-961 deterministic fixed-synapse LIF threshold-code screen."""
import csv,json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[96121,96123,96127];TASKS=16;NTEST=4096;STEPS=8;LEAK=.9;BIAS=1.5;W=1.0

def simulate(x,threshold):
    """One scalar LIF neuron, one spike count per input, exact hard threshold dynamics."""
    v=np.zeros_like(x,dtype=np.float64);count=np.zeros_like(x,dtype=np.int32)
    cur=W*x+BIAS
    for _ in range(STEPS):
        v=LEAK*v+cur
        fired=v>=threshold
        count+=fired
        v=np.where(fired,0.0,v)
    return count

def quantize4(theta):
    lo=float(theta.min());hi=float(theta.max());q=np.rint((theta-lo)/(hi-lo)*15).astype(np.uint8)
    recon=lo+q.astype(np.float64)/15*(hi-lo)
    return q,lo,hi,recon

def payload_bytes(name,state):
    p=ROOT/'source'/f'.{name}.pt';torch.save(state,p);n=p.stat().st_size
    loaded=torch.load(p,map_location='cpu',weights_only=True)
    for key,value in state.items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,loaded[key]),key
        else:assert value==loaded[key],key
    p.unlink();return n

def main():
    rows=[];raw=[];leaky_sum=sum(LEAK**t for t in range(STEPS))
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        # Sorted but nonlinear boundaries create a smooth task-index trend and a private residual.
        theta_decision=np.sort(rng.uniform(-1.25,1.25,TASKS))
        theta= (BIAS+theta_decision)*leaky_sum
        q,lo,hi,qtheta=quantize4(theta)
        idx=np.arange(TASKS,dtype=np.float64);slope=float(np.polyfit(idx,theta,1)[0]);base=float(theta.mean()-slope*idx.mean())
        affine=base+slope*idx; residual=theta-affine
        states={
          'native_fp32':{'synapse':torch.tensor([W]),'thresholds':torch.tensor(theta,dtype=torch.float32),'metadata':{'steps':STEPS,'leak':LEAK,'bias':BIAS}},
          'direct_int4':{'synapse':torch.tensor([W]),'threshold_codes':torch.tensor(q),'range':torch.tensor([lo,hi]),'metadata':{'steps':STEPS,'leak':LEAK,'bias':BIAS,'bits':4}},
          'mirror_affine':{'synapse':torch.tensor([W]),'base':torch.tensor([base]),'step':torch.tensor([slope]),'task_index':torch.arange(TASKS,dtype=torch.uint8),'metadata':{'steps':STEPS,'leak':LEAK,'bias':BIAS}},
          'mirror_private_residual':{'synapse':torch.tensor([W]),'base':torch.tensor([base]),'step':torch.tensor([slope]),'residual':torch.tensor(residual,dtype=torch.float32),'metadata':{'steps':STEPS,'leak':LEAK,'bias':BIAS}}
        }
        bytes_by={k:payload_bytes(k+str(seed),v) for k,v in states.items()}
        for task,c in enumerate(theta_decision):
            x=rng.normal(0,1,NTEST);truth=x>c
            preds={}
            for name,recon in [('native_fp32',theta),('direct_int4',qtheta),('mirror_affine',affine),('mirror_private_residual',theta)]:
                nsp=simulate(x,(BIAS+recon[task]/leaky_sum)*leaky_sum if False else recon[task])
                pred=nsp>0;preds[name]=(pred,nsp)
            for name,(pred,nsp) in preds.items():
                row={'seed':seed,'task_id':task,'condition':name,'threshold_decision':float(c),'accuracy':float(np.mean(pred==truth)),
                     'mean_spikes_per_example':float(np.mean(nsp)),'spikes_per_1000_examples':float(np.sum(nsp)/len(nsp)*1000),
                     'payload_bytes':bytes_by[name],'test_examples':NTEST,'lif_steps':STEPS}
                rows.append(row)
        raw.append({'seed':seed,'decision_thresholds':theta_decision.tolist(),'physical_thresholds':theta.tolist(),
                    'affine_base':base,'affine_step':slope,'private_residual':residual.tolist(),'quantized_codes':q.tolist(),
                    'quantized_min':lo,'quantized_max':hi})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'source/results.json').write_text(json.dumps({'rows':rows,'worlds':raw,'seeds':SEEDS,'tasks':TASKS,'steps':STEPS,
      'leak':LEAK,'bias':BIAS,'synaptic_weight':W,'device':'cpu'},indent=2)+'\n')
    for seed in SEEDS:
        rr=[x for x in rows if x['seed']==seed]
        summary={'seed':seed,'conditions':{c:{'mean_accuracy':float(np.mean([x['accuracy'] for x in rr if x['condition']==c])),
          'mean_spikes':float(np.mean([x['mean_spikes_per_example'] for x in rr if x['condition']==c])),
          'payload_bytes':rr[[x['condition'] for x in rr].index(c)]['payload_bytes']} for c in sorted({x['condition'] for x in rr})}}
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
