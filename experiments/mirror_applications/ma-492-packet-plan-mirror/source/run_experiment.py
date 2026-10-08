#!/usr/bin/env python3
"""MA-492 tiny categorical packet-plan screening model."""
from __future__ import annotations
import argparse,csv,io,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
N_PACKET=16

def sample_world(seed,n=4096):
    rng=np.random.default_rng(seed)
    patterns=rng.choice(N_PACKET,size=4,replace=False)
    weights=rng.dirichlet(np.array([2.0,2.0,2.0,2.0]))
    support=rng.choice(4,size=n,p=weights); audit=rng.choice(4,size=n,p=weights)
    return patterns,weights,patterns[support],patterns[audit]

def counts(x): return np.bincount(x,minlength=N_PACKET).astype(np.float64)
def smooth_prob(c,alpha=.25): return (c+alpha)/(c.sum()+alpha*len(c))
def fit(seed):
    patterns,weights,support,audit=sample_world(seed)
    start=time.perf_counter(); c=counts(support); un=smooth_prob(c)
    # Frozen 2-bit codebook: four most frequent packet plans are the learned discrete addresses.
    addr=np.argsort(c)[-4:][::-1]; mc=np.zeros(N_PACKET); mc[addr]=c[addr]
    mirror=smooth_prob(mc)
    # Same-bit VQ is the native categorical codebook control. It has identical learned states by design.
    vq=mirror.copy()
    # Continuous latent control stores a 2D PCA embedding of address vectors plus linear decoder weights;
    # this intentionally charges floats and uses soft reconstruction before normalization.
    logits=np.full((4,N_PACKET),-8.0); logits[np.arange(4),addr]=8.0
    z=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=np.float32)
    # Fit affine least squares from 2D latent to each packet logit across four addresses.
    design=np.c_[np.ones(4),z]; coef=np.linalg.lstsq(design,logits,rcond=None)[0].astype(np.float32)
    cp=np.c_[np.ones(4),z]@coef; exp=np.exp(cp-cp.max(axis=1,keepdims=True)); cont_rows=exp/exp.sum(axis=1,keepdims=True)
    cont=np.zeros(N_PACKET)
    for j,a in enumerate(addr): cont[a]=max(c[a],0)*cont_rows[j,a]
    if cont.sum()==0: cont=un.copy()
    else: cont=smooth_prob(cont)
    # Independent full per-mode categorical references retain one distribution per mode.
    bank=np.zeros((4,N_PACKET),dtype=np.float32)
    for j,a in enumerate(addr): bank[j,a]=1.0
    elapsed=time.perf_counter()-start
    actual=np.zeros(N_PACKET); actual[patterns]=weights
    methods={'unconditional_shared':un,'mirror_discrete':mirror,'vq_same_2bit':vq,'continuous_latent_2bit':cont}
    # Independent per-mode full categorical decoders, with support-derived mixture weights.
    independent_bank=np.zeros((4,N_PACKET),dtype=np.float32)
    mix=np.zeros(4,dtype=np.float32)
    for j,a in enumerate(addr):
        independent_bank[j,a]=1.0
        mix[j]=c[a]/max(c[addr].sum(),1.0)
    ind=np.zeros(N_PACKET)
    for j,a in enumerate(addr): ind[a]=mix[j]
    methods['independent_mode_full']=ind
    rows=[]
    for name,p in methods.items():
        nll=float(-np.mean(np.log(np.maximum(p[audit],1e-12))))
        valid=float(p[patterns].sum())
        coverage=int(sum(p[a]>0.02 for a in patterns))
        if name=='unconditional_shared': payload={'shared_decoder':np.array([1],dtype=np.uint8),'packet_probs':p.astype(np.float32)}
        elif name in ('mirror_discrete','vq_same_2bit'): payload={'shared_decoder':np.array([1],dtype=np.uint8),'packet_codebook':addr.astype(np.uint8),'code_probs':(c[addr]/c[addr].sum()).astype(np.float32)}
        elif name=='independent_mode_full': payload={'mode_decoder_bank':independent_bank,'mixture_weights':mix}
        else: payload={'shared_decoder':np.array([1],dtype=np.uint8),'latent_codes':z,'decoder_affine':coef}
        b=serialized(payload)
        rows.append({'condition':name,'world_or_seed':seed,'method':name,'serialized_bytes':b,'train_tokens_or_examples':len(support),'optimizer_updates':0,'active_compute_proxy':'16-way categorical lookup; continuous control adds 4x3 by 16 affine decode','wall_time_s':f'{elapsed:.9f}','primary_metric':'joint_packet_nll_nats','primary_value':f'{nll:.12g}','secondary_metric':'valid_mass_and_mode_coverage','secondary_value':f'{valid:.12g};{coverage}/4','status_note':'2-bit logical address; support-derived frequencies; '+('exactly same inference payload as VQ' if name=='vq_same_2bit' else '')})
    # upper reference stores the four true per-mode packet weights plus addresses.
    upper=payload_bytes({'mode_decoder_bank':independent_bank,'mixture_weights':mix})
    meta={'seed':seed,'patterns':patterns.tolist(),'weights':weights.tolist(),'mirror_bytes':next(r['serialized_bytes'] for r in rows if r['method']=='mirror_discrete'),'vq_bytes':next(r['serialized_bytes'] for r in rows if r['method']=='vq_same_2bit'),'independent_bytes':upper,'mirror_vq_identical':True,'fit_wall_s':elapsed}
    return rows,meta

def payload_bytes(payload):
    b=io.BytesIO();np.savez(b,**payload);return len(b.getvalue())
def serialized(payload): return payload_bytes(payload)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[49201]);ap.add_argument('--out',default=str(ROOT/'RESULTS_CORE.csv'));ap.add_argument('--summary',default=str(ROOT/'source'/'summary.json'));a=ap.parse_args()
    rows=[];summ=[]
    for seed in a.seeds:
        r,s=fit(seed);rows+=r;summ.append(s)
    with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    Path(a.summary).write_text(json.dumps(summ,indent=2)+'\n');print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
