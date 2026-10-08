from __future__ import annotations
import argparse, json, struct, time
from pathlib import Path
import numpy as np

D = 128
K = 8
SUPPORT = 8
FREQ = np.linspace(0.5, 1.5, D // 2, dtype=np.float32)


def q_apply(x, angle, inverse=False, frequencies=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2).copy()
    freqs = FREQ if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    a = np.asarray(angle * freqs, dtype=np.float64)
    if inverse: a = -a
    c, s = np.cos(a), np.sin(a)
    x0 = c*x[:, 0] - s*x[:, 1]
    x1 = s*x[:, 0] + c*x[:, 1]
    return np.stack([x0, x1], axis=1).reshape(-1)


def world(seed, condition):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.1, D)
    v = rng.normal(0, 0.15, D)
    sup_angles = 2*np.pi*np.arange(SUPPORT)/SUPPORT + rng.uniform(-0.05, 0.05, SUPPORT)
    eval_angles = 2*np.pi*(np.arange(K)+0.5)/K + rng.uniform(-0.05, 0.05, K)
    if condition == 'aligned_orthogonal_orbit':
        sup = np.stack([q_apply(v, a) for a in sup_angles])
        target = np.stack([q_apply(v, a) for a in eval_angles])
    else:
        sup = rng.normal(0, 0.15, (SUPPORT, D))
        target = rng.normal(0, 0.15, (K, D))
    x = rng.normal(size=(512, D))
    return {'base':base,'v':v,'sup':sup,'target':target,'sup_angles':sup_angles,'angles':eval_angles,'x':x}


def serialize(state):
    out = bytearray(b'MA296v1\0')
    meta = json.dumps(state.get('meta', {}), sort_keys=True, separators=(',', ':')).encode()
    out.extend(struct.pack('<I',len(meta))); out.extend(meta)
    arrs = state['arrays']
    out.extend(struct.pack('<I',len(arrs)))
    for name, value in sorted(arrs.items()):
        a=np.ascontiguousarray(value)
        nb=name.encode(); dt=a.dtype.str.encode()
        out.extend(struct.pack('<H',len(nb))); out.extend(nb)
        out.extend(struct.pack('<H',len(dt))); out.extend(dt)
        out.extend(struct.pack('<B',a.ndim)); out.extend(struct.pack('<'+'I'*a.ndim,*a.shape))
        raw=a.tobytes(); out.extend(struct.pack('<I',len(raw))); out.extend(raw)
    return bytes(out)


def deserialize(blob):
    view=memoryview(blob); pos=8
    (ml,)=struct.unpack_from('<I',view,pos); pos+=4
    meta=json.loads(bytes(view[pos:pos+ml])); pos+=ml
    (n,)=struct.unpack_from('<I',view,pos); pos+=4
    arrays={}
    for _ in range(n):
        (nl,)=struct.unpack_from('<H',view,pos); pos+=2; name=bytes(view[pos:pos+nl]).decode(); pos+=nl
        (dl,)=struct.unpack_from('<H',view,pos); pos+=2; dtype=np.dtype(bytes(view[pos:pos+dl]).decode()); pos+=dl
        (nd,)=struct.unpack_from('<B',view,pos); pos+=1
        shape=struct.unpack_from('<'+'I'*nd,view,pos); pos+=4*nd
        (rl,)=struct.unpack_from('<I',view,pos); pos+=4
        arrays[name]=np.frombuffer(view[pos:pos+rl],dtype=dtype).reshape(shape).copy(); pos+=rl
    if pos != len(view): raise ValueError('trailing payload bytes')
    return {'meta':meta,'arrays':arrays}


def package(method, w, rank=4, psp_seed=296101):
    sup, tg=w['sup'], w['target']; arr={'base':w['base'].astype(np.float32)}
    if method=='independent': arr['task_vectors']=tg.astype(np.float32)
    elif method=='shared_arithmetic': arr['delta']=sup.mean(0).astype(np.float32)
    elif method=='mirror_superposition':
        bound=np.stack([q_apply(d,a,inverse=True) for d,a in zip(sup,w['sup_angles'])])
        arr['bound_sum']=bound.sum(0).astype(np.float32)
        arr['frequencies']=FREQ.copy()
        arr['task_angles']=w['angles'].astype(np.float32)
        arr['support_count']=np.array([SUPPORT],dtype=np.uint8)
    elif method=='direct_orbit':
        seed=np.mean([q_apply(d,a,inverse=True) for d,a in zip(sup,w['sup_angles'])],axis=0)
        arr['orbit_seed']=seed.astype(np.float32); arr['frequencies']=FREQ.copy(); arr['task_angles']=w['angles'].astype(np.float32)
    elif method=='psp_rademacher':
        rng=np.random.default_rng(psp_seed)
        codes=(rng.integers(0,2,size=(SUPPORT+K,D),dtype=np.int8)*2-1).astype(np.int8)
        arr['packed_codes']=np.packbits((codes>0).astype(np.uint8),axis=1,bitorder='little')
        arr['superposed']=np.sum(codes[:SUPPORT].astype(np.float64)*sup,axis=0).astype(np.float32)
        arr['support_count']=np.array([SUPPORT],dtype=np.uint8)
    elif method.startswith('svd'):
        r=int(method[3:]); _,_,vt=np.linalg.svd(sup,full_matrices=False); basis=vt[:r].T
        coeff=tg@basis
        arr['basis']=basis.astype(np.float32); arr['coeff']=coeff.astype(np.float32)
    return {'meta':{'method':method,'dimension':D,'tasks':K,'support_tasks':SUPPORT,'rank':rank,'psp_seed':psp_seed},'arrays':arr}


def reconstruct(pkg, w, method):
    a=pkg['arrays']
    if method=='independent': return a['task_vectors'].astype(float)
    if method=='shared_arithmetic': return np.tile(a['delta'].astype(float),(K,1))
    if method=='mirror_superposition':
        s=a['bound_sum'].astype(float)/float(a['support_count'][0])
        return np.stack([q_apply(s,ang,frequencies=a['frequencies']) for ang in a['task_angles']])
    if method=='direct_orbit': return np.stack([q_apply(a['orbit_seed'],ang,frequencies=a['frequencies']) for ang in a['task_angles']])
    if method=='psp_rademacher':
        bits=np.unpackbits(a['packed_codes'],axis=1,count=D,bitorder='little')
        codes=(bits.astype(np.int8)*2-1).astype(float)
        return codes[SUPPORT:].astype(float)*a['superposed'].astype(float)
    if method.startswith('svd'):
        return a['coeff'].astype(float) @ a['basis'].astype(float).T
    raise ValueError(method)


def measure(w, pred):
    x=w['x']; target=w['target']; base=w['base']
    task=[]
    for i in range(K):
        yt=x@(base+target[i]); yp=x@(base+pred[i])
        task.append(float(np.mean((yt-yp)**2)/np.mean((yt-x@base)**2)))
    pair=[]
    for i in range(K):
        j=(i+1)%K
        for sign in (1.0,-1.0):
            actual=target[i]+sign*target[j]; estimate=pred[i]+sign*pred[j]
            ya=x@actual; yp=x@estimate
            pair.append(float(np.mean((ya-yp)**2)/np.mean(ya**2)))
    return float(np.mean(task)),float(np.mean(pair)),task,pair


def run_world(seed, condition, svd_rank=4, psp_seed=296101):
    w=world(seed,condition); rows=[]
    methods=['independent','shared_arithmetic','psp_rademacher',f'svd{svd_rank}','direct_orbit','mirror_superposition']
    for m in methods:
        t0=time.perf_counter(); pkg=package(m,w,svd_rank,psp_seed); blob=serialize(pkg); packed=deserialize(blob); prep=time.perf_counter()-t0
        pred=reconstruct(packed,w,m)
        tm,pm,per_task,per_pair=measure(w,pred)
        t0=time.perf_counter()
        for _ in range(10): reconstruct(packed,w,m)
        inf_s=time.perf_counter()-t0
        rows.append({'condition':condition,'world_or_seed':seed,'method':m,'serialized_bytes':len(blob),'support_task_vectors_seen':SUPPORT,'evaluation_task_vectors':K,'heldout_examples_per_task':len(w['x']),'optimizer_updates':0,'active_compute_proxy':int((D//2)*K*(2 if m in ('mirror_superposition','direct_orbit') else 1)),'construction_wall_s':prep,'inference_examples_per_s':float(10*K*len(w['x'])/max(inf_s,1e-12)),'task_output_norm_mse':tm,'signed_pair_norm_mse':pm,'per_task_mse':per_task,'per_pair_mse':per_pair,'payload_roundtrip_bytes':len(blob),'status_note':'metrics computed from deserialized payload; support and evaluation data are world-seeded and disjoint'})
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--condition',choices=['aligned_orthogonal_orbit','independent_isotropic_deltas'],required=True); p.add_argument('--svd-rank',type=int,default=4); p.add_argument('--psp-seed',type=int,default=296101); p.add_argument('--out',required=True); a=p.parse_args()
    rows=run_world(a.seed,a.condition,a.svd_rank,a.psp_seed); Path(a.out).write_text(json.dumps(rows,indent=2));
    for r in rows: print(r['condition'],r['world_or_seed'],r['method'],f"mse={r['task_output_norm_mse']:.5g}",f"pair={r['signed_pair_norm_mse']:.5g}",f"bytes={r['serialized_bytes']}",f"ips={r['inference_examples_per_s']:.3g}")
if __name__=='__main__': main()
