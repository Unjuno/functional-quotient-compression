"""Small deterministic tensorized KV cache/view mechanism screen (NumPy only)."""
from __future__ import annotations
import json, struct, time
import numpy as np

S, D, ROLES, ALIGNED, QN = 16, 8, 8, 6, 64
RANKS = (2, 4, 6, 8)
METHODS = ("raw_mha", "hard_shared", "independent_svd", "pa35_shared_sequence_basis", "mirror_shared_base_private_boundary")

def rotation(theta: float) -> np.ndarray:
    out = np.eye(D, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(0, D, 2): out[i:i+2, i:i+2] = ((c, -s), (s, c))
    return out

def svd_factor(x: np.ndarray, rank: int):
    u, sig, vt = np.linalg.svd(x.astype(np.float64), full_matrices=False)
    return (u[:, :rank] * sig[:rank]).astype('<f4'), vt[:rank].astype('<f4')

def teacher(seed: int):
    rng = np.random.default_rng(seed)
    # Rank-four shared base K/V tensors; two independent rank-four private functions.
    def low_rank(): return (rng.normal(size=(S,4)) @ rng.normal(size=(4,D)) / 2).astype(np.float32)
    bk, bv = low_rank(), low_rank()
    angles = rng.uniform(-0.8, 0.8, size=ALIGNED).astype(np.float32)
    keys, vals = [], []
    for i in range(ROLES):
        if i < ALIGNED:
            rot = rotation(float(angles[i])).astype(np.float32)
            keys.append(bk @ rot); vals.append(bv @ rot)
        else:
            keys.append(low_rank()); vals.append(low_rank())
    queries = rng.normal(size=(ROLES, QN, D)).astype(np.float32)
    return np.stack(keys), np.stack(vals), angles, queries

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e / np.sum(e, axis=-1, keepdims=True)

def attention(k, v, q):
    return softmax(q @ k.T / np.sqrt(D)) @ v

def score(khat, vhat, k, v, q):
    errs=[]; norms=[]
    for i in range(ROLES):
        y=attention(k[i],v[i],q[i]); yh=attention(khat[i],vhat[i],q[i])
        errs.append(float(np.mean((yh-y)**2))); norms.append(float(np.mean(y*y)))
    return float(np.mean(errs)/max(float(np.mean(norms)),1e-30))

def make_state(method: str, rank: int, k: np.ndarray, v: np.ndarray, angles: np.ndarray):
    a={}; meta={"method":method,"rank":int(rank),"shape":[ROLES,S,D],"dtype":"float32-le"}
    if method == 'raw_mha': a={'k':k.astype('<f4'),'v':v.astype('<f4')}
    elif method == 'hard_shared': a={'k':k.mean(axis=0).astype('<f4'),'v':v.mean(axis=0).astype('<f4')}
    elif method == 'independent_svd':
        for name,x in [('k',k),('v',v)]:
            for i in range(ROLES): a[f'{name}_u_{i:02}'],a[f'{name}_vt_{i:02}']=svd_factor(x[i],rank)
    elif method == 'pa35_shared_sequence_basis':
        for name,x in [('k',k),('v',v)]:
            u,_,_=np.linalg.svd(np.concatenate([x[i] for i in range(ALIGNED)],axis=1).astype(np.float64),full_matrices=False)
            basis=u[:,:rank].astype('<f4'); a[f'{name}_basis']=basis
            for i in range(ALIGNED): a[f'{name}_coef_{i:02}']=(basis.T@x[i]).astype('<f4')
            for i in range(ALIGNED,ROLES): a[f'{name}_private_u_{i:02}'],a[f'{name}_private_vt_{i:02}']=svd_factor(x[i],rank)
    elif method == 'mirror_shared_base_private_boundary':
        meta['aligned_roles']=ALIGNED
        a['angles']=angles.astype('<f4')
        for name,x in [('k',k),('v',v)]:
            aligned=np.stack([x[i] @ rotation(float(angles[i])).T for i in range(ALIGNED)])
            base=aligned.mean(axis=0)
            a[f'{name}_base_u'],a[f'{name}_base_vt']=svd_factor(base,rank)
            for i in range(ALIGNED,ROLES): a[f'{name}_private_u_{i:02}'],a[f'{name}_private_vt_{i:02}']=svd_factor(x[i],rank)
    else: raise ValueError(method)
    return meta,a

def serialize(meta, arrays):
    spec=[]; raw=[]
    for name in sorted(arrays):
        ar=np.ascontiguousarray(arrays[name],dtype='<f4'); b=ar.tobytes()
        spec.append({'name':name,'shape':list(ar.shape),'nbytes':len(b)}); raw.append(b)
    header=json.dumps({'meta':meta,'arrays':spec},sort_keys=True,separators=(',',':')).encode()
    return b'MA330\x01'+struct.pack('<I',len(header))+header+b''.join(raw)

def deserialize(payload):
    assert payload[:6]==b'MA330\x01'
    n=struct.unpack('<I',payload[6:10])[0]; h=json.loads(payload[10:10+n]); pos=10+n; a={}
    for item in h['arrays']:
        b=payload[pos:pos+item['nbytes']]; pos+=item['nbytes']
        a[item['name']]=np.frombuffer(b,dtype='<f4').copy().reshape(item['shape'])
    assert pos==len(payload)
    return h['meta'],a

def reconstruct(meta,a):
    method=meta['method']; rank=meta['rank']; kh=[]; vh=[]
    if method in ('raw_mha','hard_shared'):
        kk=a['k']; vv=a['v']
        if method=='hard_shared': kk=np.repeat(kk[None],ROLES,axis=0); vv=np.repeat(vv[None],ROLES,axis=0)
        return kk,vv
    if method=='independent_svd':
        for i in range(ROLES): kh.append(a[f'k_u_{i:02}']@a[f'k_vt_{i:02}']); vh.append(a[f'v_u_{i:02}']@a[f'v_vt_{i:02}'])
    elif method=='pa35_shared_sequence_basis':
        for i in range(ALIGNED): kh.append(a['k_basis']@a[f'k_coef_{i:02}']); vh.append(a['v_basis']@a[f'v_coef_{i:02}'])
        for i in range(ALIGNED,ROLES): kh.append(a[f'k_private_u_{i:02}']@a[f'k_private_vt_{i:02}']); vh.append(a[f'v_private_u_{i:02}']@a[f'v_private_vt_{i:02}'])
    else:
        for i in range(ROLES):
            if i < ALIGNED:
                r=rotation(float(a['angles'][i])).astype(np.float32)
                kh.append((a['k_base_u']@a['k_base_vt'])@r); vh.append((a['v_base_u']@a['v_base_vt'])@r)
            else:
                kh.append(a[f'k_private_u_{i:02}']@a[f'k_private_vt_{i:02}']); vh.append(a[f'v_private_u_{i:02}']@a[f'v_private_vt_{i:02}'])
    return np.stack(kh),np.stack(vh)

def proxies(method,rank,payload_bytes):
    base=2*S*rank*D
    if method=='raw_mha': decode=0; workspace=2*S*D*4
    elif method=='hard_shared': decode=0; workspace=2*S*D*4
    elif method=='mirror_shared_base_private_boundary': decode=6*base+2*(ROLES-ALIGNED)*base+ALIGNED*2*S*D*4; workspace=2*S*D*4
    elif method=='pa35_shared_sequence_basis': decode=ROLES*base; workspace=2*S*D*4
    else: decode=ROLES*base; workspace=2*S*D*4
    attn=ROLES*QN*2*S*D
    return {'decode_macs':int(decode),'attention_macs':int(attn),'peak_stream_workspace_bytes':int(workspace),'expanded_cache_bytes':int(2*ROLES*S*D*4),'modeled_payload_read_write_bytes':int(payload_bytes+2*ROLES*S*D*4)}

def evaluate(seed,rank,method):
    k,v,angles,q=teacher(seed); meta,a=make_state(method,rank,k,v,angles); payload=serialize(meta,a)
    meta2,a2=deserialize(payload); kh,vh=reconstruct(meta2,a2)
    t0=time.perf_counter(); kh,vh=reconstruct(meta2,a2)
    for i in range(ROLES): attention(kh[i],vh[i],q[i])
    wall=time.perf_counter()-t0
    return {'seed':seed,'rank':rank,'method':method,'serialized_bytes':len(payload),'normalized_attention_output_mse':score(kh,vh,k,v,q),'wall_clock_decode_attention_s':wall,**proxies(method,rank,len(payload))}

if __name__=='__main__':
    import csv, pathlib, sys
    root=pathlib.Path(__file__).resolve().parents[1]
    mode=sys.argv[1] if len(sys.argv)>1 else 'dev'
    if mode=='dev':
        rows=[]
        for seed in (33001,33002):
            for rank in RANKS:
                for method in METHODS: rows.append(evaluate(seed,rank,method))
        with (root/'source'/'development.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        by={}
        for r in rows:
            if r['method']=='mirror_shared_base_private_boundary': by.setdefault(r['rank'],[]).append(r['normalized_attention_output_mse'])
        means={r:float(np.mean(v)) for r,v in by.items()}
        qualifies=[r for r,m in means.items() if m<=1e-4]
        chosen=min(qualifies) if qualifies else min(means,key=means.get)
        (root/'source'/'frozen_config.json').write_text(json.dumps({'selected_rank':chosen,'dev_mirror_rank_mse':means,'fresh_seeds':[33031,33032,33033]},indent=2)+'\n')
        print('selected rank',chosen,'dev scores',means)
    elif mode=='fresh':
        cfg=json.loads((root/'source'/'frozen_config.json').read_text()); rank=cfg['selected_rank']; fresh=[]
        for seed in cfg['fresh_seeds']:
            for method in METHODS: fresh.append(evaluate(seed,rank,method))
        with (root/'source'/'fresh.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(fresh[0])); w.writeheader(); w.writerows(fresh)
        for r in fresh: print(r)
    else: raise SystemExit('usage: engine.py dev|fresh')
