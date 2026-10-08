#!/usr/bin/env python3
"""MA-286 fixed-subspace Mirror screen. NumPy CPU, typed actual-byte codec."""
from __future__ import annotations
import argparse,json,struct,time,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D,R,O,N,K=32,4,32,10,8
SELECTORS=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],
 [0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]],dtype=np.uint8)

def hadamard(n):
    h=np.array([[1.]],dtype=np.float64)
    while h.shape[0]<n: h=np.block([[h,h],[h,-h]])
    return h/np.sqrt(n)
U=hadamard(D)[:,:16]

def build_world(seed):
    rng=np.random.default_rng(seed); bstar=rng.normal(0,.22,(16,O)); codes=np.arange(8,dtype=np.uint8)
    matrices=[]
    for m in codes:
        s=SELECTORS[int(m)].astype(int); w=np.zeros((D,O)); w=U[:,s]@bstar[s,:]; matrices.append(w)
    for _ in range(2):
        a=rng.normal(0,.25,(D,R)); c=rng.normal(0,.25,(R,O)); matrices.append(a@c)
    worlds=[]
    for w in matrices:
        splits=[]
        for n in (256,128,512):
            x=rng.normal(size=(n,D)); y=x@w; splits.append((x,y))
        worlds.append(splits)
    return worlds,codes

def fit(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def nmse(pred,y): return float(np.mean((pred-y)**2)/(np.mean(y*y)+1e-12))

def lowrank4(x,y):
    w=fit(x,y); u,s,vt=np.linalg.svd(w,full_matrices=False)
    a=u[:,:R]*s[:R]; b=vt[:R]
    return a,b

def serialize(records):
    out=bytearray(b"MA286\x01")+bytearray(struct.pack('<I',len(records)))
    for name,val in records:
        a=np.ascontiguousarray(val); nm=name.encode()
        if a.dtype.kind=='f': typ=1; a=a.astype('<f4',copy=False)
        else: typ=2; a=a.astype('u1',copy=False)
        out+=struct.pack('<H',len(nm))+nm+struct.pack('<BB',typ,a.ndim)
        out+=struct.pack('<'+'I'*a.ndim,*a.shape)+a.tobytes()
    return bytes(out)

def deserialize(payload):
    v=memoryview(payload)
    if bytes(v[:6])!=b'MA286\x01': raise ValueError('bad magic')
    p=6; n=struct.unpack_from('<I',v,p)[0]; p+=4; out={}
    for _ in range(n):
        z=struct.unpack_from('<H',v,p)[0]; p+=2; name=bytes(v[p:p+z]).decode(); p+=z
        typ,nd=struct.unpack_from('<BB',v,p); p+=2; shape=struct.unpack_from('<'+'I'*nd,v,p) if nd else (); p+=4*nd
        dt='<f4' if typ==1 else 'u1'; size=int(np.prod(shape,dtype=np.int64))*(4 if typ==1 else 1)
        out[name]=np.frombuffer(v[p:p+size],dtype=dt).copy().reshape(shape); p+=size
    if p!=len(v): raise ValueError('trailing bytes')
    return out

def meta(method):
    return [('meta_method',np.frombuffer(method.encode(),dtype='u1')),
            ('meta_basis',np.frombuffer(b'walsh32-first16-v1',dtype='u1'))]

def rank4_private_records(t,a,b): return [(f't{t}_private_A',a),(f't{t}_private_B',b)]

def fit_shared_codes(world,codes):
    # Fit a shared 16x32 B from the supplied addresses by pooled LS.
    zs=[]; ys=[]
    for t in range(8):
        x,y=world[t][0]; s=SELECTORS[int(codes[t])].astype(int); z=np.zeros((len(x),16)); z[:,s]=x@U[:,s]; zs.append(z); ys.append(y)
    b=fit(np.concatenate(zs),np.concatenate(ys))
    return b

def build_method(method,world,codes,threshold=1e-4,seed=28600):
    start=time.perf_counter(); states=[]; records=meta(method); selectors=SELECTORS.copy(); private=0
    if method in ('mirror_index','mirror_onehot','random_shared'):
        if method=='random_shared':
            rng=np.random.default_rng(seed); used_codes=rng.permutation(K).astype(np.uint8)
        else: used_codes=np.asarray(codes,dtype=np.uint8)
        b=fit_shared_codes(world,used_codes); records.extend([('selector_dictionary',selectors),('shared_B',b)])
        if method=='mirror_index':
            addr=np.full(N,255,dtype=np.uint8); addr[:8]=codes; records.append(('task_addresses',addr))
        elif method=='mirror_onehot':
            gates=np.zeros((N,K),dtype=np.uint8); gates[np.arange(8),codes]=1; records.append(('task_onehot_gates',gates))
        else:
            addr=np.full(N,255,dtype=np.uint8); addr[:8]=used_codes; records.append(('random_task_addresses',addr))
        for t in range(8): states.append({'kind':'shared','code':int(used_codes[t])})
        if method=='random_shared':
            # Compare against frozen random addresses; fallback if the address does not fit.
            addr=records[-1][1]
            for t in range(8):
                x,y=world[t][1]; m=int(addr[t]); s=selectors[m].astype(int); pred=(x@U[:,s])@b[s]
                if nmse(pred,y)>threshold:
                    a,c=lowrank4(*world[t][0]); states[t]={'kind':'private','a':a,'b':c}; records.extend(rank4_private_records(t,a,c)); private+=1
        for t in range(8,10):
            a,c=lowrank4(*world[t][0]); states.append({'kind':'private','a':a,'b':c}); records.extend(rank4_private_records(t,a,c)); private+=1
    elif method in ('cheap_matched','cheap_fixed','cheap_random'):
        if method=='cheap_fixed': chosen=np.tile(selectors[0],(8,1))
        elif method=='cheap_random': chosen=np.random.default_rng(seed).integers(0,16,size=(8,R),dtype=np.uint8)
        else: chosen=np.array([selectors[int(c)] for c in codes],dtype=np.uint8)
        records.append(('task_selectors',chosen))
        for t in range(8):
            s=chosen[t].astype(int); x,y=world[t][0]; c=fit(x@U[:,s],y); xv,yv=world[t][1]
            if nmse((xv@U[:,s])@c,yv)>threshold:
                a,p=lowrank4(x,y); states.append({'kind':'private','a':a,'b':p}); records.extend(rank4_private_records(t,a,p)); private+=1
            else:
                states.append({'kind':'local','selector':s,'c':c}); records.append((f't{t}_local_B',c))
        for t in range(8,10):
            a,c=lowrank4(*world[t][0]); states.append({'kind':'private','a':a,'b':c}); records.extend(rank4_private_records(t,a,c)); private+=1
    elif method=='hard_shared':
        s=selectors[0].astype(int); xall=np.concatenate([world[t][0][0] for t in range(8)]); yall=np.concatenate([world[t][0][1] for t in range(8)])
        c=fit(xall@U[:,s],yall); records.extend([('shared_selector',s.astype('u1')),('shared_local_B',c)])
        states=[{'kind':'hard','selector':s,'c':c} for _ in range(8)]
        for t in range(8,10):
            a,p=lowrank4(*world[t][0]); states.append({'kind':'private','a':a,'b':p}); records.extend(rank4_private_records(t,a,p)); private+=1
    elif method=='independent_full':
        for t in range(N):
            w=fit(*world[t][0]); states.append({'kind':'full','w':w}); records.append((f't{t}_full_W',w))
    else: raise ValueError(method)
    # Kind vector is persistent routing state; charge all task roles.
    kindmap={'shared':1,'private':2,'local':3,'hard':4,'full':5}
    records.append(('task_kinds',np.array([kindmap[s['kind']] for s in states],dtype='u1')))
    payload=serialize(records); elapsed=time.perf_counter()-start
    return payload,elapsed,private

def predict(rec,method,t,x):
    kind=int(rec['task_kinds'][t])
    if kind==2: return (x@rec[f't{t}_private_A'])@rec[f't{t}_private_B']
    if kind==5: return x@rec[f't{t}_full_W']
    if method in ('mirror_index','random_shared'):
        addr=rec['task_addresses'] if method=='mirror_index' else rec['random_task_addresses']; m=int(addr[t]); s=rec['selector_dictionary'][m].astype(int); return (x@U[:,s])@rec['shared_B'][s]
    if method=='mirror_onehot':
        m=int(np.argmax(rec['task_onehot_gates'][t])); s=rec['selector_dictionary'][m].astype(int); return (x@U[:,s])@rec['shared_B'][s]
    if method in ('cheap_matched','cheap_fixed','cheap_random'):
        s=rec['task_selectors'][t].astype(int); return (x@U[:,s])@rec[f't{t}_local_B']
    if method=='hard_shared':
        s=rec['shared_selector'].astype(int); return (x@U[:,s])@rec['shared_local_B']
    raise ValueError((method,kind))

METHODS=['hard_shared','cheap_fixed','cheap_random','cheap_matched','random_shared','mirror_onehot','mirror_index','independent_full']

def run_world(seed,split):
    world,codes=build_world(seed); rows=[]
    for method in METHODS:
        payload,fit_s,private=build_method(method,world,codes); rec=deserialize(payload)
        per=[]
        for t in range(N):
            x,y=world[t][2]; per.append(nmse(predict(rec,method,t,x),y))
        t0=time.perf_counter()
        for _ in range(20):
            for t in range(N): _=predict(rec,method,t,world[t][2][0])
        sec=(time.perf_counter()-t0)/(20*N*512)
        mac_per={'hard_shared':256,'cheap_fixed':256,'cheap_random':256,'cheap_matched':256,'random_shared':256,'mirror_onehot':256,'mirror_index':256,'independent_full':1024}[method]
        private_fit=private*(256*32*32+32*32*32)
        fit_mac={'hard_shared':8*256*4*32+private_fit,'cheap_fixed':8*256*4*32+private_fit,'cheap_random':8*256*4*32+private_fit,'cheap_matched':8*256*4*32+private_fit,'random_shared':2048*16*32+2048*32*4+private_fit,'mirror_onehot':2048*16*32+2048*32*4+private_fit,'mirror_index':2048*16*32+2048*32*4+private_fit,'independent_full':N*256*32*32}[method]
        kinds=rec['task_kinds'];
        rows.append({'seed':seed,'split':split,'method':method,'normalized_test_mse_by_task':per,'mean_normalized_test_mse':float(np.mean(per)),'max_normalized_test_mse':max(per),'serialized_bytes':len(payload),'private_fallbacks':private,'train_examples':N*256,'validation_examples':N*128,'test_examples':N*512,'optimizer_updates':0,'fit_compute_proxy':fit_mac,'fit_wall_seconds':fit_s,'inference_mac_proxy':mac_per*N*512,'inference_examples_per_second':1./sec,'operator_workspace_bytes':1024 if method in ('mirror_index','mirror_onehot','random_shared') else 0,'payload_sha256':hashlib.sha256(payload).hexdigest(),'task_kinds':kinds.tolist()})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['development','fresh'],required=True); p.add_argument('--out',type=Path,default=ROOT/'source'/'results.json'); a=p.parse_args()
    seeds=[28601,28602] if a.split=='development' else [28611,28612,28613]; rows=[]
    for seed in seeds: rows.extend(run_world(seed,a.split))
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'split':a.split,'seeds':seeds,'rows':rows},indent=2)+'\n'); print('wrote',len(rows),'rows')
if __name__=='__main__': main()
