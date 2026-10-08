"""MA-643 fixed-atom CPU screen. Does not implement GaLore training."""
from __future__ import annotations
import hashlib, json, struct, time
from pathlib import Path
import numpy as np

DIM, ATOMS, ACTIVE, STEPS, LR = 64, 16, 4, 80, 0.08
METHODS = ('full_adam_state','galore_projected_adam_state','hard_shared_atom_tie','mirror_sparse_atom_address','ordinary_dense_atom_coefficients','full_gradient_upper_control')

def orthonormal_basis(seed):
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(rng.normal(size=(DIM,ATOMS)))
    return q

def world(seed):
    rng=np.random.default_rng(seed)
    basis=orthonormal_basis(64300)
    idx=np.sort(rng.choice(ATOMS,size=ACTIVE,replace=False))
    coef=rng.normal(0,0.5,size=ACTIVE)
    target=basis[:,idx]@coef
    return basis,idx,coef,target

def serialize(meta, arrays):
    spec=[]; chunks=[]
    for name in sorted(arrays):
        src=np.asarray(arrays[name]);dtype='<u1' if name=='indices' else '<f4'
        a=np.ascontiguousarray(src,dtype=dtype);raw=a.tobytes()
        spec.append({'name':name,'shape':list(a.shape),'nbytes':len(raw),'dtype':dtype});chunks.append(raw)
    header=json.dumps({'meta':meta,'arrays':spec},sort_keys=True,separators=(',',':')).encode()
    return b'MA643\x01'+struct.pack('<I',len(header))+header+b''.join(chunks)

def deserialize(blob):
    assert blob[:6]==b'MA643\x01';n=struct.unpack('<I',blob[6:10])[0]
    h=json.loads(blob[10:10+n]);pos=10+n; arrays={}
    for s in h['arrays']:
        raw=blob[pos:pos+s['nbytes']];pos+=s['nbytes']
        arrays[s['name']]=np.frombuffer(raw,dtype=s['dtype']).copy().reshape(s['shape'])
    assert pos==len(blob)
    return h['meta'],arrays

def payload(method,basis,idx,coef):
    if method in ('full_adam_state','full_gradient_upper_control'):
        arrays={'direction':(basis[:,idx]@coef).astype('f4')}
    elif method=='galore_projected_adam_state':
        arrays={'basis':basis.astype('f4'),'coordinates':np.pad(coef,(0,ATOMS-ACTIVE)).astype('f4')}
    elif method=='hard_shared_atom_tie':
        arrays={'basis':basis.astype('f4')}
    elif method=='mirror_sparse_atom_address':
        arrays={'basis':basis.astype('f4'),'indices':idx.astype('u1'),'values':coef.astype('f4')}
    else:
        arrays={'basis':basis.astype('f4'),'coefficients':np.zeros(ATOMS,dtype='f4')}
        arrays['coefficients'][idx]=coef
    blob=serialize({'method':method,'dim':DIM,'atoms':ATOMS,'active':ACTIVE},arrays)
    meta,arr=deserialize(blob)
    return blob,meta,arr

def reconstruct(method,arr,idx=None):
    if method in ('full_adam_state','full_gradient_upper_control'):
        return arr['direction'].astype(float)
    if method=='hard_shared_atom_tie': return np.zeros(DIM)
    if method=='mirror_sparse_atom_address': return arr['basis'].astype(float)[:,arr['indices'].astype(int)]@arr['values'].astype(float)
    coeff=arr['coordinates'] if method=='galore_projected_adam_state' else arr['coefficients']
    return arr['basis'].astype(float)@coeff.astype(float)

def quadratic_gap(theta,target):
    return float(np.sum((theta-target)**2)/max(np.sum(target**2),1e-30))

def evaluate(seed,method):
    basis,idx,coef,target=world(seed)
    t0=time.perf_counter();blob,meta,arr=payload(method,basis,idx,coef);serialize_s=time.perf_counter()-t0
    theta=reconstruct(method,arr);start_gap=quadratic_gap(np.zeros(DIM),target)
    gaps=[];t0=time.perf_counter()
    if method=='hard_shared_atom_tie':
        gaps=[quadratic_gap(theta,target)]*STEPS; updates=0
    else:
        for _ in range(STEPS):
            grad=2*(theta-target)
            if method in ('galore_projected_adam_state','mirror_sparse_atom_address','ordinary_dense_atom_coefficients'):
                grad=basis@(basis.T@grad)
            theta=theta-LR*grad
            gaps.append(quadratic_gap(theta,target))
        updates=STEPS
    wall=time.perf_counter()-t0
    start_bytes=len(serialize({'method':'initial','dim':DIM}, {'theta':np.zeros(DIM,dtype='f4')}))
    return {'seed':seed,'method':method,'serialized_bytes':len(blob),'examples':1,'optimizer_updates':updates,'active_compute_proxy':int(STEPS*DIM*(DIM if method in ('full_adam_state','full_gradient_upper_control') else ATOMS if method in ('galore_projected_adam_state','mirror_sparse_atom_address','ordinary_dense_atom_coefficients') else 0)),'query_wall_s':wall,'serialization_wall_s':serialize_s,'initial_objective_gap':start_gap,'final_objective_gap':gaps[-1],'gap_trajectory_hash':hashlib.sha256(np.asarray(gaps,dtype='<f8').tobytes()).hexdigest(),'payload_sha256':hashlib.sha256(blob).hexdigest()}

def run(mode):
    seeds=(64301,64302) if mode=='dev' else (64311,64312,64313)
    rows=[evaluate(s,m) for s in seeds for m in METHODS]
    root=Path(__file__).resolve().parents[1];dest=root/'source'/('development_invalid_proxy.csv' if mode=='dev' else 'fresh.csv')
    import csv
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    import sys
    for row in run(sys.argv[1] if len(sys.argv)>1 else 'dev'): print(row)
