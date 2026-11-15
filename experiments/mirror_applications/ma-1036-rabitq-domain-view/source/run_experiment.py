#!/usr/bin/env python3
"""MA-1036: FAISS RaBitQ domain-view experiment. NumPy/FAISS CPU only."""
import argparse, json, math, os, statistics, time
from pathlib import Path
import numpy as np
import faiss

D=128; K=10; DOMAINS=('isotropic','front','back','alternating')
GRID=np.arange(8,dtype=np.float64)*math.pi/8
ROOT=Path(__file__).resolve().parents[1]

def world(seed, ntrain=2048, ndb=2048, nq=256):
    rng=np.random.default_rng(seed); centers=rng.normal(size=(32,D)).astype('float32')*2.2
    scales=[]
    for j,name in enumerate(DOMAINS):
        if name=='isotropic': s=np.ones(D)
        elif name=='front': s=np.geomspace(2.8,.35,D)
        elif name=='back': s=np.geomspace(.35,2.8,D)
        else: s=np.where(np.arange(D)%2==0,2.0,.55)
        scales.append(s.astype('float32'))
    # A shared database and shared train set; queries have domain-specific residual covariance.
    labels=rng.integers(0,len(centers),size=ndb)
    db=centers[labels]+rng.normal(size=(ndb,D)).astype('float32')*np.mean(scales,axis=0)*.45
    tl=rng.integers(0,len(centers),size=ntrain)
    train=centers[tl]+rng.normal(size=(ntrain,D)).astype('float32')*.65
    qs=[]
    for j,s in enumerate(scales):
        ql=rng.integers(0,len(centers),size=nq)
        q=centers[ql]+rng.normal(size=(nq,D)).astype('float32')*s*.40
        qs.append(np.ascontiguousarray(q))
    return np.ascontiguousarray(train),np.ascontiguousarray(db),qs

def givens(x,theta):
    z=np.array(x,copy=True).reshape(-1,D//2,2)
    c=np.float32(math.cos(float(theta))); s=np.float32(math.sin(float(theta)))
    a=z[:,:,0].copy(); b=z[:,:,1].copy()
    z[:,:,0]=c*a-s*b; z[:,:,1]=s*a+c*b
    return z.reshape(-1,D)

def rotation(seed):
    r=faiss.RandomRotationMatrix(D,D); r.init(int(seed))
    # LinearTransform A is the exact inference matrix used by IndexPreTransform.
    return np.asarray(faiss.vector_to_array(r.A),dtype=np.float32).reshape(D,D,order='F').copy()

def apply_basis(x,R):
    return np.ascontiguousarray(np.asarray(x,dtype=np.float32) @ np.asarray(R,dtype=np.float32).T)

def index_for(train,db):
    idx=faiss.IndexRaBitQ(D,faiss.METRIC_L2,1)
    idx.train(np.ascontiguousarray(train,dtype=np.float32)); idx.add(np.ascontiguousarray(db,dtype=np.float32))
    return idx

def measure(idx,queries,transform=None,repeats=7):
    x=queries if transform is None else transform(queries)
    idx.search(x[:8],K)
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); idx.search(x,K); times.append(time.perf_counter()-t)
    return float(np.median(times)), float(len(x)/np.median(times))

def recall(idx,db,queries):
    exact=faiss.IndexFlatL2(D); exact.add(db); _,gt=exact.search(queries,K)
    t=time.perf_counter(); _,got=idx.search(queries,K); wall=time.perf_counter()-t
    return float((got==gt).sum()/(len(queries)*K)),float((got[:,0]==gt[:,0]).mean()),wall

def write_index(idx,path):
    blob=bytes(faiss.serialize_index(idx)); path.write_bytes(blob); return blob

def payload(root,method,seed,indices,bases,angles):
    p=root/'payload'/method/str(seed); p.mkdir(parents=True,exist_ok=True); n=0
    seen_idx=set(); seen_basis=set()
    for i,(idx,R) in enumerate(zip(indices,bases)):
        key=id(idx)
        if key not in seen_idx:
            blob=write_index(idx,p/f'index_{i}.faiss'); n+=len(blob); seen_idx.add(key)
        if R is not None:
            bh=hash(np.asarray(R).tobytes())
            if bh not in seen_basis:
                rb=np.asarray(R,dtype='<f4').tobytes(order='C'); (p/f'basis_{i}.f32').write_bytes(rb); n+=len(rb); seen_basis.add(bh)
    unique=[]; domain_to_index=[]
    for idx in indices:
        found=next((i for i,x in enumerate(unique) if x is idx),None)
        if found is None: found=len(unique); unique.append(idx)
        domain_to_index.append(found)
    meta={'method':method,'seed':seed,'angles_rad':[float(a) for a in angles], 'd':D,'physical_index_count':len(unique),'domain_to_index':domain_to_index,'faiss_version':faiss.__version__}
    mb=json.dumps(meta,sort_keys=True,separators=(',',':')).encode(); (p/'manifest.json').write_bytes(mb); n+=len(mb)
    return n

def arm(train,db,qs,base_seed,method,angles=None):
    idxs=[]; bases=[]; trans=[]; build=0.0
    shared_idx=None
    shared_R=rotation(base_seed) if method in ('shared','mirror') else None
    for j in range(len(DOMAINS)):
        if method=='unrotated': R=None; th=0.
        elif method=='shared': R=shared_R; th=0.
        elif method=='mirror': R=shared_R; th=float(angles[j])
        elif method=='independent': R=rotation(base_seed+j+1); th=0.
        else: raise ValueError(method)
        tr=givens(train,th); dbv=givens(db,th); qv=givens(qs[j],th)
        if R is not None:
            tr=apply_basis(tr,R); dbv=apply_basis(dbv,R); qv=apply_basis(qv,R)
        if method in ('unrotated','shared') and shared_idx is not None:
            idx=shared_idx
        else:
            tb=time.perf_counter(); idx=index_for(tr,dbv); build += time.perf_counter()-tb
            if method in ('unrotated','shared'): shared_idx=idx
        idxs.append(idx); bases.append(R); trans.append(qv)
    rec=[]; r1=[]; search=[]; tsearch=[]
    for j,idx in enumerate(idxs):
        r,a,_=recall(idx, (db if bases[j] is None else apply_basis(givens(db,0. if method in ('shared','independent') else float(angles[j])),bases[j])), trans[j]); rec.append(r); r1.append(a)
        st,qps=measure(idx,trans[j]); search.append(qps)
        # Transform + search includes fixed Givens and common rotation.
        def apply(x,j=j):
            th=(0. if method in ('shared','unrotated','independent') else float(angles[j]))
            z=givens(x,th)
            if bases[j] is not None: z=apply_basis(z,bases[j])
            return z
        _,qps2=measure(idx,qs[j],transform=apply); tsearch.append(qps2)
    nbytes=payload(ROOT,method,base_seed,idxs,bases,angles or [0]*len(DOMAINS))
    return {'recall10':rec,'recall1':r1,'search_qps':search,'transform_search_qps':tsearch,'build_s':build,'bytes':nbytes,'indices':idxs,'bases':bases,'transformed_queries':trans}

def dev_select():
    train,db,qs=world(103600)
    R=rotation(103600)
    angles=[]
    for j in range(4):
        scores=[]
        for th in GRID:
            tr=givens(train,th); dv=givens(db,th); qv=givens(qs[j],th)
            idx=index_for(apply_basis(tr,R),apply_basis(dv,R))
            scores.append(recall(idx,apply_basis(dv,R),apply_basis(qv,R))[0])
        angles.append(float(GRID[int(np.argmax(scores))]))
    out={'development_seed':103600,'grid_rad':GRID.tolist(),'frozen_angles_rad':angles,'selected_recall10_grid':[]}
    for j,th in enumerate(angles):
        idx=index_for(apply_basis(givens(train,th),R),apply_basis(givens(db,th),R))
        out['selected_recall10_grid'].append(recall(idx,apply_basis(givens(db,th),R),apply_basis(givens(qs[j],th),R))[0])
    (ROOT/'runs'/'DEV_SELECTION.json').write_text(json.dumps(out,indent=2)+'\n')
    return angles

def postquant_gauge(seed):
    train,db,qs=world(seed)
    idx=index_for(train,db)
    codes=np.asarray(faiss.vector_to_array(idx.codes),dtype=np.uint8).reshape(idx.ntotal,idx.code_size)
    decoded=np.ascontiguousarray(idx.sa_decode(codes))
    theta=0.73
    q=qs[0]
    exact0=faiss.IndexFlatL2(D); exact0.add(decoded); _,i0=exact0.search(q,K)
    qg=givens(q,theta); dg=givens(decoded,theta)
    exact1=faiss.IndexFlatL2(D); exact1.add(dg); _,i1=exact1.search(qg,K)
    blob=bytes(faiss.serialize_index(idx)); angle_bytes=np.asarray([theta],dtype='<f4').tobytes()
    return {'seed':seed,'theta_rad':theta,'rankings_identical':bool(np.array_equal(i0,i1)),
      'changed_neighbor_ids':int(np.count_nonzero(i0!=i1)), 'code_stream_bytes':len(blob),
      'paid_gauge_coordinate_bytes':len(angle_bytes),'total_payload_bytes':len(blob)+len(angle_bytes),
      'logical_functions_added':0 if np.array_equal(i0,i1) else None}

def audit(angles,seeds):
    rows=[]; records=[]
    for seed in seeds:
        train,db,qs=world(seed)
        methods={}
        for meth in ('unrotated','shared','mirror','independent'):
            res=arm(train,db,qs,seed,meth,angles if meth=='mirror' else None); methods[meth]=res
            for j,name in enumerate(DOMAINS):
                rows.append({'seed':seed,'domain':name,'method':meth,'recall10':res['recall10'][j],'recall1':res['recall1'][j], 'search_qps':res['search_qps'][j], 'transform_search_qps':res['transform_search_qps'][j], 'payload_bytes':res['bytes'],'build_s':res['build_s']})
        # Per-domain means; note all index/base bytes charged to each bank.
        records.append({'seed':seed,'methods':{k:{q:v for q,v in x.items() if q not in ('indices','bases','transformed_queries')} for k,x in methods.items()}})
    gauges=[postquant_gauge(seed) for seed in seeds]
    (ROOT/'runs'/'POSTQUANT_GAUGE.json').write_text(json.dumps(gauges,indent=2)+'\n')
    (ROOT/'runs'/'AUDIT_RESULTS.json').write_text(json.dumps({'seeds':seeds,'records':records,'rows':rows,'postquant_gauge':gauges},indent=2)+'\n')
    return rows,records

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['dev','audit','all'],default='all'); ap.add_argument('--angles'); ap.add_argument('--seeds',nargs='*',type=int,default=[103601,103602,103603]); a=ap.parse_args()
    if a.stage in ('dev','all'): angles=dev_select()
    else:
        p=ROOT/'runs'/'DEV_SELECTION.json'; angles=json.loads(p.read_text())['frozen_angles_rad']
    if a.angles: angles=[float(x) for x in a.angles.split(',')]
    if a.stage in ('audit','all'): audit(angles,a.seeds)
    print(json.dumps({'angles':angles,'faiss':faiss.__version__,'stage':a.stage}))
if __name__=='__main__': main()
