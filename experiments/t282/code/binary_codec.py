"""Research-only whole-weight binary codec. All learned tensors paid; tied LM head.
Little endian; SHA256 trailer; no executable deserialization in decoder.
FP32 dense execution after decoding is not native low-bit inference.
"""
from __future__ import annotations
import json,struct,hashlib,math,time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
MAGIC=b'FQCWR001'; PREFIX=struct.Struct('<8sIQQ')

def pack_bits(x,b:int)->bytes:
    x=np.asarray(x).reshape(-1)
    if b<1 or b>8 or np.any(x<0) or np.any(x>=(1<<b)):raise ValueError('invalid codes')
    x=x.astype(np.uint8,copy=False)
    if b==8:return x.tobytes()
    if b in (2,4):
        g=8//b; pad=(-len(x))%g
        xx=np.pad(x,(0,pad)).reshape(-1,g);out=np.zeros(len(xx),dtype=np.uint8)
        for i in range(g):out|=xx[:,i]<<(b*i)
        return out.tobytes()
    parts=[]
    for i in range(0,len(x),262144):
        xx=x[i:i+262144]; bits=((xx[:,None]>>np.arange(b,dtype=np.uint8))&1).reshape(-1)
        parts.append(np.packbits(bits,bitorder='little').tobytes())
    return b''.join(parts)

def unpack_bits(buf:bytes,n:int,b:int)->np.ndarray:
    if len(buf)!=(n*b+7)//8:raise ValueError('packed extent')
    x=np.frombuffer(buf,dtype=np.uint8)
    if b==8:return x.copy()
    if b in (2,4):
        g=8//b;out=np.empty(len(x)*g,dtype=np.uint8)
        for i in range(g):out[i::g]=(x>>(b*i))&((1<<b)-1)
        out=out[:n]
    else:
        bits=np.unpackbits(x,bitorder='little')[:n*b].reshape(n,b)
        out=(bits*(1<<np.arange(b,dtype=np.uint8))).sum(1,dtype=np.uint16).astype(np.uint8)
    if n*b%8 and int(x[-1])>>(n*b%8):raise ValueError('nonzero packed padding')
    return out

def fp16(x):
    q=np.asarray(x,dtype='<f2')
    if not np.isfinite(q).all():raise ValueError('FP16 overflow/nonfinite')
    return q

def encode_uniform(x:np.ndarray,bits:int,group:int=64):
    shape=x.shape;flat=x.astype(np.float32,copy=False).reshape(-1);n=len(flat);pad=(-n)%group
    a=np.pad(flat,(0,pad)).reshape(-1,group)
    lo=fp16(a.min(1));sc=fp16(np.maximum((a.max(1)-lo.astype(np.float32))/((1<<bits)-1),2**-24))
    q=np.clip(np.rint((a-lo.astype(np.float32)[:,None])/sc.astype(np.float32)[:,None]),0,(1<<bits)-1).astype(np.uint8)
    meta=np.stack((lo,sc),1).astype('<f2').tobytes();codes=pack_bits(q,bits)
    return {'kind':'uniform','shape':list(shape),'bits':bits,'group':group,'n':n,'groups':len(a),'meta_bytes':len(meta)},meta+codes

def decode_uniform(d,blob):
    ng=d['groups'];g=d['group'];n=d['n'];mb=d['meta_bytes']
    if mb!=ng*4 or ng!=(n+g-1)//g:raise ValueError('uniform shape')
    m=np.frombuffer(blob[:mb],dtype='<f2').astype(np.float32).reshape(ng,2)
    if not np.isfinite(m).all() or np.any(m[:,1]<=0):raise ValueError('uniform metadata')
    codes=unpack_bits(blob[mb:],ng*g,d['bits']).reshape(ng,g).astype(np.float32)
    return (codes*m[:,1,None]+m[:,0,None]).reshape(-1)[:n].reshape(d['shape']).copy()

def role(name:str)->str:
    if name=='transformer.wte.weight':return 'embedding'
    if name=='transformer.wpe.weight':return 'position'
    if '.mlp.c_fc.weight' in name:return 'mlp_fc'
    if '.mlp.c_proj.weight' in name:return 'mlp_proj'
    for r in ['q','k','v','out']:
        if f'.attn.attention.{r}_proj.weight' in name:return 'attn_'+r
    return 'other'

def prepare_array(name,t):
    a=t.detach().cpu().numpy()
    return a.T.copy() if role(name)=='mlp_proj' else a

def train_vq_group(items:list, K:int,block:int=32,seed:int=266, normalize_rows:bool=False, max_samples:int=32768):
    """Per-role MiniBatchKMeans; assign against SERIALIZED int4 centroids."""
    rng=np.random.default_rng(seed);arrays=[];scales=[]
    for name,t in items:
        a=prepare_array(name,t).astype(np.float32)
        if a.shape[-1]%block:raise ValueError('block size does not divide row')
        rs=fp16(np.maximum(np.sqrt(np.mean(a.astype(np.float64)**2,axis=1)),2**-24)) if normalize_rows else None
        if rs is not None:a=a/rs.astype(np.float32)[:,None]
        arrays.append(a.reshape(-1,block));scales.append(rs)
    sample=[]; each=max(K,min(max_samples//len(arrays),max(len(a) for a in arrays)))
    for a in arrays:sample.append(a[rng.choice(len(a),size=min(each,len(a)),replace=False)])
    train=np.concatenate(sample)
    with threadpool_limits(limits=2):
        km=MiniBatchKMeans(n_clusters=K,random_state=seed,n_init=1,max_iter=16,batch_size=4096,init_size=min(len(train),max(3*K,4096)),reassignment_ratio=0)
        km.fit(train)
    centers=km.cluster_centers_.astype(np.float32);z=fp16(centers.min(1));s=fp16(np.maximum((centers.max(1)-z.astype(np.float32))/15,2**-24))
    q=np.clip(np.rint((centers-z.astype(np.float32)[:,None])/s.astype(np.float32)[:,None]),0,15).astype(np.uint8)
    cb=q.astype(np.float32)*s.astype(np.float32)[:,None]+z.astype(np.float32)[:,None]
    cbblob=pack_bits(q,4)+np.stack((z,s),1).astype('<f2').tobytes();idxbits=int(math.log2(K))
    payloads=[];stats=[]
    with threadpool_limits(limits=2):
        cbnorm=(cb*cb).sum(1)
        for (name,t),a,rs in zip(items,arrays,scales):
            labels=[]
            for start in range(0,len(a),4096):
                batch=a[start:start+4096]; dist=(batch*batch).sum(1)[:,None]+cbnorm[None,:]-2*batch@cb.T
                labels.append(dist.argmin(1))
            lab=np.concatenate(labels)
            blob=pack_bits(lab,idxbits)+(rs.tobytes() if rs is not None else b'')
            d={'kind':'vq','name':name,'shape':list(t.shape),'oriented_shape':list(prepare_array(name,t).shape),'K':K,'block':block,'index_bits':idxbits,
               'transpose':role(name)=='mlp_proj','nblocks':len(a),'row_scales':normalize_rows,'nrows':len(rs) if rs is not None else 0}
            payloads.append((d,blob));stats.append({'name':name,'unique_codes':len(np.unique(lab)),'serialized_tensor_bytes':len(blob)})
    return {'kind':'codebook','K':K,'block':block,'centroid_bits':4},cbblob,payloads,stats

def decode_codebook(d,blob):
    K=d['K'];B=d['block'];nc=K*B;nb=(nc*4+7)//8
    if len(blob)!=nb+K*4:raise ValueError('codebook extent')
    q=unpack_bits(blob[:nb],nc,4).astype(np.float32).reshape(K,B)
    m=np.frombuffer(blob[nb:],dtype='<f2').astype(np.float32).reshape(K,2)
    if not np.isfinite(m).all() or np.any(m[:,1]<=0):raise ValueError('codebook metadata')
    return q*m[:,1,None]+m[:,0,None]

def write_model(path:Path,cfg:dict,sections:list,provenance:dict,pad_to:int|None=None):
    desc=[];parts=[];offset=0
    for d,b in sections:
        desc.append({**d,'offset':offset,'length':len(b)});parts.append(b);offset+=len(b)
    metadata={'schema_version':1,'config':cfg,'tied_lm_head':'transformer.wte.weight','learned_tensors':108,
              'provenance':provenance,'sections':desc,'not_native_lowbit':True,'common_tokenizer_assets_excluded':True}
    h=json.dumps(metadata,sort_keys=True,separators=(',',':')).encode();body=b''.join(parts)
    total=PREFIX.size+len(h)+len(body)+32
    if pad_to is not None and pad_to<total:raise ValueError(f'budget exceeded: {total}>{pad_to}')
    padding=0 if pad_to is None else pad_to-total
    blob=PREFIX.pack(MAGIC,len(h),len(body),padding)+h+body+bytes(padding)
    blob+=hashlib.sha256(blob).digest();path.write_bytes(blob)
    return {'path':path.name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'header_bytes':PREFIX.size+len(h),'payload_bytes':len(body),'padding_bytes':padding,'checksum_bytes':32}

def read_model(path:Path, device:str='cpu'):
    blob=path.read_bytes()
    if len(blob)<PREFIX.size+32:raise ValueError('truncated')
    if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:raise ValueError('checksum')
    magic,nh,nb,pad=PREFIX.unpack_from(blob)
    if magic!=MAGIC or nh>1000000 or PREFIX.size+nh+nb+pad+32!=len(blob):raise ValueError('header/extent')
    h=json.loads(blob[PREFIX.size:PREFIX.size+nh]);body=blob[PREFIX.size+nh:PREFIX.size+nh+nb]
    if pad and any(blob[-32-pad:-32]):raise ValueError('nonzero padding')
    if h['schema_version']!=1 or h['tied_lm_head']!='transformer.wte.weight':raise ValueError('schema')
    cbs={};out={};nextoff=0
    for d in h['sections']:
        off=d['offset'];n=d['length']
        if off!=nextoff or off+n>len(body):raise ValueError('section coverage')
        b=body[off:off+n];nextoff+=n;kind=d['kind']
        if kind=='codebook':cbs[d['name']]=decode_codebook(d,b);continue
        if kind=='fp16':
            if n!=2*math.prod(d['shape']):raise ValueError('FP16 extent')
            a=np.frombuffer(b,dtype='<f2').astype(np.float32).reshape(d['shape'])
        elif kind=='uniform':a=decode_uniform(d,b)
        elif kind=='vq':
            cb=cbs[d['codebook']];ni=(d['nblocks']*d['index_bits']+7)//8
            inds=unpack_bits(b[:ni],d['nblocks'],d['index_bits'])
            a=cb[inds].reshape(d['oriented_shape'])
            if d['row_scales']:
                if len(b)-ni!=2*d['nrows']:raise ValueError('row scale extent')
                rs=np.frombuffer(b[ni:],dtype='<f2').astype(np.float32)
                if not np.isfinite(rs).all() or np.any(rs<=0):raise ValueError('row scales')
                a=a*rs[:,None]
            elif len(b)!=ni:raise ValueError('VQ extent')
            if d['transpose']:a=a.T.copy()
        elif kind=='mlp_v2':
            from codec_v2 import decode
            w1,w2,info=decode(b)
            for i in range(8):
                out[f'transformer.h.{i}.mlp.c_fc.weight']=torch.from_numpy(w1[i].copy()).to(device)
                out[f'transformer.h.{i}.mlp.c_proj.weight']=torch.from_numpy(w2[i].copy()).to(device)
            continue
        else:raise ValueError('unknown section')
        if d['name'] in out or not np.isfinite(a).all():raise ValueError('duplicate/nonfinite tensor')
        out[d['name']]=torch.from_numpy(a.copy()).to(device)
    if nextoff!=len(body) or len(out)!=h['learned_tensors']:raise ValueError('missing model tensors')
    return h['config'],out,h

def uniform_sections(state:dict,bits:int):
    result=[]
    for name,t in state.items():
        a=t.detach().cpu().numpy()
        if bits==16 or a.ndim==1:d={'kind':'fp16','shape':list(a.shape)};b=fp16(a).tobytes()
        else:d,b=encode_uniform(a,bits)
        result.append(({**d,'name':name},b))
    return result
