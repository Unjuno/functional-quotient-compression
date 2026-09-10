"""Independent read-only decoder: no encoder, sklearn, source checkpoint or caches.
Consumes only the emitted FQCWR001 artifact. Built for the fixed TinyStories schema.
"""
from __future__ import annotations
import hashlib,json,struct,math
from pathlib import Path
import numpy as np, torch

def unpack(buf,n,b):
    if not 1<=b<=16 or len(buf)!=(n*b+7)//8:raise ValueError('bit extent')
    v=np.frombuffer(buf,dtype=np.uint8)
    if n*b%8 and int(v[-1])>>(n*b%8):raise ValueError('unused packed bits')
    if b in [2,4,8]:
        g=8//b;out=np.empty(len(v)*g,dtype=np.uint16)
        for i in range(g):out[i::g]=(v>>(b*i))&((1<<b)-1)
        return out[:n]
    bits=np.unpackbits(v,bitorder='little')[:n*b].reshape(n,b)
    return (bits.astype(np.uint16)*(1<<np.arange(b,dtype=np.uint16))).sum(1,dtype=np.uint32)

def f16(buf,shape):
    if len(buf)!=2*math.prod(shape):raise ValueError('FP16 extent')
    a=np.frombuffer(buf,dtype='<f2').astype(np.float32).reshape(shape)
    if not np.isfinite(a).all():raise ValueError('nonfinite FP16')
    return a

def centroid(buf,k,b):
    nb=k*b//2
    q=unpack(buf[:nb],k*b,4).astype(np.float32).reshape(k,b);m=f16(buf[nb:],(k,2))
    if np.any(m[:,1]<=0):raise ValueError('centroid scale')
    return q*m[:,1,None]+m[:,0,None]

def mlp(blob):
    if len(blob)<96 or hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:raise ValueError('MLP hash')
    magic,ver,L,D,H,B,k,kt,kind,n,lb,lt,total,r0,r1=struct.unpack_from('<8s14I',blob)
    if (magic,ver,L,D,H,B)!=(b'FQMLP002',2,8,512,2048,32) or r0 or r1:raise ValueError('MLP schema')
    if k not in [64,128] or kind not in [0,1,2] or n>16384 or total!=len(blob):raise ValueError('MLP header')
    if 96+lb+lt>total or any(blob[64+lb+lt:-32]):raise ValueError('MLP extent/padding')
    if kind==0 and (n or kt or lt):raise ValueError('MLP no-tail schema')
    bits=int(math.log2(k));nblocks=L*H*D//B
    expected_base=2*(20*k+(nblocks*bits+7)//8)
    if lb!=expected_base:raise ValueError('MLP base size')
    p=64;sides=[]
    for side in range(2):
        cb=centroid(blob[p:p+20*k],k,32);p+=20*k;ni=(nblocks*bits+7)//8
        ids=unpack(blob[p:p+ni],nblocks,bits);p+=ni;sides.append(cb[ids].reshape(8,2048,512).copy())
    if kind:
        tail=blob[p:p+lt]
        if kind==1:
            if tail[:8]!=b'FQCTv1\0\0':raise ValueError('VQ tail magic')
            tn,tk,tbits,tb,mode=struct.unpack_from('<5I',tail,8)
            if (tn,tk,tb,mode)!=(n,kt,32,1) or kt not in [64,128,256,512,1024] or tbits!=int(math.log2(kt)) or any(tail[28:64]):raise ValueError('VQ tail schema')
            expected=64+40*kt+(14*n+7)//8+(n*32*tbits+7)//8+4*n
            if lt!=expected:raise ValueError('VQ tail extent')
            p=64;nc=2*kt*32;nb=nc//2;codes=unpack(tail[p:p+nb],nc,4).astype(np.float32).reshape(2,kt,32);p+=nb
            meta=f16(tail[p:p+8*kt],(2,kt,2));p+=8*kt
            if np.any(meta[:,:,1]<=0):raise ValueError('tail centroid scale')
            cb=codes*meta[:,:,1,None]+meta[:,:,0,None];ib=(14*n+7)//8;ids=unpack(tail[p:p+ib],n,14);p+=ib
            ni=(n*32*tbits+7)//8;labels=unpack(tail[p:p+ni],n*32,tbits).reshape(n,32);p+=ni;scales=f16(tail[p:],(n,2))
            residuals=[(cb[s][labels[:,s*16:(s+1)*16]]*scales[:,s,None,None]).reshape(n,512) for s in range(2)]
        else:
            if tail[:8]!=b'FQCRAW01' or struct.unpack_from('<II',tail,8)!=(n,512) or any(tail[16:64]):raise ValueError('raw tail schema')
            if lt!=64+(14*n+7)//8+512*n+4*n:raise ValueError('raw tail extent')
            p=64;ib=(14*n+7)//8;ids=unpack(tail[p:p+ib],n,14);p+=ib
            q=unpack(tail[p:p+512*n],n*1024,4).astype(np.float32).reshape(n,2,512);p+=512*n
            if np.any(q>14):raise ValueError('reserved raw code')
            scales=f16(tail[p:],(n,2));r=(q-7)*scales[:,:,None];residuals=[r[:,0],r[:,1]]
        if np.any(scales<=0) or np.any(ids>=16384) or (len(ids)>1 and np.any(ids[1:]<=ids[:-1])):raise ValueError('support/scales')
        for s in range(2):sides[s].reshape(16384,512)[ids]+=residuals[s]
    out={}
    for i in range(8):
        out[f'transformer.h.{i}.mlp.c_fc.weight']=sides[0][i]
        out[f'transformer.h.{i}.mlp.c_proj.weight']=sides[1][i].T.copy()
    return out

def decode_model(path: str|Path, device:str='cpu'):
    blob=Path(path).read_bytes();prefix=28
    if len(blob)<60 or hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:raise ValueError('artifact hash')
    magic,nh,nb,npad=struct.unpack_from('<8sIQQ',blob)
    if magic!=b'FQCWR001' or nh>1000000 or prefix+nh+nb+npad+32!=len(blob):raise ValueError('artifact schema/extent')
    if npad and any(blob[-32-npad:-32]):raise ValueError('padding')
    meta=json.loads(blob[prefix:prefix+nh]);body=blob[prefix+nh:prefix+nh+nb]
    if meta['schema_version']!=1 or meta['tied_lm_head']!='transformer.wte.weight':raise ValueError('model metadata')
    cfg=meta['config'];state={};books={};offset=0
    for d in meta['sections']:
        if d['offset']!=offset or d['length']<0 or offset+d['length']>nb:raise ValueError('section extent')
        data=body[offset:offset+d['length']];offset+=d['length'];kind=d['kind'];name=d['name']
        if kind=='codebook':
            if name in books:raise ValueError('duplicate codebook')
            books[name]=centroid(data,d['K'],d['block']);continue
        if kind=='mlp_v2':
            ms=mlp(data)
            if set(ms)&set(state):raise ValueError('duplicate MLP')
            state.update(ms);continue
        if name in state:raise ValueError('duplicate parameter')
        if kind=='fp16':a=f16(data,d['shape'])
        elif kind=='uniform':
            ng=d['groups'];g=d['group'];n=d['n'];nm=d['meta_bytes'];bits=d['bits']
            if ng!=(n+g-1)//g or nm!=ng*4 or n!=math.prod(d['shape']):raise ValueError('uniform schema')
            m=f16(data[:nm],(ng,2))
            if np.any(m[:,1]<=0):raise ValueError('uniform scales')
            q=unpack(data[nm:],ng*g,bits).astype(np.float32).reshape(ng,g)
            a=(q*m[:,1,None]+m[:,0,None]).reshape(-1)[:n].reshape(d['shape'])
        elif kind=='vq':
            k=d['K'];b=d['block'];n=d['nblocks'];bits=d['index_bits'];nl=(n*bits+7)//8
            cb=books[d['codebook']]
            if cb.shape!=(k,b) or (1<<bits)!=k or n*b!=math.prod(d['oriented_shape']):raise ValueError('VQ schema')
            ids=unpack(data[:nl],n,bits);a=cb[ids].reshape(d['oriented_shape'])
            if d['row_scales']:
                sc=f16(data[nl:],(d['nrows'],))
                if np.any(sc<=0):raise ValueError('row scales')
                a=a*sc[:,None]
            elif len(data)!=nl:raise ValueError('trailing VQ data')
            if d['transpose']:a=a.T
            if list(a.shape)!=d['shape']:raise ValueError('VQ output shape')
        else:raise ValueError('unsupported section')
        if not np.isfinite(a).all():raise ValueError('nonfinite output')
        state[name]=a.copy()
    if offset!=nb or len(state)!=108 or meta['learned_tensors']!=108:raise ValueError('model coverage')
    return cfg,{n:torch.from_numpy(a.copy()).to(device) for n,a in state.items()},meta

if __name__=='__main__':
    import argparse
    from engine import forward,tensor_hash
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--probes',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(2);cfg,s,meta=decode_model(args.artifact,args.device);probes=json.loads(Path(args.probes).read_text());ids=torch.tensor([probes[0]['input_ids']],device=args.device)
    with torch.inference_mode():z=forward(s,cfg,ids)
    result={'artifact':Path(args.artifact).name,'artifact_sha256':hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest(),
            'decoded_tensors':len(s),'decoded_tensor_sha256':tensor_hash(s),'probe0_logits_sha256':hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest(),
            'source_checkpoint_opened':False,'encoder_imported':False,'device':args.device}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
