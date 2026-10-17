"""MA-1162 rate-quality follow-up: compact one-core multihead Mirror payload.

Post-hoc question from completed MA-1162 teacher trial; new worlds/seeds are
predeclared in PROTOCOL_RATE_QUALITY_FROZEN.json. No trained-language claims.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source'))
from multihead import SharedMultihead, code_pairwise_kl
from train import make_teacher

MAGIC=b'MMH1'
HEADER_FMT='<4sBBBBHHH' # magic, version, mode, phys_bits, code_bits, dmodel,dhead,heads
HEADER_BYTES=struct.calcsize(HEADER_FMT)
MODES={'linear':1,'nonlinear':2}
REVERSE_MODES={v:k for k,v in MODES.items()}
PHYS=(16,32)
CODES=(8,12,16,32)
FIT={ 'linear':'linear', 'nonlinear':'nonlinear', 'off_orbit':'linear' }


def wrap_angle(a):
    return ((float(a)+math.pi)%(2*math.pi))-math.pi


def quant_code(a,bits):
    if bits==32:
        return None,float(a)
    # Fixed quantizer covering [-pi,+pi] with signed periodic angle correction.
    m=(1<<bits)-1
    t=(wrap_angle(a)+math.pi)/(2*math.pi)
    z=min(m,max(0,round(t*m)))
    return z, (z/m)*(2*math.pi)-math.pi


def pack_codes(v,bits):
    if bits==32:
        return b''.join(struct.pack('<f',float(x)) for x in v)
    n=0
    for i,x in enumerate(v):
        z,_=quant_code(x,bits)
        n |= z << (bits*i)
    return n.to_bytes((len(v)*bits+7)//8,'little')


def unpack_codes(blob,n,bits):
    if bits==32:
        assert len(blob)==4*n
        return [struct.unpack_from('<f',blob,4*i)[0] for i in range(n)]
    assert len(blob)==(n*bits+7)//8
    q=int.from_bytes(blob,'little')
    mask=(1<<bits)-1
    return [( ( (q>>(i*bits))&mask)/mask )*(2*math.pi)-math.pi for i in range(n)]


def serialize(model,phys_bits,code_bits):
    assert model.mode in MODES
    assert phys_bits in PHYS and code_bits in CODES
    header=struct.pack(HEADER_FMT,MAGIC,1,MODES[model.mode],phys_bits,code_bits,
                       model.d_model,model.d_head,model.heads)
    coeff=torch.cat([model.Wq.reshape(-1),model.Wk.reshape(-1),model.Wv.reshape(-1)]).detach().cpu()
    coeff=coeff.to(torch.float16 if phys_bits==16 else torch.float32).numpy()
    payload=header+coeff.tobytes(order='C')
    values=torch.cat((model.q_code,model.v_code)).detach().cpu().tolist()
    return payload+pack_codes(values,code_bits)


def deserialize(data):
    import numpy as np
    assert len(data)>=HEADER_BYTES
    magic,version,mode,phys,code,dmodel,dhead,heads=struct.unpack_from(HEADER_FMT,data,0)
    if magic!=MAGIC or version!=1 or mode not in REVERSE_MODES or phys not in PHYS or code not in CODES:
        raise ValueError('invalid header')
    n=3*dmodel*dhead
    cbytes=n*(phys//8)
    codebytes=(2*heads*code+7)//8
    if len(data)!=HEADER_BYTES+cbytes+codebytes:
        raise ValueError('invalid total payload length')
    dtype='<f2' if phys==16 else '<f4'
    coeff=np.frombuffer(data,dtype=np.dtype(dtype),count=n,offset=HEADER_BYTES).astype('float32')
    angle=unpack_codes(data[HEADER_BYTES+cbytes:],2*heads,code)
    model=SharedMultihead(d_model=dmodel,d_head=dhead,heads=heads,mode=REVERSE_MODES[mode])
    a=torch.from_numpy(coeff.copy())
    with torch.no_grad():
        for i,name in enumerate(('Wq','Wk','Wv')):
            getattr(model,name).copy_(a[i*dmodel*dhead:(i+1)*dmodel*dhead].reshape(dmodel,dhead))
        model.q_code.copy_(torch.tensor(angle[:heads]))
        model.v_code.copy_(torch.tensor(angle[heads:]))
    return model.eval()


def native_full_bytes(dmodel,dhead,heads,phys_bits):
    # Full independent Q,K,V matrices per logical head, same raw codec header.
    return HEADER_BYTES+3*heads*dmodel*dhead*(phys_bits//8)


def native_folded_qv_bytes(dmodel,dhead,heads,phys_bits):
    # Exact linear Mirror function, independent Q_j,V_j matrices plus shared K0.
    return HEADER_BYTES+(2*heads+1)*dmodel*dhead*(phys_bits//8)


def shared_bytes(dmodel,dhead,phys_bits):
    return HEADER_BYTES+3*dmodel*dhead*(phys_bits//8)


def cache_bytes(T,heads,dhead,phys_bits,batch=1):
    base=2*batch*T*dhead*(phys_bits//8)
    return base,heads*base


def fit_teacher(teacher_mode,world,seed,updates=160,batch=32,lr=0.06):
    teacher=make_teacher(world,mode=teacher_mode)
    teacher.eval()
    student_mode=FIT[teacher_mode]
    torch.manual_seed(1000+seed)
    student=SharedMultihead(16,8,4,student_mode)
    with torch.no_grad():
        for name in ('Wq','Wk','Wv'):
            getattr(student,name).copy_(getattr(teacher,name))
    optim=torch.optim.Adam(student.parameters(),lr=lr)
    gen=torch.Generator().manual_seed(world*113+seed*13+17)
    for _ in range(updates):
        x=torch.randn(batch,8,16,generator=gen)
        with torch.no_grad(): y=teacher(x)
        p=student(x)
        loss=F.mse_loss(p,y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),2.)
        optim.step()
    return teacher.eval(),student.eval()


def eval_pair(teacher,original,decoded,x):
    with torch.no_grad():
        target=teacher(x)
        pred,weights=decoded(x,return_attention=True)
        pristine=original(x)
        return (F.mse_loss(pred,target).item(),F.mse_loss(pred,pristine).item(),
                code_pairwise_kl(weights), (pred-pristine).abs().max().item())


def run(phase,out,updates=160,batch=32,lr=.06,eval_n=256):
    if phase=='dev':
        worlds=[16110]; seeds=[1]
    else:
        worlds=[16111,16112,16113];seeds=[1,2]
    out=Path(out)
    if out.exists():raise FileExistsError(f'never overwrite {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    rows=[]
    for family in FIT:
        for world in worlds:
            for seed in seeds:
                teacher,model=fit_teacher(family,world,seed,updates,batch,lr)
                g=torch.Generator().manual_seed(world*113+seed*13+1000017)
                x=torch.randn(eval_n,8,16,generator=g)
                for phys in PHYS:
                    for code in CODES:
                        data=serialize(model,phys,code)
                        clone=deserialize(data)
                        mse,codec_dist,head_kl,max_diff=eval_pair(teacher,model,clone,x)
                        raw_model=shared_bytes(16,8,phys)
                        rows.append({
                            'phase':phase,'teacher':family,'world':world,'seed':seed,
                            'code_bits':code,'physical_bits':phys,
                            'learned_code_scalars':8,'physical_base_scalars':384,
                            'mirror_payload_bytes':len(data),
                            'shared_no_view_bytes':raw_model,
                            'native_full_heads_bytes':native_full_bytes(16,8,4,phys),
                            'native_folded_qv_bytes':native_folded_qv_bytes(16,8,4,phys) if model.mode=='linear' else '',
                            'cache_canonical_fp16_T8192_bytes':cache_bytes(8192,4,8,16)[0],
                            'cache_four_heads_fp16_T8192_bytes':cache_bytes(8192,4,8,16)[1],
                            'quantized_teacher_mse':mse,
                            'codec_vs_fullprecision_mse':codec_dist,
                            'head_pairwise_attention_kl':head_kl,
                            'max_decode_difference':max_diff,
                            'training_updates':updates,'batch':batch,'eval_examples':eval_n,
                            'lr':lr,'mode':model.mode,
                            'torch':torch.__version__,'threads':torch.get_num_threads(),
                        })
                print(f'{phase} {family:9} {world} seed{seed}: '+
                      f'FP16 code8 MSE={rows[-len(PHYS)*len(CODES)]["quantized_teacher_mse"]:.4g}',flush=True)
    if not rows:raise RuntimeError('empty')
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    meta={'phase':phase,'rows':len(rows),'worlds':worlds,'seeds':seeds,
          'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
          'raw_bytes_roundtrip_checked_each_row':True,
          'caveats':['aligned synthetics are not generalization','comparator raw matrix payload is structural, not pretrained natural MHA',
           'nonlinear output View cannot be treated as folded linear','KV cache assumption is identical canonical history']}
    out.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('dev','fresh'),required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    run(a.phase,a.out)

if __name__=='__main__':main()
