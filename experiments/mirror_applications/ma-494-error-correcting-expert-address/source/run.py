#!/usr/bin/env python3
"""MA-494 error-correcting logical expert address screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,E,N=16,16,10000
NOISE=(0.,.01,.05,.1,.2)
METHODS=('raw_id','random7','hamming74','repeat3')

def hamming_codes():
 out=[]
 for idx in range(E):
  d=[(idx>>b)&1 for b in range(4)];p1=d[0]^d[1]^d[3];p2=d[0]^d[2]^d[3];p4=d[1]^d[2]^d[3];out.append([p1,p2,d[0],p4,d[1],d[2],d[3]])
 return torch.tensor(out,dtype=torch.uint8)

def codebook(method,seed):
 ids=torch.arange(E)
 if method=='raw_id':return ((ids[:,None]>>torch.arange(4))&1).to(torch.uint8)
 if method=='hamming74':return hamming_codes()
 if method=='repeat3':return (((ids[:,None]>>torch.arange(4))&1).repeat_interleave(3,dim=1)).to(torch.uint8)
 gen=torch.Generator().manual_seed(seed+1494);codes=[]
 while len(codes)<E:
  x=torch.randint(0,2,(7,),generator=gen,dtype=torch.uint8)
  if not any(torch.equal(x,c) for c in codes):codes.append(x)
 return torch.stack(codes)

def world(seed):
 g=torch.Generator().manual_seed(seed+494);mats=torch.randn(E,D,D,generator=g)*.25;qg=torch.Generator().manual_seed(seed+2494);req=torch.randint(0,E,(N,),generator=qg);queries=torch.randn(N,D,generator=qg)
 target=torch.stack([queries[i]@mats[req[i]].T for i in range(N)])
 return {'matrices':mats,'requests':req,'queries':queries,'target':target}

def corrupt(bits,p,seed):
 g=torch.Generator().manual_seed(seed+round(p*100000)+3494);flip=(torch.rand(bits.shape,generator=g)<p);return (bits^flip.to(torch.uint8)),flip

def decode(method,received,codes):
 if method=='raw_id':return (received.long()*(2**torch.arange(received.shape[1]))).sum(1)
 if method=='repeat3':
  b=received.reshape(-1,4,3).sum(2)>=2;return (b.long()*(2**torch.arange(4))).sum(1)
 dist=(received[:,None,:]!=codes[None,:,:]).sum(2);return dist.argmin(1)

def pack(codes,mats,method):
 packed=np.packbits(codes.numpy().reshape(-1),bitorder='little');bio=io.BytesIO();meta={'format':'MA494-expert-address-v1','method':method,'experts':E,'dim':D,'address_bits':int(codes.shape[1]),'decoder':'binary-id' if method=='raw_id' else ('bit-majority' if method=='repeat3' else 'nearest-hamming')}
 np.savez(bio,expert_matrices=mats.numpy(),packed_codebook=packed,schema_json=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8));return bio.getvalue()

def evaluate(seed,method,w):
 codes=codebook(method,seed);payload=pack(codes,w['matrices'],method);metrics={}
 for p in NOISE:
  rec,flip=corrupt(codes[w['requests']],p,seed);t=time.perf_counter();predids=decode(method,rec,codes);decwall=time.perf_counter()-t
  start=time.perf_counter();pred=torch.stack([w['queries'][i]@w['matrices'][predids[i]].T for i in range(N)]);querywall=time.perf_counter()-start
  rmse=float(torch.sqrt(torch.mean((pred-w['target'])**2)));wrong=float((predids!=w['requests']).float().mean());coverage=int(torch.unique(predids).numel())/E;L=codes.shape[1]
  decops=N*L if method in ('raw_id','repeat3') else N*E*L;queryops=N*D*D
  metrics[str(p)]={'wrong_route_rate':wrong,'output_rmse':rmse,'decoded_expert_coverage':coverage,'requests':N,'address_bits_per_request':L,'decode_ops_proxy':decops,'query_ops_proxy':queryops,'active_ops_proxy':decops+queryops,'decode_wall_s':decwall,'query_wall_s':querywall}
 return {'inference_payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'address_bits':int(codes.shape[1]),'noise_results':metrics},payload,codes

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);w=world(seed);results={}
 for method in METHODS:
  m,raw,_=evaluate(seed,method,w);results[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
 doc={'experiment_id':'MA-494','seed':seed,'split':'dev','task':{'experts':E,'requests_per_rate':N,'dim':D,'noise_rates':NOISE},'methods':results};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
