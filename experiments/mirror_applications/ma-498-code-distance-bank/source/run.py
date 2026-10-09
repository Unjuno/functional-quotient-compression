#!/usr/bin/env python3
"""MA-498 learned metric-distance codebook screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time,math
from pathlib import Path
import numpy as np,torch
D,E,N=16,8,10000
SIGMAS=(0.,.05,.1,.2,.3,.5)
METHODS=('raw_id','random_orthogonal','random_sphere8','mirror_distance_reg8','native_metric8')

def world(seed):
 g=torch.Generator().manual_seed(seed+498);mats=torch.randn(E,D,D,generator=g)*.25;qg=torch.Generator().manual_seed(seed+2498);req=torch.randint(E,(N,),generator=qg);x=torch.randn(N,D,generator=qg);y=torch.stack([x[i]@mats[req[i]].T for i in range(N)])
 return {'matrices':mats,'requests':req,'queries':x,'target':y}

def codebook(method,seed):
 if method=='raw_id':return None,0.,0
 if method=='random_orthogonal':
  g=torch.Generator().manual_seed(seed+1498);q,_=torch.linalg.qr(torch.randn(E,E,generator=g));return q,0.,0
 if method in ('mirror_distance_reg8','native_metric8'):
  g=torch.Generator().manual_seed(seed+1498);c=torch.randn(E,E,generator=g);c=torch.nn.functional.normalize(c,dim=1);c=torch.nn.Parameter(c);opt=torch.optim.Adam([c],lr=.03);start=time.perf_counter()
  for _ in range(500):
   cn=torch.nn.functional.normalize(c,dim=1);dist=torch.cdist(cn,cn);mask=~torch.eye(E,dtype=torch.bool);loss=torch.relu(1.45-dist[mask]).square().mean();opt.zero_grad();loss.backward();opt.step()
  wall=time.perf_counter()-start;return torch.nn.functional.normalize(c.detach(),dim=1),wall,500
 g=torch.Generator().manual_seed(seed+1498);c=torch.randn(E,E,generator=g);return torch.nn.functional.normalize(c,dim=1),0.,0

def decode(method,z,codes):
 if method=='raw_id':
  bits=(z[:,:3]>0).long();return (bits*(2**torch.arange(3))).sum(1)
 return torch.cdist(z,codes).argmin(1)

def pack(method,w,codes):
 arr={'expert_matrices':w['matrices'].numpy()}
 if codes is not None:arr['address_codebook']=codes.numpy().astype(np.float32)
 meta={'format':'MA498-function-address-v1','method':method,'experts':E,'dim':D,'code_dim':0 if codes is None else E,'decoder':'binary-id' if codes is None else 'nearest-euclidean'}
 bio=io.BytesIO();a={k:np.asarray(v) for k,v in arr.items()};a['schema_json']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(bio,**a);return bio.getvalue()

def evaluate(seed,method,w):
 codes,trainwall,updates=codebook(method,seed);raw=pack(method,w,codes);results={}
 if method=='raw_id':min_dist=1.
 else:
  d=torch.cdist(codes,codes);d.fill_diagonal_(float('inf'));min_dist=float(d.min())
 for sigma in SIGMAS:
  gen=torch.Generator().manual_seed(seed+round(sigma*10000)+3498);noise=torch.randn(N,E,generator=gen)*sigma
  if method=='raw_id':
   base=((w['requests'][:,None]>>torch.arange(3))&1).float()*2-1;z=torch.zeros(N,E);z[:,:3]=base/math.sqrt(3)
  else:z=codes[w['requests']]
  rec=z+noise;start=time.perf_counter();idx=decode(method,rec,codes);decwall=time.perf_counter()-start;start=time.perf_counter();pred=torch.stack([w['queries'][i]@w['matrices'][idx[i]].T for i in range(N)]);qwall=time.perf_counter()-start
  rmse=float(torch.sqrt(torch.mean((pred-w['target'])**2)));wrong=float((idx!=w['requests']).float().mean());ops=N*E*E if codes is not None else N*E
  results[str(sigma)]={'wrong_route_rate':wrong,'output_rmse':rmse,'decoded_function_coverage':int(torch.unique(idx).numel())/E,'requests':N,'decode_ops_proxy':ops,'query_ops_proxy':N*D*D,'active_compute_proxy':ops+N*D*D,'decode_wall_s':decwall,'query_wall_s':qwall}
 return {'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'minimum_pairwise_code_distance':min_dist,'optimizer_updates':updates,'training_wall_s':trainwall,'noise_results':results},raw,codes

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);w=world(seed);results={};cache={}
 for method in METHODS:
  key='mirror_distance_reg8' if method=='native_metric8' else method
  if key not in cache:cache[key]=evaluate(seed,key,w)
  m,raw,codes=cache[key];results[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
 assert (out/'mirror_distance_reg8_payload.npz').read_bytes()==(out/'native_metric8_payload.npz').read_bytes()
 doc={'experiment_id':'MA-498','seed':seed,'split':'dev','task':{'experts':E,'requests_per_sigma':N,'dim':D,'noise_sigmas':SIGMAS},'methods':results};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
