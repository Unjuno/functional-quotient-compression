import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
INPUT,HIDDEN,OUTPUT=8,12,3
UPDATES=300

def forward(x,p,activation):
 w1,b1,w2,b2=p;z=x@w1.T+b1;h=torch.relu(z) if activation=='relu' else torch.tanh(z);return h@w2.T+b2

def world(seed,activation):
 g=torch.Generator().manual_seed(seed+1);torch.manual_seed(seed)
 x=torch.randn(1024,INPUT,generator=g);a=torch.randn(INPUT,OUTPUT,generator=g);y=torch.sin(x@a)
 init=[torch.nn.Parameter(torch.randn(HIDDEN,INPUT)*.15),torch.nn.Parameter(torch.zeros(HIDDEN)),torch.nn.Parameter(torch.randn(OUTPUT,HIDDEN)*.15),torch.nn.Parameter(torch.zeros(OUTPUT))]
 opt=torch.optim.Adam(init,lr=.01);st=time.perf_counter()
 for _ in range(UPDATES):
  loss=(forward(x,init,activation)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 trainwall=time.perf_counter()-st
 testx=torch.randn(512,INPUT,generator=torch.Generator().manual_seed(seed+22))
 return [p.detach() for p in init],testx,trainwall

def transform(p,kind,codes):
 w1,b1,w2,b2=p
 if kind=='relu_scale':
  s=codes;return [w1*s[:,None],b1*s,w2/s[None,:],b2]
 if kind=='tanh_sign':
  s=codes;return [w1*s[:,None],b1*s,w2*s[None,:],b2]
 if kind=='relu_uncoupled':
  s=codes;return [w1*s[:,None],b1*s,w2,b2]
 s=codes;return [w1*s[:,None],b1*s,w2,b2]

def symmetry_codes(activation):
 if activation=='relu':return torch.exp(torch.linspace(-.65,.65,HIDDEN))
 return torch.where(torch.arange(HIDDEN)%2==0,1.,-1.)

def pack(arr,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
  for k,v in sorted(arr.items()):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(v),allow_pickle=False)
   info=zipfile.ZipInfo(k+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;z.writestr(info,b.getvalue())
 raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def load(path):
 out={}
 with zipfile.ZipFile(path) as z:
  for f in z.namelist():out[f[:-4]]=np.load(io.BytesIO(z.read(f)),allow_pickle=False)
 return out

def weight_arrays(p):return {f'w{i}':v.detach().cpu().numpy().astype('<f2') for i,v in enumerate(p)}
def eval_arrays(a,x,activation):
 p=[torch.from_numpy(np.array(a[f'w{i}'],copy=True)).float() for i in range(4)]
 with torch.no_grad():return forward(x,p,activation)

def run(seed,split,outdir,jsonpath):
 rows=[]
 for act in ('relu','tanh'):
  model,x,wall=world(seed,act);codes=symmetry_codes(act)
  ref_path=Path(outdir)/f'{split}_{seed}_{act}_base.zip';base=weight_arrays(model);base['meta']=np.asarray([INPUT,HIDDEN,OUTPUT,UPDATES],dtype=np.uint16);bn,bh=pack(base,ref_path);base_loaded=load(ref_path);ref=eval_arrays(base_loaded,x,act)
  kind='relu_scale' if act=='relu' else 'tanh_sign';good=transform(model,kind,codes);badkind='relu_uncoupled' if act=='relu' else 'tanh_uncoupled';bad=transform(model,badkind,codes)
  with torch.no_grad():base32=forward(x,model,act);good32=forward(x,good,act)
  base_quant=float((base32-ref).abs().max())
  for name,params,cc in [(kind,good,codes),(badkind,bad,codes)]:
   arr=weight_arrays(params);arr['code']=cc.detach().cpu().numpy().astype('<f2');arr['meta']=np.asarray([INPUT,HIDDEN,OUTPUT,UPDATES],dtype=np.uint16)
   path=Path(outdir)/f'{split}_{seed}_{name}.zip';n,h=pack(arr,path);got=load(path);pred=eval_arrays(got,x,act)
   sym32=float((good32-base32).abs().max()) if name==kind else None
   qerr=float((good32-pred).abs().max()) if name==kind else None
   rows.append({'condition':split,'seed':seed,'activation':act,'method':name,'serialized_bytes':n,'payload_sha256':h,'base_payload_bytes':bn,'updates':UPDATES,'train_examples':len(x),'train_wall_s':wall,'max_abs_output_difference':float((pred-ref).abs().max()),'rms_output_difference':float(((pred-ref)**2).mean().sqrt()),'fp32_symmetry_max_abs_difference':sym32,'base_fp16_quantization_max_abs_difference':base_quant if name==kind else None,'view_fp16_quantization_max_abs_difference':qerr,'code_bytes':got['code'].nbytes})
  rows.append({'condition':split,'seed':seed,'activation':act,'method':'untransformed_base','serialized_bytes':bn,'payload_sha256':bh,'base_payload_bytes':bn,'updates':UPDATES,'train_examples':len(x),'train_wall_s':wall,'max_abs_output_difference':0.0,'rms_output_difference':0.0,'code_bytes':0})
 out={'condition':split,'seed':seed,'summaries':rows};Path(jsonpath).parent.mkdir(parents=True,exist_ok=True);Path(jsonpath).write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['development','fresh'],required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--json',required=True);a=ap.parse_args();run(a.seed,a.split,a.outdir,a.json)
