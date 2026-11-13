"""Tiny linear CsiNet-style synthetic CSI domain-adaptation screen."""
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
NANT=8;NTONE=8;D=2*NANT*NTONE;K=16;SNR=10.;FRESH=(98311,98313,98317);DEV=98301
COND=('shared','mirror','embedding','film','lora1','independent')

def channel_batch(rng,angle_center,delay_center,count):
 out=[];a=np.arange(NANT)[:,None];f=np.arange(NTONE)[None,:]
 for _ in range(count):
  l=4;ang=np.clip(angle_center+rng.normal(0,.09,l),-.95,.95);tau=np.mod(delay_center+rng.normal(0,.12,l),1.0)
  gain=(rng.normal(size=l)+1j*rng.normal(size=l))/np.sqrt(2*l)
  h=np.zeros((NANT,NTONE),np.complex128)
  for g,th,td in zip(gain,ang,tau):h+=g*np.exp(1j*np.pi*a*np.sin(th))*np.exp(-2j*np.pi*f*td/NTONE)
  out.append(h)
 h=np.asarray(out);x=np.concatenate([h.real.reshape(count,-1),h.imag.reshape(count,-1)],axis=1).astype(np.float32)
 return x

def encoder():
 rng=np.random.default_rng(98300);q,_=np.linalg.qr(rng.normal(size=(D,K)));return q.T.astype(np.float32)
E=encoder()

def code(x,enc=E):
 z=x@enc.T
 # Fixed symmetric uniform 8-bit quantizer, charged equally to all controls.
 return (np.clip(np.rint(np.clip(z,-4,4)*127/4),-127,127).astype(np.int8).astype(np.float32)*4/127)

def fit_decoder(x,lam=.02):
 z=code(x).astype(np.float64);y=x.astype(np.float64)
 return np.linalg.solve(z.T@z+lam*np.eye(K),z.T@y).T.astype(np.float32)

def source_models():
 rng=np.random.default_rng(983000);centers=[(-.68,.12),(-.25,.34),(.22,.61),(.66,.86)];xs=[]
 for a,t in centers:xs.append(channel_batch(rng,a,t,512))
 W0=fit_decoder(np.concatenate(xs));res=[]
 for x in xs:res.append(fit_decoder(x)-W0)
 mat=np.stack([v.reshape(-1) for v in res]);_,_,vh=np.linalg.svd(mat-mat.mean(0),full_matrices=False)
 B=vh[0].reshape(D,K).astype(np.float32)
 if np.linalg.norm(B)<1e-7:B=res[int(np.argmax([np.linalg.norm(x) for x in res]))]/max(np.linalg.norm(res[int(np.argmax([np.linalg.norm(x) for x in res]))]),1e-8)
 return W0,B

def adapt_target(W0,B,xcal):
 z=code(xcal);pred=z@W0.T;delta=xcal-pred;basis=z@B.T
 m=float(np.sum(basis*delta)/(np.sum(basis*basis)+1e-9))
 # Group-FiLM: eight groups over the real/imag output vector.
 gamma=np.zeros(8,np.float32);beta=np.zeros(8,np.float32)
 for g in range(8):
  sl=slice(g*16,(g+1)*16);A=np.stack([pred[:,sl].reshape(-1),np.ones(pred[:,sl].size)],axis=1);coef=np.linalg.lstsq(A,xcal[:,sl].reshape(-1),rcond=None)[0];gamma[g]=coef[0]-1;beta[g]=coef[1]
 # Rank-1 LoRA approximation to the calibration-fitted full decoder residual.
 Wt=fit_decoder(xcal);u,s,vh=np.linalg.svd(Wt-W0,full_matrices=False);A=(u[:,0]*np.sqrt(max(s[0],0))).astype(np.float32);C=(vh[0]*np.sqrt(max(s[0],0))).astype(np.float32)
 return {'m':m,'gamma':gamma,'beta':beta,'lora_A':A,'lora_B':C,'Wtarget':Wt}

def bytes_of(obj):
 f=io.BytesIO();torch.save(obj,f);return f.getvalue()

def score(x,W0,B,ad,loaded,kind):
 z=code(x,loaded['E'].numpy());base=z@loaded['W0'].numpy().T
 if kind=='shared':pred=base
 elif kind in ('mirror','embedding'):
  pred=base+float(loaded['domain_code'])* (z@loaded['basis'].numpy().T)
 elif kind=='film':
  g=loaded['gamma'].numpy();b=loaded['beta'].numpy();pred=base.copy()
  for j in range(8):pred[:,j*16:(j+1)*16]=pred[:,j*16:(j+1)*16]*(1+g[j])+b[j]
 elif kind=='lora1':pred=z@(loaded['W0'].numpy()+np.outer(loaded['lora_A'].numpy(),loaded['lora_B'].numpy())).T
 else:pred=z@loaded['Wtarget'].numpy().T
 truth=x;nmse=np.mean((pred-truth)**2)/np.mean(truth**2)
 h=truth[:,:NANT*NTONE]+1j*truth[:,NANT*NTONE:];hh=pred[:,:NANT*NTONE]+1j*pred[:,NANT*NTONE:]
 hn=np.maximum(np.linalg.norm(h,axis=1),1e-10);hnn=np.maximum(np.linalg.norm(hh,axis=1),1e-10)
 inner=np.sum(np.conj(h)*hh/hnn[:,None],axis=1);gain=np.abs(inner)**2/(hn**2)
 rate=np.log2(1+SNR*np.abs(inner)**2)
 return float(10*np.log10(max(nmse,1e-12))),float(gain.mean()),float(rate.mean()),float(np.percentile(rate,5))

def world(seed):
 t0=time.perf_counter();W0,B=source_models();rng=np.random.default_rng(seed);targets=[]
 for _ in range(4):
  ac=rng.uniform(-.82,.82);dc=rng.uniform(0,1);cal=channel_batch(rng,ac,dc,64);test=channel_batch(rng,ac,dc,256);targets.append((cal,test))
 ad=[adapt_target(W0,B,c) for c,_ in targets]
 states={
  'shared':{'E':torch.tensor(E),'W0':torch.tensor(W0),'metadata':{'kind':'shared-decoder','latent_bits':128}},
  'mirror':{'E':torch.tensor(E),'W0':torch.tensor(W0),'basis':torch.tensor(B),'domain_code':torch.tensor([a['m'] for a in ad]),'metadata':{'kind':'rank1-domain-view','latent_bits':128}},
  'embedding':{'E':torch.tensor(E),'W0':torch.tensor(W0),'basis':torch.tensor(B),'domain_code':torch.tensor([a['m'] for a in ad]),'metadata':{'kind':'rank1-domain-view','latent_bits':128}},
  'film':{'E':torch.tensor(E),'W0':torch.tensor(W0),'gamma':torch.tensor(np.stack([a['gamma'] for a in ad])),'beta':torch.tensor(np.stack([a['beta'] for a in ad])),'metadata':{'kind':'group-film','latent_bits':128}},
  'lora1':{'E':torch.tensor(E),'W0':torch.tensor(W0),'lora_A':torch.tensor(np.stack([a['lora_A'] for a in ad])),'lora_B':torch.tensor(np.stack([a['lora_B'] for a in ad])),'metadata':{'kind':'rank1-lora','latent_bits':128}},
  'independent':{'E':torch.tensor(E),'W0':torch.tensor(W0),'Wtarget':torch.tensor(np.stack([a['Wtarget'] for a in ad])),'metadata':{'kind':'target-decoders','latent_bits':128}}
 }
 out=[];details=[]
 for kind,state in states.items():
  tcond=time.perf_counter()
  payload=bytes_of(state);loaded=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
  metrics=[]
  for i,(_,test) in enumerate(targets):
   d={k:(v[i] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==4 and k not in ('E','W0') else v) for k,v in loaded.items()}
   m=score(test,W0,B,ad[i],d,kind);metrics.append(m)
  aa=np.asarray(metrics);out.append({'condition':kind,'mean_nmse_db':float(aa[:,0].mean()),'mean_gain':float(aa[:,1].mean()),'mean_rate':float(aa[:,2].mean()),'p5_rate':float(np.percentile(aa[:,3],5)),'bytes':len(payload),'feedback_bits':130,'calibration':256,'macs':K*D,'wall_s':time.perf_counter()-tcond,'target_metrics':metrics,'payload_sha256':hashlib.sha256(payload).hexdigest()})
  details.append({'condition':kind,'payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest()})
 return out,details

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args()
 d,_=world(DEV);(R/'source'/'DEVELOPMENT.json').write_text(json.dumps({'seed':DEV,'results':d},indent=2)+'\n')
 if args.dev_only:print(json.dumps({'development':d}));return
 fresh=[];rows=[]
 for seed in FRESH:
  z,detail=world(seed);fresh.append({'seed':seed,'results':z,'payloads':detail})
  for x in z:rows.append({'seed':seed,'condition':x['condition'],'mean_nmse_db':x['mean_nmse_db'],'mean_beamforming_gain':x['mean_gain'],'mean_spectral_efficiency':x['mean_rate'],'p5_spectral_efficiency':x['p5_rate'],'payload_bytes':x['bytes'],'feedback_bits_per_channel':x['feedback_bits'],'calibration_channels':x['calibration'],'mac_proxy_per_channel':x['macs'],'wall_time_s':x['wall_s']})
 with (R/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n').writerows(rows)
 (R/'source'/'FRESH.json').write_text(json.dumps(fresh,indent=2)+'\n');print(json.dumps(fresh))
if __name__=='__main__':main()
