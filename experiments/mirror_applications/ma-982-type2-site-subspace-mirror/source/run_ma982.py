"""Synthetic Type-II site-subspace adaptation screen."""
import csv,io,json
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
N=8;RANK=2;SITES=4;USERS=128;SNR=10.;FRESH=(98211,98213,98217);DEV=98201
SITE_CENTERS=np.array([-.21,-.07,.07,.21]);
n=np.arange(N)
U0=np.stack([np.exp(2j*np.pi*n*k/N)/np.sqrt(N) for k in (0,1)],axis=1).astype(np.complex64)

def qcomplex(x,bound=2.):
 scale=127/bound
 qr=np.clip(np.rint(x.real*scale),-127,127).astype(np.int8)
 qi=np.clip(np.rint(x.imag*scale),-127,127).astype(np.int8)
 return qr.astype(np.float32)/scale+1j*qi.astype(np.float32)/scale

def serialize(x):
 b=io.BytesIO();torch.save(x,b);return b.getvalue()

def world(seed):
 rng=np.random.default_rng(seed);phis=SITE_CENTERS+rng.normal(0,.007,SITES)
 # 8 pilot probes/site produce a noisy site fingerprint/phase estimate.
 estimates=phis+rng.normal(0,.018/np.sqrt(8),SITES)
 true_u=[];hs=[]
 for s in range(SITES):
  u=(np.exp(1j*np.pi*n*phis[s])[:,None]*U0).astype(np.complex64);true_u.append(u)
  c=(rng.normal(size=(USERS,RANK))+1j*rng.normal(size=(USERS,RANK))).astype(np.complex64)/np.sqrt(2*RANK)
  z=(rng.normal(size=(USERS,N))+1j*rng.normal(size=(USERS,N))).astype(np.complex64)/np.sqrt(2*N)
  z=z-(z@u.conj())@u.T
  z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
  h=c@u.T+.12*z
  hs.append(h)
 return np.asarray(phis),np.asarray(estimates),np.asarray(true_u),np.asarray(hs)

def make_payloads(estimates,true_u):
 return {
  'shared_global_subspace_no_site_code': {'U0':torch.tensor(U0),'metadata':{'kind':'shared','sites':SITES,'rank':RANK}},
  'mirror_shared_subspace_plus_site_phase_coordinate': {'U0':torch.tensor(U0),'m_site':torch.tensor(estimates.astype(np.float32)),'site_ids':torch.arange(SITES,dtype=torch.int16),'metadata':{'kind':'mirror-site-code','rank':RANK}},
  'native_phase_parametric_TypeII': {'base_subspace':torch.tensor(U0),'site_phase':torch.tensor(estimates.astype(np.float32)),'site_ids':torch.arange(SITES,dtype=torch.int16),'metadata':{'kind':'native-type2-phase','rank':RANK}},
  'stored_per_site_TypeII_subspace': {'site_subspaces':torch.tensor(true_u),'site_ids':torch.arange(SITES,dtype=torch.int16),'metadata':{'kind':'stored-type2','rank':RANK}},
  'full_quantized_CSI_feedback': {'metadata':{'kind':'full-CSI-8bit-real-imag','n_ant':N,'pilot_probes_per_site':8}},
 }

def basis_for(condition,loaded,site):
 if condition=='shared_global_subspace_no_site_code':return loaded['U0'].numpy()
 if condition=='stored_per_site_TypeII_subspace':return loaded['site_subspaces'][site].numpy()
 key='m_site' if condition.startswith('mirror') else 'site_phase'
 base='U0' if condition.startswith('mirror') else 'base_subspace'
 phi=float(loaded[key][site]);return (np.exp(1j*np.pi*n*phi)[:,None]*loaded[base].numpy()).astype(np.complex64)

def evaluate(seed):
 phis,estimates,true_u,hs=world(seed); payloads=make_payloads(estimates,true_u);out=[]
 for condition,state in payloads.items():
  blob=serialize(state);loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
  rates=[];nmses=[];macs=[]
  for s in range(SITES):
   h=hs[s]
   if condition=='full_quantized_CSI_feedback':hhat=qcomplex(h)
   else:
    u=basis_for(condition,loaded,s);coef=qcomplex(h@u.conj());hhat=coef@u.T
   nmse=float(np.mean(np.abs(h-hhat)**2)/np.mean(np.abs(h)**2));nmses.append(nmse)
   norm=np.maximum(np.linalg.norm(hhat,axis=1,keepdims=True),1e-8);w=hhat/norm
   p=np.abs(np.sum(h*np.conj(w),axis=1))**2;rates.extend(np.log2(1+SNR*p).tolist())
   macs.append(N*RANK if condition!='full_quantized_CSI_feedback' else N)
  fb=(2+2*RANK*8) if condition!='full_quantized_CSI_feedback' else (2+2*N*8)
  out.append({'condition':condition,'mean_rate':float(np.mean(rates)),'p5_rate':float(np.percentile(rates,5)),'relative_nmse':float(np.mean(nmses)),'bytes':len(blob),'feedback_bits':fb,'probes':8,'macs':int(np.mean(macs))})
 return out

def main():
 dev=evaluate(DEV);(R/'source'/'DEVELOPMENT.json').write_text(json.dumps({'seed':DEV,'results':dev},indent=2)+'\n')
 fresh=[];rows=[]
 for seed in FRESH:
  z=evaluate(seed);fresh.append({'seed':seed,'results':z})
  for r in z:rows.append({'seed':seed,'condition':r['condition'],'mean_rate':r['mean_rate'],'p5_rate':r['p5_rate'],'relative_channel_nmse':r['relative_nmse'],'persistent_payload_bytes':r['bytes'],'online_feedback_bits_per_user':r['feedback_bits'],'pilot_probes_per_site':r['probes'],'complex_mac_proxy_per_user':r['macs']})
 with (R/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n').writerows(rows)
 (R/'source'/'FRESH.json').write_text(json.dumps(fresh,indent=2)+'\n');print(json.dumps(fresh))
if __name__=='__main__':main()
