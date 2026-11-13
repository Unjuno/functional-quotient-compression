"""Synthetic clustered ULA beam codebook screen; no radio measurements."""
import csv,io,json
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
N=8; CENTERS=np.array([-.72,-.24,.24,.72]); S=4; USERS=256; SNR=10.0
DEV=(98101,); FRESH=(98111,98113,98117)

def steering(u):
 n=np.arange(N);return np.exp(1j*np.pi*n*u)/np.sqrt(N)

def channels(seed):
 rng=np.random.default_rng(seed); hs=[]; sectors=[]
 for s,c in enumerate(CENTERS):
  for _ in range(USERS):
   offsets=rng.normal(0,.055,4); gains=(rng.normal(size=4)+1j*rng.normal(size=4))/np.sqrt(8)
   h=sum(g*steering(np.clip(c+o,-.98,.98)) for g,o in zip(gains,offsets))
   hs.append(h);sectors.append(s)
 return np.asarray(hs),np.asarray(sectors)

def make_beams(count):
 offsets=np.linspace(-.18,.18,count)
 return np.stack([[steering(c+o) for o in offsets] for c in CENTERS])

def serialize(state):
 b=io.BytesIO();torch.save(state,b);return b.getvalue()

def rate(h,beams):
 # h is [users, antennas]; beams [sectors, beam, antennas]
 powers=np.abs(np.einsum('ua,sba->usb',h.conj(),beams))**2
 sel=np.argmax(powers,axis=-1);best=np.take_along_axis(powers,sel[...,None],axis=-1)[...,0]
 return np.log2(1+SNR*best),sel

def evaluate(seed):
 h,sector=channels(seed); b8=make_beams(8);b16=make_beams(16)
 global8=np.stack([np.stack([steering(u) for u in np.linspace(-.875,.875,8)])]*S)
 methods={'global_shared_DFT_codebook':global8,'mirror_sector_phase_view':b8,'native_sector_steering_codebook':b8,'independent_sector_dense_codebooks':b16}
 # Continuous matched filter is an upper bound, not a serialized beam bank.
 hnorm=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
 out=[]
 for name,bank in methods.items():
  payload={
   'global_shared_DFT_codebook':{'beams':torch.tensor(global8),'sector_router':torch.tensor(sector),'metadata':{'n_ant':N,'n_sector':S,'kind':'global'}},
   'mirror_sector_phase_view':{'base_offsets':torch.tensor(np.linspace(-.18,.18,8)),'sector_centers':torch.tensor(CENTERS),'sector_router':torch.tensor(sector),'metadata':{'n_ant':N,'n_sector':S,'kind':'mirror'}},
   'native_sector_steering_codebook':{'offsets':torch.tensor(np.linspace(-.18,.18,8)),'sector_centers':torch.tensor(CENTERS),'sector_router':torch.tensor(sector),'metadata':{'n_ant':N,'n_sector':S,'kind':'native-angle'}},
   'independent_sector_dense_codebooks':{'beams':torch.tensor(b16),'sector_router':torch.tensor(sector),'metadata':{'n_ant':N,'n_sector':S,'kind':'independent'}},
  }[name]
  blob=serialize(payload); loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
  if name in ('mirror_sector_phase_view','native_sector_steering_codebook'):
   offsets=loaded['base_offsets' if name.startswith('mirror') else 'offsets'].numpy();centers=loaded['sector_centers'].numpy()
   decoded=np.stack([[steering(c+o) for o in offsets] for c in centers])
  else: decoded=loaded['beams'].numpy()
  rates,sel=rate(h,decoded); picked=rates[np.arange(len(h)),sector]
  out.append({'condition':name,'mean_rate':float(picked.mean()),'p5_rate':float(np.percentile(picked,5)),'feedback_bits':int(np.ceil(np.log2(decoded.shape[1]))),'bytes':len(blob),'macs':int(decoded.shape[1]*N),'phase_ops':int(decoded.shape[1]*N if name in ('mirror_sector_phase_view','native_sector_steering_codebook') else 0)})
 # Matched-filter upper bound on fresh channel vectors.
 upper=np.log2(1+SNR*np.sum(np.abs(h)**2,axis=1))
 out.append({'condition':'continuous_matched_filter_upper','mean_rate':float(upper.mean()),'p5_rate':float(np.percentile(upper,5)),'feedback_bits':None,'bytes':0,'macs':N,'phase_ops':N})
 return out

def main():
 rows=[]; dev=evaluate(DEV[0]);(R/'source'/'DEVELOPMENT.json').write_text(json.dumps({'seed':DEV[0],'results':dev},indent=2)+'\n')
 fresh=[]
 for seed in FRESH:
  z=evaluate(seed);fresh.append({'seed':seed,'results':z})
  for r in z:rows.append({'seed':seed,'condition':r['condition'],'mean_spectral_efficiency':r['mean_rate'],'5th_percentile_spectral_efficiency':r['p5_rate'],'feedback_bits_per_beam':r['feedback_bits'],'payload_bytes':r['bytes'],'complex_macs_per_user':r['macs'],'phase_ops_per_user':r['phase_ops']})
 with (R/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n').writerows(rows)
 (R/'source'/'FRESH.json').write_text(json.dumps(fresh,indent=2)+'\n');print(json.dumps(fresh))
if __name__=='__main__':main()
