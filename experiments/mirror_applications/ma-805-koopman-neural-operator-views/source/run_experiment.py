"""Known Fourier semigroup screen for parameterized Koopman neural-operator Views."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80501;N=32;T=.8;C=.18;NREG=9;NF=512
KAPPA=np.linspace(.0005,.0035,NREG,dtype=np.float32);HELD=np.array([2,6]);TRAIN=np.array([i for i in range(NREG) if i not in HELD])

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def multipliers(kappa,k2,phase):return phase*np.exp(-kappa*T*k2)
def apply(x,lam):return np.fft.irfft(np.fft.rfft(x,axis=-1)*lam[None,:],n=N,axis=-1).astype(np.float32)
def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
 rng=np.random.default_rng(SEED);freq=np.fft.rfftfreq(N)*N;k2=(freq**2).astype(np.float32);phase=np.exp(-1j*C*freq*T).astype(np.complex64)
 teacher=np.stack([multipliers(float(k),k2,phase) for k in KAPPA]).astype(np.complex64)
 # Smooth random fields emphasize resolved Fourier modes and avoid an empty high-frequency score.
 raw=rng.normal(size=(NREG,NF,N)).astype(np.float32);spec=np.fft.rfft(raw,axis=-1);spec*=np.exp(-.035*k2)[None,None,:]
 fields=np.fft.irfft(spec,n=N,axis=-1).astype(np.float32);targets=np.stack([apply(fields[r],teacher[r]) for r in range(NREG)])
 mean_kernel=teacher[TRAIN].mean(axis=0).astype(np.complex64)
 # The shared physical state includes the full generator and phase basis; no matrix/count is treated as free.
 view=pack(diffusion_generator=k2,advection_phase=phase,diffusivity_codes=KAPPA,regime_ids=np.arange(NREG,dtype=np.int8),grid_size=np.array(N,np.int16),horizon=np.array(T,np.float32))
 fno=pack(fourier_multipliers=teacher,frequencies=freq.astype(np.float32),regime_ids=np.arange(NREG,dtype=np.int8),grid_size=np.array(N,np.int16),horizon=np.array(T,np.float32))
 shared=pack(fourier_multiplier=mean_kernel,frequencies=freq.astype(np.float32),train_regime_ids=TRAIN.astype(np.int8),grid_size=np.array(N,np.int16),horizon=np.array(T,np.float32))
 methods=[('independent_fno_multiplier_tables',teacher,fno,0,'fno'),('shared_unconditioned_kno',np.broadcast_to(mean_kernel,(NREG,len(freq))),shared,0,'shared'),
  ('mirror_diffusivity_views',teacher,view,len(KAPPA)*len(freq),'view'),('native_conditional_fourier_kno',teacher,view,len(KAPPA)*len(freq),'view')]
 rows=[]
 for name,kernels,payload,codeops,kind in methods:
  saved=np.load(io.BytesIO(payload))
  if kind=='fno':restored=saved['fourier_multipliers']
  elif kind=='shared':restored=np.broadcast_to(saved['fourier_multiplier'],(NREG,len(freq)))
  else:restored=np.stack([multipliers(float(k),saved['diffusion_generator'],saved['advection_phase']) for k in saved['diffusivity_codes']])
  rt=float(np.max(np.abs(restored-kernels)))
  if rt>1e-5:raise AssertionError(f'{name} payload reconstruction error {rt}')
  if name=='shared_unconditioned_kno':pred=np.stack([apply(fields[r],kernels[r]) for r in range(NREG)])
  else:
   start=time.perf_counter();pred=np.stack([apply(fields[r],kernels[r]) for r in range(NREG)]);wall=time.perf_counter()-start
  if name=='shared_unconditioned_kno':
   start=time.perf_counter();pred=np.stack([apply(fields[r],kernels[r]) for r in range(NREG)]);wall=time.perf_counter()-start
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   heldout_field_nrmse=nrmse(pred[HELD],targets[HELD]),all_regime_field_nrmse=nrmse(pred,targets),max_multiplier_magnitude=float(np.max(np.abs(kernels))),
   active_spectral_multipliers=len(freq),code_generation_exp_ops=codeops,test_fields=NREG*NF,optimizer_updates=0,wall_time_s=wall,
   payload_roundtrip_max_abs=rt,status_note='exact native physical-parameter alias' if name=='mirror_diffusivity_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
