#!/usr/bin/env python3
"""MA-372 synthetic factorized Mix'n'Match function bank."""
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; SEEDS=(37201,37202); LAYERS=4; WIDTHS=(2,4,8)
TRAIN=((0,0),(0,1),(1,1),(2,1),(2,2),(3,2)); HOLD=tuple((l,w) for l in range(4) for w in range(3) if (l,w) not in TRAIN)

def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def run(seed):
 rng=np.random.default_rng(seed); base=rng.normal(size=(8,8)).astype(np.float32); lf=rng.normal(size=(4,8)).astype(np.float32); wf=rng.normal(size=(3,8)).astype(np.float32)
 # Composition target exactly follows shared + layer * width diagonal function.
 target={(l,w):base+np.diag(lf[l]*wf[w]) for l in range(4) for w in range(3)}
 # Payloads charge actual bases/codes. Flat stores all 12 operators. Native only uses base.
 p_native=pack({'base':base,'meta':np.frombuffer(b'native-width-prefix',dtype=np.uint8)})
 p_flat=pack({**{f'op_{l}_{w}':v for (l,w),v in target.items()},'meta':np.frombuffer(b'flat-12-configs',dtype=np.uint8)})
 factor={'base':base,'layer':lf,'width':wf,'meta':np.frombuffer(b'factorized-code',dtype=np.uint8)}
 p_mirror=pack(factor);p_direct=p_mirror
 # Mirror/direct exact algebra, only description metadata differs; functional values equal.
 results=[]
 for method in ('native_width','flat_table','mirror_factorized','direct_factorized'):
  errs=[];decoded=[]
  for l,w in HOLD:
   if method=='native_width': pred=base
   elif method=='flat_table':pred=target[(l,w)]
   else:pred=base+np.diag(lf[l]*wf[w])
   errs.append(float(np.mean((pred-target[(l,w)])**2)));decoded.append(np.round(pred,6).tobytes())
  b={'native_width':len(p_native),'flat_table':len(p_flat),'mirror_factorized':len(p_mirror),'direct_factorized':len(p_direct)}[method]
  start=time.perf_counter()
  # Dense 8x8 function application over 128 probes for each heldout pair.
  probes=rng.normal(size=(128,8)).astype(np.float32)
  for l,w in HOLD:
   mat=base if method=='native_width' else (target[(l,w)] if method=='flat_table' else base+np.diag(lf[l]*wf[w]))
   _=probes@mat
  wall=time.perf_counter()-start;mac= len(HOLD)*128*8*8
  results.append({'world':seed,'method':method,'serialized_bytes':b,'train_examples':0,'optimizer_updates':0,'active_compute_proxy':mac,'wall_time_s':f'{wall:.6f}','heldout_nmse':f'{np.mean(errs):.9g}','distinct_functions':len(set(decoded)),'status_note':'oracle aligned synthetic composition; no training'})
 return results,{'native':len(p_native),'flat':len(p_flat),'mirror':len(p_mirror),'direct':len(p_direct)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();rows=[];sizes=[]
 for s in SEEDS:r,z=run(s);rows+=r;sizes.append(z)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'sizes':sizes},indent=2))
if __name__=='__main__':main()
