"""Synthetic factorized task-locus views across input, recurrent and readout maps."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=79701;D=8;N=256
LOCI=('input','recurrent','readout');TASKS=[(l,f) for l in range(3) for f in (-1.,0.,1.)]
HELD=np.array([2,5,8])

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()

def apply(stages,x):
 h=x@stages[0].T;h=h@stages[1].T;return h@stages[2].T

def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def make_stages(base,left,right,locus,factor):
 out=[z.copy() for z in base];out[locus]+=factor*np.outer(left[locus],right[locus]);return out

def run():
 rng=np.random.default_rng(SEED)
 base=[rng.normal(0,.18,(D,D)).astype(np.float32) for _ in LOCI]
 left=np.stack([rng.normal(0,.06,D) for _ in LOCI]).astype(np.float32)
 right=np.stack([rng.normal(0,.06,D) for _ in LOCI]).astype(np.float32)
 teacher=[]
 for locus,f in TASKS:teacher.append(make_stages(base,left,right,locus,f))
 teacher=np.asarray(teacher,np.float32)
 # Learn each locus update from only the observed -1 and 0 task variants.
 learned_base=np.stack([teacher[3*l+1,l] for l in range(3)]).astype(np.float32)
 dl=[];dr=[]
 for l in range(3):
  delta=teacher[3*l,l]-learned_base[l]  # observed factor -1, therefore negate to obtain unit-factor update
  u,s,vt=np.linalg.svd(-delta,full_matrices=False);scale=np.sqrt(max(s[0],0.))
  dl.append((u[:,0]*scale).astype(np.float32));dr.append((vt[0]*scale).astype(np.float32))
 dl=np.stack(dl);dr=np.stack(dr)
 codes=np.array([[f,l] for l,f in TASKS],np.float32)
 pred=np.stack([make_stages(learned_base,dl,dr,l,f) for l,f in TASKS])
 xs=rng.normal(0,.5,(9,N,D)).astype(np.float32)
 target=np.stack([apply(teacher[k],xs[k]) for k in range(9)])
 shared=np.broadcast_to(learned_base,(9,3,D,D)).copy()
 independent_payload=pack(task_stages=teacher,task_codes=codes,task_ids=np.arange(9,dtype=np.int16))
 view_payload=pack(shared_stages=learned_base,view_left=dl,view_right=dr,task_codes=codes,task_ids=np.arange(9,dtype=np.int16))
 payloads=[('independent_full_task_stages',teacher,independent_payload,0),
  ('shared_stages_no_task_code',shared,pack(shared_stages=learned_base,task_ids=np.arange(9,dtype=np.int16)),0),
  ('mirror_factorized_locus_views',pred,view_payload,9*D*D),
  ('native_lowrank_locus_conditioning',pred,view_payload,9*D*D)]
 rows=[]
 for name,st,payload,decode in payloads:
  saved=np.load(io.BytesIO(payload))
  if name=='independent_full_task_stages':restored=saved['task_stages']
  elif name=='shared_stages_no_task_code':restored=np.broadcast_to(saved['shared_stages'],(9,3,D,D))
  else:
   restored=np.stack([make_stages(saved['shared_stages'],saved['view_left'],saved['view_right'],int(l),float(f)) for f,l in saved['task_codes']])
  rt=float(np.max(np.abs(restored-st)))
  if rt>1e-5:raise AssertionError(f'{name} payload roundtrip error {rt}')
  start=time.perf_counter();out=np.stack([apply(st[k],xs[k]) for k in range(9)]);wall=time.perf_counter()-start
  locus_errors=[]
  for l in range(3):
   ab=[]
   for k,(task_l,f) in enumerate(TASKS):
    if task_l==l and f!=0:
     ab.append(nrmse(apply(learned_base,xs[k]),target[k]))
   locus_errors.append(float(np.mean(ab)))
  correct=[];wrong=[]
  for k,(task_l,f) in enumerate(TASKS):
   if f==0:continue
   for candidate in range(3):
    counter=make_stages(learned_base,dl,dr,candidate,f)
    err=nrmse(apply(counter,xs[k]),target[k])
    (correct if candidate==task_l else wrong).append(err)
  active=3*D*D+(2*D if name in ('mirror_factorized_locus_views','native_lowrank_locus_conditioning') else 0)
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   heldout_output_nrmse=nrmse(out[HELD],target[HELD]),all_task_output_nrmse=nrmse(out,target),
   input_locus_ablation_nrmse=locus_errors[0],recurrent_locus_ablation_nrmse=locus_errors[1],readout_locus_ablation_nrmse=locus_errors[2],
   correct_locus_view_nrmse=float(np.mean(correct)),wrong_locus_view_nrmse=float(np.mean(wrong)),
   active_macs_per_example=active,reconstruction_macs_total=decode,test_examples=9*N,optimizer_updates=0,wall_time_s=wall,
   payload_roundtrip_max_abs=rt,status_note='exact native low-rank alias' if name=='mirror_factorized_locus_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
