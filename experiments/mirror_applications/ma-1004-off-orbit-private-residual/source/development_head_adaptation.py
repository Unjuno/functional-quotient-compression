#!/usr/bin/env python3
"""Development-only low-rank last-readout adaptation feasibility study for MA-1004."""
import argparse, copy, hashlib, io, json, pathlib, time
import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from mace.calculators import MACECalculator, mace_mp
KCAL_MOL_TO_EV=0.0433641153087705

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=3);a=ap.parse_args()
 torch.set_num_threads(1);torch.manual_seed(100401)
 exp=pathlib.Path(__file__).resolve().parents[1]; man=json.loads((exp/'runs/ROW_MANIFEST.json').read_text()); data_root=pathlib.Path(a.data_dir)
 model=mace_mp(model='small',device='cpu',return_raw_model=True,default_dtype='float64')
 for p in model.parameters():p.requires_grad_(False)
 calc=MACECalculator(models=model,device='cpu',default_dtype='float64')
 head=model.readouts[-1].linear_2; base=head.weight.detach().clone(); bias=head.bias.detach().clone() if head.bias is not None else None
 width=base.numel(); assert width==16, width
 batches={};labels={}; counts={}
 for split in man['splits']:
  mol=split['molecule']; raw=torch.load(data_root/f'{mol}.pt',map_location='cpu',weights_only=False)
  z=raw['z'].tolist(); ptr=raw['ptr'];pos=raw['pos']; force=raw['forces']
  selected=split['selected_rows']
  # use first 8 fixed hash-selected rows for source basis and target support
  if split['role']=='basis_fit':selected=selected[:8]
  if split['role']=='target_support':selected=selected[:8]
  if split['role'] not in ('basis_fit','target_support','development'):continue
  for row in selected:
   ix=row['row'];lo=int(ptr[ix]);hi=int(ptr[ix+1]);atoms=Atoms(numbers=z,positions=pos[lo:hi].numpy())
   b=calc._atoms_to_batch(atoms).to_dict();batches[(mol,ix)]=b;labels[(mol,ix)]=force[lo:hi].clone()*KCAL_MOL_TO_EV
  counts[split['role']]=len(selected)
 dev_split=next(x for x in man['splits'] if x['role']=='development');dev=dev_split['selected_rows']
 ur=next(x for x in man['splits'] if x['role']=='target_support');target_support=ur['selected_rows'][:8]
 src=[x for x in man['splits'] if x['role']=='basis_fit']
 def activate(delta):
  def forward(x):return type(head).forward(head,x,weight=base+delta.reshape_as(base),bias=bias)
  head.forward=forward
 def pred(mol,ix,delta,train=False):
  activate(delta)
  b={k:(v.clone() if torch.is_tensor(v) else v) for k,v in batches[(mol,ix)].items()}
  b['positions'].requires_grad_(True)
  return model(b,training=train,compute_force=True,compute_virials=False,compute_stress=False)['forces']
 def train_delta(mol,rows,delta,steps,lr=.01,reg=1e-3):
  delta=torch.nn.Parameter(delta.detach().clone());opt=torch.optim.Adam([delta],lr=lr);hist=[]
  for q in range(steps):
   row=rows[q%len(rows)];ix=row['row'];out=pred(mol,ix,delta,True);gold=labels[(mol,ix)]
   loss=(out-gold).square().mean()+reg*delta.square().mean();opt.zero_grad();loss.backward();opt.step();hist.append(float(loss.detach()))
  return delta.detach(),hist
 t0=time.perf_counter();src_deltas=[];source_hist={}
 # 3 epochs over 8 structures per source task; train only the 16-value last readout weight.
 for sp in src:
  mol=sp['molecule'];rows=sp['selected_rows'][:8];d,h=train_delta(mol,rows,torch.zeros(width,dtype=base.dtype),a.epochs*len(rows));src_deltas.append(d);source_hist[mol]={'updates':len(h),'last_force_mse':h[-1]}
 D=torch.stack(src_deltas);D=D-D.mean(0,keepdim=True);_,_,vh=torch.linalg.svd(D,full_matrices=False);B=vh[:2].T.contiguous()
 # Fit rank-2 molecule code on the fixed 8-shot support; no development labels enter adaptation.
 support_rows=target_support
 m=torch.nn.Parameter(torch.zeros(2,dtype=base.dtype));opt=torch.optim.Adam([m],lr=.02);target_hist=[]
 for q in range(a.epochs*len(support_rows)):
  row=support_rows[q%len(support_rows)];ix=row['row'];out=pred('uracil',ix,B@m,True);gold=labels[('uracil',ix)]
  loss=(out-gold).square().mean()+1e-3*m.square().mean();opt.zero_grad();loss.backward();opt.step();target_hist.append(float(loss.detach()))
 mval=m.detach().clone();mirror_delta=B@mval
 # Support-gradient selected private four-coordinate residual, layered over the fitted rank-2 code.
 rfull=torch.nn.Parameter(torch.zeros(width,dtype=base.dtype));opt=torch.optim.Adam([m,rfull],lr=.02);mask=None
 # Coordinate salience from 8-shot force residual gradient, evaluated at the code fit.
 grad=torch.zeros(width,dtype=base.dtype)
 for row in support_rows:
  ix=row['row'];out=pred('uracil',ix,mirror_delta,True);loss=(out-labels[('uracil',ix)]).square().mean();g=torch.autograd.grad(loss,head.parameters(),allow_unused=True) if False else None
  # Estimate coordinate sensitivity by a local differentiable readout offset vector.
  d=torch.zeros(width,dtype=base.dtype,requires_grad=True);activate(mirror_delta+d);b={k:(v.clone() if torch.is_tensor(v) else v) for k,v in batches[('uracil',ix)].items()};b['positions'].requires_grad_(True);o=model(b,training=True,compute_force=True,compute_virials=False,compute_stress=False)['forces'];ll=(o-labels[('uracil',ix)]).square().mean();grad+=torch.autograd.grad(ll,d)[0].detach().abs()
 mask=torch.zeros(width,dtype=base.dtype);mask[torch.topk(grad,4).indices]=1
 opt.zero_grad(set_to_none=True)
 for q in range(a.epochs*len(support_rows)):
  row=support_rows[q%len(support_rows)];ix=row['row'];out=pred('uracil',ix,B@m+mask*rfull,True);loss=(out-labels[('uracil',ix)]).square().mean()+1e-3*(m.square().mean()+rfull.square().mean());opt.zero_grad();loss.backward();opt.step()
 private_delta=B@m.detach()+mask*rfull.detach()
 # Full 16-coordinate readout fine-tune is a small direct sparse-state upper control, not full-model FT.
 direct,_=train_delta('uracil',support_rows,torch.zeros(width,dtype=base.dtype),a.epochs*len(support_rows),lr=.01)
 configs={'mirror_rank2':mirror_delta,'mirror_plus_top4_private':private_delta,'direct_16_readout':direct}
 # Each config gets its own support-derived energy zero point; all force and energy quality is development-only.
 out_rows={};runtime={}
 for name,delta in configs.items():
  es=[];sums=[];t_eval=time.perf_counter()
  for row in support_rows:
   ix=row['row'];activate(delta);b={k:(v.clone() if torch.is_tensor(v) else v) for k,v in batches[('uracil',ix)].items()};b['positions'].requires_grad_(True);o=model(b,training=False,compute_force=True,compute_virials=False,compute_stress=False);pass
  # labels are read only on support/development; raw energies are available in the dataset shard.
  off=0.0
  # Force-only evaluation here; the established frozen baseline script already provides calibrated energy metrics.
  for row in dev:
   ix=row['row'];activate(delta);b={k:(v.clone() if torch.is_tensor(v) else v) for k,v in batches.get(('uracil',ix),{}).items()}
   if not b:
    raw=torch.load(data_root/'uracil.pt',map_location='cpu',weights_only=False);z=raw['z'].tolist();ptr=raw['ptr'];pos=raw['pos'];lo=int(ptr[ix]);hi=int(ptr[ix+1]);atoms=Atoms(numbers=z,positions=pos[lo:hi].numpy());b=calc._atoms_to_batch(atoms).to_dict();batches[('uracil',ix)]=b;labels[('uracil',ix)]=raw['forces'][lo:hi].clone()*KCAL_MOL_TO_EV;b={k:(v.clone() if torch.is_tensor(v) else v) for k,v in b.items()}
   b['positions'].requires_grad_(True);t=time.perf_counter();o=model(b,training=False,compute_force=True,compute_virials=False,compute_stress=False);runtime.setdefault(name,[]).append(time.perf_counter()-t);gold=labels[('uracil',ix)];err=(o['forces'].detach()-gold).abs();sums.extend(err.flatten().tolist())
  out_rows[name]={'dev_force_component_mae_eV_A':float(np.mean(sums)),'dev_rows':len(dev),'adaptation_updates':a.epochs*len(support_rows),'seconds_inference_mean':float(np.mean(runtime[name]))}
 # Save exact parameter vector and rank2 basis objects, including mask and metadata, to measure actual inference payload bytes.
 payloads={}
 objects={'mirror_rank2':{'basis':B.cpu(),'code':mval.cpu(),'model_sha256':'2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736','format':'ma1004-head-view-v1'},'mirror_plus_top4_private':{'basis':B.cpu(),'code':m.detach().cpu(),'private_values':(mask*rfull.detach())[mask.bool()].cpu(),'private_indices':torch.nonzero(mask).flatten().to(torch.int64).cpu(),'model_sha256':'2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736','format':'ma1004-head-view-private-v1'},'direct_16_readout':{'delta':direct.cpu(),'model_sha256':'2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736','format':'ma1004-head-direct-v1'}}
 for name,obj in objects.items():
  buf=io.BytesIO();torch.save(obj,buf);payloads[name]={'serialized_bytes':len(buf.getvalue()),'sha256':hashlib.sha256(buf.getvalue()).hexdigest()}
 result={'experiment_id':'MA-1004','role':'development screening; no fresh quality access','data':'rMD17 CC0','prior':'MACE-MP small','seed':100401,'target_support_shots':8,'development_rows':len(dev),'source_basis_rows_per_molecule':8,'source_training':source_hist,'rank':2,'basis_sha256':hashlib.sha256(B.cpu().numpy().tobytes()).hexdigest(),'target_updates':a.epochs*len(support_rows),'private_residual_coordinates':int(mask.sum().item()),'private_mask_indices':torch.nonzero(mask).flatten().tolist(),'development':out_rows,'payloads':payloads,'train_wall_s':time.perf_counter()-t0,'fresh_rows_accessed':False,'notes':['Force-only last-readout adaptation feasibility screen. Energy is not reported in this script.','direct_16_readout is a matched low-dimensional sparse-head control, not PA300 full EquiSparseDeltaSTR or full-model fine-tuning.','Mirror and native rank-2 factorization are the same parameterization here; any equivalent scores do not support a Mirror-specific claim.']}
 pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_training',)},indent=2))

if __name__=='__main__':main()
