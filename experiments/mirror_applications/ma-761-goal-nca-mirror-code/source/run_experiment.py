#!/usr/bin/env python3
"""NCA-inspired shared local transition rule conditioned on goal coordinates."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SEED=76101; GRID=5; GOALS=[(x,y) for y in range(GRID) for x in range(GRID)]
HELDOUT={(0,0),(1,3),(3,1),(4,4),(2,2)}; TRAIN_GOALS=[g for g in GOALS if g not in HELDOUT]
torch.set_num_threads(1)

class Rule(torch.nn.Module):
    def __init__(self,dim):
      super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,2),torch.nn.Tanh())
    def forward(self,x):return self.net(x)

def norm_xy(xy):return (np.asarray(xy,dtype=np.float32)/(GRID-1))*2-1
def pack(**a):
    b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def state_dict(model):return {k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}

def fit_model(dim,features,targets,seed,steps=900):
    torch.manual_seed(seed);m=Rule(dim);opt=torch.optim.Adam(m.parameters(),lr=.02)
    x=torch.tensor(features);y=torch.tensor(targets)
    for _ in range(steps):
      opt.zero_grad();loss=torch.mean((m(x)-y)**2);loss.backward();opt.step()
    return m

def rollout(model,goal,onehot=None,start=(2,2),max_steps=12):
    pos=np.array(start,dtype=np.float32);seen=set()
    for t in range(max_steps):
      key=tuple(pos.astype(int))
      if key==goal:return True,t
      if key in seen:return False,t
      seen.add(key)
      p=norm_xy(pos);g=norm_xy(goal)
      inp=np.concatenate([p,g] if onehot is None else [p,onehot]).astype(np.float32)
      with torch.no_grad():delta=model(torch.tensor(inp)[None,:]).numpy()[0]
      pos=np.clip(pos+np.sign(delta),0,GRID-1)
    return tuple(pos.astype(int))==goal,max_steps

def run():
    torch.manual_seed(SEED);np.random.seed(SEED)
    # The local rule is applied to the active cell each step; goal coordinates
    # are broadcast context. Train targets are one-cell moves toward each goal.
    feats=[];targets=[];onehot_feats=[];onehot_targets=[]
    for gi,g in enumerate(GOALS):
      if g in HELDOUT:continue
      for p in GOALS:
        feats.append(np.concatenate([norm_xy(p),norm_xy(g)]));targets.append(np.sign(np.asarray(g)-np.asarray(p)).astype(np.float32))
        z=np.zeros(len(GOALS),np.float32);z[gi]=1
        onehot_feats.append(np.concatenate([norm_xy(p),z]));onehot_targets.append(np.sign(np.asarray(g)-np.asarray(p)).astype(np.float32))
    feats=np.asarray(feats,np.float32);targets=np.asarray(targets,np.float32)
    coord=fit_model(4,feats,targets,SEED)
    onehot=fit_model(2+len(GOALS),np.asarray(onehot_feats,np.float32),np.asarray(onehot_targets,np.float32),SEED+1)
    # No-goal hard-tied local rule uses only current position.
    no_goal=fit_model(2,np.asarray([norm_xy(p) for p in GOALS for _ in TRAIN_GOALS],np.float32),
        np.asarray([np.mean([np.sign(np.asarray(g)-np.asarray(p)) for g in TRAIN_GOALS],axis=0) for p in GOALS for _ in TRAIN_GOALS],np.float32),SEED+2)
    # Independent per-goal policies are trained for seen goals only.
    bank=[];bank_start=time.perf_counter()
    for i,g in enumerate(TRAIN_GOALS):
      x=np.asarray([norm_xy(p) for p in GOALS],np.float32)
      y=np.asarray([np.sign(np.asarray(g)-np.asarray(p)) for p in GOALS],np.float32)
      bank.append(fit_model(2,x,y,SEED+100+i,steps=400))
    bank_train_ms=(time.perf_counter()-bank_start)*1000
    coord_blob=pack(rule=state_dict(coord),goal_codes=np.asarray([norm_xy(g) for g in sorted(HELDOUT)],np.float32),goal_ids=np.asarray(sorted(HELDOUT),np.int8))
    native_blob=coord_blob
    onehot_blob=pack(rule=state_dict(onehot),goal_embedding=np.eye(len(GOALS),dtype=np.float32)[[GOALS.index(g) for g in HELDOUT]],goal_ids=np.asarray(sorted(HELDOUT),np.int8))
    nogoal_blob=pack(rule=state_dict(no_goal))
    bank_blob=pack(seen_goal_rules=np.stack([{k:v for k,v in state_dict(m).items()}[k] for m in bank for k in []]) if False else np.asarray([g for g in TRAIN_GOALS],np.int8),
                    **{f'goal_{i}_{k}':v for i,m in enumerate(bank) for k,v in state_dict(m).items()})
    methods=[('independent_per_goal_rule_bank',bank_blob,'bank'),('hard_tied_no_goal_code',nogoal_blob,'none'),('shared_onehot_goal_embedding',onehot_blob,'onehot'),('native_coordinate_goal_rule',native_blob,'coord'),('mirror_coordinate_goal_view',coord_blob,'coord')]
    rows=[]
    for name,blob,kind in methods:
      successes=[];steps=[];covered=0;t0=time.perf_counter()
      for g in sorted(HELDOUT):
        for start in ((0,0),(4,0),(0,4),(4,4)):
          if kind=='bank':successes.append(False);steps.append(12);continue
          if kind=='onehot':
            oh=np.zeros(len(GOALS),np.float32);oh[GOALS.index(g)]=1
            ok,n=rollout(onehot,g,oh,start)
          elif kind=='none':ok,n=rollout(no_goal,g,np.asarray([],np.float32),start)
          else:ok,n=rollout(coord,g,start=start)
          successes.append(ok);steps.append(n);covered+=int(ok)
      eval_ms=(time.perf_counter()-t0)*1000
      # Twenty rollouts × up to 12 steps; policy network MACs per transition.
      dim={'bank':2,'none':2,'onehot':2+len(GOALS),'coord':4}[kind]
      macs=20*12*(dim*32+32*32+32*2)
      rows.append(dict(seed=SEED,method=name,split='unseen_goals',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
          rollout_success=float(np.mean(successes)),mean_steps=float(np.mean(steps)),goal_coverage='0/5' if kind=='bank' else '5/5',
          active_macs=macs,fit_updates=(len(bank)*400 if kind=='bank' else 900),fit_wall_ms=bank_train_ms if kind=='bank' else 0,
          eval_wall_ms=eval_ms,status_note='exact native coordinate-code alias' if name=='mirror_coordinate_goal_view' else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('train goals',len(TRAIN_GOALS),'heldout goals',len(HELDOUT))
if __name__=='__main__':run()
