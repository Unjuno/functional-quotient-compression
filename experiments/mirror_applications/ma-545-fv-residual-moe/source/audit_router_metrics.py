#!/usr/bin/env python3
"""Post-hoc descriptive route confusion/F1 from frozen MA-545 artifacts only."""
import argparse,json,time
from pathlib import Path
import numpy as np
from task_data_and_pinning import TASKS,load_model
from run_experiment import demo_set,prompt_with_query,get_hidden,route

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);ap.add_argument('--experiment-dir',required=True);a=ap.parse_args()
 model,tok,torch=load_model(a.model_dir);base=Path(a.experiment_dir);report={}
 for name in ('dev_54501','dev_54502','fresh_54511','fresh_54512','fresh_54513'):
  root=base/'results'/name
  if not root.exists():continue
  manifest=json.loads((root/'split_manifest.json').read_text())
  with np.load(root/'routed_fv.npz',allow_pickle=False) as z:router={k:z[k] for k in ('router_mean','router_std','router_weight','router_bias')}
  preds=[];truth=[];t0=time.perf_counter()
  for tid,task in enumerate(manifest):
   for x,_ in task['queries']:
    feat,_=get_hidden(model,tok,torch,prompt_with_query(x,demo_set(task['support'],x)))
    pid,_=route({'mean':router['router_mean'],'std':router['router_std'],'weight':router['router_weight'],'bias':router['router_bias']},feat[None,:])
    preds.append(int(pid[0]));truth.append(tid)
  n=len(TASKS);cm=np.zeros((n,n),dtype=np.int64)
  for y,p in zip(truth,preds):cm[y,p]+=1
  f1=[]
  for i in range(n):
   tp=cm[i,i];fp=cm[:,i].sum()-tp;fn=cm[i,:].sum()-tp
   f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
  report[name]={'route_accuracy':float(np.mean(np.asarray(preds)==np.asarray(truth))),'macro_f1':float(np.mean(f1)),'confusion_matrix':cm.tolist(),'class_order':[t[0] for t in TASKS],'queries':len(truth),'audit_seconds':time.perf_counter()-t0}
  (root/'route_audit.json').write_text(json.dumps(report[name],indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
