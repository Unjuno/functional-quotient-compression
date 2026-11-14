#!/usr/bin/env python3
"""Deserialize shards, access old_indices only, and lock row positions; do not index or inspect labels/coordinates."""
import argparse,hashlib,json,pathlib,torch

def key(role,molecule,old_idx):return hashlib.sha256(f'MA1004-v1|{role}|{molecule}|{old_idx}'.encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=pathlib.Path(a.data_dir)
 specs=[('basis_fit','benzene',0,100000,128),('basis_fit','toluene',0,100000,128),('basis_fit','naphthalene',0,100000,128),('target_support','uracil',0,80000,32),('development','uracil',80000,100000,128),('fresh_audit','uracil',100000,133770,64)]
 rows=[]
 for role,mol,lo,hi,limit in specs:
  data=torch.load(root/f'{mol}.pt',map_location='cpu',weights_only=False)
  idx=data['old_indices'].detach().cpu().tolist();selected=[(int(v),i) for i,v in enumerate(idx) if lo<=int(v)<hi]
  selected.sort(key=lambda q:key(role,mol,q[0]))
  kept=selected[:limit]
  rows.append({'role':role,'molecule':mol,'old_index_window':[lo,hi],'available_rows':len(selected),'selected_rows':[{'row':i,'old_index':v,'selection_sha256':key(role,mol,v)} for v,i in kept]})
  print(role,mol,'available',len(selected),'selected',len(kept),flush=True)
 result={'schema':'ma1004-rmd17-row-manifest-v1','selection':'hash-order by SHA256(MA1004-v1|role|molecule|old_index), no metric/position/force/energy inspected','fresh_labels_opened':False,'splits':rows}
 out=pathlib.Path(a.out);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print('manifest_sha256',hashlib.sha256(out.read_bytes()).hexdigest())
if __name__=='__main__':main()
