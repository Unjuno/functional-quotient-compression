import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'));import run_experiment as run

def test_payload_replay_bytes_and_native_exact_alias():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==20;d={(r['seed'],r['method']):r for r in rows}
 for seed in (67401,67402):
  q=np.random.default_rng(seed);D=16;E=M=16;N=512;base=(q.normal(size=(D,D))/D**.5*.15).astype('f');u=(q.normal(size=(M,D))/D**.5).astype('f');v=(q.normal(size=(M,D))/D**.5).astype('f');c=q.normal(size=(E,M)).astype('f')*.3;router=(q.normal(size=(D,E))/D**.5).astype('f');x=q.normal(size=(N,D)).astype('f')
  target=None
  for method in ('full_bank','full_top2','mirror_top2','mirror_top4','mirror_top8','mirror_top16','native_top2','native_top4','native_top8','native_top16'):
   p=ROOT/'source'/f'{seed}_{method}.npz';row=d[(seed,method)];assert p.stat().st_size==row['serialized_bytes']
   with np.load(p) as z:
    if method.startswith('full_'):
     pred=run.forward_full(x,z['base'],z['experts'],z['router'],method=='full_top2')
     if method=='full_bank':target=pred
    else:
     code=z['expert_codes'] if 'expert_codes' in z else z['expert_scales'];pred=run.forward_atoms(x,z['base'],z['atom_left'],z['atom_right'],code,z['router'],int(z['top_k'][0]))
    err=float(np.sqrt(np.mean((pred-target)**2)/np.mean(target**2)))
    assert abs(err-row['output_nrmse'])<1e-7
  for k in (2,4,8,16):
   with np.load(ROOT/'source'/f'{seed}_mirror_top{k}.npz') as a,np.load(ROOT/'source'/f'{seed}_native_top{k}.npz') as b:
    assert np.array_equal(a['atom_left'],b['atom_left']);assert np.array_equal(a['atom_right'],b['atom_right']);assert np.array_equal(a['expert_codes'],b['expert_scales'])
