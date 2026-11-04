import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run_experiment as run

def test_serialized_private_frontier_replay():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==48;d={(r['seed'],r['residual_rms'],r['method']):r for r in rows}
 for seed in (68801,68802):
  for rho in run.RHOS:
   protected,_basis,teacher,x=run.make_world(seed,rho);held=np.arange(24,32)
   names=['projected_full']+[f'mirror_q{q}' for q in run.QRANKS]+[f'native_q{q}' for q in (0,2,4)]
   for name in names:
    p=ROOT/'source'/f'{seed}_rho{rho:g}_{name}.npz';row=d[(seed,rho,name)];assert p.stat().st_size==row['serialized_bytes']
    with np.load(p) as z:decoded=run.decode(z)
    err,leak,_=run.eval_metrics(teacher,decoded,x,protected,held)
    assert abs(err-row['output_nrmse'])<1e-7
    assert abs(leak-row['protected_leakage'])<1e-7
    rank=0 if name=='projected_full' or name.endswith('q0') else int(name.split('q')[-1])
    expected_decode=0 if name=='projected_full' else run.M*(run.R+rank)*run.D*run.D
    assert row['decode_mac_proxy']==expected_decode
    assert row['active_mac_proxy']==expected_decode+8*run.N*run.D*run.D
   for q in (0,2,4):assert abs(d[(seed,rho,f'mirror_q{q}')]['output_nrmse']-d[(seed,rho,f'native_q{q}')]['output_nrmse'])<1e-10
