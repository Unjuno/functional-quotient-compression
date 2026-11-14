# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independently retrain three locked A9 conditions and check byte identity."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from experiments.mirror_capacity.A8 import a8
from experiments.mirror_capacity.A9 import a9


def replay(out:Path):
    cfg=json.loads((out/'config.json').read_text())
    rows=list(csv.DictReader((out/'train.csv').open()))
    candidates=[('mirror','scaffold_code4'),('mirror','random_full4'),('shift','scaffold_full4')]
    results=[]
    for kind,regime in candidates:
        matches=[r for r in rows if r['kind']==kind and r['regime']==regime]
        if not matches:continue
        r=matches[0]
        w=a8.make_world(int(r['world']),'ramp')
        m=a9.fit(w,kind,regime,int(r['seed']),steps=cfg['steps'],batch=cfg['batch'])['model']
        recalculated=a8.serialize(m)
        stored=(out/'models'/r['model_file']).read_bytes()
        ok=recalculated==stored
        results.append({'kind':kind,'regime':regime,'world':int(r['world']),
                        'seed':int(r['seed']),'byte_identical':ok,
                        'sha256_replayed':hashlib.sha256(recalculated).hexdigest(),
                        'sha256_stored':r['sha256']})
        if not ok:raise AssertionError('replay mismatch: '+str(r))
    # Independent functional identity: code-only scaffold students retain the same
    # family-structured function, and all three have identical task codes.
    eq=[]
    for world in cfg['worlds']:
        for seed in cfg['seeds']:
            cases=[r for r in rows if int(r['world'])==world and int(r['seed'])==seed
                   and r['regime']=='scaffold_code4']
            if len(cases)!=3:continue
            models={r['kind']:a8.deserialize((out/'models'/r['model_file']).read_bytes()) for r in cases}
            x=torch.linspace(-2,2,64).reshape(-1,1).repeat(16,1)
            task=torch.arange(16).repeat_interleave(64)
            with torch.no_grad():
                a=models['mirror'](x,task)
                deviations={k:float((m(x,task)-a).abs().max()) for k,m in models.items()}
            code_match=bool(torch.equal(models['mirror'].code.weight,models['pre'].code.weight)
                            and torch.equal(models['mirror'].code.weight,models['shift'].code.weight))
            if max(deviations.values())>1e-6 or not code_match:
                raise AssertionError('trained scaffold diverged: '+str((world,seed,deviations)))
            eq.append({'world':world,'seed':seed,'max_abs_diff':max(deviations.values()),
                       'code_bit_identical':code_match})
    doc={'replays':results,'trained_scaffold_equivalence':eq,
         'all_pass':all(x['byte_identical'] for x in results) and all(x['code_bit_identical'] for x in eq)}
    (out/'replay.json').write_text(json.dumps(doc,indent=2)+'\n')
    print('A9_REPLAY_PASS',len(results),'scaffold_equivalence',len(eq),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();replay(args.out)
