"""MA-1165 audit: recorded source hashes, checkpoint hashes, and exact fresh replay."""
from __future__ import annotations
import csv,hashlib,itertools,json,math
from pathlib import Path
import torch
from model import VirtualLoop, make_teacher
from run_experiment import inputs,task_nmse,MODES,TASKS,LOOPS,SEEDS,STEPS,M,N

def verify(root:Path):
    torch.set_num_threads(1)
    results=list(csv.DictReader((root/'results'/'RESULTS_CORE.csv').open()))
    assert len(results)==len(MODES)*len(TASKS)*len(LOOPS)*len(SEEDS)==72
    labels={r['label'] for r in results};assert len(labels)==72
    hashes=json.loads((root/'SOURCE_SHA256.json').read_text())
    for name,digest in hashes.items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest
    checks=0
    for rec in results:
        label=rec['label'];kind=rec['task'];loops=int(rec['loops']);seed=int(rec['seed']);mode=rec['mode']
        assert label==f'{kind}_{mode}_L{loops}_s{seed}'
        checkpoint=root/'checkpoints'/f'{label}.pt'
        assert checkpoint.exists()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==rec['checkpoint_sha256']
        payload=torch.load(checkpoint,weights_only=True,map_location='cpu')
        model=VirtualLoop(mode,loops,seed=seed+1200)
        model.load_state_dict(payload['model'])
        assert model.learned_count()==int(rec['parameters'])
        assert model.payload_bytes()==int(rec['inference_bytes'])
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        teacher=make_teacher(kind,loops,seed)
        fresh=inputs(4096,300000+seed)
        nmse,_=task_nmse(model,teacher,fresh)
        assert abs(nmse-float(rec['fresh_nmse']))<1e-7,(label,nmse,rec['fresh_nmse'])
        if mode=='tied':assert float(rec['mean_virtual_weight_difference'])==0.
        if mode=='switch':assert float(rec['mean_virtual_weight_difference'])>1e-4
        checks+=1
    return {
        'verification':'PASS','records':checks,'source_sha256':hashes,'checkpoint_hashes_verified':checks,
        'fresh_replays_verified':checks,'exact_unique_configurations':True,
        'single_vector_parameter_count':M,'virtual_weights_per_step':N,
        'note':'Fixed-update synthetic CPU experiment only. No general capacity or VRAM speed claim.'
    }

if __name__=='__main__':
    root=Path(__file__).parents[1]
    result=verify(root)
    (root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
