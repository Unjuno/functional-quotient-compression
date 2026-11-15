"""Check MA-1029 saved prediction metrics, artifact hashes and serialized bytes."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

E=Path(__file__).resolve().parents[1];R=E/'runs';DOMAINS=('texture_contrast','low_contrast_camouflage','partial_occlusion')

def boundary_f(pred,truth,tol=2):
    p=torch.as_tensor(pred,dtype=torch.float32)[None,None];g=torch.as_tensor(truth,dtype=torch.float32)[None,None]
    pe=F.avg_pool2d(p,3,1,1)==1;ge=F.avg_pool2d(g,3,1,1)==1;pb=(p>0)&~pe;gb=(g>0)&~ge
    k=2*tol+1;pd=F.max_pool2d(pb.float(),k,1,tol)>0;gd=F.max_pool2d(gb.float(),k,1,tol)>0
    pr=float((pb&gd).sum()/(pb.sum()+1e-8));rc=float((gb&pd).sum()/(gb.sum()+1e-8));return 2*pr*rc/(pr+rc+1e-8)

def verify():
    manifest=json.loads((R/'data_manifest_audit.json').read_text())
    assert len(manifest['rows'])==54
    for item in manifest['rows']:
        assert hashlib.sha256((R/item['image']).read_bytes()).hexdigest()==item['image_sha256']
        assert hashlib.sha256((R/item['mask']).read_bytes()).hexdigest()==item['mask_sha256']
    rows=[json.loads(x) for x in (R/'audit_rows.jsonl').read_text().splitlines()]
    assert len(rows)==18
    assert len({(r['world'],r['method']) for r in rows})==18
    for row in rows:
        payload=R/f"payload_{row['world']}_{row['method']}.pt"
        blob=payload.read_bytes()
        assert len(blob)==row['adaptation_payload_bytes']
        assert hashlib.sha256(blob).hexdigest()==row['adaptation_payload_sha256']
        all_iou=[];all_bf=[]
        for domain in DOMAINS:
            predpath=R/f"pred_{row['world']}_{row['method']}_{domain}.npz"
            assert hashlib.sha256(predpath.read_bytes()).hexdigest()==row['prediction_files'][domain]
            z=np.load(predpath);pred=z['pred'].astype(bool);truth=z['truth'].astype(bool)
            assert pred.shape==truth.shape==(4,256,256)
            ious=[];bfs=[]
            for p,g in zip(pred,truth):
                union=np.logical_or(p,g).sum();inter=np.logical_and(p,g).sum()
                ious.append(float(inter/union) if union else 1.)
                bfs.append(boundary_f(p,g))
            assert abs(float(np.mean(ious))-row['per_domain'][domain]['miou'])<1e-12
            assert abs(float(np.mean(bfs))-row['per_domain'][domain]['boundary_f'])<1e-12
            all_iou.extend(ious);all_bf.extend(bfs)
        assert abs(float(np.mean([row['per_domain'][d]['miou'] for d in DOMAINS]))-row['macro_miou'])<1e-12
        assert abs(float(np.mean([row['per_domain'][d]['boundary_f'] for d in DOMAINS]))-row['macro_boundary_f'])<1e-12
        assert row['full_deploy_bytes']==375042383+row['deployment_source_bytes']+row['adaptation_payload_bytes']
    return {'rows':len(rows),'audit_examples':len(manifest['rows']),'passed':True,'prediction_metrics_replayed':True,'payload_hashes_exact':True,'full_deploy_arithmetic_exact':True}

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
