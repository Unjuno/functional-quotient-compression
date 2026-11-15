"""Verify MA-1030 inputs, serialized session payloads, and metric replay."""
import hashlib,json
from pathlib import Path
import numpy as np
from run_experiment import ROOT,RUNS,CKPT,CONFIG,score_masks
WORLDS={'audit_1':103001,'audit_2':103002,'audit_3':103003}
METHODS={'independent','hard_shared','mirror_givens','additive_rank1','mirror_private_fallback'}

def verify():
    rows=[json.loads(x) for x in (RUNS/'audit_rows.jsonl').read_text().splitlines()]
    assert len(rows)==15 and len({(r['world'],r['method']) for r in rows})==15
    for world,seed in WORLDS.items():
        root=ROOT/world;manifest=json.loads((root/'manifest.json').read_text())
        assert manifest['seed']==seed and len(manifest['rows'])==6
        for item in manifest['rows']:
            assert hashlib.sha256((root/'frames'/item['jpeg']).read_bytes()).hexdigest()==item['jpeg_sha256']
            assert hashlib.sha256((root/'masks'/f"{item['frame']:05}.npz").read_bytes()).hexdigest()==item['gt_sha256']
        cache=RUNS/f'shared_video_cache_{world}.bin';cache_blob=cache.read_bytes()
        assert hashlib.sha256(cache_blob).hexdigest()==next(r['shared_video_cache_sha256'] for r in rows if r['world']==world)
        for row in (r for r in rows if r['world']==world):
            payload=(RUNS/row['payload_file']).read_bytes();predfile=RUNS/row['prediction_file'];predblob=predfile.read_bytes()
            assert len(payload)==row['session_payload_bytes'] and hashlib.sha256(payload).hexdigest()==row['payload_sha256']
            assert hashlib.sha256(predblob).hexdigest()==row['prediction_sha256']
            header,body=payload.split(b'\n',1);h=json.loads(header)
            assert len(body)==sum(t['bytes'] for t in h['tensors'])
            assert h['encoded_memory_tensor_bytes']==row['object_memory_payload_bytes']
            assert row['full_session_payload_bytes']==CKPT.stat().st_size+CONFIG.stat().st_size+len(cache_blob)+len(payload)
            pred=np.load(predfile)['masks'].astype(bool)
            truth=[np.load(root/'masks'/f"{i:05}.npz")['masks'].astype(bool) for i in range(6)]
            scored=score_masks(list(pred),truth)
            for key in ('mean_j','mean_boundary_f','mean_jf','id_switches','cross_object_iou'):
                assert abs(float(scored[key])-float(row[key]))<1e-12,(world,row['method'],key,scored[key],row[key])
            if row['method']=='mirror_private_fallback':
                assert all(x['object_id'] in manifest['object_ids'] for x in row['fallback_frames'])
                assert all(sum(a['name'].endswith('/private_memory') for a in h['tensors'] if a['name'].startswith(f"memory/frame_{x['frame']}/"))==1 for x in row['fallback_frames'])
    return {'passed':True,'audit_worlds':len(WORLDS),'method_world_rows':len(rows),'input_hashes_exact':True,'payload_hashes_lengths_and_manifests_exact':True,'prediction_hashes_and_metrics_replayed':True,'full_session_byte_arithmetic_exact':True}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
