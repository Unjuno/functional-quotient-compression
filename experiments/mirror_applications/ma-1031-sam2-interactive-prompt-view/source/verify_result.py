"""Verify MA-1031 fresh input, prompt-history payloads and saved metric replay."""
import hashlib,json
from pathlib import Path
import numpy as np
from run_interactive import E,RUNS,ROOT,CKPT,CONFIG,score_masks
WORLDS={'audit_1':103101,'audit_2':103102,'audit_3':103103};METHODS={'no_correction','direct_clicks','grid8_clicks','mirror_polar','full_mask_upper'}
def verify():
    rows=[json.loads(x) for x in (RUNS/'audit_rows.jsonl').read_text().splitlines()]
    assert len(rows)==15 and len({(r['world'],r['method']) for r in rows})==15
    source=sum(p.stat().st_size for p in pathlib_sam())+sum((E/'source'/x).stat().st_size for x in ('run_interactive.py','generate_video.py'))+(E/'PROTOCOL.json').stat().st_size
    assert all(r['source_bytes']==source for r in rows)
    for world,seed in WORLDS.items():
        root=ROOT/world;manifest=json.loads((root/'manifest.json').read_text());assert manifest['seed']==seed
        for m in manifest['rows']:
            assert hashlib.sha256((root/'frames'/m['jpeg']).read_bytes()).hexdigest()==m['jpeg_sha256']
            assert hashlib.sha256((root/'masks'/f"{m['frame']:05}.npz").read_bytes()).hexdigest()==m['gt_sha256']
        for row in (r for r in rows if r['world']==world):
            hb=(RUNS/row['history_file']).read_bytes();pb=(RUNS/row['prediction_file']).read_bytes()
            assert len(hb)==row['click_history_bytes'] and hashlib.sha256(hb).hexdigest()==row['history_sha256']
            assert hashlib.sha256(pb).hexdigest()==row['prediction_sha256']
            assert row['full_session_bytes']==CKPT.stat().st_size+CONFIG.stat().st_size+row['source_bytes']+row['session_state_bytes']
            assert row['session_payload_retained'] is False and len(row['session_sha256'])==64
            pred=np.load(RUNS/row['prediction_file'])['masks'].astype(bool)
            truth=[np.load(root/'masks'/f'{i:05}.npz')['masks'].astype(bool) for i in range(6)]
            scored=score_masks(list(pred),truth)
            for k in ('mean_j','mean_boundary_f','mean_jf','id_switches'):assert abs(scored[k]-row[k])<1e-12
            if row['method'] in ('direct_clicks','grid8_clicks','mirror_polar'):assert row['click_count']==40
    return {'passed':True,'audit_worlds':3,'method_world_rows':15,'inputs_and_outputs_hashed':True,'history_payloads_exact':True,'saved_metrics_replayed':True,'session_byte_arithmetic_exact':True,'session_payload_hashes_recorded_but_large_state_bytes_not_retained':True}
def pathlib_sam():
    from pathlib import Path
    return Path('/tmp/ma1027-sam2/sam2').rglob('*.py')
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
