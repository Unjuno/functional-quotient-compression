"""Full deterministic SAM2 memory-state replay and payload audit."""
import hashlib,importlib.util,json,pathlib,subprocess
import numpy as np,torch
from sam2.build_sam import build_sam2_video_predictor
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs';TOP=E.parents[2];REPO=pathlib.Path('/tmp/ma1027-sam2');CKPT=REPO/'sam2.1_hiera_tiny.pt';CONFIG=REPO/'sam2/configs/sam2.1/sam2.1_hiera_t.yaml';FR=R/'video_frames';GT=R/'gt_masks'
src=E/'source'/'run_sam2_memory.py';sp=importlib.util.spec_from_file_location('ma1027',src);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
d=json.loads((R/'development.json').read_text());scene=json.loads((R/'scene_manifest.json').read_text())
assert d['code_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==d['official_commit']
assert CKPT.stat().st_size==d['checkpoint_bytes'] and hashlib.sha256(CKPT.read_bytes()).hexdigest()==d['checkpoint_sha256']
assert CONFIG.stat().st_size==d['config_bytes'] and hashlib.sha256(CONFIG.read_bytes()).hexdigest()==d['config_sha256']
assert scene['seed']==102701 and len(scene['rows'])==5
for row in scene['rows']:
    assert hashlib.sha256((FR/row['jpeg']).read_bytes()).hexdigest()==row['jpeg_sha256']
    assert hashlib.sha256((GT/f"{row['frame']:05}.npz").read_bytes()).hexdigest()==row['gt_sha256']
# Validate common, normalized frame cache payload and replay all four predictor settings.
predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(CKPT),device='cpu',apply_postprocessing=False,hydra_overrides_extra=[f'++model.image_size={m.IMAGE_SIZE}'])
state=predictor.init_state(str(FR),offload_video_to_cpu=False,offload_state_to_cpu=False);cache=np.ascontiguousarray(state['images'].cpu().numpy().astype('<f4'));cache_raw=json.dumps({'schema':'MA1027-frame-cache-v1','shape':list(cache.shape),'dtype':'float32-le'},sort_keys=True,separators=(',',':')).encode()+b'\n'+cache.tobytes(order='C');del state
saved_cache=(R/'shared_video_cache.bin').read_bytes();assert cache_raw==saved_cache;assert hashlib.sha256(saved_cache).hexdigest()==d['common_video_cache_sha256']
replay={}
for method in m.METHODS:
    old=np.load(R/f'predictions_{method}.npz')['masks'].copy()
    result=m.run_method(predictor,method,scene,saved_cache);new=np.load(R/f'predictions_{method}.npz')['masks']
    assert np.array_equal(old,new),(method,'prediction masks differ')
    recorded=d['results'][method]
    for k in ('mean_j','mean_boundary_f','mean_jf','cross_object_iou'):
        assert abs(result[k]-recorded[k])<1e-12,(method,k,result[k],recorded[k])
    assert result['id_switches']==recorded['id_switches']
    assert result['session_payload_bytes']==recorded['session_payload_bytes']
    assert result['payload_sha256']==recorded['payload_sha256']
    assert result['prediction_sha256']==recorded['prediction_sha256']
    raw=(R/f'payload_{method}.bin').read_bytes();head,body=raw.split(b'\n',1);meta=json.loads(head);off=0
    for ent in meta['tensors']:
        off+=ent['bytes']
    assert off==len(body) and meta['method']==method
    assert d['results'][method]['full_session_payload_bytes']==d['checkpoint_bytes']+d['config_bytes']+len(saved_cache)+len(raw)
    replay[method]={'mean_jf':result['mean_jf'],'session_payload_bytes':len(raw),'payload_hash':result['payload_sha256']}
# Sampled predictions are checked against recorded GT geometry and expected four-way tensor shape.
for method in m.METHODS:
    preds=np.load(R/f'predictions_{method}.npz')['masks'];assert preds.shape==(5,4,256,256)
print(json.dumps({'official_sam2_commit_checkpoint_and_config_hashes':True,'scene_frame_and_ground_truth_hashes':True,'common_video_cache_replay_exact':True,'all_four_methods_full_forward_replay_exact':True,'self_describing_payload_byte_manifest_exact':True,'full_session_arithmetic_exact':True,'methods':replay,'decision':'FAIL: quality collapse and additive control dominance'},sort_keys=True))
