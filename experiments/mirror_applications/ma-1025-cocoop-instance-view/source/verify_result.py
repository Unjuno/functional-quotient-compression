"""Replay MA-1025 metrics from serialized prompt payloads and locked audit IDs."""
import hashlib,importlib.util,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; TOP=ROOT.parents[2]
src=ROOT/'source'/'cocoop_mirror.py'; spec=importlib.util.spec_from_file_location('ma1025',src); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=json.loads((RUNS/'development.json').read_text()); prior=json.loads((TOP/'experiments'/'mirror_applications'/'ma-1024-coop-prompt-mirror'/'runs'/'development.json').read_text())
assert d['fresh_opened'] and d['fresh_opened_after_all_prompt_freeze'] and d['disjoint_from_MA1024']
assert d['code_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
assert d['train_indices'] and d['audit_indices']
for c,ids in d['train_indices'].items(): assert set(ids).isdisjoint(prior['train_indices'][c])
for c,ids in d['audit_indices'].items(): assert set(ids).isdisjoint(prior['audit_indices'][c])
path,model,tok,proc,files,bbytes=m.get_assets(); assert bbytes==d['backbone_bytes'] and files==d['model_files']
model.requires_grad_(False); model.eval()
for c,ids in d['audit_indices'].items():
    assert [hashlib.sha256(m.load_dataset(m.DATASET,split='test')[int(i)]['img'].tobytes()).hexdigest() for i in ids]==d['audit_image_sha256'][c]
ds=m.load_dataset(m.DATASET,split='test'); x,y,_=m.feats(model,proc,ds,d['audit_indices'])
ctx=m.init_ctx(model,tok); replay={}
with __import__('torch').inference_mode():
    replay['zero_shot']=m.eval_net(None,'zero',model,tok,x,y)
for kind,cls in [('shared_coop',m.Shared),('cocoop',m.Full),('mirror',m.Mir),('additive_rank2',m.Add)]:
    raw=(RUNS/f'payload_{kind}.bin').read_bytes(); assert hashlib.sha256(raw).hexdigest()==d['payload_sha256'][kind]
    head,blob=raw.split(b'\n',1); meta=json.loads(head); assert meta['method']==kind
    assert meta['token_template']=='x x x x {class}.' and meta['score_scale']==100.0
    assert meta['image_feature_normalization']=='L2' and meta['text_feature_normalization']=='L2'
    assert 'conditioner_and_view' in meta
    if kind=='mirror': assert meta['conditioner_and_view']['token_pairs']==[[0,1],[2,3]]
    state={}; offset=0
    for t in meta['tensors']:
        shape=tuple(t['shape']); count=int(np.prod(shape)); nbytes=count*4
        state[t['name']]=np.frombuffer(blob[offset:offset+nbytes],dtype='<f4').copy().reshape(shape); offset+=nbytes
    assert offset==len(blob)
    net=cls(ctx); net.load_state_dict({k:__import__('torch').from_numpy(v) for k,v in state.items()},strict=True); net.eval()
    replay[kind]=m.eval_net(net,kind,model,tok,x,y)
    assert len(raw)==d['prompt_payload_bytes'][kind]
zraw=(RUNS/'payload_zero_shot.bin').read_bytes(); zmeta=json.loads(zraw.split(b'\n',1)[0])
assert zmeta['method']=='zero_shot' and zmeta['token_template']=='a photo of a {class}.'
assert hashlib.sha256(zraw).hexdigest()==d['payload_sha256']['zero_shot']
assert len(zraw)==d['prompt_payload_bytes']['zero_shot']
for kind,r in replay.items():
    for metric in ('base_accuracy','novel_accuracy','harmonic_mean'):
        assert abs(r[metric]-d['results'][kind][metric])<1e-12,(kind,metric,r[metric],d['results'][kind][metric])
for kind,n in d['prompt_payload_bytes'].items(): assert d['full_bundle_bytes'][kind]==bbytes+n
assert d['results']['mirror']['novel_accuracy']+0.01 >= d['results']['cocoop']['novel_accuracy']
assert d['results']['mirror']['novel_accuracy']==d['results']['additive_rank2']['novel_accuracy']
assert d['prompt_payload_bytes']['mirror']<d['prompt_payload_bytes']['additive_rank2']
print(json.dumps({'payload_and_backbone_hashes':True,'audit_hashes_and_sample_disjointness':True,'serialized_prompt_replay_exact':True,'full_bundle_arithmetic':True,'mirror_equals_rank2_additive_quality':True,'status':'FAIL per frozen specificity gate'},sort_keys=True))
