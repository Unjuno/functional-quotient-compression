"""Reconstruct MA-1026 prompt states from payloads and replay the official-test metrics."""
import hashlib,importlib.util,json,pathlib
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';TOP=ROOT.parents[2]
src=ROOT/'source'/'maple_view.py';sp=importlib.util.spec_from_file_location('ma1026',src);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
d=json.loads((RUNS/'development.json').read_text());assert d['fresh_opened_after_all_prompt_states_frozen']
assert d['code_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
prev24=json.loads((TOP/'experiments/mirror_applications/ma-1024-coop-prompt-mirror/runs/development.json').read_text())
prev25=json.loads((TOP/'experiments/mirror_applications/ma-1025-cocoop-instance-view/runs/development.json').read_text())
for c,ids in d['train_indices'].items():assert set(ids).isdisjoint(set(prev24['train_indices'][c])|set(prev25['train_indices'][c]))
for c,ids in d['audit_indices'].items():assert set(ids).isdisjoint(set(prev24['audit_indices'][c])|set(prev25['audit_indices'][c]))
p,model,tok,proc,files,bbytes=m.get_assets();assert files==d['model_files'] and bbytes==d['backbone_bytes'];model.requires_grad_(False);model.eval()
for q in model.parameters():q.requires_grad_(False)
test=m.load_dataset(m.DATASET,split='test')
for c,ids in d['audit_indices'].items():
    hashes=[hashlib.sha256(test[int(ix)]['img'].tobytes()).hexdigest() for ix in ids]
    assert hashes==d['audit_image_sha256'][c]
train=m.load_dataset(m.DATASET,split='train');records,pairs=m.data_records(train,d['train_indices'],True)
assert pairs==d['heldout_base_domain_pairs']
for di,items in enumerate(records):assert all((c,di) not in {(x['class_id'],x['domain_id']) for x in pairs} for im,c,ix in items)
init=m.initial_context(model,tok);replay={'zero_shot':m.evaluate('zero_shot',None,model,tok,proc,test,d['audit_indices'])}
methods=['shared_maple','independent_domain_maple','mirror_maple','additive_rank1_multimodal']
for method in methods:
    raw=(RUNS/f'payload_{method}.bin').read_bytes();assert hashlib.sha256(raw).hexdigest()==d['payload_sha256'][method]
    head,blob=raw.split(b'\n',1);meta=json.loads(head);assert meta['method']==method and meta['n_context']==m.NCTX and meta['prompt_depth']==m.DEPTH
    assert meta['classes']==m.CLASSES and meta['domains']==m.DOMAINS and 'prompt_mapping' in meta
    state={};offset=0
    for ent in meta['tensor_manifest']:
        shape=tuple(ent['shape']);nb=int(np.prod(shape))*4
        state[ent['name']]=torch.from_numpy(np.frombuffer(blob[offset:offset+nb],dtype='<f4').copy().reshape(shape));offset+=nb
    assert offset==len(blob) and len(raw)==d['prompt_payload_bytes'][method]
    net=m.PromptState(init,method);net.load_state_dict(state,strict=True);net.eval()
    replay[method]=m.evaluate(method,net,model,tok,proc,test,d['audit_indices'])
raw=(RUNS/'payload_zero_shot.bin').read_bytes();meta=json.loads(raw.split(b'\n',1)[0]);assert meta['method']=='zero_shot' and hashlib.sha256(raw).hexdigest()==d['payload_sha256']['zero_shot'];assert len(raw)==d['prompt_payload_bytes']['zero_shot']
for method,r in replay.items():
    for key in ('base_accuracy','novel_accuracy','harmonic_mean','heldout_base_domain_pair_accuracy'):
        assert abs(r[key]-d['results'][method][key])<1e-12,(method,key,r[key],d['results'][method][key])
for method,size in d['prompt_payload_bytes'].items():assert d['full_bundle_bytes'][method]==bbytes+size
assert d['results']['mirror_maple']['harmonic_mean']<d['results']['additive_rank1_multimodal']['harmonic_mean']
print(json.dumps({'asset_hashes':True,'previous_experiment_train_audit_disjointness':True,'audit_hashes':True,'heldout_domain_class_cells':True,'payload_hashes_and_full_bundle_arithmetic':True,'serialized_prompt_replay_exact':True,'mirror_specific_gate':'FAIL: additive multimodal control has higher H-mean'},sort_keys=True))
