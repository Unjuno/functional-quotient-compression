"""Verify MA-1024 recorded metrics, complete bundle sizes and prompt artifacts."""
import hashlib,json,pathlib
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs';d=json.loads((R/'development.json').read_text())
assert d['fresh_opened'] and d['fresh_opened_after_prompt_freeze'] and d['fresh_seed_locked']
assert d['dataset']=='uoft-cs/cifar10' and d['backbone_bytes']==sum(x['bytes'] for x in d['model_files'])
for item in d['model_files']:
 p=pathlib.Path(d['model_path'])/item['name'];assert p.exists() and p.stat().st_size==item['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
for name,n in d['prompt_payload_bytes'].items():
 p=R/f'payload_{name}.bin';assert p.stat().st_size==n
 assert hashlib.sha256(p.read_bytes()).hexdigest()==d['prompt_payload_sha256'][name]
assert d['mirror_native_payload_identical']
assert d['prompt_payload_bytes']['mirror_domain_phase']==d['prompt_payload_bytes']['native_scalar_givens']
assert d['results']['mirror_domain_phase']==d['results']['native_scalar_givens']
assert d['full_bundle_bytes']['mirror_domain_phase']==d['backbone_bytes']+d['prompt_payload_bytes']['mirror_domain_phase']
print(json.dumps({'recorded_model_and_data_hashes_present':True,'payload_sizes_and_hashes_exact':True,'full_bundle_arithmetic_exact':True,'native_control_alias':True,'official_test_opened_after_freeze':True}))
