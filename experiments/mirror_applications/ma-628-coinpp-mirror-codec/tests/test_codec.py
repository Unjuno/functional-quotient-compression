import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def test_freeze_hash_and_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_quality_storage_and_native_alias():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [62801,62802]:
  for b in [4,6,8,10,12]:
   m=[r for r in rows if r['seed']==s and r['method']=='mirror_quantized_code' and r['bits_per_amplitude_phase']==b]
   n=[r for r in rows if r['seed']==s and r['method']=='native_coinpp_modulation' and r['bits_per_amplitude_phase']==b]
   assert len(m)==len(n)==4
   assert all(a['serialized_bytes']==b2['serialized_bytes'] and a['heldout_nrmse']==b2['heldout_nrmse'] for a,b2 in zip(m,n))
  q=[r for r in rows if r['seed']==s and r['method']=='mirror_quantized_code' and r['bits_per_amplitude_phase']==12]
  g=next(r for r in rows if r['seed']==s and r['method']=='independent_grid_u8')
  assert max(r['heldout_nrmse'] for r in q)<=1e-3 and q[0]['serialized_bytes']<=.25*g['serialized_bytes']
  assert (P/'source'/f'{s}_mirror_native_b12.bin').stat().st_size==q[0]['serialized_bytes']
