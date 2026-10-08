import hashlib,json,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA408(unittest.TestCase):
 def test_payloads(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),105)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_fresh_gates(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x];self.assertEqual(len(rows),45)
  by={m:[r for r in rows if r['method']==m] for m in ('condconv','output_mix')}
  import statistics
  delta=abs(statistics.mean(r['accuracy'] for r in by['condconv'])-statistics.mean(r['accuracy'] for r in by['output_mix']))
  self.assertLessEqual(delta,.01)
  self.assertLessEqual(by['condconv'][0]['active_mac_proxy_per_example']/by['output_mix'][0]['active_mac_proxy_per_example'],.70)
  self.assertLessEqual(by['condconv'][0]['serialized_bytes']/by['output_mix'][0]['serialized_bytes'],1.1)
if __name__=='__main__':unittest.main()
