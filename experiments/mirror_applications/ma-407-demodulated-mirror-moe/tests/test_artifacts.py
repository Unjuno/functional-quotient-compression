import hashlib,json,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA407(unittest.TestCase):
 def test_payload_hashes_and_bytes(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),126)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_fresh_demod_is_functionally_null_here(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines()];rows=[r for r in rows if r['split']=='fresh']
  by={(r['world'],r['seed'],r['method']):r for r in rows}
  for w in (40710,40711,40712):
   for s in range(3):
    a=by[(w,s,'mirror_raw')];b=by[(w,s,'mirror_demod')];self.assertEqual(a['accuracy'],b['accuracy']);self.assertLess(abs(a['activation_rms_cv']-b['activation_rms_cv']),1e-7)
if __name__=='__main__':unittest.main()
