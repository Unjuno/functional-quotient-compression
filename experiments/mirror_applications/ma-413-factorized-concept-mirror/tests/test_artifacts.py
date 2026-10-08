import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA413(unittest.TestCase):
 def test_payloads(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),126)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_locked_combination_and_gate(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x];self.assertEqual(len(rows),54)
  methods={m:[r for r in rows if r['method']==m] for m in ('factor_mirror','factor_film','direct_table')}
  means={m:statistics.mean(r['fresh_heldout_11_accuracy'] for r in rs) for m,rs in methods.items()}
  self.assertLess(max(means['factor_mirror']-means['factor_film'],means['factor_mirror']-means['direct_table']),.05)
  self.assertEqual({r['world'] for r in methods['factor_mirror']},{41310,41311,41312})
if __name__=='__main__':unittest.main()
