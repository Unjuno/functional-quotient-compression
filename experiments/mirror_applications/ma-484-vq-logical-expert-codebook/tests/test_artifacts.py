import csv,hashlib,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class TestArtifacts(unittest.TestCase):
 def test_fresh_row_and_payload_integrity(self):
  with (ROOT/'RESULTS_CORE.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),144)
  self.assertEqual({int(x['world']) for x in rows},{48410,48411,48412})
  for r in rows:
   b=(ROOT/r['path'].split('ma-484-vq-logical-expert-codebook/',1)[1]).read_bytes()
   self.assertEqual(len(b),int(r['payload_bytes']))
   self.assertEqual(hashlib.sha256(b).hexdigest(),r['hash'])
if __name__=='__main__':unittest.main()
