import csv,hashlib,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class TestArtifacts(unittest.TestCase):
 def test_fresh_rows_hash_and_sizes(self):
  with (ROOT/'RESULTS_CORE.csv').open() as f:r=list(csv.DictReader(f))
  self.assertEqual(len(r),144);self.assertEqual({int(x['world']) for x in r},{48610,48611,48612})
  for x in r:
   b=(ROOT/x['path'].split('ma-486-sparse-dictionary-mirror/',1)[1]).read_bytes();self.assertEqual(len(b),int(x['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),x['hash'])
if __name__=='__main__':unittest.main()
