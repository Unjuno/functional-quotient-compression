import csv,hashlib,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class T(unittest.TestCase):
 def test_a1_hashes(self):
  with (ROOT/'RESULTS_CORE.csv').open() as f:r=list(csv.DictReader(f))
  self.assertEqual(len(r),63);self.assertEqual({int(x['world']) for x in r},{50120,50121,50122})
  for x in r:
   b=(ROOT/x['path'].split('ma-501-loreft-mirror-coordinate/',1)[1]).read_bytes();self.assertEqual(len(b),int(x['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),x['hash'])
if __name__=='__main__':unittest.main()
