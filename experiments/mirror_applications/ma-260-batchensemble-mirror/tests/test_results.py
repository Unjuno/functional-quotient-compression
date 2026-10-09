import csv,hashlib,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class Audit(unittest.TestCase):
 def test_27_rows_and_worlds(self):
  q=list(csv.DictReader((R/"FRESH_RESULTS.csv").open()));self.assertEqual(len(q),27);self.assertEqual({int(x["world"]) for x in q},{26010,26011,26012})
 def test_checkpoint_bytes_hash(self):
  q=list(csv.DictReader((R/"FRESH_RESULTS.csv").open()))
  for x in q:
   p=R.parents[2]/x["path"];b=p.read_bytes();self.assertEqual(len(b),int(x["payload_bytes"]));self.assertEqual(hashlib.sha256(b).hexdigest(),x["sha256"])
if __name__=="__main__":unittest.main()
