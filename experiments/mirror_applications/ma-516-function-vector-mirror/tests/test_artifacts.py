import csv,hashlib,io,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class T(unittest.TestCase):
 def test_rows_payloads_and_vector_metric_replay(self):
  with (ROOT/'RESULTS_CORE.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),54)
  self.assertEqual({int(r['world']) for r in rows},{51610,51611,51612})
  by={(int(r['world']),int(r['seed']),r['method'],int(r['rank'])):r for r in rows}
  for key,row in by.items():
   if not row['path']: continue
   blob=(ROOT/row['path'].split('ma-516-function-vector-mirror/',1)[1]).read_bytes()
   self.assertEqual(len(blob),int(row['intervention_payload_bytes']))
   self.assertEqual(hashlib.sha256(blob).hexdigest(),row['hash'])
   obj=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
   if row['method']=='mirror_pca':
    refrow=by[(key[0],key[1],'explicit',0)]
    refblob=(ROOT/refrow['path'].split('ma-516-function-vector-mirror/',1)[1]).read_bytes()
    ref=torch.load(io.BytesIO(refblob),map_location='cpu',weights_only=False)['vectors']
    rec=obj['mean']+obj['coords'].float()@obj['basis'].T
    err=float((rec-ref).norm()/(ref.norm()+1e-12))
    self.assertAlmostEqual(err,float(row['vector_nrmse']),places=6)
if __name__=='__main__':unittest.main()
