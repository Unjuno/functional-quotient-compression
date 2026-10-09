import csv,hashlib,io,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class TestArtifacts(unittest.TestCase):
 def test_fresh_serialized_rows(self):
  with (ROOT/'RESULTS_CORE.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),18)
  self.assertEqual({int(r['world']) for r in rows},{48310,48311,48312})
  for r in rows:
   b=(ROOT/r['path'].split('ma-483-adaptive-rvq-depth/',1)[1]).read_bytes()
   self.assertEqual(len(b),int(r['payload_bytes']))
   self.assertEqual(hashlib.sha256(b).hexdigest(),r['hash'])
   o=torch.load(io.BytesIO(b),weights_only=False)
   codes=o['codes'];depth=o['depth']
   if codes.ndim==1:
    full=torch.zeros(o['N'],4,dtype=torch.uint8);p=0
    for i,d in enumerate(depth.tolist()):full[i,:d]=codes[p:p+d];p+=d
    codes=full
   y=torch.zeros(o['N'],16);sc=[1.,.5,.25,.125]
   for j in range(4):
    mask=depth>j;idx=codes[:,j].long();live=mask & (idx<32);ph=idx*(2*torch.pi/32)
    y[live,2*j]=sc[j]*ph[live].cos();y[live,2*j+1]=sc[j]*ph[live].sin()
   # errors are replayed against deterministic fresh teacher in standalone runner; integrity/hashes checked here.
if __name__=='__main__':unittest.main()
