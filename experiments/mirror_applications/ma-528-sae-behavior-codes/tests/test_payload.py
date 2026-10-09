import importlib.util,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('ma528_run',ROOT/'source'/'run_behavior_codes.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def test_uncompressed_code_payload_roundtrip():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'code.npz';a={'pool':np.arange(64,dtype=np.int16),'indices':np.arange(16,dtype=np.uint8),'coefficients':np.ones((16,16),np.float32)};r.save(p,a);b=np.load(p,allow_pickle=False)
  assert all(np.array_equal(a[k],b[k]) for k in a)
  with zipfile.ZipFile(p) as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
