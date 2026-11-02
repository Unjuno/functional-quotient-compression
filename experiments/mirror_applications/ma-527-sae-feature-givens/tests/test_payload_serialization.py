import importlib.util, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma527_run',ROOT/'source'/'run_givens.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def test_paid_state_npz_is_uncompressed_and_exact():
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'state.npz'; a={'pool':np.arange(16,dtype=np.int16),'code':np.arange(16,dtype=np.float32)}
  runner.save(path,a); b=np.load(path,allow_pickle=False)
  assert all(np.array_equal(a[k],b[k]) for k in a)
  import zipfile
  with zipfile.ZipFile(path) as z: assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
