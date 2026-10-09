import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_screen.py'; spec=importlib.util.spec_from_file_location('ma546',P); ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
def test_invertible_coordinate_transforms_preserve_function():
 w1,w2,x,views=ma.make(54601); base=x+(x@w1.T)@w2.T
 for g in views.values():
  out=x+((x@(g@w1).T)@(w2@np.linalg.inv(g).astype(np.float32)).T)
  assert np.max(np.abs(out-base)) < 1e-5
def test_unpaired_transform_changes_function_and_payload_pays_views(tmp_path):
 w1,w2,x,views=ma.make(54602);g=views['orthogonal']; base=x+(x@w1.T)@w2.T; out=x+(x@(g@w1).T)@w2.T
 assert np.linalg.norm(out-base)/np.linalg.norm(base)>1e-5
 n,_=ma.write_payload(tmp_path/'bank.npz',{'W1':g@w1,'W2':w2,'G':g,'G_inv':np.linalg.inv(g).astype(np.float32)})
 assert n>(w1.nbytes+w2.nbytes)
