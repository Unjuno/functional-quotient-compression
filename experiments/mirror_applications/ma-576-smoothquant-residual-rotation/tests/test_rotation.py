import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py';sp=importlib.util.spec_from_file_location('ma576',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def test_rotation_inverse_preserves_matrix():
 r=np.random.default_rng(576);w=r.normal(size=(4,512)).astype(np.float32);c=m.new_code(r);z=m.rotate(w,c);x=m.unrotate(z,c);np.testing.assert_allclose(x,w,atol=2e-6,rtol=2e-6)
def test_payload_quant_roundtrip_shape():
 w=np.random.default_rng(7).normal(size=(3,512)).astype(np.float32);p,s=m.quant(w);d=m.dequant(p,s);assert d.shape==w.shape;assert np.linalg.norm(d-w)/np.linalg.norm(w)<.12
