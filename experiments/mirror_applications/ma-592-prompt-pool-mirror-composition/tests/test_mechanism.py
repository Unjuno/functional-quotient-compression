import importlib.util
from pathlib import Path
import numpy as np
SRC=Path(__file__).parents[1]/'source'/'run_experiment.py';spec=importlib.util.spec_from_file_location('ma592',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_weighted_view_matches_native_top2_composition():
 rng=np.random.default_rng(592);p=rng.normal(size=(8,m.PLEN,m.HID)).astype(np.float32);q=rng.normal(size=m.HID).astype(np.float32);k=rng.normal(size=(8,m.HID)).astype(np.float32)
 ix,w,_=m.route_weights(q,k);native=np.einsum('k,kph->ph',w,p[ix]);mirror=m.mix_prompt(p,ix,w)
 assert np.array_equal(native,mirror)
def test_route_coordinates_are_normalized_and_bounded():
 q=np.ones(16,np.float32);keys=np.eye(16,dtype=np.float32)[:8];ix,w,sim=m.route_weights(q,keys)
 assert len(ix)==m.K and np.isclose(w.sum(),1.0) and np.all(w>=0) and sim.shape==(8,)
