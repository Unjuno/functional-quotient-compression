import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma539',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_split_disjoint_and_world_reproducible():
 a=m.make_world(53901);b=m.make_world(53901);assert np.array_equal(a[0],b[0]);assert np.array_equal(a[1],b[1]);assert np.array_equal(a[2],b[2])
 for i in range(m.FUNCTIONS):assert set(a[1][i,:,0]).isdisjoint(set(a[2][i,:,0]))
def test_packet_construction_four_slots():
 _,sup,_=m.make_world(42);x,y=m.packets(sup,11);assert x.shape==(16,2,4);assert y.shape==x.shape
