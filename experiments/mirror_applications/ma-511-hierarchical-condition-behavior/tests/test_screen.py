import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_screen.py'
spec=importlib.util.spec_from_file_location('ma511',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_split_is_whole_pair_and_connected():
 w=m.make_world(51101)
 assert len(w['visible'])==48 and len(w['held'])==16
 assert set(map(tuple,w['visible'])).isdisjoint(set(map(tuple,w['held'])))
 assert len(set(w['visible'][:,0]))==m.C and len(set(w['visible'][:,1]))==m.B
def test_additive_fit_generalizes_zero_private():
 w=m.make_world(51101); f=m.fit(w,4,4,0)
 x=m.eval_metrics(w,f,0)
 assert x['heldout_relative_rmse']<=.05
def test_private_pair_residual_is_not_composed():
 w=m.make_world(51101); f=m.fit(w,4,4,.25)
 x=m.eval_metrics(w,f,.25)
 assert x['heldout_relative_rmse']>.05
