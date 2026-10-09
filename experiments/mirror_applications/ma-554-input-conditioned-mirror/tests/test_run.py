import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma554',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_rows_and_finite_quality():
 r=m.run(55401);assert len(r)==5;assert all(float(x['heldout_output_nmse'])>=0 for x in r)
def test_direct_control_is_mirror_equivalent():
 r=m.run(55402);d={x['method']:x for x in r};assert abs(float(d['direct_two_basis']['heldout_output_nmse'])-float(d['mirror_angle']['heldout_output_nmse']))<1e-8
