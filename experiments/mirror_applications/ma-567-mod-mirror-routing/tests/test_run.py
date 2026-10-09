import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma567',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_route_and_control_rows():
 r=m.run(56701);assert len(r)==4;assert {x['method'] for x in r}=={'mod_shared','mirror_view','direct_coeff','independent'}
def test_mirror_direct_equal_quality():
 d={x['method']:x for x in m.run(56702)};assert float(d['mirror_view']['routed_nmse'])==float(d['direct_coeff']['routed_nmse'])
