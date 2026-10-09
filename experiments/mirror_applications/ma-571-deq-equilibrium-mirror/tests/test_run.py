import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma571',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_methods_stable_and_rows():
 r=m.run(57101);assert len(r)==4;assert all(x['stable_tasks']==8 for x in r)
def test_direct_code_exact_alias():
 d={x['method']:x for x in m.run(57102)};assert d['mirror_angle']['serialized_bytes']==d['direct_coeff']['serialized_bytes'];assert d['mirror_angle']['nmse']==d['direct_coeff']['nmse']
