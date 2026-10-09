import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma586',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_methods_and_packed_bytes():
 r=m.run(58601);assert len(r)==4;assert all(int(x['serialized_bytes'])>0 for x in r)
def test_mirror_direct_alias():
 d={x['method']:x for x in m.run(58602)};assert d['mirror_givens']['serialized_bytes']==d['direct_givens']['serialized_bytes'];assert d['mirror_givens']['attention_output_nmse']==d['direct_givens']['attention_output_nmse']
