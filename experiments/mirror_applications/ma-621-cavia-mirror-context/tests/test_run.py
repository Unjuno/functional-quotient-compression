import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma621',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_context_methods_and_disjoint_support_query():assert len(m.run(62101))==5
def test_direct_angle_alias():
 d={x['method']:x for x in m.run(62102)};assert d['mirror']['serialized_bytes']==d['direct_angle']['serialized_bytes'];assert d['mirror']['heldout_nmse']==d['direct_angle']['heldout_nmse']
