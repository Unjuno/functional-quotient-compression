import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma565',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_control_rows_and_alias():
 r=m.run(56501);assert len(r)==12
 a=[x for x in r if x['method']=='mirror_role'];b=[x for x in r if x['method']=='direct_gate'];assert [x['serialized_bytes'] for x in a]==[x['serialized_bytes'] for x in b]
def test_depths_locked():assert m.DEPTHS==(2,3,4)
