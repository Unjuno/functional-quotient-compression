import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma557',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_direct_gain_alias_and_rows():
 r=m.run(55701);assert len(r)==15
 a=[x for x in r if x['method']=='mirror_roles'];b=[x for x in r if x['method']=='direct_gains']
 assert [x['serialized_bytes'] for x in a]==[x['serialized_bytes'] for x in b]
def test_widths_locked():assert m.WIDTHS==(2,4,8)
