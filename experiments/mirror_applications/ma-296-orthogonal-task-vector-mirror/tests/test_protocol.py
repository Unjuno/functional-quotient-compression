import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py'
s=importlib.util.spec_from_file_location('ma296',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_task_generator_shapes_and_determinism():
 a=m.make(29600,0);b=m.make(29600,0)
 assert a[0].shape==(8,512) and a[3].shape==(8,768,512)
 assert (a[0]==b[0]).all()
def test_payload_accounts_for_address():
 p=m.pack([__import__('torch').zeros(8,512)],'raw')
 q=m.pack([__import__('torch').zeros(8,512),__import__('torch').zeros(8,8)],'coded')
 assert len(q)>len(p)
