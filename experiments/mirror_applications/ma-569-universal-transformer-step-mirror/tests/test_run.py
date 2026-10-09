import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma569',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_controls_and_rows():assert len(m.run(56901))==5
def test_direct_time_code_matches_mirror():
 d={x['method']:x for x in m.run(56902)};assert d['mirror_step']['nmse']==d['direct_time_code']['nmse'];assert d['mirror_step']['serialized_bytes']==d['direct_time_code']['serialized_bytes']
