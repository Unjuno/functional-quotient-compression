import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma553',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_factorized_direct_and_mirror_are_exact_cost_alias():
 rows,b=m.run(55301);assert b['direct']==b['mirror'];assert len(rows)==6
def test_heldout_pairs_locked():
 assert len(m.HOLD)==6 and len(m.TRAIN)==18 and not set(m.HOLD)&set(m.TRAIN)
