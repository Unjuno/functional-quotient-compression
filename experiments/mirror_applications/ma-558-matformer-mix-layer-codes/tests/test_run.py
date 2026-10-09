import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma558',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_exact_direct_alias():
 r=m.run(55801);d={x['method']:x for x in r};assert d['mirror_factorized']['serialized_bytes']==d['direct_factorized']['serialized_bytes'];assert d['mirror_factorized']['heldout_nmse']=='0'
def test_six_locked_mixes():assert len(m.HOLD)==6
