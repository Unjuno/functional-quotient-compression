import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma371',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_direct_and_mirror_storage_formula_is_same():
 rows,s=m.runworld(37101)
 assert s['direct']==s['mirror'] and len(rows)==24
def test_fixed_mix_count():
 assert len(m.TRAIN_CONFIGS)==3 and len(m.MIXES)==3
