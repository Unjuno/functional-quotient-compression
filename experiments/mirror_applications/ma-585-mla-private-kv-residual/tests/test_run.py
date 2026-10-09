import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma585',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_frontier_rows_finite():
 r=m.run(58501);assert len(r)==4*(4+6);assert all(float(x['attention_output_nmse'])>=0 for x in r)
def test_sweeps_locked():assert len(m.RHOS)==4 and len(m.FRACS)==6
