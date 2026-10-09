import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma556',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_frontier_rows_and_exact_independent():
 r=m.run(55601);assert len(r)==4*(4+6);assert all(float(x['nmse'])>=0 for x in r)
def test_rho_and_fraction_sweep_locked():
 assert m.RHOS==(0.0,.1,.3,.6) and len(m.FRACS)==6
