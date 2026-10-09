import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma372',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_factorized_view_recovers_heldout_and_direct_control():
 rows,s=m.run(37201); d={x['method']:x for x in rows}
 assert float(d['mirror_factorized']['heldout_nmse'])==0
 assert s['mirror']==s['direct']
def test_six_locked_mixes():
 assert len(m.HOLD)==6 and not set(m.HOLD)&set(m.TRAIN)
