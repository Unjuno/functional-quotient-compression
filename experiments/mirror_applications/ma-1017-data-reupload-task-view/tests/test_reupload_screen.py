import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import reupload_screen as q

def test_circuit_has_fixed_reupload_schedule():
    assert len(q.inputs())==16 and q.BLOCKS==3
    assert q.pack('mirror_reupload',q.angles())==q.pack('native_scalar_schedule',q.angles())

def test_phase_view_changes_function_on_grid():
    t=q.angles(); x=q.inputs()[3]
    assert abs(q.expectation(x,q.PHASES[0],t)-q.expectation(x,q.PHASES[2],t))>1e-8

def test_data_and_gate_resources_match_all_views():
    assert q.pack('mirror_reupload',q.angles())==q.pack('native_scalar_schedule',q.angles())

def test_task_codes_select_four_distinct_classifiers():
    t=q.angles(); functions=[]
    for phi in q.PHASES: functions.append(tuple(1 if q.expectation(x,phi,t)>=0 else -1 for x in q.inputs()))
    assert len(set(functions))==4

def test_payload_charges_every_schedule_and_code():
    m=q.pack('mirror_reupload',q.angles());i=q.pack('independent_full',q.angles())
    assert len(m)==379 and len(i)==523
    assert len(i)>len(m)
