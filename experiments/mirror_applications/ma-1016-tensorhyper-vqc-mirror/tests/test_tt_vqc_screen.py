import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import tt_vqc_screen as q

def test_givens_task_code_reconstructs_rank2_tt_core():
    native=q.task_codes_native(q.PHASES)
    mirror=q.task_codes_native(q.PHASES)
    assert (native==mirror).all()

def test_tt_control_and_mirror_match_teacher_expectations():
    shared=q.cores(); a=q.circuit_table('native_tt',shared); b=q.circuit_table('mirror_phase',shared)
    for task in range(q.TASKS):
        for x in q.patterns(): assert abs(q.expectation(x,a[task])-q.expectation(x,b[task]))<1e-12

def test_mirror_and_native_scalar_control_have_identical_paid_bytes():
    assert q.pack('mirror_phase',q.cores())==q.pack('native_scalar_phase',q.cores())

def test_payload_roundtrip_contains_all_inference_state():
    header,values=q.unpack(q.pack('mirror_phase',q.cores()))
    assert header['schema']=='MA1016-TTVQC-inference-v1'
    assert values['task_phase'].shape==(q.TASKS,)
    assert {'tt_layer_core','tt_qubit_core','tt_axis_core'}<=set(values)

def test_teacher_contains_four_distinct_logical_label_functions():
    shared=q.cores(); angles=q.circuit_table('mirror_phase',shared); labels=[]
    for task in range(q.TASKS): labels.append(tuple(1 if q.expectation(x,angles[task])>=0 else -1 for x in q.patterns()))
    assert len(set(labels))==4

def test_serialized_mirror_state_reconstructs_angles():
    _,v=q.unpack(q.pack('mirror_phase',q.cores()))
    shared=(v['tt_layer_core'],v['tt_qubit_core'],v['tt_axis_core'])
    angles=[q.tt_angles(q.task_codes_native([phi])[0],shared) for phi in v['task_phase']]
    assert len(angles)==q.TASKS and all(a.shape==(q.LAYERS,q.NQ,2) for a in angles)
