import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import Teacher, Student, make_data, bytes_and_hash

def test_teacher_and_student_shapes():
    t=Teacher(7); x,y=make_data(8,t,11)
    assert x.shape==(11,12) and y.shape==(11,8)
    for m in ('static','condconv','mirror','mlp_gate'):
        s=Student(m,9)
        assert s(x).shape==y.shape

def test_payload_is_deterministic():
    a=Student('mirror',9); b=Student('mirror',9)
    assert bytes_and_hash(a,'mirror',9)==bytes_and_hash(b,'mirror',9)
