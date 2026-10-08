from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_training_split_covers_each_axis_and_withholds_six_pairs():
    assert len(run.TRAIN)==18 and len(run.HELD)==6
    assert all(any(t==i for t,l in run.TRAIN) for i in range(run.TASKS))
    assert all(any(l==i for t,l in run.TRAIN) for i in range(run.LAYERS))
    assert not set(run.TRAIN)&set(run.HELD)

def test_teacher_adapter_rank_is_at_most_two():
    _,target=run.teacher(46101)
    assert torch.linalg.matrix_rank(target.reshape(-1,run.D*run.D),atol=1e-5).item()<=2

def test_native_lowrank_and_mirror_start_as_same_parameterization():
    a=run.make_model('mirror',46101);b=run.make_model('native_lowrank',46101)
    for k in ('task','layer','fc1','fc2'):
        for x,y in zip(getattr(a['net'],k).parameters(),getattr(b['net'],k).parameters()):assert torch.equal(x,y)
    assert torch.equal(a['basis'],b['basis'])
    assert torch.equal(run.output('mirror',a,2,1),run.output('native_lowrank',b,2,1))
