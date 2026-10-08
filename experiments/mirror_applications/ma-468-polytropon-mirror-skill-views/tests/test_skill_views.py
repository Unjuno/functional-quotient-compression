from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_aligned_logical_skills_share_rank_one_view_basis():
    w=run.world(46801)[0]
    for p in range(run.PHYS):
        delta=w['skills'][2*p+1]-w['skills'][2*p]
        assert torch.linalg.matrix_rank(delta,atol=1e-5).item()==1

def test_task_skill_allocations_are_sparse():
    assert len(run.ALLOC)==run.TASKS
    assert all(len(row) in (1,2) for row in run.ALLOC)
    assert all(max(row)<run.NSKILL for row in run.ALLOC)

def test_native_lowrank_control_is_exact_view_map():
    teacher,data=run.world(46801);a=run.init('mirror',46801);b=run.init('native_lowrank',46801)
    for k in a:assert torch.equal(a[k],b[k])
    x=data['xq'][0]
    assert torch.equal(run.predict('mirror',a,0,x,teacher['allocation']),run.predict('native_lowrank',b,0,x,teacher['allocation']))
