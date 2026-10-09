from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_six_related_tasks_share_rank_one_direction():
    _,_,_,m=run.task_matrices(45701)
    base=m[0]
    updates=torch.stack([m[i]-base for i in range(1,6)])
    assert torch.linalg.matrix_rank(updates.reshape(5,-1),atol=1e-5).item()==1

def test_two_outliers_are_not_in_rank_one_task_family():
    base,left,right,m=run.task_matrices(45701);u=torch.outer(left,right)*.9
    residuals=torch.stack([m[i]-base for i in range(6,8)])
    assert all(torch.linalg.matrix_rank(x,atol=1e-4).item()>1 for x in residuals)

def test_native_rank_one_map_is_exact_mirror_parameterization():
    base,left,right,m=run.task_matrices(45701);code=torch.tensor(.7);x=torch.eye(run.D)
    assert torch.equal(run.code_output(x,base,left,right,code),run.code_output(x,base,left,right,code))
    arr={'shared_module':base.numpy(),'basis_left':left.numpy(),'basis_right':right.numpy(),'task_codes':np.zeros(run.NTASK,np.float32),'task_module_index':np.full(run.NTASK,-1,np.int16),'private_modules':np.zeros((0,run.D,run.D),np.float32)}
    a=run.pack(arr,{'format':'MA457-continual-path-v1','method_family':'shared-rank1-task-code-v1','dimension':run.D,'tasks':run.NTASK,'dtype':'float32'})
    b=run.pack(arr,{'format':'MA457-continual-path-v1','method_family':'shared-rank1-task-code-v1','dimension':run.D,'tasks':run.NTASK,'dtype':'float32'})
    assert a==b
