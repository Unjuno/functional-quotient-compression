from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_shared_codes_replay_aligned_functions():
 w=run.world(31,8);obj=run.make_objects('shared_mirror',w)
 assert run.score('shared_mirror',obj,w)['relative_output_rmse']<1e-4

def test_native_shared_control_is_byte_identical_and_codes_unique():
 w=run.world(32,16);obj=run.make_objects('shared_mirror',w)
 assert run.pack('shared_mirror',obj)==run.pack('native_shared_code',obj)
 assert torch.unique(obj['codes'],dim=0).shape[0]==16

def test_shared_payload_amortizes_by_eight_tasks():
 w=run.world(33,8);sm=run.pack('shared_mirror',run.make_objects('shared_mirror',w));ind=run.pack('task_loreft_r4',run.make_objects('task_loreft_r4',w))
 assert len(sm)<=.30*len(ind)
