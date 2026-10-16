import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from train import create,train_single,TRAINED,HELDO


def test_selection_does_not_train_on_heldout_pairs():
    assert sorted(TRAINED+HELDO)==list(range(9))
    assert len(set(TRAINED)&set(HELDO))==0


def test_identical_base_initializations():
    a=create('mirror12',0);b=create('gain12',0);c=create('independent',0)
    assert torch.equal(a.wg,b.wg)
    assert torch.equal(a.wg,c.wg[0])
    assert torch.equal(a.wu,c.wu[0])


def test_function_preservation_and_autogrow_storage():
    a=create('auto_mirror',123)
    assert a.active_q==1
    x=torch.randn(6,8)
    y=a(x).clone()
    a.set_active(4)
    assert torch.equal(a(x),y)
    a.set_active(12)
    assert torch.equal(a(x),y)


def test_short_trial_reproducibility(tmp_path):
    z=(73001,0,'aligned_holdout','mirror4',20,0.018,str(tmp_path))
    a=train_single(z)
    p=tmp_path/'runs'/f'{a["name"]}.json'
    s=p.read_bytes()
    b=train_single(z)
    assert a==b
    assert p.read_bytes()==s or abs(a['test']-b['test'])<1e-12
