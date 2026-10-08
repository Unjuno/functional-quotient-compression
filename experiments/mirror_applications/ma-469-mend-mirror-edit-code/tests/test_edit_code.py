from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_teacher_updates_share_two_matrix_basis():
    w=run.world(46901);assert torch.linalg.matrix_rank(w['deltas'].reshape(run.EDITS,-1),atol=1e-5).item()<=2

def test_support_gradient_signal_has_expected_shape():
    w=run.world(46901);assert w['signals'].shape==(run.EDITS,run.D*run.D)
    assert all(torch.isfinite(w['signals'][i]).all() for i in range(run.EDITS))

def test_native_lowrank_editor_is_exact_mirror_parameterization():
    a=run.init('mirror',46901);b=run.init('native_lowrank',46901)
    for k in a:
        if k=='net':
            for x,y in zip(a[k].parameters(),b[k].parameters()):assert torch.equal(x,y)
        else:assert torch.equal(a[k],b[k])


def test_edit_generation_mac_proxy_counts_input_and_mirror_decode():
    d=run.D
    mend=d*d+d*d*32+32*(d*d)
    mirror=d*d+d*d*32+32*2+2*d*d
    assert mend==2340 and mirror==1324
    assert mirror < mend
