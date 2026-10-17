import sys
from pathlib import Path
import io
import torch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from run import NativeWidth, payload, replay_payload, convert_full_teacher_to_width, HIDDEN, MS, KR, DEV_WORLDS, FRESH_WORLDS
from core import state_bytes

@pytest.mark.parametrize('h',HIDDEN)
def test_native_param_formula_and_replay(h):
    torch.manual_seed(4)
    m=NativeWidth(h).eval()
    assert m.trainable==75*h+10
    x=torch.randn(5,64,dtype=torch.float64)
    p=payload(m,torch.float64)
    with torch.no_grad():assert torch.equal(m(x),replay_payload(p,x))
    assert 0<state_bytes(p)<100000


def test_quantized_storage_smaller():
    m=NativeWidth(10)
    assert state_bytes(payload(m,torch.float16))<state_bytes(payload(m,torch.float64))


def test_fresh_not_dev_and_params_nearest():
    assert set(DEV_WORLDS).isdisjoint(FRESH_WORLDS)
    assert MS==(256,512,768,1024)
    assert 75*6+10 <=512<75*7+10
    assert 75*10+10<=768<75*11+10
    assert KR[768]==(8,4)
    assert FRESH_WORLDS==(618450,618451,618452)


def test_full_teacher_adapter_replays_identically():
    from core import NativeMLP
    torch.manual_seed(5)
    teacher=NativeMLP()
    adapted=convert_full_teacher_to_width(teacher)
    x=torch.randn(7,64,dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(teacher(x),adapted(x))
        assert torch.equal(adapted(x),replay_payload(payload(adapted),x))
    assert adapted.trainable==1210
