import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_ma790", ROOT / "source" / "run_ma790.py")
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


def test_draw28_replay_and_composition_split():
    draw=json.loads((ROOT/"source"/"draw28_exclusions.json").read_text())
    assert len(draw["pool_ids"])==519
    assert hashlib.sha256("\n".join(draw["pool_ids"]).encode()).hexdigest()==draw["pool_sha256"]
    digest=hashlib.sha256(bytes.fromhex(draw["seed_hex"])+bytes.fromhex(draw["pool_sha256"])+draw["rejection_counter"].to_bytes(4,"big")).digest()
    limit=(1<<256)-((1<<256)%len(draw["pool_ids"]))
    v=int.from_bytes(digest,"big")
    assert v<limit
    idx=v%len(draw["pool_ids"])
    assert idx==draw["selection_index_zero_based"]==251
    assert draw["pool_ids"][idx]=="MA-790"
    assert len(ma.TRAIN_PAIRS)==len(ma.AUDIT_PAIRS)==8
    assert not set(ma.TRAIN_PAIRS)&set(ma.AUDIT_PAIRS)
    assert {m for m,_ in ma.TRAIN_PAIRS}=={0,1,2,3}
    assert {s for _,s in ma.AUDIT_PAIRS}=={0,1,2,3}


def test_teacher_coefficients_are_sparse_probabilities():
    for mode in ("aligned","offorbit"):
        a,b,c=ma.make_world(23,mode)
        assert a.shape==(12,8,16) and b.shape==(12,8) and c.shape==(16,12)
        assert np.allclose(c.sum(1),1)
        assert np.all((c>0).sum(1)==3)
        a2,b2,c2=ma.make_world(23,mode)
        assert np.array_equal(a,a2) and np.array_equal(b,b2) and np.array_equal(c,c2)


def test_straight_through_topk_has_sparse_forward_and_dense_gradient():
    logits=torch.tensor([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2],requires_grad=True)
    coeff=ma.sparse_topk_softmax(logits,straight_through=True)
    assert int((coeff.detach()>0).sum())==3
    assert torch.allclose(coeff.detach().sum(),torch.tensor(1.0))
    coeff.square().sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum().item()>0


def test_coefficient_methods_emit_sparse_rows_and_receive_gradients():
    for method in ma.METHODS:
        ma.set_seed(31);model=ma.CoefficientModel(method)
        c=model.coefficients(1,2,straight_through=True)
        assert c.shape==(12,)
        assert int((c.detach()>0).sum())==3
        assert torch.allclose(c.detach().sum(),torch.tensor(1.0),atol=1e-6)
        c.square().sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.parameters())


def test_actual_payload_measurement(tmp_path,monkeypatch):
    monkeypatch.setattr(ma,"ART",tmp_path)
    anchors=np.zeros((12,8,16),np.float32);bias=np.zeros((12,8),np.float32)
    model=ma.CoefficientModel("mirror_factorized")
    a,g,c,t,f=ma.serialize_model(model,"mirror_factorized",anchors,bias,"aligned",11,31)
    assert t==a+g+c
    assert a>anchors.nbytes+bias.nbytes
    assert c==(tmp_path/"aligned_w11_s31_mirror_factorized_factor_codes.safetensors").stat().st_size
    assert f==(tmp_path/"aligned_w11_s31_mirror_factorized_free16_coefficient_reference.safetensors").stat().st_size
    with safe_open(str(tmp_path/"aligned_w11_s31_mirror_factorized_generator.safetensors"),framework="pt") as sf:
        assert sf.metadata()["schema"]=="MA790-PAYLOAD-V2"
        assert "bias[12]" in sf.metadata()["layout"]
    assert not (tmp_path/"aligned_w11_s31_mirror_factorized_metadata.json").exists()
