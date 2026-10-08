import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from model import Config,LayerKV,Teacher


def test_all_layer_kv_methods_return_finite_layer_outputs():
    cfg=Config();q=torch.randn(8,cfg.layers,cfg.head_dim);mem=torch.randn(8,cfg.memory_tokens,cfg.model_dim)
    for method in LayerKV.METHODS:
        model=LayerKV(cfg,method);y=model(q,mem)
        assert y.shape==(8,cfg.layers,cfg.head_dim)
        assert torch.isfinite(y).all()


def test_mirror_reconstructs_aligned_teacher_exactly():
    cfg=Config();teacher=Teacher(cfg,22);model=LayerKV(cfg,"mirror")
    with torch.no_grad():
        model.wk.copy_(teacher.wk.unsqueeze(0));model.wv.copy_(teacher.wv.unsqueeze(0));model.angles.copy_(teacher.angles)
    q=torch.randn(5,cfg.layers,cfg.head_dim);mem=torch.randn(5,cfg.memory_tokens,cfg.model_dim)
    assert torch.equal(model(q,mem),teacher(q,mem))


def test_physical_cache_tensor_bytes_match_group_count():
    cfg=Config();mem=torch.randn(1,cfg.memory_tokens,cfg.model_dim)
    expected={"mha":4,"mlkv2":2,"mlkv1":1,"mirror":1,"gate":1}
    for method,groups in expected.items():
        model=LayerKV(cfg,method)
        assert model.measured_cache_bytes(mem)==groups*2*cfg.memory_tokens*cfg.head_dim*4
        assert model.cache_bytes()==model.measured_cache_bytes(mem)


def test_payloads_serialize_and_mirror_metadata_is_paid():
    cfg=Config();mirror=LayerKV(cfg,"mirror");hard=LayerKV(cfg,"mlkv1")
    assert mirror.parameter_count()==hard.parameter_count()+cfg.layers*2
    assert mirror.inference_payload_bytes()>hard.inference_payload_bytes()
