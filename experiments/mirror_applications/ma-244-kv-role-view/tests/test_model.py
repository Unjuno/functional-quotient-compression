import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from model import AttentionVariant,Config,Teacher


def test_all_projection_variants_produce_expected_attention_shape():
    cfg=Config();q=torch.randn(16,cfg.heads,cfg.head_dim);mem=torch.randn(16,cfg.memory_tokens,cfg.model_dim)
    for method in AttentionVariant.METHODS:
        model=AttentionVariant(cfg,method);out=model(q,mem)
        assert out.shape==(16,cfg.model_dim)
        assert torch.isfinite(out).all()


def test_mirror_family_can_exactly_reconstruct_aligned_teacher():
    cfg=Config();teacher=Teacher(cfg,11);model=AttentionVariant(cfg,"mirror_mqa")
    with torch.no_grad():
        model.w.copy_(teacher.w)
        model.angles.copy_(teacher.angles)
    q=torch.randn(23,cfg.heads,cfg.head_dim);mem=torch.randn(23,cfg.memory_tokens,cfg.model_dim)
    assert torch.equal(model(q,mem),teacher(q,mem))


def test_byte_near_gate_and_mirror_share_one_projection():
    cfg=Config();mirror=AttentionVariant(cfg,"mirror_mqa");gate=AttentionVariant(cfg,"gate_mqa")
    assert mirror.parameter_count()==gate.parameter_count()
    assert mirror.cache_bytes()==gate.cache_bytes()==cfg.memory_tokens*cfg.head_dim*4
    assert mirror.inference_payload_bytes()>0


def test_standard_and_shared_kv_cache_sizes_match_structure():
    cfg=Config();
    assert AttentionVariant(cfg,"mha").cache_bytes()==cfg.heads*2*cfg.memory_tokens*cfg.head_dim*4
    assert AttentionVariant(cfg,"kv_equal_mha").cache_bytes()==cfg.heads*cfg.memory_tokens*cfg.head_dim*4
    assert AttentionVariant(cfg,"gqa2").cache_bytes()==2*2*cfg.memory_tokens*cfg.head_dim*4
    assert AttentionVariant(cfg,"mqa").cache_bytes()==2*cfg.memory_tokens*cfg.head_dim*4
    assert AttentionVariant(cfg,"mirror_mqa").cache_bytes()==cfg.memory_tokens*cfg.head_dim*4
    mem=torch.randn(2,cfg.memory_tokens,cfg.model_dim)
    for method in AttentionVariant.METHODS:
        model=AttentionVariant(cfg,method)
        assert model.measured_cache_bytes(mem)==2*model.cache_bytes()
