import importlib.util
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma331_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_permutation_is_exact_function_symmetry():
    base, models, _, x = m.world(33101)
    for i in range(m.ALIGNED):
        assert torch.allclose(m.mlp(x, *models[i]), m.mlp(x, *m.align_to_base(models[i], base)[0]), atol=1e-6, rtol=1e-6)


def test_rebasin_recovers_shared_hidden_basis():
    base, models, _, _ = m.world(33102)
    for model in models[:m.ALIGNED]:
        aligned, _ = m.align_to_base(model, base)
        assert torch.allclose(aligned[0], base[0], atol=1e-6, rtol=1e-6)
        assert torch.allclose(aligned[1], base[1], atol=1e-6, rtol=1e-6)


def test_phase_and_direct_coefficients_reconstruct_same_orbit(tmp_path):
    base, models, _, x = m.world(33103)
    fit = m.aligned_deltas(base, models)
    direct_path=tmp_path/'direct.zip'; phase_path=tmp_path/'phase.zip'
    m.serialize('rebasin_direct',base,models,fit,direct_path)
    m.serialize('mirror_phase',base,models,fit,phase_path)
    da=m.load_arrays(direct_path);pa=m.load_arrays(phase_path)
    dm=m.metrics('rebasin_direct',da,base,models,x)
    pm=m.metrics('mirror_phase',pa,base,models,x)
    assert max(dm['task_nmse']) < 1e-5
    assert max(pm['task_nmse']) < 1e-5


def test_private_unrelated_tasks_are_preserved(tmp_path):
    base, models, _, x = m.world(33104)
    fit=m.aligned_deltas(base,models);p=tmp_path/'p.zip'
    m.serialize('mirror_phase',base,models,fit,p)
    metrics=m.metrics('mirror_phase',m.load_arrays(p),base,models,x)
    assert max(metrics['task_nmse']) < 1e-5
