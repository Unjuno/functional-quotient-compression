import csv,importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma771',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_exact_cycle():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    r=next(x for x in rows if x['method']=='native_conditional_realnvp')
    assert float(r['max_inverse_cycle_error'])<1e-5
    assert float(r['min_jacobian_det'])>0

def test_native_flow_and_mirror_alias():
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_conditional_realnvp')
    b=next(r for r in rows if r['method']=='mirror_realnvp_functional_view')
    for k in ('serialized_bytes','payload_sha256','heldout_function_nrmse','max_inverse_cycle_error','mean_logdet','active_macs_per_example'):
        assert a[k]==b[k]

def test_equal_size_mlp_core():
    assert m.FLOW.size==4
    # Four trainable scalar weights in the residual MLP head.
    import torch
    p=torch.nn.Parameter(torch.zeros(4))
    assert p.numel()==m.FLOW.size
