import importlib.util
from pathlib import Path
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'train.py'
spec=importlib.util.spec_from_file_location('ma1064',SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def base_model():
    model=m.B.Model('dhe_shared',0,1)
    model.hash_u=torch.randn(m.MAXU+1,m.B.K); model.hash_i=torch.randn(m.MAXI+1,m.B.K); model.eval()
    return model

def test_old_ma1063_audit_rows_are_excluded_before_new_split():
    pair=next((u,i) for u in range(1,40) for i in range(1,40) if m.legacy_audit(u,i))
    assert m.partition_bucket(*pair) is None
    other=next((u,i) for u in range(1,40) for i in range(1,40) if not m.legacy_audit(u,i))
    assert m.partition_bucket(*other) in range(10)

def test_all_private_views_start_as_the_shared_dhe_function():
    base=base_model(); ids=[[2,4],[7,9]]; u=torch.tensor([2,3,4]); i=torch.tensor([7,8,9])
    with torch.no_grad(): expected=base(u,i)
    for method in m.ADAPTERS:
        model=m.PrivateView(base,method,ids); model.eval()
        with torch.no_grad(): actual=model(u,i)
        assert torch.allclose(actual,expected,atol=1e-6)

def test_per_id_code_storage_shapes_match_method_definitions():
    base=base_model(); ids=[[2,4,6],[7]]
    expected={'id_residual':128,'private_replace':128,'id_mirror':64,'mirror_residual':192}
    for method,n in expected.items():
        model=m.PrivateView(base,method,ids)
        assert sum(p.numel() for p in model.codes)==n

def test_selected_id_maps_are_rebuilt_from_the_paid_id_list():
    model=m.PrivateView(base_model(),'id_residual',[[3,11],[5]])
    assert model.slot_maps[0][3].item()==0 and model.slot_maps[0][11].item()==1
    assert model.slot_maps[0][4].item()==-1 and model.slot_maps[1][5].item()==0
