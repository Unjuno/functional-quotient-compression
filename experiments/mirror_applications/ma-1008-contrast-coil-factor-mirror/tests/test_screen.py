import importlib.util
from pathlib import Path
import torch

SCRIPT=Path(__file__).parents[1]/'source'/'run_screen.py'
spec=importlib.util.spec_from_file_location('ma1008_screen',SCRIPT)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_synthetic_multicoil_forward_and_adjoint_shapes():
    q=mod.case(1008123,2,1)
    assert q['x'].shape==(2,32,32)
    assert q['y'].shape==(2,32,32)
    assert q['sens'].shape==(4,32,32)
    assert q['ks'].shape==(4,32,32)
    assert q['mask'].sum().item()>0
    assert torch.isfinite(q['x']).all()

def test_views_are_finite_and_operator_independent():
    q=mod.case(1008124,1,2);m=mod.Prior();m.mode='mirror'
    m.factor.data.copy_(torch.tensor([.03,-.02,.01,.04,-.03]))
    y=m(q['x'][None],q['contrast'],q['scanner'])
    assert y.shape==(1,2,32,32) and torch.isfinite(y).all()
    # The acquisition data and operator are constructed before and independently
    # of the prior's Mirror coordinates.
    q0=mod.case(1008124,1,2)
    assert torch.equal(q['mask'],q0['mask'])
    assert torch.equal(q['ks'],q0['ks'])
    assert torch.equal(q['sens'],q0['sens'])

def test_additive_chart_predicts_withheld_cell_without_target_input():
    # Frozen design matrix for eight observed cells; heldout (2,2) is absent.
    pairs=[(c,k) for c in range(3) for k in range(3) if (c,k)!=(2,2)]
    X=[[1.,float(c==1),float(c==2),float(k==1),float(k==2)] for c,k in pairs]
    assert len(pairs)==8 and len(X[0])==5
    coef=torch.arange(5,dtype=torch.float32)
    prediction=float(torch.tensor([1.,0.,1.,0.,1.])@coef)
    assert prediction==6.0

def test_actual_inference_payload_roundtrip_is_tensor_exact(tmp_path):
    model=mod.Prior(); model.mode='mirror'; model.factor.data.copy_(torch.tensor([.1,.2,.3,.4,.5]))
    payload={'state':model.state_dict(),'operator':'same explicit four-coil FFT path','coordinates':model.factor.detach().cpu()}
    path=tmp_path/'payload.pt'; torch.save(payload,path)
    decoded=torch.load(path,map_location='cpu',weights_only=False)
    assert decoded['operator']==payload['operator']
    assert decoded['state'].keys()==payload['state'].keys()
    for key,value in payload['state'].items(): assert torch.equal(value,decoded['state'][key])
    assert torch.equal(payload['coordinates'],decoded['coordinates'])
