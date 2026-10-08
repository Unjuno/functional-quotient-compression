from pathlib import Path
import importlib.util
import sys
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'source' / 'experiment.py'

def test_00_implementation_exists():
    assert SOURCE.is_file(), 'The registered experiment implementation is not yet present.'

@pytest.fixture(scope='module')
def m():
    assert SOURCE.is_file(), 'Implementation absent'
    spec=importlib.util.spec_from_file_location('ma403', SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    torch.set_num_threads(1)
    return module

def test_pair_rotation_identity_and_norm(m):
    torch.manual_seed(9101)
    x=torch.randn(3,5,24,dtype=torch.float64)
    assert torch.equal(m.rotate_pairs(x,torch.zeros(3,5,12,dtype=x.dtype)),x)
    a=torch.randn(3,5,12,dtype=x.dtype)
    y=m.rotate_pairs(x,a)
    torch.testing.assert_close(y.square().sum(-1),x.square().sum(-1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(m.rotate_pairs(y,-a),x,rtol=1e-12,atol=1e-12)

def test_pair_rotation_matches_complex_phase(m):
    x=torch.randn(2,7,24,dtype=torch.float64);a=torch.randn(2,7,12,dtype=x.dtype)
    z=torch.complex(x[...,0::2],x[...,1::2])*torch.exp(1j*a)
    expected=torch.stack((z.real,z.imag),-1).flatten(-2)
    torch.testing.assert_close(m.rotate_pairs(x,a),expected,rtol=1e-12,atol=1e-12)

def test_rotation_gradient(m):
    x=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    a=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(m.rotate_pairs,(x,a))

def test_backbone_and_context_have_no_future_leakage(m):
    cfg=m.default_config(); b=m.Backbone(cfg,9191).eval()
    x=torch.randint(cfg['vocab'],(4,12));y=x.clone();y[:,6:]=(y[:,6:]+3)%cfg['vocab']
    h,c,l=b.features(x);hh,cc,ll=b.features(y)
    torch.testing.assert_close(h[:,:6],hh[:,:6],rtol=0,atol=0)
    torch.testing.assert_close(c[:,:7],cc[:,:7],rtol=0,atol=0)
    assert torch.equal(c[:,0],torch.zeros_like(c[:,0]))

def test_all_adapters_start_at_identity(m):
    cfg=m.default_config();h=torch.randn(2,12,24);c=torch.randn_like(h);l=torch.randn_like(h)
    for mode in m.METHODS:
        a=m.Adapter(cfg,mode,9281)
        torch.testing.assert_close(a(h,c,l),h,rtol=0,atol=0)

def test_split_and_seed_reproducibility(m):
    x=m.make_tokens(32,12,24,9471,avoid=True)
    assert not m.heldout_mask(x).any()
    assert torch.equal(x,m.make_tokens(32,12,24,9471,avoid=True))
    y=m.make_tokens(128,12,24,9472,avoid=False)
    assert m.heldout_mask(y).any()
    assert not torch.equal(x,y[:32])

def test_serialization_roundtrip_and_true_bytes(m,tmp_path):
    cfg=m.default_config();b=m.Backbone(cfg,9501);a=m.Adapter(cfg,'mirror',9502)
    with torch.no_grad():a.generator[-1].bias.fill_(.31)
    p=tmp_path/'model.bin';summary=m.save_model(p,b,a)
    bb,aa=m.load_model(p)
    x=m.make_tokens(3,12,24,9503,avoid=False)
    torch.testing.assert_close(m.forward(b,a,x),m.forward(bb,aa,x),rtol=0,atol=0)
    assert summary['bytes']==p.stat().st_size
    assert summary['bytes']>summary['tensor_bytes']
    assert all(not q.requires_grad for q in bb.parameters())

def test_training_reduces_loss_without_changing_backbone(m):
    cfg=m.default_config();b=m.Backbone(cfg,9601)
    t=m.make_teacher(cfg,'rotation',9602)
    x=m.make_tokens(48,12,24,9603,avoid=True)
    data=m.cache_data(b,t,x,torch.ones_like(x,dtype=torch.bool))
    a=m.Adapter(cfg,'mirror',9604)
    before=m.metrics(b,a,data)['excess_kl'];state={k:v.clone() for k,v in b.state_dict().items()}
    m.fit(b,a,data,updates=80,batch=16,lr=.02,seed=9605,record_every=40)
    after=m.metrics(b,a,data)['excess_kl']
    assert after < before*.75, (before,after)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in state.items())

def test_shape_validation(m):
    with pytest.raises(ValueError):m.rotate_pairs(torch.ones(2,5),torch.ones(2,2))
    with pytest.raises(ValueError):m.Adapter(m.default_config(),'not-a-method',1)
