from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
try:
    import core
except ImportError:
    core = None

def need():
    assert core is not None, 'core implementation not written yet'
    return core

def test_basis_partition_and_identity():
    m=need(); k=np.linspace(-1,1,17)
    assert np.allclose(m.basis(k,17),np.eye(17))
    b=m.basis(np.linspace(-1,1,129),17)
    assert np.allclose(b.sum(1),1) and b.min()>=0
    with pytest.raises(ValueError): m.basis(np.array([1.1]),17)

def test_pcgrad_zero_opposite_and_nonconflicting():
    m=need()
    for g in [np.zeros((3,4)),np.array([[1.,0],[-1.,0]])]:
        assert np.max(np.abs(m.pcgrad(g,np.random.default_rng(1))))<1e-14
    g=np.array([[1.,0],[0,1.]])
    assert np.array_equal(m.pcgrad(g,np.random.default_rng(1)),g.mean(0))
    g=np.array([[2.,0],[-1.,1.]])
    p=m.pcgrad(g,np.random.default_rng(1))
    assert np.all(g@p>0), 'a descent for both exists in this fixture'

def test_cosine_zero_marked_missing_and_cancellation():
    m=need()
    s=m.diagnostics(np.zeros((2,3)))
    assert s['conflict_fraction'] is None and s['cancel_ratio'] is None
    s=m.diagnostics(np.array([[1.,0],[-1.,0]]))
    assert s['conflict_fraction']==1 and s['cancel_ratio']==0

def test_model_gradients_match_torch():
    m=need(); import torch
    torch.set_num_threads(1)
    w=m.make_world('partial',901,32,24,40)
    for name in ['shared','mirror_r2','rank3','selected_private','film','independent']:
        p=m.initialize(name,8,17,3,private_ids=[2,5])
        if 'P' in p: p['P'][:]=np.random.default_rng(8).normal(size=p['P'].shape)*.2
        g,shared,task=m.gradients(name,p,w['H'],w['Q'])
        tp={k:torch.tensor(v,requires_grad=True) for k,v in p.items() if k!='ids'}
        if name=='independent': c=tp['C']
        elif name=='film': c=tp['a'][:,None]*tp['c0'][None,:]+tp['b'][:,None]
        elif 'D' in tp:
            c=tp['c0'][None,:]+tp['M']@tp['D'].T
            if 'P' in tp:
                sel=torch.nn.functional.one_hot(torch.tensor(p['ids'].astype(int)),8).T.double()
                c=c+sel@tp['P']
        else: c=tp['c0'][None,:].expand(8,-1)
        pred=torch.tensor(w['Btrain'])@c.T
        loss=((pred-torch.tensor(w['Ytrain']))**2).mean(); loss.backward()
        for k in tp:
            assert np.allclose(g[k],tp[k].grad.numpy(),atol=2e-12), (name,k)
        expected=np.concatenate([g[k].ravel() for k in m.shared_keys(name)]) if m.shared_keys(name) else np.empty(0)
        assert np.allclose(shared.mean(0), expected,atol=2e-12)

def test_weighted_oracle_and_rank_boundary():
    m=need()
    for family in ['identical','aligned','partial','unrelated']:
        w=m.make_world(family,903,64,32,40)
        full=m.oracle(w['Btrain'],w['Ytrain'],7)
        assert np.mean((w['Btrain']@full.T-w['Ytrain'])**2)<1e-24
        two=m.oracle(w['Btrain'],w['Ytrain'],2)
        e=np.mean((w['Btrain']@two.T-w['Ytrain'])**2)
        if family in ['identical','aligned']: assert e<1e-24
        else: assert e>1e-6
        if family=='partial':
            c=m.oracle(w['Btrain'],w['Ytrain'],2,private_ids=w['private_truth'])
            assert np.mean((w['Btrain']@c.T-w['Ytrain'])**2)<1e-24
        assert np.allclose(m.oracle(w['Btrain'],w['Ytrain'],0),full.mean(0)[None,:])

def test_dead_factors_and_native_alias():
    m=need(); w=m.make_world('aligned',904,64,32,40)
    p=m.initialize('mirror_r2',8,17,1)
    p['c0'][:]=m.oracle(w['Btrain'],w['Ytrain'],0)[0]
    p['D'][:]=0; p['M'][:]=0
    g,shared,_=m.gradients('mirror_r2',p,w['H'],w['Q'])
    assert np.max(np.abs(g['D']))==0 and np.max(np.abs(g['M']))==0
    assert np.linalg.norm(g['c0'])<1e-12
    assert np.linalg.norm(shared)>1e-3
    a=m.initialize('native_lowrank',8,17,1)
    b=m.initialize('mirror_r2',8,17,1)
    assert all(np.array_equal(a[k],b[k]) for k in a)

def test_codec_roundtrip_and_bad_file(tmp_path):
    m=need()
    for name in ['shared','mirror_r2','selected_private','film','independent']:
        p=m.initialize(name,8,17,3,private_ids=[2,5]); f=tmp_path/(name+'.gic')
        m.save_model(f,name,p); decoded,params=m.load_model(f)
        assert np.array_equal(m.coefficients(name,p),m.coefficients(decoded,params))
    a=tmp_path/'mirror_r2.gic'; b=tmp_path/'native.gic'
    m.save_model(b,'native_lowrank',m.initialize('native_lowrank',8,17,3))
    assert a.read_bytes()==b.read_bytes()
    a.write_bytes(b'bad')
    with pytest.raises(ValueError): m.load_model(a)

def test_adam_matches_torch():
    m=need(); import torch
    p={'v':np.array([1.,-2.,.5])}; opt=m.Adam(p,.03)
    q=torch.tensor(p['v'],requires_grad=True); topt=torch.optim.Adam([q],lr=.03)
    rng=np.random.default_rng(71)
    for _ in range(50):
        g=rng.normal(size=3); opt.update(p,{'v':g})
        q.grad=torch.tensor(g); topt.step(); topt.zero_grad()
    assert np.allclose(p['v'],q.detach().numpy(),atol=1e-14)

def test_runner_functional_training_and_oracle():
    m=need()
    try: import run
    except ImportError: run=None
    assert run is not None, 'runner implementation not written yet'
    w=m.make_world('aligned',951,64,32,64)
    p,row,h=run.fit(w,'mirror_r2',1,.03,steps=300,prefix=75)
    assert row['test_mean_mse']<1e-3
    assert row['train_mse']>=row['oracle_train_mse']-1e-10
    assert all(np.isfinite(v).all() for v in p.values())

def test_allocation_pair_has_identical_prefix():
    m=need()
    try: import run
    except ImportError: run=None
    assert run is not None, 'runner implementation not written yet'
    w=m.make_world('partial',952,64,32,64)
    states=[]
    for name in ['selected_private','selected_private_pcgrad','random_private','random_private_pcgrad']:
        p,row,h=run.fit(w,name,1,.03,steps=81,prefix=75)
        states.append(row['prefix_hash'])
    assert len(set(states))==1
