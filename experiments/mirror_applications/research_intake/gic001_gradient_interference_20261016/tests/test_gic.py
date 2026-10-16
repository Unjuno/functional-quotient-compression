import importlib.util
import numpy as np
import pytest

def test_module_exists():
    assert importlib.util.find_spec('gic') is not None, 'implement the experiment core'

def test_basis_partition_and_bounds():
    from gic import basis
    b=basis(np.linspace(-1,1,301),12)
    assert b.shape==(301,12)
    np.testing.assert_allclose(b.sum(1),1,atol=1e-14)
    assert b.min()>=0
    with pytest.raises(ValueError): basis(np.array([1.1]),12)

def test_pcgrad_opposing_and_aligned():
    from gic import pcgrad
    rng=np.random.default_rng(1)
    np.testing.assert_allclose(pcgrad(np.array([[1.,0.],[-1.,0.]]),rng),0,atol=1e-14)
    g=np.array([[1.,2.],[2.,4.]])
    np.testing.assert_allclose(pcgrad(g,rng),g.mean(0),atol=1e-14)
    np.testing.assert_allclose(pcgrad(np.array([[1.,2.],[1.,-2.]]),rng),[1.6,0.],atol=1e-14)

def test_gradients_against_autograd():
    import torch
    from gic import Model, stats, gradients
    rng=np.random.default_rng(1)
    b=rng.normal(size=(9,12)); y=rng.normal(size=(9,8)); st=stats(b,y)
    for rank,kind in [(0,'factor'),(2,'factor'),(0,'independent')]:
        m=Model(8,12,rank,11,kind)
        if kind=='factor': m.allocate([1,4])
        p={k:torch.tensor(v,requires_grad=True) for k,v in m.p.items()}
        if kind=='independent': c=p['C']
        else:
            c=p['c'][None,:]+p['M']@p['D'].T
            lift=torch.zeros((8,len(m.ids)),dtype=torch.float64)
            lift[m.ids,np.arange(len(m.ids))]=1
            c=c+lift@p['P']
        loss=((torch.tensor(b)@c.T-torch.tensor(y))**2).mean()
        loss.backward()
        g,_,_=gradients(m,st)
        for key in p:
            np.testing.assert_allclose(g[key],p[key].grad.numpy(),rtol=1e-11,atol=1e-11)

def test_oracle_representation_limits():
    from gic import make_world, optimal_coeff, population_mse
    for regime in ['identical','related','partial','unrelated']:
        w=make_world(123,regime,noise=0)
        c=w['truth']; h=w['population_H']; masks=w['exceptions']
        rank=0 if regime=='identical' else 2
        fit=optimal_coeff(c,h,rank,masks if regime=='partial' else [])
        mse=population_mse(fit,c,h).mean()
        if regime!='unrelated': assert mse<1e-20
        else: assert mse>1e-3
        if regime=='related':
            assert population_mse(optimal_coeff(c,h,0,[]),c,h).mean()>1e-2

def test_codec_allocation_and_exact_replay(tmp_path):
    from gic import Model,save_model,load_model,basis
    for kind,rank in [('factor',0),('factor',2),('independent',0)]:
        m=Model(8,12,rank,7,kind); prev=m.coeff().copy()
        if kind=='factor':
            m.allocate([1,5]); np.testing.assert_array_equal(m.coeff(),prev)
            m.p['P']+=0.125
        f=tmp_path/'x.bin'; save_model(m,f); data=f.read_bytes()
        n=load_model(f); save_model(n,tmp_path/'y.bin')
        assert data==(tmp_path/'y.bin').read_bytes()
        np.testing.assert_array_equal(n.coeff(),m.coeff())
        np.testing.assert_array_equal(n.predict(np.array([-.2,.5])),m.predict(np.array([-.2,.5])))

def test_zero_gradient_diagnostic():
    from gic import diagnostic
    d=diagnostic(np.zeros((8,36)))
    assert d['conflict_fraction'] is None and d['cancellation'] is None

def test_population_quadrature_matches_dense_grid():
    from gic import population_gram,basis
    b=basis(np.linspace(-1,1,100001),12)
    h=np.trapezoid(b[:,:,None]*b[:,None,:],dx=2/100000,axis=0)/2
    np.testing.assert_allclose(population_gram(12),h,atol=1e-8)
