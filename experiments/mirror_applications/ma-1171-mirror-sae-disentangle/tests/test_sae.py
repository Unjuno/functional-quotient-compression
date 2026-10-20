"""Pretraining mathematical smoke tests for the MA-1171 pilot."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from experiment import SAE, METHODS, generate, ground_truth, true_feature_f1, sparse_relu_topk, fixed_maps, FNUM, D, K

torch.set_num_threads(1)

def test_shapes_sparsity_gradients():
    torch.manual_seed(11)
    x=torch.randn(10,D)
    for method in METHODS:
        m=SAE(method)
        z,y=m.encode_and_reconstruct(x)
        assert (z.shape,y.shape)==((10,FNUM),(10,D))
        assert torch.all(z>=0) and torch.all(z.count_nonzero(dim=-1)<=K)
        assert torch.isfinite(y).all()
        loss=y.square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        if method=='mirror_loop':
            for key in ['core','address','gain']:
                assert getattr(m,key).grad.norm()>0,key
    print('PASS shapes, sparsity, all gradient paths')

def test_deterministic_pages():
    x=fixed_maps(); y=fixed_maps()
    assert all(torch.equal(a,b) for a,b in zip(x,y))
    h=x[-1]
    assert torch.allclose(h@h.T,torch.eye(D),atol=1e-6)
    print('PASS seeded pages and orthogonal Mirror')

def test_controls_parameter_counts():
    x={m:sum(p.numel() for p in SAE(m).parameters()) for m in METHODS}
    assert x['affine_loop']==x['mirror_loop']
    assert x['affine_loop']<x['lowrank_loop']<x['dense_loop']
    print('PASS matched parameter sizes',x)

def test_id_and_ood_splits():
    for world in ['iid','correlated_pairs']:
        a,i=generate(11,world,'train',n=128)
        b,j=generate(11,world,'dev',n=128)
        assert not torch.allclose(a,b)
        assert i.unique().numel()>20
        assert (i[:,0]!=i[:,1]).all() and (i[:,0]!=i[:,2]).all() and (i[:,1]!=i[:,2]).all()
    print('PASS split independence and unique sparse IDs')

def test_metric_exact_permutation():
    w=ground_truth(11,'iid')
    zz=torch.zeros(10,FNUM)
    indices=torch.stack([torch.randperm(FNUM)[:K] for _ in range(10)])
    zz.scatter_(1,indices,1.)
    f1,cosine,l0=true_feature_f1(zz,indices,w,w)
    assert abs(f1-1.)<1e-9 and abs(cosine-1.)<1e-5 and abs(l0-K)<1e-7
    print('PASS oracle permutation and support-F1')

if __name__=='__main__':
    test_shapes_sparsity_gradients()
    test_deterministic_pages()
    test_controls_parameter_counts()
    test_id_and_ood_splits()
    test_metric_exact_permutation()
