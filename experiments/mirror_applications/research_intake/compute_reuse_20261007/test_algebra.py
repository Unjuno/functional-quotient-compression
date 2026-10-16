"""Numerical invariants and falsifiers. Not a trained-model benchmark."""
import math
import unittest
import torch
from torch.nn import functional as F
from algebra import sign_explicit, sign_fused, mirror, attention, lazy_attention, partition_attention

torch.set_num_threads(1)
class AlgebraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(701)
        self.dtype = torch.float64
    def rand(self, *shape):
        return torch.randn(*shape, dtype=self.dtype)
    def close(self, x, y):
        torch.testing.assert_close(x, y, rtol=2e-11, atol=2e-11)
    def test_sign_ffn_forward(self):
        v, w = self.rand(7, 12), self.rand(12, 5)
        signs = torch.randint(2, (8,12)).to(self.dtype)*2-1
        a = self.rand(7,8).softmax(-1)
        for act in (F.gelu, F.silu, F.relu):
            self.close(sign_explicit(v,w,signs,a,act), sign_fused(v,w,signs,a,act))
    def test_sign_ffn_gradients_including_router(self):
        x, up, down, router = [z.requires_grad_() for z in (self.rand(7,4),self.rand(4,12),self.rand(12,5),self.rand(4,8))]
        signs = torch.randint(2,(8,12)).to(self.dtype)*2-1
        v, a = x@up, (x@router).softmax(-1)
        out1 = sign_explicit(v,down,signs,a)
        out2 = sign_fused(v,down,signs,a)
        g1=torch.autograd.grad(out1.square().mean(),(x,up,down,router),retain_graph=True)
        g2=torch.autograd.grad(out2.square().mean(),(x,up,down,router))
        for p,q in zip(g1,g2): self.close(p,q)
    def test_signed_unnormalized_mixture(self):
        v,w,a=self.rand(7,12),self.rand(12,5),self.rand(7,8)
        s=torch.randint(2,(8,12)).to(self.dtype)*2-1
        self.close(sign_explicit(v,w,s,a),sign_fused(v,w,s,a))
    def test_activation_identity(self):
        x=self.rand(128)
        for act in (F.gelu, lambda z:F.gelu(z,approximate='tanh'), F.silu,F.relu):
            self.close(act(x)-act(-x),x)
    def test_antipodal_general_invertible(self):
        v=self.rand(7,12)
        q=torch.linalg.qr(self.rand(12,12)).Q@torch.diag(torch.linspace(.7,1.3,12,dtype=self.dtype))
        for act in (F.gelu,F.silu,F.relu):
            self.close((mirror(v,q,act)+mirror(v,-q,act))/2,v/2)
    def test_permutation_is_functional_null(self):
        p=torch.eye(12,dtype=self.dtype)[torch.randperm(12)]
        v=self.rand(7,12)
        self.close(mirror(v,p),F.gelu(v))
    def test_nonlinear_view_precomposition_is_not_general(self):
        v=torch.tensor([[1.,-2.],[.3,.8]],dtype=self.dtype)
        q1=torch.eye(2,dtype=self.dtype);q2=torch.tensor([[1.,1.],[0.,1.]],dtype=self.dtype)
        error=((mirror(v,q1)+mirror(v,q2))/2-mirror(v,(q1+q2)/2)).abs().max()
        self.assertGreater(float(error),1e-3)
    def test_lazy_cache_exact(self):
        q,k,v,a,b=self.rand(3,8),self.rand(31,8),self.rand(31,6),self.rand(8,8),self.rand(6,6)
        self.close(attention(q,k@a,v@b),lazy_attention(q,k,v,a,b))
    def test_query_key_compensation_is_null(self):
        q,k,v=self.rand(3,8),self.rand(31,8),self.rand(31,6)
        a=torch.linalg.qr(self.rand(8,8)).Q@torch.diag(torch.linspace(.7,1.3,8,dtype=self.dtype))
        self.close(attention(q@a,k@torch.linalg.inv(a).T,v),attention(q,k,v))
    def test_value_only_mixture_one_attention(self):
        q,k,v=self.rand(3,8),self.rand(31,8),self.rand(31,6)
        b=self.rand(5,6,6);weights=self.rand(3,5).softmax(-1)
        naive=torch.stack([attention(q,k,v@be) for be in b],1)
        naive=(naive*weights[:,:,None]).sum(1)
        out=attention(q,k,v)
        fused=torch.einsum('bd,bde->be',out,torch.einsum('bk,kde->bde',weights,b))
        self.close(naive,fused)
    def test_different_keys_cannot_merge_softmax(self):
        q,k,v=self.rand(3,8),self.rand(31,8),self.rand(31,6)
        a1,a2=self.rand(8,8),self.rand(8,8)
        actual=(attention(q,k@a1,v)+attention(q,k@a2,v))/2
        wrong=attention(q,k@((a1+a2)/2),v)
        self.assertGreater(float((actual-wrong).abs().max()),.01)
    def test_token_dependent_value_view_cannot_pull_out(self):
        q,k,v=self.rand(1,8),self.rand(31,8),self.rand(31,6)
        scale=torch.linspace(.1,2.,31,dtype=self.dtype)
        actual=attention(q,k,v*scale[:,None]);wrong=attention(q,k,v)*scale.mean()
        self.assertGreater(float((actual-wrong).abs().max()),.01)
    def test_partition_attention_lse_merge(self):
        q,k,v=self.rand(3,8),self.rand(31,8),self.rand(31,6)
        self.close(attention(q,k,v),partition_attention(q,k[:20],v[:20],k[20:],v[20:]))
    def test_missing_projection_information_impossible(self):
        h=torch.tensor([[0.,0.],[0.,1.]],dtype=self.dtype)
        old=h@torch.tensor([[1.],[0.]],dtype=self.dtype)
        target=h@torch.tensor([[0.],[1.]],dtype=self.dtype)
        self.close(old[0],old[1]);self.assertNotEqual(float(target[0]),float(target[1]))
    def test_projection_sidecar_recovers_missing_direction(self):
        h=self.rand(21,8);p=torch.eye(8,dtype=self.dtype)[:,:3];u=torch.eye(8,dtype=self.dtype)[:,3:5]
        a,b=self.rand(3,4),self.rand(2,4)
        self.close(h@(p@a+u@b),(h@p)@a+(h@u)@b)
    def test_quantization_error_not_view_invariant(self):
        x=self.rand(19,8)
        q=torch.round(x/.1)*.1;e=q-x
        stretch=torch.eye(8,dtype=self.dtype);stretch[0,0]=10.
        self.assertGreater(float(torch.linalg.vector_norm(e@stretch)),float(torch.linalg.vector_norm(e)))
    def test_independent_dropout_invalidates_sign_fusion(self):
        v,w=self.rand(7,12),self.rand(12,5);s=torch.randint(2,(8,12)).to(self.dtype)*2-1;a=self.rand(7,8).softmax(-1)
        branches=s[None,:,:]*F.gelu(v[:,None,:]*s[None,:,:])
        dropped=F.dropout(branches,p=.5,training=True)
        actual=torch.einsum('bk,bkh,ho->bo',a,dropped,w)
        self.assertGreater(float((actual-sign_fused(v,w,s,a)).abs().max()),.01)
    def test_shared_down_projection_fusion_general_views(self):
        from algebra import mirror_mixture_fused
        v,w=self.rand(7,12),self.rand(12,5)
        qs=torch.stack([torch.linalg.qr(self.rand(12,12)).Q for _ in range(4)])
        a=self.rand(7,4).softmax(-1)
        expected=sum(a[:,i:i+1]*(mirror(v,qs[i])@w) for i in range(4))
        self.close(expected,mirror_mixture_fused(v,w,qs,a))
    def test_rectangular_latent_preserves_logical_temperature(self):
        q,k,v,a=self.rand(3,8),self.rand(31,5),self.rand(31,6),self.rand(5,8)
        expected=attention(q,k@a,v)
        correct=((q@a.T)@k.T/math.sqrt(8)).softmax(-1)@v
        wrong=attention(q@a.T,k,v)
        self.close(expected,correct)
        self.assertGreater(float((expected-wrong).abs().max()),1e-3)
if __name__=='__main__': unittest.main(verbosity=2)
