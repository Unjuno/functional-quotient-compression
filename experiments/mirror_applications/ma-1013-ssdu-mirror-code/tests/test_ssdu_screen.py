import ast,inspect,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from ssdu_screen import A,AH,sens,masks,phantom,Prior,adapt_scan

def test_ssdu_masks_are_disjoint_and_full_sample_partition():
    m,tr,ho=masks(1013101,1)
    assert torch.equal(tr|ho,m) and not torch.any(tr&ho)
    assert tr.sum()==7*32 and ho.sum()==32

def test_sense_operator_adjoint():
    g=torch.Generator().manual_seed(2);x=torch.complex(torch.randn(32,32,generator=g),torch.randn(32,32,generator=g));q=torch.complex(torch.randn(2,32,32,generator=g),torch.randn(2,32,32,generator=g));m,_,_=masks(1013102,2);s=sens(2);xx=torch.stack([x.real,x.imag]);ax=A(xx,s,m);lhs=(ax.conj()*q).sum();rhs=(x.conj()*AH(q*m[None],s)).sum();assert torch.allclose(lhs,rhs,atol=1e-4,rtol=1e-5)

def test_ssdu_training_paths_never_read_truth_or_holdout():
    source=Path(inspect.getsourcefile(adapt_scan)).read_text();tree=ast.parse(source);targets={'adapt_scan','private_tune'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in targets:
            body=ast.get_source_segment(source,node)
            assert "['x']" not in body and "['ho']" not in body and '["x"]' not in body and '["ho"]' not in body

def test_mirror_rotation_is_orthogonal():
    m=Prior();x=torch.randn(1,2,32,32);f=m.feat(x);a,b=torch.tensor(.2),torch.tensor(-.4);ca,sa=torch.cos(a),torch.sin(a);cb,sb=torch.cos(b),torch.sin(b);q=f.clone();q[:,0]=ca*f[:,0]-sa*f[:,1];q[:,1]=sa*f[:,0]+ca*f[:,1];q[:,2]=cb*f[:,2]-sb*f[:,3];q[:,3]=sb*f[:,2]+cb*f[:,3];assert torch.allclose(q.square().sum(),f.square().sum(),atol=1e-4)
