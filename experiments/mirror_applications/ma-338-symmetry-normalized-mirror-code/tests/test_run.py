import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"source"/"run.py";spec=importlib.util.spec_from_file_location("ma338",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_activation_matching_removes_exact_hidden_gauges():
    base,tasks,off,xcal,xeval,phase=m.world(33811);href=np.tanh(xcal@base["W1"].T+base["b1"])
    for g in tasks+[off]:
        can,e=m.canonicalize(g,xcal,href)
        assert e<1e-5
        assert np.max(np.abs(m.infer(can,xeval)-m.infer(g,xeval)))<1e-5

def test_normalized_rank_two_recovers_heldout_phase_functions():
    base,tasks,off,xcal,xeval,phase=m.world(33812);href=np.tanh(xcal@base["W1"].T+base["b1"])
    can=[m.canonicalize(n,xcal,href)[0] for n in tasks]
    d=np.stack([n["W2"]-base["W2"] for n in can]);_,s,vt=np.linalg.svd(d[:m.NTRAIN].reshape(m.NTRAIN,-1),full_matrices=False);B=vt[:2]
    coeff=d.reshape(m.NTASK,-1)@B.T;recon=(coeff@B).reshape(m.NTASK,m.DOUT,m.NH)
    assert np.max(np.abs(recon-d))<1e-5

def test_raw_rank_two_is_not_gauge_invariant():
    base,tasks,off,xcal,xeval,phase=m.world(33802);raw=np.stack([m.vec(n) for n in tasks]);mean=raw[:m.NTRAIN].mean(0);_,s,vt=np.linalg.svd(raw[:m.NTRAIN]-mean,full_matrices=False);B=vt[:2]
    recon=mean+(raw-mean)@B.T@B
    preds=[m.infer(m.unvec(v),xeval) for v in recon[m.NTRAIN:]]
    expected=[m.infer(n,xeval) for n in tasks[m.NTRAIN:]]
    assert np.mean([m.nerr(a,b) for a,b in zip(preds,expected)])>0.1
