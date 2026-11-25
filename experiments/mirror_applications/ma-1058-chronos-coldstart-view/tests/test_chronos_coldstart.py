import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import ma1058

def test_predeclared_series_groups_are_disjoint():
    assert set(ma1058.FIT_IDS).isdisjoint(ma1058.DEV_IDS)
    assert set(ma1058.FIT_IDS+ma1058.DEV_IDS).isdisjoint(ma1058.AUDIT_IDS)
    assert ma1058.AUDIT_IDS==list(range(62,78))

def test_support_and_query_targets_do_not_overlap_and_are_chronological():
    x=np.arange(52116,dtype=np.float32)
    e=ma1058.examples(x)
    support=[o for _,_,o in e['support']]; query=[o for _,_,o in e['query']]
    assert len(support)==6 and len(query)==3
    assert max(o+ma1058.PRED_LEN for o in support)<=int(len(x)*.60)
    assert min(query)>=int(len(x)*.80)
    assert support==sorted(support) and query==sorted(query)

def test_rotation_is_orthonormal_and_identity_at_zero():
    z=torch.tensor([[.3,-.7],[2.,1.]])
    for t in (-1.2,0.0,.7):
        c,s=np.cos(t),np.sin(t)
        r=torch.tensor([[c,-s],[s,c]],dtype=z.dtype)
        out=z@r.T
        assert torch.allclose(out.norm(dim=-1),z.norm(dim=-1),atol=1e-6)
        if t==0: assert torch.allclose(out,z,atol=1e-6)
