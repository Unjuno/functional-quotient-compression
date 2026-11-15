import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import dc_audit as d

def test_projection_satisfies_acquired_measurements():
    c=d.case(1014101,1); x=d.phantom(1014102,True); p=d.project(x,c)
    assert d.metrics(p,c)[0] < 1e-6

def test_nonzero_mirror_after_dc_breaks_consistency():
    c=d.case(1014101,1); model=d.Prior(); a=torch.tensor([.37,-.29])
    p=d.project(model(c['zf'][None])[0],c); q=model(p[None],True,a)[0]
    assert d.metrics(q,c)[0] > 1e-4

def test_payload_charges_mirror_coordinate():
    model=d.Prior(); plain=len(d.pack(model,'prior_then_dc',None)); mirrored=len(d.pack(model,'mirror_before_prior_then_dc',[.37,-.29]))
    assert mirrored > plain
