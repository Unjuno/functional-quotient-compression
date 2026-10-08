import csv,importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"source"/"run.py";spec=importlib.util.spec_from_file_location("ma341",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_sinusoidal_client_tasks_and_unseen_linear_generator():
    w0,u,v,amp,z,target,off,support,truth,xeval,offdata=m.world(34111)
    design=np.column_stack([np.ones(m.NTRAIN),z[:m.NTRAIN]])
    gen=np.linalg.lstsq(design,target[:m.NTRAIN].reshape(m.NTRAIN,-1),rcond=None)[0]
    out=(np.column_stack([np.ones(m.NCLIENT),z])@gen).reshape(m.NCLIENT,m.DOUT,m.DIN)
    assert np.max(np.abs(out-target))<1e-5

def test_private_state_restores_offorbit_client():
    w0,u,v,amp,z,target,off,support,truth,xeval,offdata=m.world(34112)
    design=np.column_stack([np.ones(m.NTRAIN),z[:m.NTRAIN]])
    gen=np.linalg.lstsq(design,target[:m.NTRAIN].reshape(m.NTRAIN,-1),rcond=None)[0]
    pred=(np.r_[1.,1.,0.]@gen).reshape(m.DOUT,m.DIN)
    assert m.nerr(offdata[2]@pred.T,offdata[3])>0.1
    assert m.nerr(offdata[2]@off.T,offdata[3])<1e-12

def test_mirror_and_native_scalar_code_payloads_identical_in_dev():
    p=Path(__file__).parents[1]/"artifacts"/"development.csv";rows=list(csv.DictReader(p.open()))
    for seed in ("34101","34102"):
        a=next(r for r in rows if r['seed']==seed and r['method']=='mirror_phase_with_private')
        b=next(r for r in rows if r['seed']==seed and r['method']=='native_scalar_phase_with_private')
        assert a['sha256']==b['sha256'] and a['payload_bytes']==b['payload_bytes']
