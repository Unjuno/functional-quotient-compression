from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_values_have_exact_shared_rank_eight_basis():
    w=run.world(47501)
    assert torch.linalg.matrix_rank(w['values'],atol=1e-5).item()==run.RANK

def test_shared_basis_fit_uses_train_values_and_recovers_heldout():
    w=run.world(47501);m,b,c=run.make_values('mirror_pca',w)
    assert torch.max(torch.abs(m-w['values']))<1e-5
    assert b.shape==(run.D,run.RANK) and c.shape==(run.N,run.RANK)

def test_fixed_key_router_selects_queries_and_avoids_locality():
    w=run.world(47501);ok=[]
    for e,qs in enumerate(w['queries']):
        for q in qs:
            i,on=run.route(q,w['keys']);ok.append(i==e and on)
    assert sum(ok)/len(ok)>=.995
    assert sum(run.route(q,w['keys'])[1] for q in w['locality'])/run.NLOCAL<=.005

def test_shared_pca_serialization_aliases_native_control():
    w=run.world(47501)
    _,basis,codes=run.make_values('mirror_pca',w)
    a=run.serialize('mirror_pca',w,basis,codes)
    b=run.serialize('native_pca',w,basis,codes)
    assert a==b


def test_no_edit_payload_does_not_store_value_arrays():
    w=run.world(47501);raw=run.serialize('no_edit',w,None,None)
    import io,numpy as np
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        assert 'values' not in z.files
        assert 'keys' in z.files
