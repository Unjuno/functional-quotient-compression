from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_values_share_rank_eight_basis():
    w=run.world(47601);assert torch.linalg.matrix_rank(w['values'],atol=1e-5).item()==run.R

def test_vq_is_fit_on_train_codes_only_and_indices_are_bounded():
    w=run.world(47601);s,_,_=run.prepare('vq16',w,47601)
    assert s['centers'].shape==(16,run.R)
    assert int(s['labels'].max())<16
    assert torch.max(torch.abs(s['centers']-torch.nan_to_num(s['centers'])))==0

def test_fixed_router_is_same_locality_for_value_methods():
    w=run.world(47601);ok=[]
    for e,qs in enumerate(w['queries']):
        for q in qs:
            i,on=run.route(q,w['keys']);ok.append(i==e and on)
    assert sum(ok)/len(ok)>=.995
    assert sum(run.route(q,w['keys'])[1] for q in w['locality'])/run.NLOCAL<=.005

def test_continuous_basis_aliases_native_pca():
    w=run.world(47601);s,_,_=run.prepare('mirror_pca',w,47601)
    assert run.serialize('mirror_pca',w,s)==run.serialize('native_pca',w,s)
