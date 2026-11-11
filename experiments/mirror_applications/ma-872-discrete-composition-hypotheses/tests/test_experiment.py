import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from experiment import evaluate,make_world,fit_models,serialize


def test_mirror_address_exactly_aliases_native_codebook():
    row=evaluate(821)
    for n in ('1','2','4','8','16','32'):
        m=row['by_support'][n]
        assert m['mean_mse']['mirror']==m['mean_mse']['native_codebook']
        assert m['per_task_mse']['mirror']==m['per_task_mse']['native_codebook']
        assert m['indices']['mirror']==m['indices']['native_codebook']
        assert m['payload_bytes']['mirror']==m['payload_bytes']['native_codebook']


def test_actual_payload_lengths_charge_codebook_and_per_task_codes():
    world=make_world(821); model=fit_models(world)
    codes=[0]*16
    a=len(serialize(world,model,'mirror',indices=codes))
    b=len(serialize(world,model,'independent',codes=world['true_prototypes'][[0]*16]))
    assert a==569
    assert b==809
    assert a < b
    assert a==len(serialize(world,model,'native_codebook',indices=codes))
