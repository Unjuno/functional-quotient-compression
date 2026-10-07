import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tm001 import make_world, sample_batch, TinyAR, PeriodModel, full_ar_logits, cached_ar_logits, encode_model, decode_model

def base_cfg(P=4):
    return dict(states=32,rules=12,d=32,heads=4,layers=2,ff=64,max_len=16,P=P)

def test_world_hidden_branch_same_context_two_futures():
    w=make_world(7,32,12)
    b0=sample_batch(w,64,4,'hidden',11,forced_branch=0);b1=sample_batch(w,64,4,'hidden',11,forced_branch=1)
    assert torch.equal(b0['context'],b1['context']);assert (b0['targets']!=b1['targets']).any()

def test_revealed_branch_context_differs_and_future_is_deterministic():
    w=make_world(7,32,12)
    b0=sample_batch(w,64,4,'revealed',11,forced_branch=0);b1=sample_batch(w,64,4,'revealed',11,forced_branch=1)
    assert not torch.equal(b0['context'],b1['context'])
    assert torch.equal(b0['targets'],sample_batch(w,64,4,'revealed',11,forced_branch=0)['targets'])

def test_period_shapes():
    c=base_cfg();m=PeriodModel(c,True);w=make_world(3,32,12);x=sample_batch(w,5,4,'deterministic',4)['context']
    assert m(x).shape==(5,4,m.vocab)

def test_direct_and_mixer_masks_differ():
    c=base_cfg();a=PeriodModel(c,False);b=PeriodModel(c,True);md=a.build_mask(3,4,torch.device('cpu'));mm=b.build_mask(3,4,torch.device('cpu'))
    assert torch.isneginf(md[4,3]);assert mm[4,3]==0;assert torch.isneginf(md[3,4]) and torch.isneginf(mm[3,4])

def test_cached_ar_matches_full_forward():
    torch.manual_seed(9);c=base_cfg();m=TinyAR(c).eval();w=make_world(4,32,12);batch=sample_batch(w,3,4,'revealed',8)
    seq=torch.cat([batch['context'],batch['targets']],1)
    torch.testing.assert_close(full_ar_logits(m,seq,4,4),cached_ar_logits(m,seq,4,4),rtol=1e-5,atol=1e-5)

def test_serialization_roundtrip_exact_logits():
    torch.manual_seed(5);c=base_cfg();m=PeriodModel(c,True).eval();w=make_world(1,32,12);x=sample_batch(w,4,4,'deterministic',2)['context'];y=m(x)
    raw=encode_model(m);n=decode_model(raw).eval();torch.testing.assert_close(y,n(x),rtol=0,atol=0);assert raw==encode_model(n)

def test_latent_period_shapes_and_single_latent_path():
    from tm001 import LatentPeriodModel
    c=base_cfg();c['latents']=2;m=LatentPeriodModel(c,True);w=make_world(3,32,12);x=sample_batch(w,5,4,'hidden',4)['context']
    assert m(x).shape==(5,2,4,m.vocab);assert m(x,latent_id=torch.zeros(5,dtype=torch.long)).shape==(5,4,m.vocab)

def test_latent_joint_loss_is_sequence_mixture_not_token_mixture():
    from tm001 import latent_joint_nll
    logits=torch.full((1,2,2,2),-4.0);logits[0,0,:,0]=4.0;logits[0,1,:,1]=4.0;target=torch.tensor([[0,0]])
    nll=latent_joint_nll(logits,target,torch.zeros(1,2));assert 0.65<float(nll)<0.75

def test_hard_em_sequence_loss_selects_one_latent_for_whole_packet():
    from tm001 import latent_hard_em_loss
    logits=torch.full((2,2,3,2),-5.0);logits[0,0,:,0]=5.0;logits[0,1,:,1]=5.0;logits[1,0,:,0]=5.0;logits[1,1,:,1]=5.0
    loss,assign=latent_hard_em_loss(logits,torch.tensor([[0,0,0],[1,1,1]]),0.0);assert assign.tolist()==[0,1];assert float(loss)<0.01

def test_hidden_oracle_entropy_accounts_for_identical_branch_trajectories():
    from engine import evaluate
    c=base_cfg(P=2);m=TinyAR(c).eval();w=make_world(9,32,12);w['table'][1].copy_(w['table'][0])
    assert abs(evaluate(m,w,'hidden',2)['oracle_joint_entropy_nats'])<1e-12
