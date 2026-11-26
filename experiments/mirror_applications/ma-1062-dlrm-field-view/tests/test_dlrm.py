import importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma1062',ROOT/'source/train.py')
ma=importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)

def test_hash_is_deterministic_and_field_specific():
    x=torch.arange(1,100,dtype=torch.long)
    assert torch.equal(ma.field_hash(x,0,512),ma.field_hash(x,0,512))
    assert not torch.equal(ma.field_hash(x,0,512),ma.field_hash(x,1,512))

def test_givens_view_preserves_each_embedding_norm_at_initialization():
    torch.manual_seed(7); model=ma.DLRM('mirror_512',0,1)
    cats=torch.tensor([[1,1,1,0],[943,1682,12,6]],dtype=torch.long)
    before=torch.stack([model.shared(ma.field_hash(cats[:,i],i,512)) for i in range(4)],dim=1)
    after=model.embed_fields(cats)
    assert torch.equal(before,after)
    assert torch.allclose(before.norm(dim=-1),after.norm(dim=-1),atol=1e-7)

def test_dlrm_output_shape_for_each_parameterization():
    cats=torch.tensor([[1,1,1,0],[2,3,12,6]],dtype=torch.long); dense=torch.zeros(2,1)
    for method in ma.METHODS:
        m=ma.DLRM(method,0,1)
        assert m(cats,dense).shape==(2,)

def test_train_dev_split_is_pair_disjoint():
    rows=[(i%3+1,i%7+1,1,0,1_000_000+i,float(i%2)) for i in range(100)]
    train,dev=ma.split_train_dev(rows)
    train_pairs={(r[0],r[1]) for r in train}; dev_pairs={(r[0],r[1]) for r in dev}
    assert train_pairs.isdisjoint(dev_pairs)

def test_protocol_freezes_all_controls_and_audit_outputs_do_not_save_labels():
    import json
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert 'sixteen per-coordinate diagonal gains' in protocol['controls'][-2]
    assert 'diag16_512' in (ROOT/'source/train.py').read_text()
    assert 'per-coordinate diagonal' in protocol['success_gate']
    source=(ROOT/'source/train.py').read_text()
    assert "np.savez_compressed(adir/f'{method}_seed{seed}.npz',logits=preds)" in source
    assert "np.savez_compressed(adir/f'{method}_seed{seed}.npz',logits=preds,labels=labels)" not in source
    assert protocol['amendments'][0]['audit_access']=='ua.test remains unopened.'
    assert 'no audit metric or prediction was computed' in protocol['amendments'][-1]['audit_access']

def test_final_train_path_accepts_no_eval_rows():
    source=(ROOT/'source/train.py').read_text()
    assert 'if eval_rows is not None: ec,ed,ey=arrays(eval_rows,mean,std)' in source
    assert 'else: ec=ed=ey=None' in source
    protocol=__import__('json').loads((ROOT/'PROTOCOL.json').read_text())
    assert protocol['amendments'][-1]['id']=='AM-02'
    assert 'no audit metric or prediction was computed' in protocol['amendments'][-1]['audit_access']
