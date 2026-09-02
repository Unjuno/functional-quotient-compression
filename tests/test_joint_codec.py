from fqc.decoder_dag import Atom
from fqc.joint_codec import JointOption, LayoutSpec, exact_joint_codec_allocate, coordinate_best_from_incumbent, evaluate_joint_candidate


def _layouts():
    per=(JointOption('p2',5,(),mode='private',precision=2), JointOption('s3',2,('root',),mode='shared',precision=3))
    return [LayoutSpec('cross',(per,per)),LayoutSpec('aligned',(per,per))]


def _error(layout,opts):
    states=tuple(o.option_id for o in opts)
    if layout=='aligned' and states==('s3','s3'): return 8.0
    if layout=='cross' and states==('s3','s3'): return 10.5
    if layout=='aligned' and states==('p2','p2'): return 10.0
    if states==('p2','p2'): return 10.0
    return 11.0


def test_joint_layout_precision_root_move_escapes_coordinate_dead_zone():
    atoms=[Atom('root','PAID',6)]
    layouts=_layouts(); budget=10
    coord=coordinate_best_from_incumbent(layouts,atoms,budget,_error,'cross',('p2','p2'))
    assert coord['layout_only'].error==10.0
    assert coord['local_only'].error==10.0
    best=exact_joint_codec_allocate(layouts,atoms,budget,_error)
    assert best is not None
    assert best.layout_id=='aligned' and best.option_ids==('s3','s3')
    assert best.total_bits==10 and best.error==8.0


def test_shared_root_is_union_charged_once_in_joint_state():
    atoms=[Atom('root','PAID',6)]
    c=evaluate_joint_candidate(_layouts()[1],atoms,[1,1],_error)
    assert c.private_bits==4 and c.shared_paid_bits==6 and c.total_bits==10


def test_joint_error_can_be_nonseparable():
    per=(JointOption('a',0),JointOption('b',0))
    layout=LayoutSpec('L',(per,per))
    def err(_,opts):
        return 0.0 if tuple(o.option_id for o in opts)==('b','b') else 1.0
    best=exact_joint_codec_allocate([layout],[],0,err)
    assert best.option_ids==('b','b') and best.error==0.0
