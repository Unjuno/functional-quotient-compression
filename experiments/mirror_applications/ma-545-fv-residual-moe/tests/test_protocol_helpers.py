import sys
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'source'
sys.path.insert(0,str(SOURCE))
from run_experiment import demo_set, prompt_with_query, route, npz_bytes


def test_demo_set_excludes_query_and_keeps_four_context_examples():
    support=[(f'x{i}',f'y{i}') for i in range(8)]
    demos=demo_set(support,'x7')
    assert len(demos)==4
    assert all(x!='x7' for x,_ in demos)
    assert 'Input: x7\nOutput:' in prompt_with_query('x7',demos)


def test_router_argmax_and_stored_normalization():
    router={'mean':np.zeros(2,np.float32),'std':np.ones(2,np.float32),
            'weight':np.array([[1,0],[0,1]],np.float32),'bias':np.zeros(2,np.float32)}
    ids,logits=route(router,np.array([[0,2],[3,0]],np.float32))
    assert ids.tolist()==[1,0]
    assert logits.tolist()==[[0.0,2.0],[3.0,0.0]]


def test_npz_payload_counts_actual_serialized_bytes(tmp_path):
    arr={'weights':np.arange(32,dtype=np.float32),'ids':np.arange(3,dtype=np.int16)}
    p=tmp_path/'payload.npz'
    size=npz_bytes(p,arr)
    assert size==p.stat().st_size
    with np.load(p) as loaded:
        assert np.array_equal(loaded['weights'],arr['weights'])
        assert np.array_equal(loaded['ids'],arr['ids'])
