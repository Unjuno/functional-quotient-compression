import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import Model,payload,world_data

def test_serialized_payload_and_shapes():
    for method in ['tied','compacter','scalar','mirror','independent']:
        m=Model(method,1); assert m.matrices().shape==(6,16,16); assert payload(m)>0

def test_mirror_coordinate_changes_function():
    m=Model('mirror',2)
    x=torch.randn(7,16); before=torch.einsum('bd,tdh->tbh',x,m.matrices())
    with torch.no_grad(): m.angles[0,0]=0.7
    after=torch.einsum('bd,tdh->tbh',x,m.matrices())
    assert not torch.equal(before[0],after[0])

def test_world_data_shapes_and_disjoint_generators():
    _,xv,y,yv=world_data(7); assert xv.shape==(2048,16) and y.shape==(6,8192,16) and yv.shape==(6,2048,16)
    assert not torch.equal(y[:,:2048],yv[:,:2048])
