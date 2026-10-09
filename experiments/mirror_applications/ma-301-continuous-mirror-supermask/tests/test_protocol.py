import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma301',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_teacher_shapes_and_determinism():
 a=m.make(30100,0);b=m.make(30100,0)
 assert a[0].shape==(4096,2) and a[1].shape==(64,2) and a[2].shape==(64,4096)
 assert (a[2]==b[2]).all()
def test_binary_masks_are_bit_packed():
 import torch
 raw=m.pack_masks(torch.zeros((64,4096),dtype=torch.uint8),'mask')
 expected=6+4+len(__import__('json').dumps({'method':'mask','shape':[64,4096],'bitorder':'little'},separators=(',',':')))+64*4096//8
 assert len(raw)==expected
