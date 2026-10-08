import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import SharedDecoder,make_tasks,kmeans,D,K

def test_shared_decoder_shapes():
 d=SharedDecoder();x=torch.randn(3,40,2);z=torch.randn(3,1,D)
 assert d(x,z).shape==(3,40)

def test_task_world_has_declared_factors():
 book,train,held,levels,g=make_tasks(41701)
 assert book.shape==(K,D) and train.shape==(64,D) and held.shape==(64,D)
 assert levels.count(0.0)==16 and levels.count(.30)==16

def test_kmeans_codebook_dimensions():
 z=torch.randn(64,D);b=kmeans(z)
 assert b.shape==(K,D)
