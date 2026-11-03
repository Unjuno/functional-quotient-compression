import importlib.util
from pathlib import Path
import numpy as np
SRC=Path(__file__).parents[1]/'source'/'run_experiment.py';spec=importlib.util.spec_from_file_location('ma591',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_prompt_flatten_shape_and_lowrank_decode():
 rng=np.random.default_rng(591);p=rng.normal(size=(m.TASKS,m.PROMPT_LEN,m.HIDDEN)).astype(np.float32)
 mu=p.reshape(m.TASKS,-1).mean(0);_,_,vt=np.linalg.svd(p.reshape(m.TASKS,-1)-mu,full_matrices=False);b=vt[:m.RANK].T;c=(p.reshape(m.TASKS,-1)-mu)@b
 dec=c@b.T+mu
 assert b.shape==(m.PROMPT_LEN*m.HIDDEN,m.RANK)
 assert dec.shape==(m.TASKS,m.PROMPT_LEN*m.HIDDEN)
 assert np.isfinite(dec).all()
def test_article_task_boundary_and_window_constants():
 assert m.WIN==64 and m.PROMPT_LEN==8
 assert len(set([1024,1152,1280,1408]))==4


def test_serialized_coordinate_decode_shape():
 rng=np.random.default_rng(592);mean=rng.normal(size=m.PROMPT_LEN*m.HIDDEN).astype(np.float16);basis=rng.normal(size=(m.PROMPT_LEN*m.HIDDEN,m.RANK)).astype(np.float16);codes=rng.normal(size=(m.TASKS,m.RANK)).astype(np.float16)
 out=m.decode_task_prompt(mean,basis,codes,0)
 assert out.shape==(m.PROMPT_LEN*m.HIDDEN,) and np.isfinite(out).all()
