import importlib.util,zipfile
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'source'/'run_atoms.py';spec=importlib.util.spec_from_file_location('ma526_atoms',SRC);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_pool_selection_uses_fit_ids_and_has_fixed_count():
    rng=np.random.default_rng(2);v=rng.normal(size=(16,512)).astype(np.float32);w=rng.normal(size=(2048,512)).astype(np.float32);b=np.zeros(2048,np.float32)
    ids,score=mod.select_pool(v,w,b)
    assert ids.shape==(64,) and len(set(ids.tolist()))==64 and score.shape==(2048,)

def test_matching_pursuit_is_sparse_and_reconstructs_selected_atoms():
    rng=np.random.default_rng(3);atoms=rng.normal(size=(20,512)).astype(np.float32);x=atoms[[2,8]]*np.array([[1.5],[-.7]],dtype=np.float32)
    ids,vals,dec=mod.matching_pursuit(x,atoms,k=2)
    assert ids.shape==(2,2) and vals.shape==(2,2) and dec.shape==x.shape
    assert len(set(ids[0].tolist()))==2 and np.isfinite(dec).all()

def test_shared_pool_payload_is_uncompressed_npz(tmp_path):
    meta={'task_ids':np.arange(16,dtype=np.int16)};parts={'pool':np.arange(64,dtype=np.int16),'ids':np.zeros((16,8),np.int16),'values':np.zeros((16,8),np.float32)}
    n=mod.save_payload(tmp_path/'x.npz','mirror_pool64_omp8',meta,parts)
    assert n==(tmp_path/'x.npz').stat().st_size
    with zipfile.ZipFile(tmp_path/'x.npz') as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
