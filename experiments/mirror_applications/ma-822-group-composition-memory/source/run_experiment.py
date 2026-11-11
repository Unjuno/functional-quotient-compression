"""Exact cyclic group matrix memory versus native group indexing."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82201;N=16;D=2

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def rot(k):a=2*np.pi*k/N;return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]],np.float32)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 table=np.stack([rot(k) for k in range(N)]);pairs=[((i*7)%N,(i*11+3)%N) for i in range(32)];truth=np.stack([table[a]@table[b] for a,b in pairs]);ind=pack(element_matrices=table,group_pairs=np.array(pairs,np.int8));mirror=pack(generator=table[1],modulus=np.array(N,np.int8),element_codes=np.arange(N,dtype=np.uint8),group_pairs=np.array(pairs,np.int8));native=pack(generator=table[1],order=np.array(N,np.int8),group_indices=np.arange(N,dtype=np.uint8),group_pairs=np.array(pairs,np.int8));rows=[]
 for name,payload,kind in [('independent_matrix_table',ind,'ind'),('mirror_group_element_code',mirror,'code'),('native_cyclic_group_index',native,'code')]:
  z=np.load(io.BytesIO(payload));decoded=z['element_matrices'] if kind=='ind' else np.stack([np.linalg.matrix_power(z['generator'],k) for k in range(N)])
  pred=np.stack([decoded[a]@decoded[b] for a,b in pairs]);err=nrmse(pred,truth);rt=float(np.max(np.abs(decoded-table)));st=time.perf_counter();_=np.stack([decoded[a]@decoded[b] for a,b in pairs]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_composition_nrmse=err,composition_count=len(pairs),group_order=N,representation_dim=D,active_compute_proxy=2 if kind=='ind' else 4,payload_roundtrip_max_abs=rt,max_orthogonality_error=float(max(np.linalg.norm(m.T@m-np.eye(D)) for m in decoded)),max_determinant_error=float(max(abs(np.linalg.det(m)-1) for m in decoded)),wall_time_s=wall,status_note='native cyclic group law alias' if name=='mirror_group_element_code' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
