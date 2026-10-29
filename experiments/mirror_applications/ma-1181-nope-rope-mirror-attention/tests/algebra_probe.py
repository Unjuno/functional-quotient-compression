import numpy as np
rng=np.random.default_rng(20261030)
def rope(t, angle=0.37):
 c,s=np.cos(t*angle),np.sin(t*angle)
 return np.array([[c,-s],[s,c]],dtype=np.float64)
q=rng.normal(size=2); k=rng.normal(size=2)
v=np.array([0.8,0.6]); M=np.eye(2)-2*np.outer(v,v)
P1=rope(2); P2=rope(7)
original_nope=q@k; mirrored_nope=(M@q)@(M@k)
original_rope=(P1@q)@(P2@k); mirrored_rope=(P1@M@q)@(P2@M@k)
query_only_nope=(M@q)@k
q2=np.array([1.,-2.]); k2=np.array([0.25,1.]);
print({'nope_paired_delta':mirrored_nope-original_nope,'rope_paired_delta':mirrored_rope-original_rope,'nope_query_delta':query_only_nope-original_nope,'commutator_norm':np.linalg.norm(M@rope(5)-rope(5)@M)})
assert abs(mirrored_nope-original_nope)<1e-12
assert abs(mirrored_rope-original_rope)>0.1
assert abs(query_only_nope-original_nope)>0.1
