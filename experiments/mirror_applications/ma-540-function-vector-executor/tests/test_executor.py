import importlib.util
from pathlib import Path
import numpy as np
import torch
p=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma540',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_affine_rules_are_bijections_and_closed():
 table,tr,te,rules=m.make_world(54001)
 assert table.shape==(24,16)
 assert all(len(set(row.tolist()))==16 for row in table)
 assert set(map(tuple,tr)).isdisjoint(set(map(tuple,te)))
 for a,b in te:
  assert (int(a),int(b)) in map(tuple,te)
  assert (int(b),int(a)) in map(tuple,te)
def test_extracted_fv_exactly_aliases_operator_embedding():
 torch.manual_seed(7);net=m.StepNet();support=np.tile(np.arange(8),(m.R,1));fv=net.extract_fv(support)
 x=torch.arange(m.N).repeat(m.R);op=torch.arange(m.R).repeat_interleave(m.N)
 torch.testing.assert_close(net.forward_ids(x,op),net.forward_fv(x,fv[op]),rtol=1e-5,atol=5e-5)
def test_composition_order_can_differ():
 table,_,_,_=m.make_world(54002)
 pairs=[(a,b) for a in range(m.R) for b in range(m.R) if a!=b]
 assert any(not np.array_equal(table[b,table[a]],table[a,table[b]]) for a,b in pairs)
