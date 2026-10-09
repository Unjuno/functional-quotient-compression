import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_screen.py'
spec=importlib.util.spec_from_file_location('ma517',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_fixed_composition_pairs_are_inverses():
 c=m.load_common()
 for a,b in m.PAIR_IDS:
  left=dict(c.TASKS[a][1]);right=dict(c.TASKS[b][1])
  assert all(right[left[x]]==x for x in left)
def test_mirror_native_product_algebra_is_same():
 x=np.arange(16*512,dtype=np.float32).reshape(16,512)/1000
 c=m.load_common();a,parts=m.code_product(x,4,c);b,parts2=m.code_product(x,4,c)
 assert np.array_equal(a,b)
 assert all(np.array_equal(x,y) for x,y in zip(parts,parts2))
def test_split_keys_are_disjoint_for_registered_dev_seeds():
 c=m.load_common()
 for seed in (51701,51702):
  for tid in range(16):
   support,query=c.split_task(tid,seed)
   assert set(support).isdisjoint(set(query))


def test_heldout_path_is_hidden_from_both_inverse_supports():
 c=m.load_common()
 for seed in (51701,51702):
  manifest=m.aligned_manifest(seed,c)
  for a,b in m.PAIR_IDS:
   sa={x for x,_ in manifest[a]['support']}
   qa={x for x,_ in manifest[a]['evaluation']}
   sb={x for x,_ in manifest[b]['support']}
   qb={x for x,_ in manifest[b]['evaluation']}
   assert not (sa&qa)
   assert not ({y for x,y in manifest[a]['evaluation']} & sb)
   assert {y for x,y in manifest[a]['evaluation']}==qb
