import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_screen.py'
spec=importlib.util.spec_from_file_location('ma516',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_task_registry_is_16_unique_function_ids():
 assert len(m.TASKS)==16
 assert len({n for n,_ in m.TASKS})==16
 assert all(len(pairs)==16 for _,pairs in m.TASKS)
def test_seeded_support_query_split_is_disjoint():
 for seed in (51601,51602):
  for tid in range(16):
   a,b=m.split_task(tid,seed)
   assert len(a)==len(b)==8
   assert set(a).isdisjoint(set(b))
def test_basis_holdout_ids_do_not_overlap_fit_ids():
 assert set(m.BASIS_TASKS).isdisjoint(m.HELD_TASKS)
 assert m.BASIS_TASKS==tuple(range(12)) and m.HELD_TASKS==tuple(range(12,16))

def test_every_relation_is_one_to_one():
 for _,pairs in m.TASKS:
  assert len({x for x,_ in pairs})==16
  assert len({y for _,y in pairs})==16


def test_rank_one_mirror_is_compressed_not_explicit():
 assert m.representation_kind('mirror_r1',1)==(2,1)
 assert m.representation_kind('native_pca_r1',-1)==(2,1)
 assert m.representation_kind('explicit_fv',1)==(1,0)
