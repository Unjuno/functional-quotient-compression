from __future__ import annotations
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ma1183_real_tabular import real_dataset,SEEDS,CHECKS,MAX_UPDATES

class RealTabularTests(unittest.TestCase):
 def test_splits_and_shapes(self):
  for name in ('breast_cancer','wine_binary'):
   for seed in SEEDS:
    (x,y),(v,z),(t,q),h=real_dataset(name,seed)
    self.assertEqual(x.shape[1],32)
    self.assertEqual(v.shape[1],32)
    self.assertEqual(t.shape[1],32)
    self.assertTrue(np.all(np.isfinite(x)))
    self.assertEqual(len(x)+len(v)+len(t),569 if name=='breast_cancer' else 178)
    self.assertEqual(len(h),64)
    self.assertEqual(set(np.unique(y)),{0,1})
 def test_training_schedule(self):
  self.assertEqual(CHECKS,(25,50,100,150,200))
  self.assertEqual(MAX_UPDATES,200)
 def test_source_data_determinism(self):
  a=real_dataset('wine_binary',302);b=real_dataset('wine_binary',302)
  for k in range(3):
   np.testing.assert_array_equal(a[k][0],b[k][0])
   np.testing.assert_array_equal(a[k][1],b[k][1])

if __name__=='__main__':unittest.main(verbosity=2)
