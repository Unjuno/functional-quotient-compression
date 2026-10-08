import sys
from pathlib import Path
import numpy as np
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run_screen as r

class ScreenTests(unittest.TestCase):
 def test_views_are_orthogonal(self):
    for m in (-1.2,-.2,0.,.8):
        for t in (r.monarch(m),r.butterfly(m)):
            self.assertLess(np.max(np.abs(t@t.T-np.eye(8))),1e-12)

 def test_serializer_roundtrip_layout_and_stability(self):
    x=np.arange(12,dtype=np.float32).reshape(3,4)
    payload=r.serialize([('x',x)])
    self.assertEqual(payload,r.serialize([('x',x)]))
    got=r.deserialize(payload)
    self.assertEqual(got[0][0],'x'); np.testing.assert_array_equal(got[0][1],x)
    self.assertEqual(len(payload),71)

 def test_aligned_monarch_reconstructs(self):
    a,b,_,_=r.init_world(28211,True); m=.37; x=np.random.default_rng(4).normal(size=(20,8))
    y=r.target(x,a,b,m,True); T=r.monarch(m)
    self.assertLess(np.max(np.abs(y-r.relu_ffn(x@T.T,a,b)@T)),1e-12)

 def test_independent_full_reconstructs_each_teacher(self):
    _,_,codes,teachers=r.init_world(28211,False)
    x=np.random.default_rng(5).normal(size=(20,8))
    for w1,w2,_ in teachers:
        y=r.relu_ffn(x,w1,w2)
        self.assertLess(np.max(np.abs(y-r.relu_ffn(x,w1,w2))),1e-12)

 def test_independent_full_upper_reconstructs_aligned_and_unrelated_teachers(self):
    for aligned in (True,False):
        rows=r.eval_world(28241,aligned,'upper_fixed_fresh',4)
        upper=next(x for x in rows if x['method']=='independent_full')
        self.assertLess(upper['mse'],1e-20)

 def test_frozen_world_metrics_replay(self):
    rows=r.eval_world(28221,True,'amended_fresh',4)
    mirror=next(x for x in rows if x['method']=='mirror_monarch')
    self.assertLess(mirror['mse'],1e-20)
    independent=r.eval_world(28221,False,'amended_fresh',4)
    upper=next(x for x in independent if x['method']=='independent_full')
    self.assertLess(upper['mse'],1e-20)
    self.assertLess(mirror['payload_bytes'],.5*upper['payload_bytes'])

 def test_independent_monarch_executes_charged_view_codes(self):
    rows=r.eval_world(28231,False,'final_fresh',4)
    row=next(x for x in rows if x['method']=='ind_monarch')
    self.assertGreater(row['mse'],0.)
