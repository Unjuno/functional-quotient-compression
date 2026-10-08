"""Tests for Stage0 MA1171 physical codec and joint objective."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ma1171_joint_codec import (
 BITS, METHODS, PRIV, N_TASK, N_BLOCK, D, packbits, unpackbits, rotate2,
 _source_and_targets, _fit_block, make_full_blob, decode_full, decode_block,
 enum_exact, input_batch, run_world)

class JointTests(unittest.TestCase):
 def test_int2_int4_int8_roundtrip_and_bounds(self):
  rng=np.random.default_rng(7)
  for bits in BITS:
   for n in [1,2,3,5,6,7,9,12,100]:
    q=rng.integers(0,1<<bits,n,dtype=np.uint8)
    blob=packbits(q,bits)
    self.assertEqual(len(blob),(bits*n+7)//8)
    np.testing.assert_equal(unpackbits(blob,bits,n),q)

 def test_orbit_rotation_and_null_gauge(self):
  b=np.array([1.25,-0.32]);a=.53
  rec=rotate2(b,a)
  self.assertAlmostEqual(np.linalg.norm(rec),np.linalg.norm(b),places=12)
  np.testing.assert_allclose(rotate2(rec,-a),b,atol=1e-12)

 def test_codec_all_methods_and_private(self):
  rng=np.random.default_rng(123)
  src=rng.normal(size=(6,2));tgt=rng.normal(size=(6,2))
  for meth in METHODS:
   for bits in BITS:
    for private in PRIV:
     opt=_fit_block(src,tgt,meth,bits,private)
     self.assertEqual(opt['byte_count'],len(opt['blob'])+2)
     np.testing.assert_allclose(decode_block(opt['blob']),opt['rec'],rtol=1e-6,atol=1e-6)
     self.assertGreaterEqual(opt['byte_count'],9)

 def test_full_codec_length_and_reject_trailing(self):
  src,support,_,_,_=_source_and_targets(3,1.)
  options=[_fit_block(src[:,2*i:2*i+2],support[:,2*i:2*i+2],'mirror_orientation',4,int(i==2)) for i in range(4)]
  b=make_full_blob(options)
  self.assertEqual(len(b),7+sum(o['byte_count'] for o in options))
  np.testing.assert_allclose(decode_full(b),np.concatenate([o['rec'] for o in options],axis=1),atol=1e-6,rtol=1e-6)
  for corrupt in [b'',b[:5],b[:-1],b+b'\x00',b'XXXXX'+b[5:]]:
   with self.assertRaises(ValueError):decode_full(corrupt)

 def test_source_only_calibrated_and_coupled(self):
  source,support,truth,probe,test=_source_and_targets(11,.5)
  self.assertEqual(source.shape,(6,8));self.assertEqual(support.shape,(6,8))
  self.assertEqual(truth.shape,(6,8));self.assertEqual(test.shape,(6,512,8))
  G=probe.T@probe/len(probe)
  self.assertGreater(np.linalg.eigvalsh(G).min(),0)
  self.assertGreater(np.linalg.norm(G-np.diag(np.diag(G))),.1)
  u=np.arange(8,dtype=np.float64)*.15
  exact=np.mean((probe@u)**2)
  approx=float(u@G@u)
  self.assertAlmostEqual(exact,approx,places=12)
  self.assertGreater(abs(approx-u@np.diag(np.diag(G))@u),1e-6)

 def test_exact_enumeration_matches_all_sampled_bits(self):
  rng=np.random.default_rng(97)
  source,support,truth,probe,test=_source_and_targets(12,1.)
  options=[]
  for b in range(4):
   options.append([_fit_block(source[:,2*b:2*b+2],support[:,2*b:2*b+2],m,q,p)
                   for m in METHODS for q in BITS for p in PRIV])
  G=probe.T@probe/len(probe)
  limits=[108,111,139,159]
  opt=enum_exact(options,G,support,limits)
  for budget,allx,nativex,_ in opt:
   for chosen in [allx,nativex]:
    if chosen is None:continue
    objective,bytescost,_,spec,blob,rec=chosen
    self.assertLessEqual(bytescost,budget)
    self.assertEqual(bytescost,len(blob))
    delta=support-rec
    exact=np.einsum('ti,ij,tj->',delta,G,delta)/N_TASK
    self.assertAlmostEqual(objective,exact,places=9)
    np.testing.assert_allclose(decode_full(blob),rec,rtol=1e-6,atol=1e-6)

 def test_all_blockwise_realized_mse_computes(self):
  rows=run_world(21,1.)
  self.assertEqual(len(rows),12)
  self.assertTrue(all(r['bytes']<=r['budget_bytes'] for r in rows if r['feasible']))
  self.assertEqual(sum(r['kind']=='joint_with_mirror' for r in rows),4)

if __name__=='__main__':unittest.main(verbosity=2)
