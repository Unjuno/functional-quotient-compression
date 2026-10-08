import unittest
import torch
from run_sfm003 import world,rotate,rotate_stored,fit,predict,base,model_bytes,diversity_ratio
class TestSFM003(unittest.TestCase):
    def test_true_orbit_exact_and_diverse(self):
        a=world(11,5,0.)
        self.assertTrue(torch.equal(rotate(a['zs'][2].unsqueeze(0).expand(5,-1,-1),a['true_angles']),a['ys'][2]))
        self.assertGreater(diversity_ratio(a['ys'][2],a['ys'][2])[0],.999)
    def test_stored_sin_cos_equivalence(self):
        a=world(12,4,0.)
        y=a['zs'][0].unsqueeze(0).expand(4,-1,-1)
        w=a['true_angles']
        self.assertLess(float((rotate(y,w)-rotate_stored(y,w.cos(),w.sin())).abs().max()),1e-7)
    def test_no_audit_training_and_shapes(self):
        a=world(13,4,.3)
        for m in ('mirror','diag','linear'):
            p,loss=fit(m,a['zs'][0],a['ys'][0])
            self.assertEqual(predict(a['zs'][2],m,p).shape,(4,512,8))
            self.assertGreater(model_bytes(a,m,p),0)
    def test_no_extra_heavy_pass_in_packed_output(self):
        a=world(11,4,0.)
        count=[0]
        def heavy(x):
            count[0]+=1;return base(x,a['U'],a['D'])
        z=heavy(a['xs'][2]); y=rotate(z.unsqueeze(0).expand(4,-1,-1),a['true_angles'])
        self.assertEqual(count[0],1)
        self.assertEqual(y.shape,(4,512,8))
if __name__=='__main__':unittest.main(verbosity=2)
