"""Regression tests for reference RC, BatchEnsemble and operator splitting."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
from algebra_reference import reverse_complement,strand_audit,member_audit,pde_audit,rows_for,DEV,FRESH

class AuditTests(unittest.TestCase):
    def test_rc_is_involution(self):
        J=reverse_complement()
        self.assertEqual(J.shape,(64,64))
        self.assertLess(np.max(np.abs(J@J-np.eye(64))),1e-14)
    def test_rc_symmetry_exact(self):
        for s in DEV+FRESH:
            rc=strand_audit(s)
            self.assertLess(max(rc[:3]),1e-11)
            self.assertGreater(rc[3],1e-3)
    def test_member_collision_and_post_equal(self):
        for s in DEV+FRESH:
            same,collision,post=member_audit(s)
            self.assertLess(same,1e-11)
            self.assertGreater(collision,1e-4)
            self.assertLess(post,1e-11)
    def test_commutation(self):
        for s in DEV+FRESH:
            for dt in [.05,.1,.2]:
                c,lie,corrected,strang,commute,ls,lc=pde_audit(s,dt)
                self.assertGreater(c,1e-4)
                self.assertLess(commute,1e-11)
                self.assertLess(corrected,lie)
                self.assertLess(strang,corrected)
    def test_frozen_splits(self):
        self.assertEqual(len(rows_for("dev",DEV)),9)
        self.assertEqual(len(rows_for("fresh",FRESH)),15)

if __name__=="__main__":
    unittest.main(verbosity=2)
