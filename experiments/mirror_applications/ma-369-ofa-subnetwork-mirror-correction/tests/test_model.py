import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from model import ARCHS, OFASupernet, rotate_first_pairs

class ModelTests(unittest.TestCase):
    def test_givens_is_orthogonal_and_zero_is_identity(self):
        x=torch.randn(5,16)
        z=rotate_first_pairs(x,torch.zeros(4))
        self.assertTrue(torch.equal(x,z))
        angles=torch.tensor([.1,-.2,.3,-.4])
        y=rotate_first_pairs(x,angles)
        self.assertAlmostEqual(float((x[:,:8].square().sum()-y[:,:8].square().sum()).abs()),0.0,places=4)

    def test_nested_subnet_shapes(self):
        m=OFASupernet()
        x=torch.randn(7,64)
        for arch in ARCHS:
            self.assertEqual(tuple(m(x,arch).shape),(7,10))

    def test_required_architectures(self):
        self.assertEqual(ARCHS,((16,1),(16,2),(32,1),(32,2),(64,1),(64,2)))

if __name__=="__main__":
    unittest.main()
