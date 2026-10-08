import importlib.util
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma945_model',ROOT/'source'/'model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)

class VideoTokenTests(unittest.TestCase):
    def test_coordinate_grid_ranges_and_count(self):
        video=torch.zeros(model.FRAMES,model.HEIGHT,model.WIDTH,3)
        coords,targets=model.all_coords(video)
        self.assertEqual(tuple(coords.shape),(model.FRAMES*model.HEIGHT*model.WIDTH,3))
        self.assertEqual(tuple(targets.shape),(len(coords),3))
        self.assertGreaterEqual(float(coords.min()),0.0)
        self.assertEqual(float(coords[:,-1].max()),(model.FRAMES-1)/model.FRAMES)
        self.assertEqual(float(coords[:,:2].max()),1.0)
    def test_decoder_coordinate_token_contract(self):
        torch.manual_seed(11);d=model.CoordinateDecoder();coords=torch.rand(19,3);tokens=torch.randn(model.TOKENS,model.TOKEN_DIM)
        y=d(coords,tokens)
        self.assertEqual(tuple(y.shape),(19,3));self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(((y>=0)&(y<=1)).all())
    def test_factor_generators_return_slot_aligned_tokens(self):
        torch.manual_seed(12)
        for kind in ('additive','mirror'):
            g=model.SharedFactorGenerator(kind,4,4);z=torch.randn(4)
            a=g.table_from_code(z);b=g.table_from_code(z+0.1)
            self.assertEqual(tuple(a.shape),(model.TOKENS,model.TOKEN_DIM))
            self.assertFalse(torch.allclose(a,b))
    def test_split_is_deterministic_and_disjoint(self):
        a,b=model.split_indices(1024,9451);aa,bb=model.split_indices(1024,9451)
        self.assertTrue(torch.equal(a,aa));self.assertTrue(torch.equal(b,bb))
        self.assertEqual(len(set(a.tolist())&set(b.tolist())),0)
        self.assertEqual(len(a)+len(b),1024)

if __name__=='__main__':unittest.main()
