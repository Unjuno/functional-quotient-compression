import importlib.util
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma921',ROOT/'source'/'run_experiment.py')
ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)

class FlowProtocolTests(unittest.TestCase):
    def test_heldout_intervals_and_composition_components(self):
        self.assertEqual(len(ma.ALL_PAIRS),210)
        self.assertEqual(len(ma.TRAIN_PAIRS),198)
        self.assertEqual(len(ma.HOLDOUT),12)
        for i,k,j in ma.COMPOSITIONS:
            self.assertIn((i,k),ma.TRAIN_PAIRS);self.assertIn((k,j),ma.TRAIN_PAIRS)
            self.assertIn((i,j),ma.HOLDOUT)

    def test_reference_flow_has_semigroup(self):
        maps=ma.make_world(991)
        for i,k,j in ma.COMPOSITIONS:
            composed=maps[(k,j)]@maps[(i,k)]
            self.assertTrue(torch.allclose(composed,maps[(i,j)],atol=2e-5,rtol=2e-5))

    def test_view_endpoints_compose_by_product(self):
        model=ma.SharedView('mirror',2,13)
        with torch.no_grad():
            model.start.zero_();model.end.zero_();model.decode.copy_(torch.eye(2,4))
            model.start[0]=torch.tensor([2.,3.]);model.end[20]=torch.tensor([4.,5.])
        coeff=model.start[0]*model.end[20]@model.decode
        self.assertTrue(torch.allclose(coeff,torch.tensor([8.,15.,0.,0.])))

    def test_payload_exact_reconstruction(self):
        for method,rank in [('fmm',4),('independent',4),('native_basis',4),('additive',2),('mirror',2)]:
            model=ma.build_model(method,rank,41)
            blob,digest=ma.serialize(model,method,rank);restored,meta=ma.deserialize(blob)
            self.assertEqual(len(digest),64);self.assertEqual(meta['method'],method)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values())))

if __name__=='__main__':unittest.main()
