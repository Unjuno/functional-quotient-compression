import importlib.util
import unittest
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ma742', ROOT / 'source' / 'run_experiment.py')
ma742 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma742)

class ProtocolTests(unittest.TestCase):
    def test_relation_split_is_four_heldout_combinations(self):
        train, held = ma742.split_relations()
        self.assertEqual(len(train), 12)
        self.assertEqual(held, [0, 6, 11, 13])
        self.assertFalse(set(train) & set(held))

    def test_mirror_uses_composed_attribute_code(self):
        model = ma742.RelationModel('mirror', 2, 7)
        with __import__('torch').no_grad():
            model.attr_a.copy_(__import__('torch').tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]))
            model.attr_b.copy_(__import__('torch').tensor([[2.,3.],[4.,5.],[6.,7.],[8.,9.]]))
            model.decode.copy_(__import__('torch').eye(2,4))
        rel=torch.tensor([0])
        coeff=model.relation_coeff(rel)
        self.assertTrue(__import__('torch').allclose(coeff,torch.tensor([[2.,6.,0.,0.]])))

    def test_serialized_payload_contains_all_coordinate_state(self):
        import torch
        for method in ('native','additive','mirror','independent'):
            model=ma742.RelationModel(method,2,11)
            whole,marginal,common,digest=ma742.state_payload(model,method)
            self.assertGreater(len(whole),len(marginal))
            self.assertEqual(len(digest),64)
            self.assertGreater(len(common),0) if method!='independent' else self.assertEqual(common,b'')

if __name__=='__main__': unittest.main()
