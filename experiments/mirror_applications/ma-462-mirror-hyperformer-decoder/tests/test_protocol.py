import importlib.util
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma462',ROOT/'source'/'run_experiment.py')
ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)

class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.world=ma.make_world(462099)
    def test_fixed_embeddings_are_identical_and_frozen(self):
        methods=['linear','hyper','additive','mirror','mirror_private']
        models=[ma.build_decoder(m,4,self.world,17) for m in methods]
        for model in models:
            self.assertFalse(model.emb_a.requires_grad);self.assertFalse(model.emb_b.requires_grad)
            self.assertTrue(torch.equal(model.emb_a,models[0].emb_a));self.assertTrue(torch.equal(model.emb_b,models[0].emb_b))
    def test_alpha_teacher_interpolates_view_and_private(self):
        c0=ma.task_codes(self.world,0.0);c1=ma.task_codes(self.world,1.0)
        self.assertTrue(torch.equal(c0,self.world['private_code']))
        self.assertTrue(torch.equal(c1,self.world['view_code']))
        self.assertFalse(torch.allclose(c0,c1))
    def test_mirror_coordinate_changes_task_function(self):
        model=ma.build_decoder('mirror',4,self.world,7)
        with torch.no_grad():
            model.proj_a.copy_(torch.eye(4));model.proj_b.copy_(torch.eye(4));model.decode.copy_(torch.randn(4,ma.K))
        c=model.all_codes()
        self.assertFalse(torch.allclose(c[0],c[1]))
    def test_payload_roundtrip_and_size(self):
        for method,rank in [('zero',4),('hyper',4),('additive',2),('mirror',4),('mirror_private',4),('flat',4),('independent',4)]:
            model=ma.build_decoder(method,rank,self.world,13);blob,digest=ma.serialize(model,method,rank)
            obj=torch.load(__import__('io').BytesIO(blob),map_location='cpu',weights_only=True)
            restored=ma.build_decoder(method,rank,self.world,13);restored.load_state_dict(obj['state_dict'])
            self.assertEqual(len(digest),64);self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values())))
            restored_blob,restored_digest=ma.serialize(restored,method,rank)
            self.assertEqual(blob,restored_blob);self.assertEqual(digest,restored_digest)

if __name__=='__main__':unittest.main()
