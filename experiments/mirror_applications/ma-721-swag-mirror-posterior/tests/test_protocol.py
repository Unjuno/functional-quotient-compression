import json,unittest,importlib.util,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("ma721_model",ROOT/"source"/"model.py");model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
data_splits,DigitsMLP,flatten,load_flat,posterior_logits=model.data_splits,model.DigitsMLP,model.flatten,model.load_flat,model.posterior_logits
class MA721Tests(unittest.TestCase):
 def test_random_draw_and_manifest_are_fixed(self):
  d=json.loads((ROOT/'source/random_draw.json').read_text());p=json.loads((ROOT/'PROTOCOL.json').read_text())
  self.assertEqual(d['selected_id'],'MA-721');self.assertEqual(p['experiment_id'],d['selected_id']);self.assertEqual(d['pool_size'],528)
 def test_locked_split_counts(self):
  p=json.loads((ROOT/'source/dataset_manifest.json').read_text());xtr,ytr,xd,yd,xa,ya=data_splits()
  self.assertEqual((len(ytr),len(yd),len(ya)),(p['counts']['train'],p['counts']['development'],p['counts']['fresh_locked']))
  self.assertFalse(p['fresh_labels_or_metrics_accessed'])
 def test_flat_checkpoint_roundtrip(self):
  torch.manual_seed(3);m=DigitsMLP();x=torch.rand(5,64);want=m(x);got=load_flat(DigitsMLP(),flatten(m))(x)
  self.assertTrue(torch.equal(want,got))
 def test_fresh_gate_is_preregistered(self):
  p=json.loads((ROOT/'PROTOCOL.json').read_text());self.assertTrue(p['fresh']['locked_before_access'])
  self.assertIn('only if',p['gates']['fresh_opening'].lower())
if __name__=='__main__':unittest.main()
