import sys,unittest
from pathlib import Path
import torch
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'source'))
import run_experiment as r

class PrivateFallbackTests(unittest.TestCase):
    def test_online_residual_selector_assigns_one_full_slot_to_outlier(self):
        torch.manual_seed(5)
        mem=torch.randn(4,1,8,8,8,dtype=torch.bfloat16)*.01
        mem[3]+=torch.arange(8,dtype=torch.bfloat16)[None,:,None,None]
        r.PRIVATE_THRESHOLD=.01
        rep,decoded=r.encode_memory('mirror_private_fallback',mem)
        self.assertEqual(int(rep['private_index'][0]),3)
        self.assertTrue(torch.equal(decoded[3],mem[3]))
        self.assertEqual(rep['private_memory'].numel(),mem[3].numel())
        self.assertEqual(sum(k=='private_memory' for k in rep),1)
    def test_threshold_can_leave_all_objects_shared(self):
        torch.manual_seed(6);mem=torch.randn(4,1,8,8,8,dtype=torch.bfloat16)
        r.PRIVATE_THRESHOLD=100.
        rep,decoded=r.encode_memory('mirror_private_fallback',mem)
        self.assertEqual(int(rep['private_index'][0]),-1)
        self.assertNotIn('private_memory',rep)
        replay=r.decode_memory('mirror_private_fallback',rep)
        torch.testing.assert_close(replay,decoded,rtol=0,atol=0)
    def test_serialized_index_uses_integer_bytes(self):
        import numpy as np
        parts=[];r.arr_record(parts,'memory/fallback_index',np.asarray([2],dtype='<i8'))
        out,manifest=r.serialize_parts({},parts)
        self.assertEqual(manifest[0]['dtype'],'<i8');self.assertEqual(manifest[0]['bytes'],8)
if __name__=='__main__':unittest.main()
