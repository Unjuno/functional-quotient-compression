import sys,unittest
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'source'))
from run_interactive import click_history_blob,decode_polar,point_pair
class PromptCodecTests(unittest.TestCase):
    def test_polar_view_reconstructs_shared_pair_geometry(self):
        r=np.asarray([9.,12.],dtype=np.float16);theta=np.asarray([0.,np.pi/2],dtype=np.float16)
        c=np.asarray([[40.,50.],[80.,90.]],dtype=np.float16)
        out=decode_polar(c.astype(np.float32),theta.astype(np.float32),r.astype(np.float32))
        np.testing.assert_allclose(out[0,0],[49,50],atol=.02);np.testing.assert_allclose(out[0,1],[31,50],atol=.02)
        np.testing.assert_allclose(out[1,0],[80,102],atol=.02);np.testing.assert_allclose(out[1,1],[80,78],atol=.02)
    def test_polar_history_is_smaller_than_float32_raw_clicks(self):
        events=[]
        for f in range(1,6):
            pairs=np.zeros((4,2,2),dtype=np.float32);pairs[:,0,0]=30+f;pairs[:,0,1]=40;pairs[:,1,0]=20+f;pairs[:,1,1]=40
            events.append({'frame':f,'pairs':pairs.tolist()})
        r=np.full(4,5,dtype=np.float16)
        direct=click_history_blob('direct_clicks',events)
        mirror=click_history_blob('mirror_polar',events,r)
        grid=click_history_blob('grid8_clicks',events)
        self.assertLess(len(mirror),len(direct));self.assertGreater(len(mirror),len(grid))
    def test_current_frame_clicks_are_selected_from_error_pixels(self):
        gt=np.zeros((32,32),bool);gt[8:20,8:20]=True
        pred=gt.copy();pred[10:12,10:12]=False;pred[22:24,22:24]=True
        pair=point_pair(pred,gt)
        self.assertEqual(pair.shape,(2,2));self.assertTrue(gt[round(pair[0,1]),round(pair[0,0])]);self.assertFalse(gt[round(pair[1,1]),round(pair[1,0])])
if __name__=='__main__':unittest.main()
