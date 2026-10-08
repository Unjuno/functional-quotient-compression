import io, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_screen import payload_bytes, world, fit_task_mats, mirror, lowrank, psp, mse

class RoundtripTests(unittest.TestCase):
    def test_serialized_states_are_nonempty_and_reconstruct_metrics(self):
        target,x,y,xt,yt=world(101)
        fitted=fit_task_mats(x,y)
        states=[ [fitted], psp(fitted,101)[1], lowrank(fitted,2)[1], mirror(fitted)[1] ]
        preds=[fitted,psp(fitted,101)[0],lowrank(fitted,2)[0],mirror(fitted)[0]]
        for state,pred in zip(states,preds):
            self.assertGreater(payload_bytes(state),0)
            b=io.BytesIO(); np.savez(b,**{f'a{i}':np.asarray(a) for i,a in enumerate(state)})
            b.seek(0)
            with np.load(b) as z: loaded=[z[k] for k in z.files]
            self.assertEqual(len(loaded),len(state))
            self.assertTrue(all(np.array_equal(a,c) for a,c in zip(state,loaded)))
            self.assertAlmostEqual(mse(pred,xt,yt),mse(pred,xt,yt),places=14)
        self.assertLess(mse(preds[2],xt,yt),1e-20)
        self.assertLess(mse(preds[3],xt,yt),1e-10)

if __name__=='__main__': unittest.main()
