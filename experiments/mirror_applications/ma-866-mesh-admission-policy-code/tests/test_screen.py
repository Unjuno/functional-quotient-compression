import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class MeshPolicyTests(unittest.TestCase):
 def test_policy_address_equals_native_best_reliability_rule(self):
  payload=run.mirror_policy_payload()
  for q in range(run.DIMS):self.assertEqual(payload[6+q],run.best_agent(q))
 def test_wire_message_has_only_id_type_and_float(self):
  msg=run.MESSAGE.pack(7,3,1.25);self.assertEqual(len(msg),6);self.assertEqual(run.MESSAGE.unpack(msg),(7,3,1.25))
 def test_central_posterior_has_lower_theoretical_mse_than_single(self):
  central=1/(1+sum(1/(s*s) for s in run.SIGMAS));best=1/(1+1/(run.SIGMAS[0]**2));self.assertLess(central,best);self.assertLess(best,.35)
if __name__=='__main__':unittest.main()
