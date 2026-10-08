import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import FUNCS,TOPOLOGIES,execute,function_bank,make_inputs,payload_for,execute_payload

class ProgramTests(unittest.TestCase):
    def test_all_compositions_exact_including_heldout(self):
        bank=function_bank(82401);x=make_inputs(82401,'development',64)
        for method in ('mirror_factors','native_interpreter','hard_shared_indices','independent_programs'):
            p=payload_for(method,82401,bank)
            if method=='native_interpreter':self.assertEqual(len(p['representation']['program_signatures']),3)
            for tid,top in enumerate(TOPOLOGIES):
                for fid,_ in enumerate(FUNCS):
                    expected=execute(x,bank,tid,fid);actual=execute_payload(x,p,top,fid)
                    self.assertTrue(torch.equal(expected,actual))

    def test_factor_axes_select_expected_multiplicity(self):
        p=payload_for('mirror_factors',82402,function_bank(82402))
        self.assertEqual(len(p['representation']['topology_codes']),2)
        self.assertEqual(len(p['representation']['function_codes']),2)
        self.assertNotIn('program_signatures',p['representation'])

if __name__=='__main__':unittest.main()
