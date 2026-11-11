import unittest
from pathlib import Path
class AuditTests(unittest.TestCase):
 def test_exploratory_results_are_marked_invalid(self):
  t=(Path(__file__).parents[1]/'RESULTS.md').read_text()
  self.assertIn('NOT ESTABLISHED',t)
  self.assertIn('oracle leakage',t)
if __name__=='__main__': unittest.main()
