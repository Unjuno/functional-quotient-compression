import csv
import hashlib
import io
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class TestMA517Artifacts(unittest.TestCase):
    def test_rows_payloads_and_fresh_split(self):
        with (ROOT / 'RESULTS_CORE.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 54)
        self.assertEqual({int(r['world']) for r in rows}, {51710, 51711, 51712})
        self.assertEqual({int(r['seed']) for r in rows}, {0, 1, 2})
        self.assertEqual({r['method'] for r in rows}, {'query_only', 'direct_icl', 'raw_sum', 'factorized', 'f1_only', 'f2_only'})
        for row in rows:
            if not row['path']:
                self.assertEqual(row['hash'], '')
                continue
            blob = (ROOT / row['path'].split('ma-517-function-vector-composition/', 1)[1]).read_bytes()
            self.assertEqual(len(blob), int(row['payload_bytes']))
            self.assertEqual(hashlib.sha256(blob).hexdigest(), row['hash'])
            obj = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
            self.assertEqual(obj['method'], row['method'])
            self.assertEqual(len(blob), 99945 if row['method'] in ('raw_sum', 'factorized') else 50793)

    def test_fresh_task_quality_replay_gate(self):
        with (ROOT / 'RESULTS_CORE.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        for method in ('query_only', 'direct_icl', 'raw_sum', 'factorized', 'f1_only', 'f2_only'):
            selected = [r for r in rows if r['method'] == method]
            self.assertEqual(len(selected), 9)
            self.assertTrue(all(float(r['accuracy']) == 0.0 for r in selected))
        self.assertTrue(all(float(r['mean_target_nll']) > 0 for r in rows))


if __name__ == '__main__':
    unittest.main()
