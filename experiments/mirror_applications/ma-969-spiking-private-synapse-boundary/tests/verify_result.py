"""Preservation and consistency checks for the invalid MA-969 screen."""
import csv, json
from pathlib import Path

R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text())
rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()))
fresh=json.loads((R/'source/FRESH.json').read_text())
dev=json.loads((R/'source/DEVELOPMENT.json').read_text())
assert p['protocol_frozen'] is True and p['experiment_id']=='MA-969'
assert {int(r['seed']) for r in rows}=={96911,96913,96917}
assert len(rows)==30 and len(fresh)==3 and len(dev)==1
assert {r['condition'] for r in rows}=={'zero','direct','mirror','sparse','independent'}
assert all(int(r['joint_payload_bytes'])>0 for r in rows)
assert all(r['evaluation_split']=='fresh' for r in fresh)
assert (R/'RESULTS.md').exists()
# This is a transparent defect check, not an attempted correction of the recorded run.
source=(R/'source/run_ma969.py').read_text()
assert 'reshape(-1).scatter(0,self.mask,self.delta)' in source
assert 'torch.sigmoid' not in source
print('MA-969 artifact integrity passed: 30 preserved rows, 3 fresh optimizer seeds, 1 development seed; invalid implementation defects disclosed.')
