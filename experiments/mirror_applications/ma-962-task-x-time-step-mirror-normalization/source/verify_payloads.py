"""After a run, verify actual serialized bytes/hashes and strict state reloadability."""
import hashlib, json
from pathlib import Path
import torch
from model import SharedSNN, IndependentTaskSNN
ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/'source'
summary=json.loads((SOURCE/'development_summary.json').read_text())
checked=[]
for seed,conditions in summary['per_seed'].items():
    for condition,result in conditions.items():
        rec=result['payload']; path=ROOT/rec['path']; raw=path.read_bytes()
        assert len(raw)==rec['bytes']
        assert hashlib.sha256(raw).hexdigest()==rec['sha256']
        obj=torch.load(path,map_location='cpu',weights_only=False)
        model=IndependentTaskSNN() if condition=='independent_snn_per_task' else SharedSNN(condition)
        model.load_state_dict(obj['state_dict'],strict=True)
        checked.append({'seed':int(seed),'condition':condition,'bytes':len(raw),'sha256':rec['sha256'],'strict_reload':'PASS'})
print(json.dumps({'payloads_checked':len(checked),'status':'PASS','payloads':checked},indent=2))
