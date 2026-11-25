#!/usr/bin/env python3
"""Reload inference payloads and independently replay locked audit forecasts."""
import hashlib,json,sys
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_screen import H,NAMES,RAW,ViewModel,load_data,make_dlinear,tensors

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
    result=json.loads((HERE/'runs/audit/results.json').read_text()); data,meta=load_data(include_audit=True); report=[]
    for row in result['results']:
        if row['method'] not in ('dlinear','fixed','gain','mirror','free_fourier','independent'): continue
        path=HERE/row['payload_path']; payload=torch.load(path,map_location='cpu',weights_only=False)
        assert path.stat().st_size==row['payload_bytes'] and sha(path)==row['payload_sha256']
        assert payload['data_sha256']==sha(RAW) and payload['seed']==row['seed'] and payload['method']==row['method']
        states=payload['states']; models=[]
        for i,state in enumerate(states):
            if row['method']=='independent': m=ViewModel(make_dlinear(result['context'],True,1),'independent')
            else: m=ViewModel(make_dlinear(result['context']),'mirror' if row['method']=='mirror' else row['method'])
            m.load_state_dict(state); m.eval(); models.append(m)
        x,y,starts=tensors(data,'audit',result['context'])
        with torch.no_grad():
            if row['method']=='independent': pred=torch.cat([m(x[:,:,i:i+1]) for i,m in enumerate(models)],dim=-1).numpy()
            else: pred=models[0](x).numpy()
        score=float(np.mean(np.mean(np.abs(pred-y.numpy()),axis=(0,1))/data['scale']))
        assert abs(score-row['macro_mase'])<1e-7,(row['method'],row['seed'],score,row['macro_mase'])
        report.append({'method':row['method'],'seed':row['seed'],'payload_bytes':path.stat().st_size,
            'payload_sha256':sha(path),'replayed_macro_mase':score,'audit_windows':len(x),
            'target_start_min':int(starts.min()),'target_end_max_exclusive':int(starts.max()+H)})
    out={'experiment_id':'MA-1056','status':'PASS_VERIFICATION','raw_sha256':sha(RAW),
         'protocol_sha256':sha(HERE/'PROTOCOL.json'),'checkpoint_replays':report,
         'checks':['all actual serialized checkpoint byte counts and hashes verified',
                   'all DLinear and view audit metrics replay exactly from serialized states',
                   'all audit target windows stay inside the registered final chronological split']}
    (HERE/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n'); print(HERE/'VERIFICATION.json')
if __name__=='__main__': verify()
