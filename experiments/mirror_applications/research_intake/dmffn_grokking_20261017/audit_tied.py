"""Post-primary shared-only baseline data and per-model independent NumPy metrics."""
from pathlib import Path
import hashlib,json,numpy as np,torch
from model import Config,MirrorSwiGLU,checkpoint_bytes
from train import split_data
from audit import numpy_forward,evaluate
ROOT=Path(__file__).resolve().parent

def main():
    path=ROOT/'results'/'tied_followup';rows=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines()]
    assert len(rows)==6 and len({(r['world'],r['seed'],r['method']) for r in rows})==6
    for r in rows:
        file=path/f"model_{r['world']}_{r['seed']}_tied.pt"
        raw=file.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['payload_sha256']
        state=torch.load(file,weights_only=True,map_location='cpu')
        model=MirrorSwiGLU(Config(),method='tied');model.load_state_dict(state['model'])
        assert checkpoint_bytes(model)==r['serialized_inference_bytes']
        train,test,_,_=split_data(31,r['world'],0.6)
        logits=numpy_forward(test[0],state['model'],'tied',state['cfg'])
        metr=evaluate(logits,test[1].numpy(),31)
        assert abs(metr['accuracy']-r['final_test_accuracy'])<1e-6
        assert abs(metr['ce']-r['final_test_ce'])<1e-4
    values={w:float(np.mean([r['final_test_accuracy'] for r in rows if r['world']==w])) for w in sorted({r['world'] for r in rows})}
    outcome={'checked':len(rows),'all_numpy_metrics_and_payloads_verified':True,
        'median_world_mean_test_accuracy':float(np.median(list(values.values()))),
        'world_mean_accuracy':values,'median_inference_bytes':int(np.median([r['serialized_inference_bytes'] for r in rows])),
        'posthoc_only':True}
    (path/'AUDIT_TIED.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print('TIED AUDIT',json.dumps(outcome,ensure_ascii=False))
if __name__=='__main__':main()
