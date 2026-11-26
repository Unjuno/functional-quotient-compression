#!/usr/bin/env python3
"""Validate MA-1062 fresh metrics and reproduce predictions from serialized payloads."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma1062',ROOT/'source/train.py'); ma=importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    results=json.loads((ROOT/'runs/audit/results.json').read_text())
    freeze=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    assert sha(ROOT/'PROTOCOL.json')==freeze['protocol_sha256']
    assert sha(ROOT/'source/train.py')==freeze['runner_sha256']
    assert sha(Path('/tmp/ma1062-ml-100k.zip'))==freeze['dataset_archive_sha256']
    audit=ma.read_audit_after_freeze(); base=ma.read_base()
    assert not ({(r[0],r[1]) for r in base}&{(r[0],r[1]) for r in audit})
    maximum_metric_difference=0.; maximum_prediction_difference=0.; payloads={}; replay_count=0
    for method in ma.METHODS:
        payloads[method]={}
        for seed in ma.FRESH_SEEDS:
            sid=str(seed); checkpoint=ROOT/'runs/audit'/f'{method}_seed{seed}.pt'; meta=ROOT/'runs/audit'/f'{method}_seed{seed}.json'; predfile=ROOT/'runs/audit'/f'{method}_seed{seed}.npz'
            model,obj=ma.load_model_from_payload(checkpoint); recomputed,pred=ma.evaluate(model,audit)
            saved=np.load(predfile)
            assert set(saved.files)=={'logits'}
            expected=saved['logits']; pdiff=float(np.max(np.abs(pred-expected)))
            maximum_prediction_difference=max(maximum_prediction_difference,pdiff)
            assert np.allclose(pred,expected,atol=5e-7,rtol=0),(method,seed,pdiff)
            expected_metrics=results['methods'][method][sid]
            mdiff=max(abs(recomputed[k]-expected_metrics[k]) for k in ('logloss','auc'))
            maximum_metric_difference=max(maximum_metric_difference,mdiff)
            assert mdiff<1e-9,(method,seed,mdiff)
            meta_bytes=meta.stat().st_size; model_bytes=checkpoint.stat().st_size; total=model_bytes+meta_bytes
            registered=results['payloads'][method][sid]
            assert total==registered['complete_payload_bytes']
            assert sha(checkpoint)==expected_metrics['payload_sha256']
            assert sha(predfile)==expected_metrics['prediction_sha256']
            payloads[method][sid]={'model_bytes':model_bytes,'metadata_bytes':meta_bytes,'complete_payload_bytes':total,'checkpoint_sha256':sha(checkpoint),'prediction_sha256':sha(predfile)}
            replay_count+=1
    out={'status':'PASS','payload_predictions_replayed':replay_count,'saved_metric_rows_replayed':replay_count,'max_prediction_difference':maximum_prediction_difference,'max_metric_difference':maximum_metric_difference,'audit_train_pairs_disjoint':True,'audit_labels_or_ids_persisted':False,'payload_bytes':payloads}
    (ROOT/'runs/audit/verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'payload_predictions_replayed':replay_count,'max_prediction_difference':maximum_prediction_difference,'max_metric_difference':maximum_metric_difference,'audit_labels_or_ids_persisted':False},indent=2))
if __name__=='__main__': main()
