#!/usr/bin/env python3
"""Verify MA-598 freeze hashes, deterministic dev replays, bytes, and gates."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    assert sha(ROOT/'PROTOCOL.json')==freeze['protocol_sha256']
    assert sha(ROOT/'source/run_hash_experts.py')==freeze['source_sha256']
    assert sha(ROOT/'tests/test_hash_experts.py')==freeze['tests_sha256']
    replays={}
    for seed in (59801,59802):
        a=ROOT/f'runs/dev_{seed}';b=ROOT/f'runs/replay_{seed}'
        ma=json.loads((a/'metrics.json').read_text())
        subprocess.run([sys.executable,str(ROOT/'source/run_hash_experts.py'),'--seed',str(seed),'--split','dev','--out',str(b)],check=True,stdout=subprocess.DEVNULL)
        mb=json.loads((b/'metrics.json').read_text())
        assert ma['dataset_tensor_sha256']==mb['dataset_tensor_sha256']
        stable=['method','serialized_bytes','payload_sha256','train_examples','test_examples','optimizer_updates','examples_seen','initial_and_final_train_loss','base_macs_per_example','extra_view_ops_per_example','hash_expansion_lookups_per_model_load','router_selected_accuracy','router_selected_cross_entropy','oracle_expert_accuracy','oracle_expert_cross_entropy','router_accuracy','expert_route_fractions','per_expert_oracle_accuracy','pairwise_raw_logit_cosine_mean','pairwise_raw_argmax_disagreement_mean','pairwise_first_layer_weight_cosine_mean','pairwise_hash_map_overlap_mean','per_expert_collision','expert_ablation_accuracy','active_experts_per_example','route_confusion']
        for x,y in zip(ma['methods'],mb['methods']):
            assert all(x[k]==y[k] for k in stable)
        for x,y in zip(ma['methods'],mb['methods']):
            p=a/(x['method']+'.npz');q=b/(x['method']+'.npz')
            assert p.stat().st_size==x['serialized_bytes']
            assert sha(p)==x['payload_sha256']==sha(q)
        d={x['method']:x for x in ma['methods']}
        assert d['mirror_givens']['router_selected_accuracy'] < d['tied_shared_hash']['router_selected_accuracy']+.01
        assert d['mirror_givens']['router_selected_accuracy'] < d['salted_shared_hash']['router_selected_accuracy']+.01
        assert d['mirror_givens']['serialized_bytes'] <= 1.05*d['salted_shared_hash']['serialized_bytes']
        assert max(d['diagonal_gate']['router_selected_accuracy'],d['rank1_residual']['router_selected_accuracy']) >= d['mirror_givens']['router_selected_accuracy']-.01
        replays[str(seed)]={'all_seven_payloads_byte_exact':True,'mirror_accuracy':d['mirror_givens']['router_selected_accuracy'],'salted_accuracy':d['salted_shared_hash']['router_selected_accuracy'],'mirror_bytes':d['mirror_givens']['serialized_bytes'],'salted_bytes':d['salted_shared_hash']['serialized_bytes']}
    assert not any((ROOT/f'runs/fresh_{s}').exists() for s in (59811,59812,59813))
    out={'experiment_id':'MA-598','status':'FAIL','protocol_sha256':freeze['protocol_sha256'],'source_sha256':freeze['source_sha256'],'development_seeds':[59801,59802],'fresh_seeds_accessed':[],'fresh_artifacts_present':False,'serialization_roundtrip_checked':True,'metric_replay_checked':True,'payloads_replayed':14,'replay_results':replays,'router_selected_cross_entropy':'NOT USABLE: all non-expert classes are masked to -1e9 before CE; identical huge values across all methods','tests':{'command':'PYTHONDONTWRITEBYTECODE=1 python -m pytest -q experiments/mirror_applications/ma-598-hash-addressed-logical-experts/tests','expected_passed':3},'gate_decision':'FAIL: Mirror misses +1pp versus tied/salted controls in both dev seeds. It passes the 1.05x salted byte cap, but byte-near diagonal/rank-one controls and native salt match its quality.'}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
