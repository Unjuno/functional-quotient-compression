import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_fisher_mirror_matches_native_and_accounts_bytes():
    rows=run();assert len(rows)==5;d={r['method']:r for r in rows}
    m=d['mirror_fisher_kmeans'];n=d['native_fisher_kmeans'];ind=d['independent_full_updates']
    assert m['payload_sha256']==n['payload_sha256']
    assert m['heldout_fisher_nrmse']==n['heldout_fisher_nrmse']
    assert m['serialized_bytes']<ind['serialized_bytes']
    assert m['cluster_retrieval_accuracy']>=d['euclidean_kmeans']['cluster_retrieval_accuracy']
