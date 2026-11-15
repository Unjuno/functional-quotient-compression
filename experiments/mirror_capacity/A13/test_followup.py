from followup import enrich_parents


def test_parent_record_joins_model_identity():
    parents=[{'path':'parents/tent_w1200_s0_mirror_meta2400.a11w','sha256':'abc','stored_bytes':1004}]
    rows=[{'family':'tent','world':1200,'seed':0,'method':'mirror','parent_sha256':'abc'}]
    res=enrich_parents(parents,rows)
    assert res[0]['family']=='tent'
    assert res[0]['world']==1200
    assert res[0]['seed']==0
    assert res[0]['method']=='mirror'


def test_numeric_evidence_fields_are_json_safe():
    import numpy as np,json
    from followup import json_native
    payload={'mean':np.float32(0.25),'count':np.int64(2),'nested':[np.float64(0.1)]}
    clean=json_native(payload)
    assert clean['count']==2
    assert abs(clean['mean']-.25)<1e-12
    json.dumps(clean)
