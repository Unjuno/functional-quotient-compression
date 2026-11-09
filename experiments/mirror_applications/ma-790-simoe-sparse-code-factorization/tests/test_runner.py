import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma790',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_sparse_interpolation_beats_factorized_view_quality_bytes():
    rows=m.run();by={(r['method'],r['rank']):r for r in rows}
    for rank in (1,2,4,8,16):
        a=by[('mirror_factorized_code',rank)];b=by[('native_pca_coefficient_code',rank)]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['heldout_output_nrmse']==b['heldout_output_nrmse']
    raw=by[('simoe_raw_sparse_coefficients',2)]
    passing=[r for (method,_),r in by.items() if method=='mirror_factorized_code' and r['heldout_output_nrmse']<=.02 and r['serialized_bytes']<.75*raw['serialized_bytes']]
    assert not passing
    assert by[('mirror_factorized_code',1)]['heldout_output_nrmse']>.9
    assert by[('mirror_factorized_code',16)]['serialized_bytes']>raw['serialized_bytes']
