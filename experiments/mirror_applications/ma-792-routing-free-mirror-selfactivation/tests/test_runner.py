import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma792',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_selfactivation_mirror_alias_and_heterogeneity():
    rows=m.run();by={(r['heterogeneity_sigma'],r['method']):r for r in rows}
    for sig in (0.0,.05):
        a=by[(sig,'mirror_selfactivation_views')];b=by[(sig,'native_lowrank_plus_selfactivation')]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['mode_routing_accuracy']==b['mode_routing_accuracy']
        assert a['output_nrmse']==b['output_nrmse']
    m0=by[(0.0,'mirror_selfactivation_views')];cent=by[(0.0,'central_top1_independent')]
    assert m0['mode_routing_accuracy']>=.95 and m0['output_nrmse']<=.02
    assert m0['serialized_bytes']>cent['serialized_bytes']
    assert by[(.05,'mirror_selfactivation_views')]['output_nrmse']>.02
