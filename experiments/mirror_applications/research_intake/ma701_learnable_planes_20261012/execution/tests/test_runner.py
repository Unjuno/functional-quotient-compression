from source.runner import configs

def test_calibration_budget():
    a=configs('learned');assert len(a)==8
    assert {c['sigma'] for c in a}=={.03,.12}
    assert {c['warmup'] for c in a}=={0,100}
    for method in ['fixed','native4','film','householder','lora2']:
        assert len(configs(method))==8

def test_no_matching_or_private_in_primary_config():
    for c in configs('learned'):
        assert set(c)=={'lr','axis_lr','warmup','sigma','steps'}
        assert c['steps']==800 and c['sigma']>0
