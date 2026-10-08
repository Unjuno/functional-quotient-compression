import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import fit

def test_discrete_and_vq_are_same_function_and_bytes():
    rows,meta=fit(49201); x={r['method']:r for r in rows}
    assert x['mirror_discrete']['primary_value']==x['vq_same_2bit']['primary_value']
    assert x['mirror_discrete']['serialized_bytes']==x['vq_same_2bit']['serialized_bytes']
    assert meta['mirror_vq_identical']

def test_payload_positive_and_valid_mass():
    rows,_=fit(49211)
    assert all(int(r['serialized_bytes'])>0 for r in rows)
    assert all(float(r['secondary_value'].split(';')[0])>=0 for r in rows)
