import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma539", path)
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


def test_support_recovers_all_rule_addresses():
    rules, ids, ins, outs, _, _ = ma.make_world(60101)
    codes, recovered = ma.extract_fv(rules, ids, ins, outs)
    assert np.all(recovered == np.arange(ma.NRULE))
    assert np.all(codes.sum(axis=1) == 1)


def test_packet_is_ordered_repeated_function_application():
    rule = np.roll(np.arange(ma.NSYM), 1)
    assert ma.packet(rule, 0) == (15, 14, 13, 12)
