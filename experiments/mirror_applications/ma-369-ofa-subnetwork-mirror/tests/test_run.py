import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/"source"/"run.py"
spec=importlib.util.spec_from_file_location("ma369_run",P); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class TestMA369:
    def test_mirror_and_direct_payloads_are_exact_functional_control(self):
        rows,payloads=mod.fit(36901,4)
        assert payloads["direct_coeff"] == payloads["mirror_gate"]
        assert len(rows)==4

    def test_all_methods_serialize_nonempty_payloads(self):
        _,payloads=mod.fit(36902,8)
        assert all(len(v)>0 for v in payloads.values())
