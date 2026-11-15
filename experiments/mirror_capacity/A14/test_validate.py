import sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import validate

def test_dependency_sha_guard_rejects_mutated_source(tmp_path):
    f=tmp_path/'a13.py';f.write_text('untampered')
    import hashlib
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    assert validate.verify_dependency({'a13.py':digest},tmp_path)==1
    f.write_text('tampered')
    with pytest.raises(RuntimeError,match='dependency'):
        validate.verify_dependency({'a13.py':digest},tmp_path)
