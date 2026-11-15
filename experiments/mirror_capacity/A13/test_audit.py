from pathlib import Path
import json
import pytest
from audit import verify_frozen


def test_audit_rejects_corrupt_model(tmp_path):
    model=tmp_path/'m.a11w';model.write_bytes(b'hello')
    rec={'path':'m.a11w','sha256':'0'*64,'stored_bytes':5}
    with pytest.raises(RuntimeError,match='hash'):
        verify_frozen(tmp_path,[rec])


def test_audit_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError,match='missing'):
        verify_frozen(tmp_path,[{'path':'missing.a11w','sha256':'0'*64,'stored_bytes':12}])
