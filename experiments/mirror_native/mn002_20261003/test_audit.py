# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
import pytest
from audit import freeze,verify_manifest

def setup(tmp_path):
    protocol={'seeds':[201], 'variants':[{'name':'mirror_s4'}]}
    pp=tmp_path/'protocol.json'; pp.write_text(json.dumps(protocol))
    td=tmp_path/'train';td.mkdir()
    return pp,td

def test_incomplete_suite_cannot_lock(tmp_path):
    pp,td=setup(tmp_path)
    with pytest.raises(ValueError): freeze(td,pp,tmp_path/'lock.json')

def test_locked_artifact_tampering_and_protocol_change_rejected(tmp_path):
    pp,td=setup(tmp_path); cp=td/'mirror_s4_seed201.pt';cp.write_bytes(b'byte fixture')
    (td/'mirror_s4_seed201.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(), 'protocol_sha256':hashlib.sha256(pp.read_bytes()).hexdigest()}))
    lock=tmp_path/'lock.json';freeze(td,pp,lock)
    assert len(verify_manifest(td,pp,lock))==1
    with pytest.raises(FileExistsError): freeze(td,pp,lock)
    cp.write_bytes(b'changed fixture')
    with pytest.raises(ValueError): verify_manifest(td,pp,lock)
    cp.write_bytes(b'byte fixture');pp.write_text(pp.read_text()+'\n')
    with pytest.raises(ValueError): verify_manifest(td,pp,lock)
