from fqc.decoder_dag import Atom, compile_decoder_dag

def test_d59_synthetic_rate_ledger_reproduces_2032_paid_bits():
    atoms=[
        Atom('dict_atoms','PAID',864),
        Atom('coeff_payload','PAID',432,('dict_atoms',)),
        Atom('sparse_residuals','PAID',672,('dict_atoms','coeff_payload')),
        Atom('codec_header','PAID',64),
        Atom('rope_profile_rule','EXTERNAL_FIXED',0),
        Atom('reconstructed_ops','DERIVED',0,('dict_atoms','coeff_payload','sparse_residuals','rope_profile_rule','codec_header')),
    ]
    r=compile_decoder_dag(atoms,['reconstructed_ops'])
    assert r.valid
    assert r.total_paid_bits==2032
    assert r.reachable_atoms==frozenset(a.atom_id for a in atoms)

def test_shared_dependency_is_counted_once():
    atoms=[Atom('Q','PAID',20),Atom('D_A','PAID',12,('Q',)),Atom('D_B','PAID',10,('Q',))]
    r=compile_decoder_dag(atoms,['D_A','D_B'])
    assert r.valid and r.total_paid_bits==42

def test_cross_class_cycle_is_invalid():
    r=compile_decoder_dag([Atom('root','PAID',8,('derived',)),Atom('derived','DERIVED',0,('root',))],['derived'])
    assert not r.valid
    assert any('cycle' in e for e in r.errors)
