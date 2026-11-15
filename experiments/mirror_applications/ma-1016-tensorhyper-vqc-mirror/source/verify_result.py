"""Deterministic replay for MA-1016; does not open fresh seeds."""
import hashlib,json,math,pathlib,subprocess,sys
E=pathlib.Path(__file__).resolve().parents[1]; R=E/'runs'; lock=json.loads((R/'development_lock.json').read_text())
subprocess.run([sys.executable,str(E/'source'/'tt_vqc_screen.py')],check=True,stdout=subprocess.DEVNULL)
now=json.loads((R/'development.json').read_text()); assert now['code_sha256']==lock['code_sha256']
assert now['fresh_opened'] is False and now['teacher_pairwise_label_disagreement']==lock['teacher_pairwise_label_disagreement']
assert now['native_mirror_payload_identical_to_scalar_phase_control'] is True
for a,b in zip(now['results'],lock['results']):
    for k in ('method','teacher_agreement','shot_accuracy','serialized_payload_bytes','qubits','logical_gates','two_qubit_gates','depth','shots_per_pattern','total_shots','max_teacher_expectation_delta'):
        assert a[k]==b[k],(a['method'],k,a[k],b[k])
    assert math.isfinite(a['wall_s'])
assert not list(R.glob('fresh*'))
lock['replay_sha256']=hashlib.sha256((R/'development.json').read_bytes()).hexdigest()
(R/'development.json').write_text(json.dumps(lock,indent=2)+'\n')
print(json.dumps({'statevector_and_shot_replay_exact':True,'payload_bytes_exact':True,'mirror_equals_native_phase_control':True,'fresh_opened':False}))
