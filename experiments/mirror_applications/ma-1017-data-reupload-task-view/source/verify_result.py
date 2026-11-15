"""Exact deterministic replay for MA-1017."""
import hashlib,json,math,pathlib,subprocess,sys
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs';lock=json.loads((R/'development_lock.json').read_text())
subprocess.run([sys.executable,str(E/'source'/'reupload_screen.py')],check=True,stdout=subprocess.DEVNULL)
now=json.loads((R/'development.json').read_text());assert now['code_sha256']==lock['code_sha256']
assert now['fresh_opened'] is False and now['distinct_task_functions']==4
assert now['mirror_native_scalar_payload_identical'] is True
for a,b in zip(now['results'],lock['results']):
 for k in ('method','teacher_agreement','shot_accuracy','serialized_payload_bytes','qubits','logical_gates','two_qubit_gates','depth','shots_per_input','total_shots'):assert a[k]==b[k],(k,a[k],b[k])
 assert math.isfinite(a['wall_s'])
lock['replay_sha256']=hashlib.sha256((R/'development.json').read_bytes()).hexdigest();(R/'development.json').write_text(json.dumps(lock,indent=2)+'\n')
print(json.dumps({'statevector_and_shot_replay_exact':True,'payload_bytes_exact':True,'mirror_native_control_alias':True,'fresh_opened':False}))
