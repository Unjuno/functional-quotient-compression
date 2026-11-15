#!/usr/bin/env python3
"""Hash upstream inputs, external replay caches, and serialized MA-1038 payloads."""
import hashlib, importlib.metadata, json, pathlib, subprocess

EXP=pathlib.Path(__file__).resolve().parents[1]
ROOT=pathlib.Path(__file__).resolve().parents[4]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def file_entry(path): return {"bytes":path.stat().st_size,"sha256":sha(path)}

model=pathlib.Path('/tmp/ma1038/colbertv2.0')
scifact=pathlib.Path('/tmp/ma1037-scifact')
nfroot=pathlib.Path('/tmp/ma1038/nfcorpus/nfcorpus')
cache=pathlib.Path('/tmp/ma1038/cache')
payload=pathlib.Path('/tmp/ma1038/payload')
inputs={
 "official_colbertv2_archive":{"url":"https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz","sha256":"d07df4b2a9fe1848fbed333127a1a5b4f2d96106f420948f0db4450b018cb119","bytes":405924985},
 "extracted_model_files":{p:file_entry(model/p) for p in ["pytorch_model.bin","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt"]},
 "colbert_reference_source":{"git_commit":subprocess.check_output(["git","-C","/tmp/ma1038/ColBERT","rev-parse","HEAD"],text=True).strip(),"repository":"https://github.com/stanford-futuredata/ColBERT.git"},
 "scifact_development_files":{p:file_entry(scifact/p) for p in ["corpus.parquet","queries.parquet","train.tsv"]},
 "nfcorpus_audit":{"archive":{"url":"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip","sha256":"efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b","bytes":pathlib.Path("/tmp/ma1038/nfcorpus.zip").stat().st_size},"files":{p.relative_to(nfroot).as_posix():file_entry(p) for p in sorted(nfroot.rglob('*')) if p.is_file()}},
}
cache_files={p.name:file_entry(p) for p in sorted(cache.iterdir()) if p.is_file()}
arms={}
for arm in ["audit_full_float","audit_residual_native","audit_mirror_role_view","audit_native_query_givens"]:
    folder=payload/arm
    arms[arm]={p.name:file_entry(p) for p in sorted(folder.iterdir()) if p.is_file()}
    assert sum(x['bytes'] for x in arms[arm].values())>0
summary=EXP/"runs"/"summary.json"
obj={
 "experiment":"MA-1038",
 "protocol_commit":"4ae21b523c310c4bdc6b60d7234b2e2f6b8725fd",
 "frozen_runner_commit":"bf7f7052d9ce67d83d7ff9e05bd58c814ca8a85d",
 "summary_sha256":sha(summary),
 "inputs":inputs,
 "external_vector_cache":{"path":"/tmp/ma1038/cache","files":cache_files},
 "actual_serialized_inference_payloads":{"external_path":"/tmp/ma1038/payload","arms":arms},
 "environment":{"python":"3.12","torch":importlib.metadata.version('torch'),"transformers":importlib.metadata.version('transformers'),"numpy":importlib.metadata.version('numpy'),"scikit_learn":importlib.metadata.version('scikit-learn'),"pyarrow":importlib.metadata.version('pyarrow'),"cpu_threads_used":4,"device":"CPU"},
 "retention":"Large model/index/vector arrays are intentionally kept outside git. This manifest records each serialized file's actual byte length and SHA256."
}
(EXP/"ARTIFACT_PROVENANCE.json").write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
print(f"wrote {EXP/'ARTIFACT_PROVENANCE.json'} ({sum(len(v) for v in arms.values())} payload files, {sum(x['bytes'] for a in arms.values() for x in a.values())} arm-bytes hashed)")
