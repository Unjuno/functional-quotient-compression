# MA-1062 — DLRM shared embedding table with feature-field Views

**Result: scoped FAIL.** A shared H=512 table plus field-wise rotations used 50,595 serialized bytes, 73.0% below full-cardinality DLRM, but failed the preregistered quality and lookup-latency gates. The full table had higher AUC, and a per-coordinate diagonal matched the Mirror within 0.001 logloss. See RESULTS.md and VERIFICATION.json.

The experiment is a compact CPU DLRM screen for explicit MovieLens rating preference, not CTR or a production DLRM reproduction. MovieLens use terms and citation are in DATASET_TERMS.md. The raw archive is not committed. Audit labels and row IDs were never saved; serialized prediction logits and model payloads are local artifacts whose sizes and hashes are checked in.
