# MA-567 — Mixture-of-Depths + Mirror compute/function routing

PA111 allocates compute token-wise. This screen separates the token depth route from a role-dependent function choice. It is a synthetic CPU mechanism study, not a trained MoD Transformer or language-quality experiment.

## Final audited result (A6)

A6 replays the fixed seed IDs after audit corrections. It restores the original float64 shared/input precision, loads all serialized inference state, computes block 2 only for selected tokens except the dense upper, charges route/view/role/basis metadata, and compares against an independently computed high-precision teacher. A0–A5 files are retained, but A6 is authoritative. Reused seed IDs make A6 an audited replay rather than an independent fresh replication.

MoD-only mean NRMSE is 0.1346. Mirror, direct role-angle, and fused rank-2 controls are about 3.52e-9. Mirror and the direct-angle control are exact aliases in output, payload bytes, and MACs. **Disposition: FAIL for incremental Mirror benefit; the real trained-MoD application is NOT ESTABLISHED.**


Astra independently replayed the original teacher and all A6 serialized payloads: all 15 metric rows and hashes matched exactly. MACs are a matrix/view proxy; nonlinear operations are excluded. A0–A5 are retained for provenance but superseded, and any earlier speed or quality comparison from those variants is withdrawn.
