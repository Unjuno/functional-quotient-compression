# MA-1035 — QINCo residual-stage Views

Status: SCREENING. This experiment tests whether a single residual-stage generator plus a small stage coordinate can approximate independent QINCo stages on vector reconstruction and nearest-neighbor retrieval.

The native reference uses the official QINCo v1 model structure at pinned upstream commit `5a324954d5c9b3700d4407d6cc24c3db6e52890e`. A small local quantization helper replaces Faiss only for nearest codeword assignment; all scoring uses brute-force L2. The Mirror shares one `QINCoStep` and applies a per-stage Givens rotation to the decoded-prefix conditioning vector. Controls include ordinary residual quantization, native independent QINCo stages, hard tying, and a shared stage generator with a rank-1 additive conditioning control.

This is a CPU screen on newly generated clustered 32-D vector worlds with small codebooks. It is not a published QINCo checkpoint reproduction or a SIFT/BigANN result. Full serialized inference bytes include self-describing model tensors and packed database codes; decoded database vectors are regenerated at load time from paid state and their resident RAM and decode latency are reported.

See `PROTOCOL.json`, `RESULTS.md`, `VERIFICATION.json`, and `ARTIFACT_PROVENANCE.json` for the frozen protocol and evidence.
