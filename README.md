# Functional Quotient Compression (FQC)

Research on task-relevant equivalence, shared representations, private corrections,
and codec optimization with actual serialized-byte accounting. Formerly Vector Mirror.

## Current evidence boundary — through T282 (2026-09-11)

**Reproducible engineering pilot, not a demonstrated high-quality 64x codec.**

- A 28M-checkpoint whole-weight 64x binary was built and independently decoded; its quality gate **failed**.
- The latest frozen scalar baseline plus standard XZ occupies 25,507,600 bytes (3.925 bits per unique paid scalar; 4.076x versus 16-bit weights). It executes after dense FP32 decoding, not low-bit arithmetic.
- Its comparison with the same-compressor control used 44 public-derived, manually transcribed prompt prefixes, not official TinyStories validation. Do not promote this to a general quality or SOTA claim.
- Functional sharing has limited historical attention-only evidence. Its **whole-model advantage over strong non-sharing baselines remains unresolved**.
- Official installed-Transformers parity, independent external validation, MPS/CUDA measurements and independent training replicas remain unverified.

Read [current state](docs/RESEARCH_STATE.md), [evidence ledger](claims/T248_T282_STATUS.json),
and the [local handoff](docs/handoff/LOCAL_RUNBOOK.md).

## Clone-to-local starting point

The curated `experiments/t282/` lane includes the unchanged measured codec/decoder/quantizer cores,
149 artifact-free unit tests, eight frozen calibration probes, three artifact locks, and small result witnesses.
Model checkpoints and large compressed weights are **not committed**. Complete historical ZIPs are indexed
by SHA256 in `provenance/t282/SOURCE_ARCHIVES.json`, not dumped into Git.

Use an isolated Python environment; observed versions and local dependency guidance are in the runbook.
Run from the repository root:

```bash
python scripts/fqc_preflight.py --output runs/preflight-001.json
python -m pytest -q experiments/t282/tests
python scripts/fqc_rebuild_frozen.py --models-root ./models --output runs/rebuild-001
```

The third command requires the exact trusted `models/28M/` checkpoint and tokenizer assets.
It regenerates three frozen artifacts on CPU and checks their bytes and decoded tensors.
It is **not** a new quality evaluation or an MPS implementation.
Never overwrite a historical result or replace an expected hash to manufacture a pass.

## Agent handoff

For OpenCode, start in the repository root and use [the Japanese task prompt](docs/handoff/OPENCODE_PROMPT_JA.md).
`AGENTS.md` preserves evidence and operational boundaries; `opencode.json` uses conservative permissions.
The next task is CPU/official-runtime/data validation, then an explicitly tested MPS evaluation path,
then a matched-byte sharing ablation. Do not start with a large 64x sweep or a public announcement.

## Existing research organization

`claims/`, `docs/`, `roadmap/`, `provenance/`, `src/`, `tests/`, and older `experiments/`
retain the prior canonical research. The new lane does not silently replace their APIs.
The pre-T282 README and research state are retained under `docs/handoff/` for provenance.

## License and scope

Repository license: Apache-2.0. Checkpoint, tokenizer and external data licenses remain their own.
The parser is a research implementation, not a general security-audited untrusted-file service.
