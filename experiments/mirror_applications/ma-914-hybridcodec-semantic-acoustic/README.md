# MA-914 — HybridCodec semantic × acoustic Mirror state

Status: UNTESTED (official implementation smoke tests pass; registered Mirror comparison not run)
Evidence lane: STORAGE / QUALITY / RUNTIME
Base commit: `research/mirror-application-worker-ready-20261007`

## H — falsifiable hypothesis

Sharing HybridCodec stream transforms with a small semantic-role × acoustic-role view can preserve semantic specialization and reconstruction quality while reducing actual deployed model bytes or improving the bitrate/quality/runtime frontier over native stream-specific transforms and byte-near ordinary controls.

## T — performed

- Inspected PA259, arXiv:2606.06743, the official `hybridcodec/hybridcodec` source at `16b0158932f5d3f6735fa45595684d846f83bfbc`, and the published HF model files.
- Built an isolated CPU environment with PyTorch/torchaudio 2.7.1+cpu, installed the official package, and ran its upstream tests: 5 passed.
- The registered Mirror/native codec comparison, held-out speech quality evaluation, bitrate and deployed-byte measurement were not run.

## D — UNTESTED

The upstream tests cover synthetic module shapes and checkpoint formats only. No task-quality, stream-sharing, Mirror or storage claim follows from them.

## C — strongest counter-hypothesis

Native semantic/acoustic specialization may already be the efficient representation; a shared view may add compute and harm content or reconstruction without saving full deployed bytes.

## U — unresolved

The public repository README describes a discrete semantic stream plus optional continuous residual codec and cites a differently titled Interspeech paper. The registry's PA259 is the arXiv paper describing separate semantic and acoustic branches. Exact source-to-paper correspondence and a runnable matched stream-level experiment must be established before testing the Mirror hypothesis.

See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md) and [source/UPSTREAM.md](source/UPSTREAM.md).
