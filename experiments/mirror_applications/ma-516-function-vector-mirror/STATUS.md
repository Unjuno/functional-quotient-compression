# MA-516 status

- Status: SCREENING
- Branch: `research/ma-516-function-vector-mirror-compression-20261009`
- Base commit: `873dd449`
- Model: pinned GPT-2, CPU float32; model/tokenizer provenance is recorded separately
- Development complete: yes; PCA basis fitted on worlds 51600/51601 × seeds 0-2
- Fresh opened: no; locked worlds 51610-51612 × seeds 0-2
- Protocol/source freeze commit: pending

## Next action

Commit the frozen protocol and runner, then execute fresh once and update results/verification/registry/claim ledger.

## Blockers

None. No CUDA device is present, so the pinned 124M-parameter model runs on CPU.

## Decisions

- The LoReFT representation-view queue (MA-503 onward) is temporarily deferred after two consecutive failures with the same VQ-vs-FP16 distortion cause. Those candidates remain UNTESTED.
- The active queue pointer moves to the distinct Function Vectors family at MA-516.
