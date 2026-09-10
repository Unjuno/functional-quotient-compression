# Research state through T282 — 2026-09-11

This supersedes the old summary's statement that no actual whole-model serialized codec exists.
It does **not** supersede the failed 64x quality gate. Historical theory and reconstruction notes
are retained in [RESEARCH_STATE_PRE_T282.md](handoff/RESEARCH_STATE_PRE_T282.md).

| Lane | Established within the recorded scope | Not established |
|---|---|---|
| T248–T265 | Synthetic support-selection, joint-allocation, gain and sparse-compilation results | Real-model generality or GPU speed |
| T266–T272 | Supplied checkpoints audited; complete whole-weight binary and independent decode; 64x rate witness | Quality-preserving 64x (failed) |
| T273–T278 | Scalar baselines, 27 candidates, historical same-environment regeneration | FQC-specific novelty; fresh held-out evaluation |
| T279–T282 | Frozen 3-candidate prefix pilot; long-context engineering tests; same-lossless-codec comparison and replay | Official data-byte parity, official runtime, independent replication |
| H282 packaging audit | 149 relocated unit tests passed; 3 frozen artifacts rebuilt with exact bytes and tensor hashes | All prior experiments rerun in this handoff, or any Mac/GPU certification |

## The most useful numerical anchors

The supplied model named 28M contains 51,987,968 unique paid learning scalars.
The 16-bit weight baseline is 103,975,936 bytes, so the 64x ceiling is 1,624,624 bytes.
T271's selected-tail 64x candidate had mean KL about 4.8167 and mean delta NLL about
+4.4596 nat/token on 16 authored documents. This is a **quality failure**, not a near-lossless codec.

For T279's 44 transcribed prompt prefixes (2,682 predicted tokens):

| Frozen candidate | Same-XZ final bytes | Document-mean KL | Document-mean delta NLL, nat/token |
|---|---:|---:|---:|
| RTN, 4-bit, group 64 | 27,064,624 | 0.091881 | +0.094847 |
| Activation-weighted, 4-bit, group 128 | 25,507,600 | 0.071836 | +0.068774 |
| Activation-weighted, 4-bit, group 64 | 27,343,380 | 0.056781 | +0.041954 |

All quantities are checkpoint/input-conditional descriptive measurements. Same XZ includes envelope overhead;
metadata is charged. Common tokenizer/software assets are excluded equally. Storage is not runtime precision.
Per-prompt numerical rows and envelope hashes are in `experiments/t282/records/`.

## Corrections that must survive handoff

Reversible mirror changes alone are not a fundamental rate reduction. Weight error need not track task error.
More private corrections can hurt. Better KL need not mean better NLL. Fixed code width need not mean equal
final lossless-compressed bytes. Same-machine repeatability is not unknown-data validity. Avoid subjective
completion percentages: they are not a measurement of research maturity.

Limited historical Q/K sharing evidence exists; it must not be erased. However, it does not demonstrate
whole-model FQC superiority against the current non-sharing controls. The recent fourfold result uses
ordinary activation-aware scalar fitting and standard entropy coding; it is not evidence for a new
functional-quotient mechanism.

## Immediate local sequence

1. Verify trusted model/tokenizer hashes and reproduce frozen CPU artifacts without changing their locks.
2. Establish parity with an installed, version-recorded official Transformers runtime and tokenizer.
3. Acquire official data with revision, license, raw hash and deterministic disjoint split manifests.
4. Validate any MPS evaluation path numerically against CPU before using it; keep float64 calibration on CPU.
5. Freeze a small non-sharing / sharing / private-correction / joint-allocation experiment at matched actual bytes.
6. Evaluate one locked candidate on an untouched test set. Publish failures and uncertainty as well as gains.

No MPS speed or memory guarantee follows merely from having a 64GB Mac. The exact chip/OS/backend must be
recorded locally. Neither 256 documents nor any other fixed count guarantees statistical power.

## Archival and reproducibility boundary

The curated Git lane retains measured cores byte-for-byte but introduces a new packaging-only CPU entry point.
It does not pretend all old runners were refactored into one API. Full legacy packages remain SHA-indexed
external source archives. Historical record JSON is immutable and may contain its original `github_write_performed: false`;
that field describes that old experiment, not the later handoff commit.
