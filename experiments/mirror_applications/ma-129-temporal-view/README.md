# MA-129 — Mirror verifier views

Status: FAIL at development gate
Evidence lane: MECHANISM
Base commit: `1ccd2583fa05619acbb87ce8f2c2531dce04cfcc`

## Hypothesis

H: A shared verifier plus per-branch Mirror views can recover several correlated branch decisions with smaller actual payload than MTP verifier heads while preserving exact valid-path quality.

## Prior-art delta

PA09/PA10 motivate MTP and packet-conditioned joint prediction. This screen isolates binary branch verification and includes tied, rank-2 shared branch-code, independent MTP, and sequential branch-scoring controls. Sequential scoring does not include a Transformer KV cache.

## Development results and decision

Two development seeds, 500 updates at LR 0.02. On the aligned teacher, Mirror quality passed its development comparison to MTP, but the actual-payload requirement (≤0.80× MTP) failed in both seeds: **Mirror 1,957B vs MTP 1,833B** (1.068×). Mirror has fewer trainable scalars, but `torch.save` stores separate angle and shared-weight tensors with more metadata. Therefore the preregistered failure gate was triggered and fresh should have remained sealed.

| Method | Median aligned NLL | Exact path accuracy | Payload |
|---|---:|---:|---:|
| tied | 0.1357 | 0.873 | 1,705B |
| rank-2 branch-code | 0.0269 | 0.961 | 2,209B |
| Mirror view | 0.0963 | 0.971 | 1,957B |
| independent MTP | 0.0976 | 0.949 | 1,833B |
| sequential shared score | 0.1357 | 0.873 | 1,705B |

### Exploratory runs after the gate failure

I incorrectly opened seeds 12911–12913 after the development byte failure. These rows are retained in `RESULTS_CORE.csv` and marked `exploratory_protocol_deviation`; they are **not confirmatory evidence** and do not change the FAIL decision. Descriptively, the aligned exploratory median Mirror NLL was 0.0966 with 0.949 exact path accuracy; MTP was 0.1228 and 0.910. The independent teacher still favored MTP. Eager Mirror branch throughput was about 0.049× MTP in these runs.

## Verification

Twenty development rows and thirty exploratory rows were replayed with exact serialized bytes; maximum metric delta was 4.95e-11. Tests passed 2/2. Verification records that fresh-split integrity was compromised by the procedural error.

## Decision

**FACT:** The actual-byte development gate failed on both seeds. Exploratory worlds were opened contrary to the locked failure rule and are excluded from the status decision.

**INTERPRETATION:** This pytorch serializer gives no storage improvement for the tested Mirror verifier, despite fewer learned scalars. The exploratory quality signal is not a fresh confirmation.

**HYPOTHESIS:** A packed tensor format could change the storage result; it requires a separately preregistered experiment. Independent branch functions need MTP capacity.

**BOUNDARY:** Synthetic binary labels, no real speculative decoding or Transformer KV cache, and a disclosed fresh-split protocol violation.
