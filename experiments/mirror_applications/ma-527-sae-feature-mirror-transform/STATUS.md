# MA-527 status

- Status: **FAIL (development gate; no fresh evaluation)**
- Branch: `research/ma-527-sae-mirror-transform-20261009`
- Base commit: `c935a903`
- Protocol: fixed Givens scale 0.1 and intervention scale 0.5; amendment 1 corrected an initial explicit-FV amplitude mismatch before any fresh evaluation.
- Development: seeds 52701 and 52702 complete.
- Fresh seeds 52711–52713 were not evaluated. Task IDs 14–15 were inadvertently extracted and serialized by the dev harness, so these identities are not sealed.
- Actual serialized method payloads and metrics: stored under `runs/dev/`.
- Verification: 5 tests passed; registry integrity check passed after reconciliation.

## H / T / D / C / U

- **H:** A shared 32-atom SAE bank with a fixed sparse Givens transform can preserve held-out relation steering while using fewer bytes than explicit FVs, native sparse coding and LoReFT controls.
- **T:** Pinned Pythia-70m and layer-3 tied SAE; fit task IDs 0–11, dev IDs 12–13, seeds 52701/52702. Both seeds evaluated no intervention, explicit FV at the same alpha=0.5, native SAE top-8, shared-bank sparse top-8, Givens Mirror, global OMP-8 and rank-8 LoReFT. Each method used 128 query candidate sequences; no optimizer updates. The inherited extraction loop also constructed/serialized codes for task IDs 14–15 under dev seeds; fresh seeds 52711–52713 were never run.
- **D:** **FAIL** under the frozen dev gate. Mirror loses 2.094 and 1.309 gold-logprob nats against same-scale explicit FVs, exceeding the 0.10-nat tolerance. It meets the incremental byte gate (3,474 B vs 34,742 B explicit) and uses 8 nonzeros/task, but seed 52701 also loses accuracy by 0.0625 and is dominated by a simple control. No fresh causal metrics exist.
- **C:** The strongest counter-hypothesis is that pretrained SAE coordinates discard task-relevant residual directions; rotation plus top-eight projection cannot recover those directions. LoReFT is substantially closer to explicit FV quality, while global OMP is better than Mirror in seed 52701.
- **U:** No fresh-seed causal behavior was measured. Task IDs 14–15 are contaminated for confirmatory use; a future audit needs unseen task identities. Other SAE dictionaries, layers, angle patterns, routing, and full training/convergence frontiers remain untested.

## Facts

- Explicit same-scale mean gold-candidate log-probability: -11.524 (seed 52701), -11.621 (seed 52702).
- Mirror: -13.618 / -12.931; accuracy 0.125 / 0.250 vs explicit 0.1875 / 0.125.
- Mirror incremental serialized payload: 3,474 B; explicit FV payload: 34,742 B; global OMP: 2,988 B; shared-bank sparse: 3,184 B; rank-8 LoReFT: 21,388 B.
- Complete Pythia base: 168,144,624 B. The SAE artifact adds 4,204,391 B. Mirror standalone total: 172,352,489 B; explicit-FV deployment: 168,179,366 B.
- Method evaluation wall time was approximately 0.93–1.21 s per dev seed/method; support extraction time is recorded in per-seed metrics.
- Five tests pass, including Givens orthogonality, sparsity bound, serialized payload byte/hash accounting, and confirmation that fresh seeds were not run after the dev failure.

## Interpretation

The sparse code provides a large incremental storage reduction when the SAE is already resident, but this fixed transform fails to preserve the explicit-FV causal effect and adds 4.17 MB to standalone deployment. This scoped result does not support SAE-feature Mirror steering for these relation functions. The dev extraction exposed task IDs 14–15 without evaluating their causal metrics; those IDs cannot serve as a sealed audit.

## Hypothesis for follow-up

MA-526 and MA-527 both indicate that this SAE representation is poorly aligned to the tested function-vector tasks. Treat the SAE shared-feature family as paused pending redesign; do not open MA-528/530 under the same representation assumption without a preregistered change that addresses feature/task alignment.
