# MA-517 — Function-vector composition with product coordinates

Status: **FAIL** (frozen development screen; fresh seeds sealed).
Branch: `research/ma-517-function-vector-composition-20261008`
Prior art: PA99, Function Vectors in Large Language Models.

## H — Hypothesis

On six inverse relation pairs in pinned Pythia-70m, multiplying shared-basis codes of operand function vectors would improve held-out identity-composition accuracy by at least 0.10 absolute over both raw FV sum and difference in both development worlds, within half the explicit-FV payload, and with a useful margin over the native PCA-code product.

## T — Execution

We used `EleutherAI/pythia-70m-deduped`, pinned revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (weights SHA-256 `3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd`), layer output 4, six fixed inverse relation pairs, 48 held-out queries per world, development seeds 51701 and 51702, and zero optimizer updates. Controls were no intervention; either operand FV; explicit sum, difference and Hadamard product; and identical native PCA-code products at ranks 2/4/8/12. Fresh seeds 51711–51713 remain sealed. CPU, five-thread limit. The shared model/config/tokenizer payload is 168,144,624 B. `RESULTS_CORE.csv` reports actual uncompressed NPZ bytes, support tokens, operation proxy, and extraction + composition-build + query-scoring wall time.

## D — Decision

**FAIL** under the frozen gates. No-intervention accuracy was already 0.938 and 0.979. Raw sum scored 0.813/0.813 and raw difference 0.833/0.875. Product-code accuracy across ranks was 0.854/0.958 (r2), 0.833/0.958 (r4), 0.833/0.917 (r8), and 0.833/0.875 (r12). Thus none exceeded both arithmetic controls by the required 0.10 in both worlds, and none improved over the already strong no-intervention control. Mirror product payloads were 8,456 B (r2), 12,680 B (r4), 21,128 B (r8), and 29,576 B (r12), below the 34,446 B explicit operand bank, but native PCA-code products matched every Mirror product payload and metric exactly. The product quality gate and Mirror-attribution gate therefore fail; fresh evaluation was not opened.

## C — Strongest counter-hypothesis

Identity behavior is largely present without an intervention in this fixed, constrained ranking screen. Raw FV arithmetic does not implement sequential function composition, while product-code arithmetic has the same behavior as ordinary PCA-coordinate multiplication. Any small accuracy differences can reflect noisy vectors and pair-specific errors rather than useful general composition.

## U — Scope limits

This is a six-pair held-out query-ranking screen on one 70M model, not broad reasoning or open-ended generation. It does not establish composition capacity, production latency, or cross-model generality. Runtime measurements are CPU wall times for this harness and show no robust runtime advantage.

## Evidence classification

- **Facts:** the frozen metrics, actual payloads, exact native aliases, and deterministic replay are recorded in RESULTS_CORE.csv and verification reports; fresh seeds were not accessed.
- **Interpretation:** compressed product Views fit under the explicit operand bank here, but the frozen quality and attribution gates fail.
- **Hypothesis:** a different task with lower no-intervention accuracy or a composition-specific learned operator may be informative, but this experiment does not test that redesign.

Amendment 1 corrected the no-state serializer; Amendment 2 added composition-build timing. Both pre-amendment run states are preserved under `runs/`.
