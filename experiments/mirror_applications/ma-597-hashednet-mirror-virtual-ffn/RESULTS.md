# MA-597 results — development screen

Status: **FAIL** for the registered Mirror claim. Fresh splits stayed sealed because the Givens method missed the native-HashedNet improvement gate in both development seeds and a simple control matched or beat it at byte-near cost.

## H — Hypothesis

Thirty-two learned input Givens angles should reduce harmful collisions in a 2,048-bucket HashedNet FFN, giving at least +1.0 percentage point test accuracy over native hashing at ≤1.10× native bytes and beating byte-near diagonal and rank-one controls.

## T — Execution

- Dataset: `sklearn.load_digits`, sklearn 1.8.0, 1,797 examples, 64 normalized features, 10 classes. Tensor SHA-256 `faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1`.
- Model: 64→128 ReLU→10 MLP. Only the first 8,192-weight matrix was hash-shared; output matrix and all biases were charged in every payload.
- Methods: dense upper, native 2,048/4,096 bucket HashedNets, 2,048 buckets plus learned 32-angle Givens input View, 64-value diagonal gate, and rank-one first-layer residual. All ran 800 AdamW updates, batch 128, identical stratified 75/25 splits per seed.
- Development seeds: 59701/59702. Fresh seeds 59711–59713 stayed unopened.
- Inference payloads are uncompressed NPZ files containing FP16 trained tensors, paid hash seed, method/schema and shape metadata. Hash indices/signs regenerate deterministically from the paid seed and committed algorithm. Inference metrics were measured after serialization/load.

## D — Decision

**FAIL.** The Mirror Givens method used 9,753 B versus 9,441 B native (1.033×), within the byte cap, and stayed within two points of the dense model. It did not improve native accuracy by the required point: seed 59701 tied at 97.56%; seed 59702 was 98.22% versus native 98.67%. Cross-entropy was +0.0084 nat worse than native on seed 59701 and 0.0023 better on seed 59702.

Simple controls explain or beat it. The diagonal gate tied Mirror accuracy on seed 59701 at 62 B more and exceeded it by 0.22 points on seed 59702 at 62 B more. Rank-one residual was 0.44 points more accurate on seed 59701 and tied seed 59702 at 545 B more, still within 1.10× Mirror bytes. A 4,096-bucket native HashedNet reached 98.44%/98.67% with 13,537 B and reduced the collision rate from 75.37%/75.44% to 56.31%/56.73%.

The Givens method increased training time versus native 2,048 buckets (0.851/0.761 s vs 0.583/0.614 s) and measured 564k/416k versus 621k/660k examples/s. These are short CPU measurements; treat throughput as diagnostic, not optimized-kernel evidence.

## C — Strongest counter-hypothesis

The small accuracy differences come from optimization and split variation. Native bucket expansion already improves accuracy while reducing collisions, and diagonal scaling or rank-one residual gives comparable or better accuracy with similar bytes. The learned rotation is an extra coordinate without a demonstrated Pareto gain.

## U — Boundaries

One small image dataset, one FFN matrix, two development splits and 800 updates do not establish LM capacity or near-convergence behavior. Fresh splits, larger models, learned/per-layer hash functions, and optimized inference kernels remain untested.

## Evidence classes

- **Facts:** Givens accuracy is 97.56%/98.22% at 9,753 B; native hash is 97.56%/98.67% at 9,441 B. Givens test CE is 0.07281/0.04446 versus native 0.06440/0.04675. All six method payloads and metrics replay exactly for each dev seed.
- **Interpretation:** the Givens view did not improve the storage/quality frontier over HashNet plus simple controls in this screen. The 4x virtual connection count is not evidence of 4x functional capacity.
- **Hypothesis:** a collision-aware sparse exception mechanism may outperform global rotation; that is registered separately as MA-602.
