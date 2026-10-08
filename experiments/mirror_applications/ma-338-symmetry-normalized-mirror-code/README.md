# MA-338 — Symmetry-normalized functional code learning

Status: **FAIL for the frozen Mirror-specific storage margin at the development gate; fresh seeds remain sealed**
Dedicated branch: `research/ma-338-symmetry-normalized-mirror-code-20261008`
Prior art: PA37 Git Re-Basin; PA47 monomial weight-space symmetries.

## H — Hypothesis

Function-preserving hidden permutation/sign gauges obscure a small task-function subspace in raw checkpoint coordinates. Activation-based canonicalization should expose a rank-2 functional basis. A one-angle Mirror phase code may then store a task view more compactly than direct two-coefficient codes. The comparison must separate symmetry normalization gains from Mirror-specific code gains.

## T — Test

Synthetic 8-12-4 tanh MLPs, 64 tasks on a constant-radius rank-2 output-head delta orbit, plus one unrelated off-orbit task. Each checkpoint was obfuscated by a random hidden permutation and tanh sign gauge. We recovered hidden-unit correspondences by signed activation correlation on 256 calibration examples per checkpoint, then normalized task deltas. A rank-2 SVD basis was fit on the first 48 tasks and evaluated on 16 held-out task checkpoints. Controls were independent checkpoints, hard tying, raw full-checkpoint PCA rank 2, normalized direct coefficients, one-angle normalized phase, and phase without private off-orbit residual. Evaluation used 1,024 separate inputs per task. Development seeds: 33801 and 33802. Fresh seeds 33811–33813 were not accessed.

Serialized ZIP/NPY FP32 payloads were reloaded before inference. The two-world development run recorded 12 payloads, metrics, hashes, calibration/evaluation examples, SVD proxy, optimizer updates (zero), and wall time.

## D — Decision

**FAIL at development for the preregistered Mirror-specific byte gate.** Symmetry normalization recovered the exact hidden gauges and reduced the functional delta rank to 2. The raw checkpoint PCA required rank 44 to explain 99% of development parameter variance, and rank-2 raw PCA had held-out function nMSE 0.92–0.93. Normalized direct and phase codes both reconstructed the 16 held-out aligned tasks with nMSE below 1e-14. However, actual phase payload was 3,033B/3,037B, while direct coefficient payload was 2,987B/3,012B: Mirror was larger in both development worlds and missed the frozen requirement of at least 10% fewer bytes. Fresh seeds remain sealed under the documented development-stop amendment.

The no-private phase representation had unrelated-task nMSE 0.59–0.95; paying a private output residual restored exact quality. Independent 64-task payload was approximately 87.8KB.

## Fact / Interpretation / Hypothesis

**Fact:** Activation matching canonicalized every hidden permutation/sign state to max hidden activation error 0. The aligned task deltas had rank 2. Raw rank-2 PCA did not retain held-out functions. Actual payload and replay evidence are in `artifacts/development.csv`.

**Interpretation:** Quotienting exact parameter gauges can expose low-dimensional function structure. The reduction comes from symmetry normalization plus a shared low-rank basis; the one-angle Mirror encoding was slightly larger than the direct control under actual compressed serialization.

**Hypothesis:** A different code alphabet or task family might amortize phase codes better, but it requires a new preregistered candidate/protocol. This run does not support a Mirror-specific win.

## C — Strongest counter-hypothesis

The improvement over raw PCA is entirely due to known exact gauge removal and ordinary rank-2 task-vector factorization. The phase code's apparent scalar advantage disappears in serialized bytes because the direct coefficient array compresses well.

## U — Unconfirmed

No trained network, learned optimizer, online task adaptation, natural task distribution, or fresh-world replication was run. Fresh seeds stayed sealed because the fixed-shape development comparison missed the frozen 10% margin in both worlds.

## Reproduction

```bash
python experiments/mirror_applications/ma-338-symmetry-normalized-mirror-code/source/run.py --dev-only
python -m pytest -q experiments/mirror_applications/ma-338-symmetry-normalized-mirror-code/tests
python experiments/mirror_applications/ma-338-symmetry-normalized-mirror-code/source/verify.py
```
