# AMA-001 — Adaptive Mirror activation, shared axes and selective private fallback
Date: 2026-10-14. **VERDICT: FAIL_DEV_GATE.** Scoped CPU, source-locked experiment; **no canonical MA status or main changes**. The five fresh seeds 5201–5205 were never accessed.

## Purpose (H)
Jointly train (a) two shared learnable Givens-like rotation planes and compact per-task angles, (b) a shared learned activation composed of tanh odd + GELU even basis, and (c) a sigmoid-gated task-private rank-1 branch. Assess whether these jointly allocated freedoms are superior to fixed planes, native FiLM, native rank-one residual, and ordinary independent task heads. The transform family itself is fixed in this first experiment; no learned discrete transform-family choice or exact L0-pruned inference state is claimed. Learnable Givens is a known native transform, not uniquely Mirror.

## Test and scope (T)
- 8 concurrent logical tasks; shared 16-dimensional input and 16-hidden-unit FFN; synthetic output dimension 4.
- Worlds: **orbit** (teacher uses related random plane rotations), **partial** (2/8 tasks have one rank-1 off-orbit component), **unrelated** (8 unrelated nonlinear teachers), and **digits** (1797 real sklearn digit images with eight correlated binary prediction targets; train-only PCA16).
- 10 methods × 4 worlds × 3 development seeds (5101,5102,5103) = **120 trained and serialized models, 60,000 AdamW updates**. Each train step sees the same batch stream within a seed/method comparison. Validation-only selection at steps 125/250/375/500. 256 synthetic train / 128 val / 384 test inputs per world; digit image identities separated 60/20/20.
- Frozen protocol and science source SHA256 registered in `PRE_RUN_LOCK.json` **before development** (commit `316068bf97ecfdd0ddc29ea2702179bcb684b61b`). No fresh selection or retuning.
- Single-thread FP32 PyTorch2.10 CPU Xeon Platinum 8573C; 40 warm-up forwards and 50 measured batch1/64 calls; saved lossless-compressed inference NPZ files, full state/gates/axes counted.

## 3-development-seed means (D)
| Method | Orbit nMSE | Partial nMSE | Unrelated nMSE | Digits BCE (nat/label) | Partial full NPZ (byte) | Partial batch64 eager (ms) |
|---|---:|---:|---:|---:|---:|---:|
| fixed_mirror | .0995 | .1557 | .4536 | .3312 | 3459 | .170 |
| mirror | .0619 | .1363 | .3397 | .2916 | 3454 | .167 |
| mirror_activation | **.0403** | .1385 | .5371 | .2865 | 3680 | .205 |
| mirror_activation_private | .0426 | .0770 | .2761 | .1705 | 4981 | .279 |
| mirror_private (fixed GELU) | .0597 | **.0637** | .2552 | .1650 | 4762 | .225 |
| native FiLM | .0911 | .1159 | .1755 | .1693 | 3984 | **.040** |
| native shared rank1 private | .1094 | .1198 | .4669 | **.1631** | 3867 | .068 |
| native multihead shared trunk | .0874 | .0893 | **.0877** | .1769 | 4685 | **.038** |

- **H1 PASS 3/3 dev:** private branch improves partial-world error relative to same learned activation+Mirror without private by 36.0%, 41.7%, 55.2%. This is a mechanism-only, not storage-quality success.
- **H2 PASS 2/3 dev:** adaptive activation gives >=10% improvement on aligned orbit vs fixed-GELU Mirror; one development seed worsened.
- **H3 FAIL 0/3:** partial-world candidate costs ~4981 B, versus FiLM ~3984 B and native private ~3867 B; learned activation+private is even worse than its simpler fixed-GELU+private version on three-seed mean. No native-quality/byte dominance.
- **H4 FAIL 0/3:** unrelated functions candidate nMSE .2761 versus native independent head .0877.
- **H5 numeric BCE subgate 3/3:** digit candidate BCE .1705 within +.025 of native best .1631. The original prereg also required “no worst-task collapse” without a numeric threshold, so **a formal complete H5 PASS is NOT ESTABLISHED**. Native is smaller and better quality.
- **H6 FAIL 0/3:** batch64 eager about .279ms candidate vs .040ms FiLM, ratio ~7x; no speed gain.
- **Selective state diagnostic:** two off-orbit task IDs (2,6) learned average private sigmoid use **.420** vs other six tasks **.242**, even though the model was not told these IDs were outliers. However every private factor stays paid in the NPZ: NOT an actual sparse storage win.

**Overall: FAIL_DEV_GATE. No 5201–5205 fresh data accessed.** No language-model, independent natural-task, production-speed, new-capacity or algorithmic novelty claim.

## Counter-explanations (C)
Synthetic orbit teacher belongs to candidate's own family and favors Mirror; learned Givens is a native mathematical alias; extra learned coefficients can hurt off-family quality. The single sigmoid private gate does not achieve hard zeros. Multihead is a strong shared-trunk native control, *not* fully independent FFN upper bound. Eager CPU code uses unfused plane loops. Full transform-family learning, activation usage gates, exact L0 pruning, per-channel adaptive rank and dynamic byte multiplier learning remain untested.

## Verification, reproducibility (U)
10/10 unit tests PASS, 120 saved NPZ byte-length/SHA256 files and their 8 task-wise metrics reloaded and replayed with **max recorded error 0**, and one seed 5102 complete training replay reproduced the *identical NPZ bytes/SHA256*. The source and test files were frozen before dev and hashed before/after with exact SHA matches.

Evidence archives from the ChatGPT conversation (not yet hosted as GitHub binaries):
- `AMA001_2026-10-14_FULL_EVIDENCE.zip` SHA256 `d42996f64475ff308c2031767ee184eb912424e60b7b1ccec95b28e8a246b576` (all 120 NPZ, tests, logs, source)
- `AMA001_2026-10-14_SOURCE.zip` SHA256 `4107bc7c765d7ebb6d188a45232a9e7dc67434998d773284ce171e4686cf09c4`
- `AMA001_2026-10-14_EXECUTABLE_CORE.zip` SHA256 `84ad4a64ea0065643c28720209aa8bf2f048a100ae9980704beccb7544f9532a`
Scientific source core `source/model.py` SHA256 `2497b99a8fc9b444982c89013253e26592f66983e48d4f63d450f2e2deb7956d` and other exact file checksums are in `PRE_RUN_LOCK.json`.

These source/model archives are conversation artifacts, **not GitHub-uploaded source yet**; use them for complete numerical replication. The GitHub branch is a research evidence index, not a merged main branch.

## Suggested next test — not performed
Prioritize a hard-concrete/exact L0 **per-group Mirror-use gate** with actual physical pruning and a worst-task quality constraint; preserve native FiLM/multihead controls and absolute byte and runtime accounting. Do NOT re-use sealed 5201–5205 as development splits.
