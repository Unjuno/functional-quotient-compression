# MA-540 — Ordered sequential execution of per-step function vectors

Status: FAIL. Scope: two-step composition of synthetic affine state transformations.

## H — hypothesis

A support-derived function vector for each primitive operation, passed in order through one shared transition block with the intermediate state fed back, will compose held-out operator pairs while saving bytes against independent step blocks and adding value over ordinary tied operator-ID conditioning.

## T — execution

Two development seeds (54001, 54002), 24 affine bit-permutation/XOR rules over all 16 four-bit states, and 110 held-out unordered rule pairs per world with both orders included. The shared MLP transition block saw all atomic function/state mappings. Each FV was extracted as the mean hidden pre-activation delta over eight support states. The two-step executor passed the predicted intermediate state and the second FV into the same block.

At 1600 updates/LR .003, mean held-out pair accuracy was highest across the frozen grid, so that common setting was selected. Controls included native tied operator IDs, the host external two-call executor, two independently trained untied blocks, a one-pass sum-of-vectors endpoint model, and exact affine-table lookup. Three numbered implementation amendments are recorded; earlier runs remain in `results/pre_amendment_2/` and `results/pre_amendment_3/`. Fresh seeds 54011–54013 remain sealed.

## D — FAIL

The selected FV executor reached 97.95% and 99.20% held-out ordered-pair accuracy, with atomic accuracy 99.22% and 99.74%. It used 9,126 B versus 17,182 B for two independently trained blocks (46.9% fewer bytes), but missed the frozen 99% pair-accuracy gate in world 54001.

The native tied operator-ID and external two-call controls reproduced the same pair accuracies exactly while using 9,112 B—14 B fewer than the FV payload. Across methods, the extracted FV logits differed from native operator-ID logits by at most 2.9e-6. The one-pass endpoint sum reached only 18.98%/15.11%; exact affine lookup reached 100% at 1,286 B.

## Fact / interpretation / hypothesis

- **Fact:** Tied FV execution nearly composes the held-out pairs and uses 46.9% fewer bytes than two untied blocks. The native tied-ID and external two-call controls match its outputs and use 14 fewer bytes. The exact structured upper is both smaller and exact.
- **Interpretation:** Passing an intermediate state in order supports logical composition in this fixture. The measured storage reduction comes from ordinary block tying; the FV coordinate adds no useful function beyond a native operator embedding.
- **Strongest counter-hypothesis:** The world-54001 miss of the 99% gate may reflect optimization or initialization variance; world 54002 clears it. This cannot establish Mirror-specific value because the native control exactly matches the FV outputs on both worlds.
- **U — limits:** Synthetic affine operators, two steps, a small MLP and externally supplied operator order only. No language-model task or optimized recurrent kernel was evaluated.

## Compute and verification

Each method's training and evaluation wall time, optimizer updates, training/inference operation proxies, atomic/pair accuracy and actual NPZ bytes are recorded in `RESULTS_CORE.csv` and per-world `metrics.json`. The corrected final run reproduced both prior corrected worlds' split, all payload arrays and pair metrics exactly; timing and atomic diagnostics were added by a reporting-only amendment. Tests: `python -m pytest -q experiments/mirror_applications/ma-540-function-vector-executor/tests` (3 passed). Full verification is in `VERIFICATION.json`.
