# MA-1180 — Mirror code optimizer geometry ablation

**Mirror insertion:** retain the exact 196-coefficient shared 4→16→4 tanh Mirror student from MA-1179. A low-description per-rule hidden Householder direction `m` changes the shared function; **this experiment modifies only the learning algorithm for `m`**, keeping its semantics and core optimizer fixed.

## H / main objective
The scale-invariance of Householder directions means radial code motion does not affect the represented function but can affect optimizer trajectories. Compare Euclidean AdamW (weight decay and LR ablations) to projected/normalized sphere-oriented code optimizers, and test whether improved Mirror optimization beats equally sized FiLM. One should NOT infer optimizer mismatch merely from an exact geometric invariance.

## T / experimental design
Reuse MA-1179's SHA-frozen 5 teacher bundles (7 ordinary teachers/seed, 35 total), 3 worlds: exact Householder-positive, independently fitted correlated and unrelated. Use a NEW, separately seeded train/dev/fresh/OOD sampling stream; 5 seeds × 3 worlds × 8 arms = 120 cold-start students, each 750 updates batch128, all 196 parameters. All training outputs generated from frozen teacher functions without transferring pretrained teacher weights to students. The common core is AdamW(lr=.012), only code optimizer differs. Compare code AdamW standard, zero-decay, low LR, high LR, per-rule fixed-norm renormalization, sphere-projected scalar-moment Adam variant, sphere-tangent momentum SGD, FiLM AdamW. Full method definitions are frozen in PROTOCOL.json.

## D / frozen pass gates
1. Fixed sphere-scalar Adam wins ≥10% Fresh NMSE over ordinary code AdamW in ≥4/5 paired correlated-world seeds.
2. DEV-selected geometric winner (three variants) wins ≥10% Fresh NMSE over DEV-selected Euclidean AdamW winner (four variants) in ≥4/5 correlated seeds; development search budget of all candidates charged.
3. DEV-selected geometric Mirror winner wins ≥10% against FiLM in ≥4/5 correlated seeds.

## C / U / interpretation
A result restricted to artificially Mirror-aligned teachers does not establish natural generality. Even a PASS against default AdamW may reflect hyperparameters rather than manifold geometry, so matched LR/zero-decay controls are mandatory. 5 previously examined teacher identities, synthetic 4D data, different optimizer computational overhead and CPU-only execution limit evidence. New fresh sampling seeds were not used for method selection. Negative results are preserved.

Primary precursors: MA-1178 and MA-1179 reports. Prior-art: Loshchilov/Hutter AdamW (arXiv:1711.05101); Bécigneul/Ganea Riemannian adaptive optimization (arXiv:1810.00760). The custom sphere algorithm here is **not** an exact published Riemannian Adam.

The teacher checkpoints are not automatically in Git history; `--teacher-root` takes MA-1179 checkpoints and validates exact SHA256. Use `--rebuild-teachers` if necessary for a numerically compatible source version, and compare resulting SHA before interpreting any reproduction.
