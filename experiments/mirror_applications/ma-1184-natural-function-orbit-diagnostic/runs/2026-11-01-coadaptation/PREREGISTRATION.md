# MA-1184 — four-way coadaptation L0/L1 run

Frozen before research training on 2026-11-01 JST (2026-10-31T22:46:59 UTC). Local source commit: `08ea72ff43674b19abba63511dced2d14c648faa`.

## Scope and amendment
The latest user-approved experiment separates A: fixed core/fixed code; B: fixed core/trainable code; C: trainable core/fixed code; D: both trainable. Unlike the earlier draft stopping rule, joint training is evaluated even if fixed-core code fitting fails. This is **development L0/L1 only**, not new-task-family L2 confirmation, natural-data evidence, or a Mirror+Private/MoE/Loop experiment. Previous MA results are unchanged; no main merge.

## Frozen design
- Input8 / hidden16 / output3 / tasks3; ReLU.
- Three non-Mirror-generated analytic families: perturbed feature directions with common thresholds/readout; directions plus task-specific thresholds/sensitivities; independently drawn ReLU/tanh/quadratic ridge functions. Independently train an ordinary width32 MLP for each of 3 target tasks and 1 source task. Student only sees teacher outputs, never teacher weights or analytic coefficients.
- Teacher: 4096 train / 1024 DEV, 1000 AdamW updates, batch128, lr0.006, weight_decay0.0001, clip1; choose DEV checkpoint at500/1000.
- Shared source core: distill source teacher outputs into width16, 1024 train / 256 DEV, 400 updates; select200/400. Charge this separately. Same source core reused within each world across all arms.
- Students: per task768 train / 256 SELECT-DEV / 512 delayed input-audit; batch128, 500 AdamW updates, lr0.006, weight_decay0.0001, clip1; checkpoint selection0/125/250/375/500. Same indexed minibatches across arms. Equal updates are NOT equal FLOPs.
- Conditioners: single Householder Mirror (243 scalars), gain-only FiLM (243), encompassing coupled-rank1 with free beta (246). Mirror and coupled are identical initially at beta=-2. Fixed and trainable codes are all counted in inference storage.
- Each A/B/C/D shares exactly the same initial core and code within a method/world/restart. Mirror directions begin at normalized coordinate-swap normals plus0.2 times unit Gaussian noise, preserving nontrivial fixed-view behavior. FiLM begins at1+0.2*unit_noise. Initial functions across Mirror and FiLM are NOT assumed identical: their actual input-distribution changes and gradients are measured.
- Reference controls: shared-only and independent width6/7 byte-brackets; independent width32 upper control is not equal-budget. No BatchEnsemble/SiLU claim in this minimal diagnostic; those remain deferred.
- Pilot: seed91001, three families, one code restart. Development: seeds1201/1213/1231/1249/1277, three families, two conditioner restarts. Restarts selected on SELECT-DEV only. Five world seeds remain n=5, not n=10.
- Development count: 360 conditioner configurations (270 trained,90 untrained A),60 trained references,60 teacher fits,15 source-core fits. Pilot is separate.

## Preflight and decisions
18 pretraining tests pass (including frozen-weight invariance, identical four-way initial predictions, mirror/coupled equality, FP64 gradient checks and packet roundtrip). Pilot GO requires finite nonzero Mirror-code gradients and, in at least2/3 families, maximum teacher DEV NMSE<0.20, independent32 DEV NMSE<0.30, and best capable non-Mirror DEV NMSE strictly between0.005 and0.30. If unmet, stop without changing task/hyperparameters in this run.

For each family, a developmental joint-learning signal requires Mirror-D audit NMSE <=0.90 times BOTH Mirror-B and Mirror-C in at least4/5 worlds. Mirror-specific developmental signal requires Mirror-D <=0.90 times EACH FiLM-D, coupled-D and SELECT-DEV-chosen independent6/7 in at least4/5 worlds. These are development screens, not confirmatory significance tests or universal impossibility verdicts. Report every paired comparator and uncertainty.

## Evaluation/integrity
Macro NMSE uses task/output-specific TRAIN target variance, not audit variance. Store all checkpoints and exact compact inference packets including fixed tensors and normalization metadata. Seal all checkpoint/asset hashes before generating delayed audit inputs. Audit is still development on the same task identities: no Fresh/OOD or new-family confirmation claim. Replay every checkpoint and packet. At selected D checkpoints, separately measure original-core/learned-code, learned-core/original-code, wrong-task codes and switches evaluated against the intended alternative targets. Batch128 inference timing: single-thread CPU FP32 eager,20 warmups,7 blocks of40 forwards; report median/IQR. Two parallel single-thread training workers; training wall is not isolated benchmarking.

## Pretraining SHA256 commitments
- PROTOCOL.json: `64f4a8901b5a758a7abc5739e88b5bd30134ac526ce2461d3bc7888e845fa4cf`
- README.md: `e40d6272dfc4475e43471cda56a9a3bee47e77a0969c0a62907d65515c663681`
- source/core.py: `18c2abb0cf70e50e0e41a864415291eeee6de0e78b75e2c630496bcee956601a`
- source/run.py: `75500193ca868d90d0aa7462a00e52bf95fef7223bc9c91297eb7626d929506d`
- tests/test_core.py: `ce5617fe6d1ae17107fd830a725fb67d478314c9705782020939fe55154ca7e3`
- tests/test_runner.py: `fe68587f645658f478820a3eacd51d84adb09a2f0c4b87f2dfab1c05916cee55`

Complete source, protocol, test logs and all artifacts will be included in the conversation's reproducibility bundle. This registration contains no results and authorizes no deployment claims.
