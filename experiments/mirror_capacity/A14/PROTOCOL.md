# A14 — bounded code-manifold oracle diagnostic (frozen before outcome inspection)

Status: exploratory, registered before executing A14; motivated by the already-inspected A13 failure.

## Goal
Distinguish ordinary support-only ES failure from the inability of a frozen, trained 2D code manifold to reproduce triangle functions. This is NOT a fair deployable result: the diagnostic oracle **deliberately fits codes to IID+OOD labels** and has test-label leakage by design. Such oracle values cannot be claimed as held-out accuracy, proof of capacity, or new compression rates.

## Inputs
- Exact A13 main & replication archived parents, both families' parent hashes already fixed. Focus first on **tent** only. 6 independent world draws: 1200–1202, 1300–1302; 2 seeds/world; 3 methods (full Mirror, direct Shift, FiLM), 36 frozen parent models.
- For every parent, import the corresponding **A13 support-only ES model** (960 candidates × 32 support examples/task). Verify both archived SHA256 and parent-child invariant before diagnosis.
- Teacher: A11 `make_world`. New code addresses 16–23, 8 held-out tasks. Each oracle code sees exactly **256 IID + 256 OOD labeled examples** from the original independent A13 audit splits. The labels are seen only by the privileged oracle; A13 scores remain genuine heldout estimates.

## Oracle
- Same 2-dimensional box [-4,4]^2 and CEM-ES rule `a13.es_optimize` (32 candidates × 30 generations = 960 new tested candidates/task); warm-start from each A13 support-only ES code. The original ES code remains an incumbent and is counted separately (a small advantage to oracle). No changes to shared weights, optimizer settings or model topology. Selection minimizes joint IID/OOD MSE. Evaluate both oracle-selected and support-selected codes on existing IID/OOD **as in-sample diagnostic fit** (oracle) vs heldout (support), and record support MSE as diagnostic only.
- Exact FP32 model bytes frozen before audit; SHA256/roundtrip required. Check parent's shared tensors and original train codes bitwise unchanged.
- Old-dev quality is fixed by parent; observed joint pass criterion: old-dev, new IID and new OOD each NRMSE <=0.20. This check is **oracle-on-test existence witness**, not a generalization/pass count.

## Hypothesis/gates
- H1: privileged oracle can make frozen Mirror pass in >=3/6 worlds; if no, evidence favors its frozen low-description manifold or optimizer landscape being the bottleneck, not only 32-example estimation.
- H2: Shift oracle has >=4/6 passing worlds and Mirror has <=1/6; this favors architecture-specific shortcomings under fixed learned parent.
- C: even 960 new candidates may miss good Mirror codes; a bounded 2D box and finite CEM are NOT a global optimizer or a lower bound on the true best attainable error. Distinguish search failure from provable unrepresentability.
- U: 6 synthetic worlds, 2 seeds each, A13-selected parents, privileged train-on-test, mismatch in FP32/FP64, code box, no exact equal-FLOP. Do not compute p-values as if task/seed observations were independent worlds.

## Units and implementation
All inputs, outputs and codes dimensionless; evaluated MSE has dimension 1, NRMSE dimension 1. Serialized FP32 model includes full train/new code tables, all shared parameters, architecture header and checksum, excluding Python decoder and runtime. CPU NumPy FP64 score, final PyTorch FP32 replay.

## Stopping and source boundaries
- Deterministic fixed 6-world population, 2 seeds, 3 methods, 8 tasks each; no outcome-based search extension or family switching.
- Lock all 36 oracle model files and config before generating the published comparison. Report failures and discrepancies. A14 cannot establish a real-LM or Mirror-specific rate-quality win.
