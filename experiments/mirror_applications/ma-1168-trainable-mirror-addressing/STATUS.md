# MA-1168 STATUS — 2026-10-20 JST

**Experimental stage: COMPLETE, VERIFIED; Mirror-specific quality gate FAIL.** All 24 CPU runs performed at 900 updates and matched the preregistered 2-world×3-seed×4-method comparison. 7/7 tests, 24/24 checkpoint fresh/OOD metric replays, 24/24 serialized payloads, 3/3 aligned oracle replays, 18/18 final VJP decompositions, 36 initial/final gradient conditions and 12 code-swap ablations verified.

- With one 289-scalar trainable object and 75 concurrently executed candidate MLP edges across each of three synchronous Loop steps, **learned physical-to-virtual addressing beat fixed-address learned Mirror orientation 6/6** in paired fresh NMSE (median ratio 0.677). This is an improvement of parameterization, not demonstrated intrinsic advantage of Mirror.
- Against **the same learnable addressing with a non-Mirror affine decoder**, learned-address Mirror wins 3/6 but wins by >=5% in **0/6** (FAIL).
- Against **equal-289-coefficient native MLP**, wins **0/6** (FAIL) and uses more compute. All initial-to-final gradient conflict findings and code-swap ablations remain local/synthetic.
- Physical learned payload 1,176 B with header; deterministic runtime fixed buffers 298,760 B and temporary virtual weights 33,000 B must be accounted separately; not deployed compression.
- Source and all exact checkpoints/checkpoint hashes are in the verified conversation artifact, not claimed to be fully present in this GitHub branch.

Evidence: [Japanese report](REPORT_JA.md), [result table](results/RESULTS_FRESH.csv), [verification manifest](VERIFICATION.json), [frozen protocol](PROTOCOL.json). Parent MA-1167 verdict and main branch unchanged.

Scientific label: M0 for Mirror-specific value; MECHANISM established for synchronous batching and trained address specialization; none of the evidence establishes unrestricted correspondence search, pruning or efficient production inference.
