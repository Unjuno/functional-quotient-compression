# MA-821 — task-demanded matrix-memory rank

## H / T / D / C / U

**H:** Task-specific rank allocation preserves group-composition behavior while reducing state and read compute versus fixed-width matrix memories, with a Mirror-specific advantage over native truncation.

**T:** Synthetic 16×16 matrix memories, eight tasks with prescribed ranks 1,1,2,2,4,4,6,8; two-memory compositions on heldout vectors; seed 82101; zero updates. Compared independent full matrices, fixed rank-8 decomposition, variable-rank Mirror codes, and native per-task SVD truncation.

**D: FAIL for Mirror attribution.** Adaptive Mirror uses 4,471 B versus 7,967 B independent matrices (43.9% fewer) and 8,890 B fixed-rank-8 factors, with heldout nRMSE 3.16e-8, composition relative error 1.32e-8, and average read MAC proxy 368 versus 512. Native SVD rank allocation reaches the same errors, state dimensions, and compute at 4,473 B. Thus task-demanded rank is a strong compression/compute point, but the mechanism is standard native rank truncation, not Mirror-specific.

**C:** The task's effective rank is directly represented by ordinary per-task SVD truncation. The 2 B Mirror difference is NPZ field naming/layout.

**U:** Learned matrix memories, rank inference from task behavior, natural group-composition tasks, optimizer dynamics and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Serialized bytes, hashes, rollout/composition error, rank, compute proxy and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** Variable rank improves the fixed-width frontier but adds no benefit over standard variable-rank decomposition.
- **Hypothesis:** A learned rank allocator may matter when task rank is unknown, but that is not tested here.
