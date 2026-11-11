# Structural Composition Mirror Family Diagnostic — 2026-11-11

## Scope

This diagnostic covers the PA234 structural-composition sequence MA-871 and MA-872 and pauses the adjacent MA-873 candidate pending a mechanism redesign. Both are completed synthetic screens on the cumulative research branch; their dedicated reports and fresh evidence remain authoritative for their scoped outcomes.

## Facts

- **MA-871:** Twelve training tasks generated from a rank-2 recombination family supported six fresh tasks at 8-shot function MSE <= 0.000801 in all three worlds. Six-task payload was 408 B for the Mirror coordinate and 488 B for independent coefficients. Native rank-2 PCA had exactly the same outputs and bytes. Rank-2 decode used a 3x proxy MAC count (120 vs 88 total with shared module evaluations).
- **MA-872:** A discrete eight-prototype codebook passed its fresh quality gate at 2 shots, versus 4 for continuous rank-2 fitting and 16 for independent coefficient fitting. Sixteen-task payload was 569 B for discrete selection, 489 B for continuous PCA, and 809 B for independent coefficients. Native nearest-codebook had identical selected addresses, outputs, and bytes as the Mirror address.
- Both screens used synthetic low-dimensional task families and did not test unseen composition rules.

## Interpretation

The shared structure can improve support efficiency and reduce storage relative to independent per-task coefficients in these generated families. The added Mirror coordinate does not distinguish itself from a standard native representation: PCA/bilinear coordinates reproduce the continuous case, and an ordinary codebook index reproduces the discrete case. The repeated failure has the same structural cause: a low-description task address is only a renaming of a native coordinate or index when it does not change how the task family is learned, inferred, or decoded.

## Family ruling

Pause the current MA-873 “module identity × recombination-rule factorization” protocol as written. Preserve its registry row as UNTESTED. Resume only after the hypothesis changes to test an additional mechanism beyond coordinate/factorized lookup, such as compositional rule learning from held-out factor combinations with a measured adaptation process and a native tensor-factorization control. Do not delete or reclassify the candidate as a result.

The pause applies to the current PA234 coordinate/codebook family. It does not block the separate PA235 stable-synapse adaptation lane.

## Hypotheses for redesign

- Learn or infer recombination structure online from examples, and compare the number of examples and updates against native low-rank and codebook methods.
- Hold out module × rule combinations and test whether a compositional learner predicts genuinely unseen combinations from independently observed factors.
- Charge basis discovery, candidate tables, task addresses, adaptation state, and inference arithmetic together; compare storage-quality-compute frontiers.

These are redesign directions, not claims established by MA-871/872.
