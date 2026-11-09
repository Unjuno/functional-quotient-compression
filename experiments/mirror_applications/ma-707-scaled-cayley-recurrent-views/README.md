# MA-707 — Scaled-Cayley recurrent Mirror Views

Status: FAIL  
Evidence lane: MECHANISM  
Base commit: `322e8039`

## Hypothesis

H: A shared orthogonal recurrent transition plus small low-rank skew/Cayley task codes can provide stable logical dynamics at lower serialized bytes than independent matrices, with solve overhead amortized across the sequence.

## Mirror insertion

> **Mirror insertion:** adds per-task skew-symmetric coordinates at the recurrent transition interface and reconstructs the task view through a Cayley map.

- Native baseline: independent dense transitions.
- Hard tying: one shared Q0 for every task.
- Cheapest native equivalent: standard per-task low-rank Cayley parameterization, included exactly.

## Prior-art delta

PA171 establishes Householder and scaled-Cayley orthogonal parameterizations. MA-707 tests a shared physical transition plus per-task skew codes, including sequence stability and inversion/solve cost.

## Gates

Frozen dimensions, ranks, seeds, and gates are in [PROTOCOL.json](PROTOCOL.json). Actual payload includes all factors and metadata; reconstruction cost is charged/amortized separately.

## Results and decision

**FACT:** Across three corrected audit seeds, serialized candidate replay matched independent trajectories exactly (NRMSE 0); maximum spectral radius of the scaled transition was .95000011. At skew rank 1/2/4, actual payload was 28.1%/40.3%/64.7% of independent matrices. Reconstruction solve/composition overhead was 43.2%/44.8%/47.9% of the 128-step matvec work. The hard-tied transition had substantial trajectory error. Native low-rank Cayley payload hash and transition matched exactly.

**INTERPRETATION:** Shared low-rank Cayley state compresses stable task transitions at rank 1, but matrix reconstruction exceeds the frozen 25% compute overhead gate. Higher ranks exceed byte limits. Native Cayley exactly explains the functional freedom, so no Mirror-specific value is established.

**HYPOTHESIS:** Low-rank skew/Cayley coordinates can preserve stable recurrent Views with a state/solve tradeoff, but this setting misses the combined frontier.

**BOUNDARY:** Synthetic driven linear recurrence only; no trained RNN/SSM, sequence prediction quality, or GPU solver claim. The initial stability/replay metrics are preserved as invalid attempt; corrected audit replay used unchanged seeds and settings.

## H/T/D/C/U

- **H:** Shared orthogonal transition plus compact low-rank skew codes recover multiple stable task dynamics with fewer bytes.
- **T:** D=32, 8 tasks, 128 steps, skew rank 1/2/4, three audit seeds; independent matrices, hard tie, native Cayley.
- **D:** FAIL for Mirror-specific value and combined frontier: stable/exact at rank1 with 28.1% bytes, but solve/reconstruction overhead is 43.2%; native Cayley exactly aliases.
- **C:** The result is conventional low-rank scaled-Cayley recurrent parameterization over shared Q0.
- **U:** Learned sequence quality, long-context stability under training, and fused/GPU solver runtime remain untested.
