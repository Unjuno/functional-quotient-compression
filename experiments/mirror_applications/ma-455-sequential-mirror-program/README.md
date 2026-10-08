# MA-455 — Sequential Mirror program over one physical block

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/RUNTIME  
Branch: `research/ma-455-sequential-mirror-program-20261008`  
Base commit: `04554b5`  
Prior art: PA81 (Routing Networks)

## H — Hypothesis

One 3×3 physical block reused for three ordered steps, with a small three-angle Givens View at each step, can fit a non-commutative three-step function with low error and at least 25% fewer actual bytes than three independent blocks. A reversed-code evaluation checks whether the learned sequence encodes order. A direct native Givens conditioner tests attribution.

## T — Frozen setup

See `PROTOCOL.json` for the frozen details. This is a 3D synthetic linear regression screen with an aligned teacher `M₂ M₁ M₀`, where each `Mⱼ = Rⱼ W Rⱼᵀ`. Development seeds are 45501 and 45502; fresh seeds 45511–45513 remain sealed. The registered controls are a tied block repeated three times, three independent matrices, per-step rank-1 residuals, direct native Givens conditioning, and reversed View order. Each condition receives 600 Adam updates of 16 examples (9,600 examples), and 512 held-out examples. Payload metric is the actual uncompressed `.npz` inference payload. No results are available until the frozen development runs complete.

## Mirror insertion

> **Mirror insertion:** add a learned rotation address `mⱼ` at each repeated block call, so the effective step is `R(mⱼ) W R(mⱼ)ᵀ` while only one physical matrix is stored.

- Native method: one weight matrix reused at every step.
- Exact insertion: per-step three-angle coordinate before and after the shared matrix.
- Persistent `m`: three coordinates for each of three ordered calls.
- Cheapest explanatory control: per-step rank-1 residual and direct native Givens conditioner.

## Prior-art delta

PA81 establishes reusable functions composed by task/input routers. MA-455 tests whether ordered per-step Views on one block reduce physical block storage while retaining the useful composition. It is a synthetic mechanism screen, not a learned router study.

## Gates and tuning boundary

PASS/PROMISING, FAIL and NOT ESTABLISHED conditions, allowed seeds, and metric definitions are frozen in `PROTOCOL.json`. No fresh/audit data may tune the configuration.

## Results

Pending frozen development runs.
