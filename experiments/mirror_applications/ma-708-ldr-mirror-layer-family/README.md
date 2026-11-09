# MA-708 — Low-displacement-rank Mirror layer family

Status: FAIL (scoped synthetic mechanism result)  
Evidence lane: MECHANISM  
Base commit: `8538db5a9645f5a52267a36a9397cff52ea49dc3`

## Hypothesis

H: A shared bank of low-displacement-rank generators with small layer codes reconstructs held-out logical dense maps at lower serialized payload and matvec cost than independent dense layers, while matching native LDR controls may remove any Mirror-specific advantage.

## Mirror insertion

> **Mirror insertion:** per-map coefficients select combinations of shared structured displacement atoms to realize logical dense linear maps.

- Native baseline: independent dense matrices.
- Existing structured method: LDR generator plus coefficients.
- Mirror proposal: one shared atom bank and compact map codes.
- Strong attribution control: identical native shared-atom coefficient representation.

## Prior art

PA173 covers neural networks with low-displacement-rank matrices; PA174 covers learned compressed transforms with LDR. This screen asks whether the coordinate naming adds a benefit beyond the ordinary LDR coefficient representation.

## Results

**FACT:** Three audit seeds were run in the frozen synthetic span, D=64, eight maps, ranks 1/2/4/8, with 64 held-out vectors per map. Serialized replay matched the independently materialized matrices to numerical zero. At ranks 1/2/4 candidate payloads were 13.1%/25.6%/50.6% of independent dense bytes; rank 8 was 100.6% and therefore larger than independent storage. Reconstruction plus apply arithmetic ranged from 1.016x to 1.125x dense apply. Isolated apply wall time was measured locally but is not a portable runtime claim. Mirror and native shared-atom coefficient control had identical payload hashes and outputs. The current compute proxy charges matrix reconstruction plus dense application; no optimizer updates were performed.

**INTERPRETATION:** The shared structured atom bank compresses maps in this aligned synthetic setting, but the exact native coefficient alias means no Mirror-specific function or storage advantage is established. This is generic LDR feasibility only.

**HYPOTHESIS:** Explicit LDR structure may be useful when task maps lie in a shared atom span; task heterogeneity and model quality remain decisive limits.

**BOUNDARY:** Synthetic linear operators sampled from the shared atom span. No trained neural network, natural task, capacity, GPU runtime, or learned atom optimization claim.

## H/T/D/C/U

- **H:** shared displacement generators plus compact map codes recover logical operators at lower bytes and acceptable compute.
- **T:** D=64, 8 maps, ranks 1/2/4/8; development seed 70801 and fresh audit seeds 70811/70812/70813; independent dense, native shared-atom coefficient, and hard-tied single-map baseline (included as a conceptual comparator in the protocol; no task quality target was trained).
- **D:** **FAIL for Mirror-specific value** because the native LDR coefficient control is an exact payload/function alias. Synthetic compression feasibility is observed; rank 1/2/4 compute proxy is below the 1.25x gate, while rank 8 fails the 50% byte gate; these are synthetic arithmetic proxies, not practical neural runtime evidence.
- **C:** This is ordinary LDR/shared atom coding, already represented by PA173/174.
- **U:** Natural or trained-task quality, learned generator optimization, true displacement-rank formulation, and fused/GPU runtime remain untested.
