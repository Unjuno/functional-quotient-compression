# MLA / KV latent Mirror family diagnostic — 2026-10-09

## Scope and decision

MA-581 and MA-582 are consecutive registered P0 screens in the MLA/KV-latent family. Both fail the frozen next-token quality gate for the same structural reason: compressing the already-produced Pythia K/V cache into rank-128 post-hoc latent reconstruction loses task-relevant information, and a small residual code does not restore it. Pause the nearby MA-583, MA-585 and MA-586 variants pending a redesigned quality-first protocol. They remain UNTESTED; no result is inferred for them.

## Evidence (facts)

| Experiment | Native/shared variant | NLL delta vs FP16 (dev worlds) | 8-session actual payload | Attribution |
|---|---|---:|---:|---|
| MA-581 | Per-layer rank-128 PCA + shared rank-4 head residual | +1.036 / +0.412 nat/token | 2,778,308 B (44.1% of FP16) | Shared residual state/cache exactly matched ordinary shared-dictionary coding. |
| MA-582 | Group-of-three rank-128 PCA + rank-4 layer View | +2.165 / +2.044 nat/token | 1,734,908 B (27.6% of FP16) | Mirror and native grouped residual cache payloads were byte-identical in both worlds. |

MA-582 without a residual is smaller (1,322,106 B for eight sessions) but has even worse NLL (+2.255/+1.964). Per-layer MLA in MA-582 also misses the gate (+1.041/+0.375), so the quality loss begins before cross-layer sharing. Each experiment used four calibration prefixes and one-token next-token evaluation; neither establishes full-sequence perplexity or an optimized serving result. Test splits stayed sealed.

## Interpretation

The two results indicate a capacity/representation problem in this specific post-hoc cache compressor, not evidence that all cross-layer KV sharing is ineffective. The rank-128 projection ignores how strongly a small subset of K/V directions can affect the next-token distribution. Grouping layers raises cache reconstruction MSE, while the rank-4 residual improves NLL inconsistently and provides no Mirror-specific mechanism beyond native shared-basis coefficients. Storage gains are real only when amortized over multiple active sessions, and do not satisfy quality constraints.

## Structural cause and redesign requirements

Before reopening this family, use a quality-first design that preserves high-impact directions and tests task-weighted cache error. Compare native MLA projection training or higher-rank per-layer/group controls, an equal-byte native residual, and private residual upper bounds. Report full-sequence/per-token quality, cache bytes at multiple session counts, and reconstruction/decode compute. Do not open fresh data until the redesigned development controls meet the quality gate and Mirror beats its native control.

## Boundary

This diagnostic applies only to post-hoc rank-128 PCA compression of Pythia-70M K/V with rank-4 residuals and short WikiText-2 prefixes. It does not falsify trained MLA, other ranks, longer contexts, or architectures that natively generate compact K/V latents.
