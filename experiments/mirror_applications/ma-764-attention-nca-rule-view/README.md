# MA-764 — Attention-generated dynamic local-rule Views

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

On 12 heldout context values, a native attention-generated kernel and Mirror rule Views have one-step nRMSE `4.0e-8`, 10-step rollout nRMSE `2.34e-7`, state norm ratio 1.0 and payload 1,037 B. Their outputs, MACs, hashes and bytes are identical. Independent context-specific kernels use 1,576 B and 2,211,840 MACs. Dynamic kernel mixing uses 4,423,680 MACs (2×). Ordinary attention over separately nonlinear branch features uses the same 1,037 B and 4,423,680 MACs, with one-step/rollout error `.00451/.00465`, within the `.02` quality gate. Hard tying has rollout error `.997`.

## Interpretation

Attention-selected interpolation inside the rule kernel recovers this designed family and compresses its context bank by 34.2%, with twice the convolution work. However, ordinary feature attention nearly matches quality at the same bytes and compute, and a native kernel hypernetwork exactly matches Mirror. The experiment therefore shows a useful operator-order comparison, not a Mirror-specific advantage.

## Hypothesis result

**FAIL** for Mirror-specific value and the full Pareto gate. The native hypernetwork is an exact alias, and feature attention meets the same quality threshold at equal storage/compute. Initial mismatched-input metrics were excluded and are preserved.

## Frozen protocol / scope

PA197 motivates using attention context to choose local update behavior. We generate 32 synthetic context values; the teacher interpolates between two shared scalar 3×3 kernels with `sigmoid(3c)` inside the kernel, followed by tanh. Fit the attention score from 20 contexts; evaluate 12 heldout contexts on matched random maps and 10-step recurrent rollouts. Compare independent effective kernels, hard tying, attention over nonlinear branch features, native kernel hypernetwork and Mirror. Actual NPZ bytes, output/rollout error, state norm, MACs and wall time are recorded.

No trained attention-NCA, emergent morphology, biological task or accelerator runtime is evaluated. Context and teacher kernels are designed analytically.

## H / T / D / C / U

- **H:** Attention-produced rule coordinates preserve heldout contextual dynamics and beat attention that only mixes features at matched storage/compute.
- **T:** Seed 76401; 32 contexts; 20 fit / 12 heldout; two 3×3 kernels; 8 maps/context; 10 recurrent steps; five controls.
- **D:** **FAIL.** Mirror aliases the native hypernetwork exactly. Feature attention also meets the `.02` gate at equal bytes and MACs; rule Views cost 2× independent MACs.
- **C:** Ordinary attention over two nonlinear rule features gives near-equivalent heldout behavior; standard kernel hypernetwork fully reproduces Mirror.
- **U:** Learned attention on real NCA tasks, long-horizon emergent behavior, context shift, morphology and accelerator kernels.
