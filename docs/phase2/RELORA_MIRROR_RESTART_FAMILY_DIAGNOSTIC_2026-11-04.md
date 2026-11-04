# ReLoRA Mirror restart-coordinate family diagnostic — 2026-11-04

## Scope

This diagnostic covers MA-646 (fixed/reusable rank-4 basis and phase block codes) and MA-647 (shared butterfly schedule with scalar restart coordinates). Both are synthetic mechanism screens; neither trains a language model or establishes ReLoRA convergence behavior.

## Facts

- **MA-646:** Four rank-4 updates produce a rank-16 target. Reusing one static rank-4 basis caps the reconstructed cumulative update at rank 4 and yields relative Frobenius error .813–.837. An overcomplete shared rank-16 dictionary plus phase block codes recovers rank 16 at error about 7.6e-8, but its 8,780B payload is the same as phase-wise ReLoRA restart factors and larger than merged ReLoRA's 8,297B. The native block-ID control is elementwise identical to Mirror.
- **MA-647:** A shared five-stage butterfly angle schedule with one scalar per phase recovers rank 16 exactly at 7,412–7,421B, 7.6% below independent ReLoRA and about 4.8% below independent BOFT. Native shared-angle BOFT has exactly the same state, outputs, bytes and operation count. A static unrotated rank-4 basis misses rank growth with relative error .813–.831.
- Both experiments used two development seeds, passed their verification tests, and kept fresh seeds sealed after the registered native-control failure gate.

## Interpretation

Across these two candidates, the tested small restart coordinates implement ordinary native structures: block IDs into an overcomplete basis or scalar modulation of BOFT angles. Fixed basis reuse does reduce state but prevents rank growth; allowing phase-specific subspaces restores rank growth but removes the fixed-basis compression, and the best observed byte gain is reproduced exactly by BOFT.

## Hypothesis and queue decision

The current ReLoRA restart-coordinate proposals do not establish a distinct Mirror mechanism. Pause the direct ReLoRA restart-coordinate lane until a proposal identifies a function/control that cannot be represented by ordinary block IDs, shared BOFT angles, or another matched structured adapter. This pause is limited to this lane. Continue the global P0 queue at MA-649; the learned optimizer candidate MA-648 is P1 and has a different hypothesis.

## Unresolved

Natural update trajectories may occupy structured subspaces not represented by these synthetic tasks. Any future restart-coordinate design must charge transform state and compute, compare against native structured controls, and measure real training quality and actual serialized bytes.
