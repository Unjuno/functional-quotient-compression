# Model manifold Mirror-coordinate diagnostic — 2026-11-11

## Scope

This diagnostic covers the direct claim that a Mirror coordinate over a learned low-loss line, simplex, or Bezier surface yields functional freedom beyond the native chart coordinate. It does not rule out compression from learning the manifold itself, quantizing its code, projecting a new task into it, or aligning symmetries before fitting a chart.

## Evidence

| MA | Native chart | Mirror chart | Fresh result relevant to this diagnostic |
|---|---|---|---|
| MA-856 | line interpolation alpha | m = 2 alpha - 1 | Exact output equality; selected line also missed the all-seed quality gate and did not beat the two-model bank |
| MA-857 | simplex barycentric weights | m = 2 a - 1 | Maximum logit difference 2.86e-6, consistent with float32 reparameterization; quality gate failed and independent bank was smaller |
| MA-858 | quadratic Bezier (u,v) | m = 2(u,v) - 1 | Exact output equality on the tested grid; quality failed; nine control points were larger than four independent models |

All three were tested on small rotated Digits screens, so their quality findings do not generalize to language models or unseen tasks.

## Interpretation

An invertible coordinate change cannot increase the represented function set when the decoder composes it back to the same native alpha, barycentric, or Bezier coordinates. Coordinate labels alone therefore do not establish Mirror-specific capacity or quality gains. This is a structural consequence of the implementation, not evidence that learned manifolds or compressed codes are useless.

## Family disposition

Pause experiments whose only new operation is renaming a continuous native chart coordinate as Mirror m. Resume such a lane only when it adds a distinct operation, such as entropy-constrained quantization, task projection, symmetry alignment, or an off-manifold residual, and compares against the native operation with actual payload accounting.

Continue MA-860 as a distinct question: whether a new task can be fit inside an existing subspace using a compact coordinate, compared with random intrinsic subspaces, native subspace projection, and full fine-tuning. The experiment must not claim the m reparameterization itself as a gain.

## H / T / D / C / U

- **H:** Native-coordinate reparameterization with an invertible Mirror chart adds no functions or outputs beyond the native chart.
- **T:** Three independent application screens: learned line (MA-856), jointly trained simplex (MA-857), and quadratic Bezier surface (MA-858), each with fresh seeds and native-coordinate comparisons.
- **D:** **FAIL** for a Mirror-specific coordinate advantage in these screens; exact or float32-equivalent native outputs each time.
- **C:** Quality/storage failures may arise from small rotated Digits task design and training, not from the coordinate transform. This does not falsify useful learned manifolds or code compression.
- **U:** Larger natural tasks, entropy-coded coordinates, symmetry-aligned interpolation, and new-task projection remain distinct and untested here.
