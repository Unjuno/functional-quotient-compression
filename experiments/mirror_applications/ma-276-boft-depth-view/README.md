# MA-276 — BOFT depth views over a tied block

Status: SCREENING  
Evidence lane: MECHANISM / DEPTH / STORAGE / RUNTIME  
Base commit: `19f0b5a`

## H — hypothesis

One physical nonlinear/linear block plus a single butterfly-angle coordinate per depth step can recover a family of distinct logical depth operators with less state than untied blocks. The view must beat static per-step LoRA and generated shared-basis modulation on a useful quality/byte/compute frontier; unrelated operators should locate the need for private residuals.

## Selection

The eligible pool was `[MA-274, MA-276, MA-278, MA-282, MA-286, MA-296, MA-299]`. `secrets.randbelow(7)` returned index 1, selecting MA-276. Remote branch check found no MA-276 branch.

## T — protocol

The CPU screen uses a tied 8×8 linear block across four depth steps. Each aligned teacher layer is a Givens-butterfly conjugation `B(m_l) W B(m_l)^T`, where a single scalar `m_l` controls all 12 pair rotations in the three butterfly stages. The independent condition uses unrelated layer matrices. We measure each layer output and the composed four-step map on held-out Gaussian inputs.

Controls: hard tying, scalar per-step gate, static rank-2 per-step LoRA, input-conditioned generated shared residual basis, independent per-depth BOFT factors, and untied full matrices. All physical weights, butterfly addresses, residual bases, depth codes, tensor metadata and headers are included in deterministic serialized inference payloads. The exact serialized-state operators drive quality and runtime.

Development seeds 27601/27602 selected generated residual-basis rank 4 (mean composed MSE 0.014618 vs 0.014648 at rank 2 and 0.014699 at rank 1). Fresh seeds 27611–27613 use the locked rank. Initial timings that regenerated the BOFT matrices per query are retained as invalidated diagnostics; the corrected path prepares each fixed depth operator once, then applies it input-side.

## Gates

PASS for the aligned screen if all fresh worlds have composed-map MSE <=1e-5, Mirror actual payload <=0.5× untied full blocks, and coordinate-path throughput >=0.5× generated shared-residual control. Mirror-specific value additionally requires a better quality/byte/compute frontier than static per-step LoRA and generated modulation. Independent layers locate the private-parameter boundary. This is a fixed linear operator screen, not Transformer language-model or capacity evidence.

## D — decision

Pending development and fresh results.

## Fact / interpretation / hypothesis

**Fact:** pending.  
**Interpretation:** limited to this butterfly-conjugated linear block family.  
**Hypothesis:** one scalar phase per depth may span a useful tied-block orbit, while arbitrary layer maps need residual state.

## C — strongest counter-hypothesis

The aligned teacher is constructed directly from the chosen butterfly coordinate; the result may show only that the transform reproduces its own generator. Static or generated low-rank controls may reach the same function with lower runtime.

## U — unresolved

No nonlinear block, optimizer training, learned depth controller, natural text, long-sequence memory, or GPU fused butterfly kernel is tested. The prior MA-247 Givens-depth failure remains relevant evidence but does not substitute for this structured BOFT screen.
