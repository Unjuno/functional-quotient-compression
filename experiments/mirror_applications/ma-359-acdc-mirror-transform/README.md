# MA-359 — ACDC structured Mirror transform

Status: **FAIL for Mirror-specific value; fresh sealed**  
Branch: `research/ma-359-acdc-mirror-transform-20261009`
Base: `c935a90`  
Prior art: PA45 ACDC structured transforms.

## H — Hypothesis

Two Mirror coordinates over shared ACDC diagonal basis factors can preserve held-out map quality while reducing serialized bytes and transform compute versus independent ACDC factors, dense maps, and direct coefficient controls.

## Mirror insertion

> **Mirror insertion:** this experiment adds two low-description coefficients to the diagonal factors surrounding a fixed DCT transform so one shared transform basis expresses many task-specific linear operators.

## T — Test

For each development seed, generated 64 dimension-32 operators of form `W_task=diag(d0+alpha_task*d1) Q diag(e0+beta_task*e1)`, with fixed orthonormal DCT-II Q. Compared dense independent matrices, independent ACDC diagonal pairs, shared ACDC bases with Mirror coefficients, direct two-coefficient control using the same bases, and diagonal-only maps. Measured 512 Gaussian probes per task; 48 tasks designated train and 16 held out. Development seeds 35901/35902; fresh 35911–35913 sealed. Basis values were oracle supplied; no optimization was performed. Actual deterministic ZIP/NPY bytes were measured after reload.

## D — Decision

**FAIL for Mirror-specific value; fresh sealed.** Shared ACDC/Mirror reconstructed all 64 maps exactly at 2,077–2,079B, compared with 11,131–11,137B independent ACDC and ~240KB dense maps. However, direct two-coefficient control was only 8B larger at equal quality. Both shared and independent ACDC use the same theoretical FFT MAC proxy (12.58M across the evaluation bank); dense costs 33.55M and diagonal-only 1.05M but had nMSE about 2.0. No Mirror-specific compute gain was measured.

## Fact / Interpretation / Hypothesis

**Fact:** Shared ACDC basis state represented 64 distinct operators with zero held-out nMSE at ~2.08KB; independent ACDC used ~11.1KB. Direct coefficients differed by 8B only. Dense maps used ~240KB; diagonal-only quality failed.

**Interpretation:** The storage decrease comes from sharing ACDC factors across an aligned task bank. Ordinary coefficients recover the same operator family and match total bytes. Fixed DCT mixing was useful compared with diagonal-only, but is attributable to ACDC structure.

**Hypothesis:** A learned non-aligned bank with paid residuals could test the private-state boundary, but this aligned oracle result does not motivate a Mirror-specific claim.

## C — Strongest counter-hypothesis

The task bank was generated directly from the shared ACDC parameterization with known basis and coefficients; it is an exact-fit oracle setting. It does not show that ACDC can discover the basis or unseen task functions from data.

## U — Unconfirmed

No learned ACDC transform, natural data, Transformer task, actual FFT kernel timing, noisy/unrelated maps, or fresh replication.
