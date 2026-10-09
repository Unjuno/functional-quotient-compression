# MA-156 status

- Status: PROMISING; secondary tied-quality gate missed
- Branch: `research/ma-156-compression-view-20261007`
- Base commit: `35f6bb0646d566011554222851e3b4473306ca57`
- Last verified commit: `c21413fbfd4c4280aee6103a711105f54359638c`
- Development complete: yes; seeds 15601–15602
- Fresh/audit opened: yes; seeds 15611–15613 after the independent-int4 development gates passed
- Results committed: yes (`c21413fbfd4c4280aee6103a711105f54359638c`)
- Verification committed: yes (`c21413fbfd4c4280aee6103a711105f54359638c`)
- Registry row updated: yes

## H / T / D / C / U

- **H:** One packed int4 base plus a small charged Mirror coordinate can represent four aligned logical matrices with activation quality close to independent int4 at much lower actual payload bytes.
- **T:** Four 16×16 matrices; independent int4, hard tied int4, Mirror int4, shared int4+rank-2 residual, independent QER rank-2, and fp32 upper control. Two dev plus three fresh seeds on aligned and independent teachers; no optimizer updates; exact packed serializer and Gaussian activation audit.
- **D:** PROMISING on storage/quality versus independent int4: quality/byte gates passed 3/3 fresh worlds, 187B vs 585B. The stricter preregistered tied-quality margin (≤0.10× hard-tie MSE) missed: observed 0.235×. Independent arbitrary weights needed private reconstruction.
- **C:** The teacher was generated from the same Givens orbit; this is a favorable mechanism test. Independent QER was more accurate at larger bytes, and Mirror's decode MAC proxy was 32× individual int4 dequantization.
- **U:** Larger and real Transformer weights, language-model quality, quantizer families/bit widths, optimized view kernels, and private residual allocation.

**FACT:** 60 rows replayed; exact serialized byte lengths; decode round-trip max tensor difference 0; tests 3/3 pass. Fresh Mirror activation MSE was within 1.01× independent int4 in every aligned world. Payload: Mirror 187B, independent int4 585B, tied int4 161B, shared rank-2 residual 745B, independent QER rank-2 1,159B, fp32 4,128B.

**INTERPRETATION:** The shared quantized physical object produces four useful logical functions on its view orbit and lowers storage relative to independent int4. It does not cover arbitrary layer matrices and increases reconstruction arithmetic.

**HYPOTHESIS:** A fused Givens decode or coordinate-specific low-cost representation could preserve the storage gain with a smaller compute penalty.

## Next action

Update the registry, claim ledger, status board, and queue; commit/push; then continue to MA-160.

## Blockers

None for the synthetic post-training screen.

## Decisions / rulings

Status remains PROMISING because the principal independent-int4 storage/quality frontier passed; the missed hard-tie threshold is explicitly recorded, so this is not a full gate PASS.
