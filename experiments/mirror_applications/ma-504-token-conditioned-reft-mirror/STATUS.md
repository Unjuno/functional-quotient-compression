# MA-504 status

- Status: PROMISING (Givens-aligned continuous-context synthetic task only)
- Branch: `research/ma-504-token-conditioned-reft-mirror-20261009`
- Base commit: `e87c50a3`
- Protocol frozen before development: yes (`f525b439`)
- Development complete: yes; LR 0.01 selected on validation intervention NRMSE
- Fresh/audit opened: yes; 3 worlds × 3 seeds, 45 method rows
- Results committed: yes (`92d76da8`)
- Verification committed: yes
- Registry row updated: yes after verification; artifact replay test passed (1/1)

## Decision

**H:** Token-conditioned Mirror ReFT can express continuous context-specific activation interventions with less state than FiLM/generic generators.

**T:** Synthetic 64D activations, continuous 8D contexts, shared rank-4 basis, exact Givens teacher; shared, linear LoReFT, Mirror, generic MLP LoReFT, FiLM; 1,000 updates; 3 fresh worlds × 3 seeds.

**D:** PROMISING within the aligned synthetic family. Mirror intervention NRMSE mean 4e-6 (maximum 7.704e-6), payload 4,257B vs generic MLP 13,401B and FiLM 20,381B. MAC proxy 536 vs 2,688/4,416; measured CPU throughput 3.52M vs 3.86M/2.68M tokens/s.

**C:** Teacher is generated from the exact Mirror Givens circuit; generic MLP is evaluated at fixed updates and may be undertrained. This is not a capacity or natural-language claim.

**U:** Natural/pretrained representations, near-convergence/equal-byte generic controls, and GPU runtime are untested.

## Next action

Continue to MA-508 on its own research branch.

## Blockers

None.

## Decisions / rulings

The shared no-op baseline has zero trainable parameters and therefore zero optimizer updates; this is stated explicitly in its result rows.
