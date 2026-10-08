# MA-361 — Mixture-of-Depths with Mirror block-role views

Status: SCREENING; frozen before development  
Branch: `research/ma-361-mixture-depths-mirror-role-20261008`  
Base: `c935a90`  
Prior art: PA49 Mixture-of-Depths.

## H — Hypothesis

With the same token routing capacity, a shared block equipped with small role coordinates can preserve token prediction quality while using fewer physical block parameters than ordinary Mixture-of-Depths blocks, without increasing active FLOPs beyond 10%.

## Mirror insertion

> **Mirror insertion:** this experiment adds a compact block-role coordinate to the shared residual MLP that selected tokens traverse, so a smaller physical block bank can provide multiple logical depth roles.

Native control: Mixture-of-Depths top-capacity token routing through separate depth blocks. Candidate shares one block across routed roles and adds one scalar coordinate per role. Direct FiLM/scalar-gate control has identical coordinates and routing.

## T — Frozen protocol

Synthetic tokenwise four-class task, 16 tokens per example, width 16, 3 candidate depth roles, top-50% routing capacity per MoD layer, 1,024 train and 512 test sequences per world; 1,000 AdamW updates, batch 64. Development seeds 36101/36102; fresh 36111/36112/36113 locked. All methods use the same fixed input-derived router and routed token identities. Compare dense 3-block network, ordinary MoD with 3 role-specific blocks, tied MoD with one shared block, tied MoD plus direct scalar gates, tied MoD plus Mirror role coordinates. Measure token NLL, per-token-position accuracy, actual serialized weight bytes, active MACs after routing, routing capacity/overflow, and latency. This is fixed-budget mechanism evidence, not capacity evidence.

## Gates

PASS both development worlds if Mirror NLL is within 0.01 of ordinary MoD, at least 10% lower weight bytes than MoD, active MACs <=1.10× MoD, and direct gate control is at least 10% larger or worse quality for Mirror-specific value. FAIL if quality misses by >0.03, active MACs >1.10×, or direct gate is within 10%. Fresh remains sealed after dev fail.

## Boundaries

Synthetic token classification only. Router is fixed and identical; this isolates block-role parameterization, not learned routing. No language model or token-dependent natural sequence claim.

## Result — development

**FAIL for Mirror-specific value and frozen quality margin; fresh sealed.** At the same top-50% routing budget, Mirror tied MoD used 5,298/5,312B versus native MoD 11,076/11,077B, and all models routed exactly 8 tokens/sequence with equal active-MAC proxy. However, direct-gate control had identical payload hashes, NLL and accuracy to Mirror in both worlds. Mirror NLL was 0.1001/0.1150 versus native 0.0952/0.1023; seed 36102 missed the frozen 0.01 NLL margin. Fresh 36111–36113 remained sealed.

**Fact:** routing identities/capacity were fixed; Mirror and direct scalar gates were exactly the same implementation state/function.  
**Interpretation:** parameter sharing reduced archive bytes relative to independent MoD blocks, but the gain is ordinary shared-block/gate compression, not Mirror-specific; quality slipped in one seed.  
**C:** this simple tokenwise task and fixed router may not expose block-role benefits under learned routing.  
**U:** no trained MoD router, language model, near-convergence capacity study, or fresh replication.

## Result — development

**FAIL for Mirror-specific value and frozen quality margin; fresh sealed.** At the same top-50% routing budget, Mirror tied MoD used 5,298/5,312B versus native MoD 11,076/11,077B, and all models routed exactly 8 tokens/sequence with equal active-MAC proxy. However, direct-gate control had identical payload hashes, NLL and accuracy to Mirror in both worlds. Mirror NLL was 0.1001/0.1150 versus native 0.0952/0.1023; seed 36102 missed the frozen 0.01 NLL margin. Fresh 36111–36113 remained sealed.

**Fact:** routing identities/capacity were fixed; Mirror and direct scalar gates were exactly the same implementation state/function.  
**Interpretation:** parameter sharing reduced archive bytes relative to independent MoD blocks, but the gain is ordinary shared-block/gate compression, not Mirror-specific; quality slipped in one seed.  
**C:** this simple tokenwise task and fixed router may not expose block-role benefits under learned routing.  
**U:** no trained MoD router, language model, near-convergence capacity study, or fresh replication.
