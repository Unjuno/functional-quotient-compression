# MA-360 — Reversible shared block with Mirror depth views

Status: SCREENING; protocol frozen before development  
Branch: `research/ma-360-reversible-mirror-block-20261008`  
Base: `c935a90`  
Prior art: PA48 reversible PEFT.

## H — Hypothesis

A shared additive-coupling block with one small layer-specific Mirror coordinate can retain a multi-step synthetic task while reducing saved activation memory versus an ordinary shared tied block, at lower serialized weight bytes than untied blocks, with recomputation overhead below 2×.

## Mirror insertion

> **Mirror insertion:** this experiment adds a scalar coordinate `m_l` to each repeated application of one shared reversible coupling block so logical depth steps can differ without separate physical block weights.

Controls: untied coupling blocks, ordinary tied block, tied block plus direct layer scalar gates, reversible tied block, and reversible tied block plus Mirror scalar views. Mirror and direct scalar views use identical parameter counts and initialization.

## T — Frozen protocol

Small CPU PyTorch mechanism screen, 4 seeds: dev 36001/36002, fresh 36011/36012 locked. Dimension 32, depth 8, batch 64, synthetic regression from an untied additive-coupling teacher. 300 AdamW updates, lr=1e-3. Record task MSE, parameter archive bytes, peak live saved activation bytes excluding parameters, MAC proxy, train time and forward/backward time. The custom reversible backward reconstructs each prior coupling input from its output and recomputes local gradients. No CUDA device is present, so conclusions are CPU-scoped.

## Gates

PASS if both dev worlds: Mirror MSE within 5% of untied upper control, at least 30% lower activation bytes than ordinary tied unrolled execution, at least 10% fewer parameter bytes than untied blocks, and runtime <=2× non-reversible tied. Direct scalar gate must be at least 10% larger or worse quality for Mirror-specific claim. FAIL if quality misses, activation saving <30%, runtime >2×, or direct gate matches within 10%. Fresh remains sealed after any dev fail.

## Boundaries

Synthetic CPU screen only; no Transformer, long-sequence training, GPU throughput, or scaling evidence. Report recomputation cost separately from activation bytes and weight storage.

## Result — development

**FAIL for Mirror-specific value; reversible tied block is a narrow PROMISING memory point.** On both dev seeds, direct scalar gates and Mirror views had identical payload hashes and task MSE. They used 2,598/2,608B versus tied 2,388/2,398B. The reversible tied block used 2,385/2,401B, matched tied task MSE, and reduced peak live saved activations from 139,264B to 81,956B (41.1%). Training wall time was 1.28–1.33s vs 0.68–0.76s tied (~1.8×), within the frozen 2× bound. Untied models had lower MSE but ~19.1KB payload.

The first exploratory run used generic checkpointing rather than inverse reconstruction; it is retained in `artifacts/development.log` and excluded. Three tests checked inverse reconstruction, gradient agreement with direct autograd, and deterministic serialization. Fresh seeds remain sealed because the Mirror/direct gate failed.
