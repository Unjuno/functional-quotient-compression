# MSR001 — Mirror-code self-routing: CPU experimental result

**Status: COMPLETED SYNTHETIC EXPERIMENT; STRICT MIRROR-TIED ROUTER GATE FAIL.**

[Japanese research report and hypotheses](../../../../docs/phase2/MSR001_MIRROR_ROUTING_RESULTS_2026-10-17.md)

## Mirror insertion

A shared 8→16→8 SwiGLU yields four logical expert views from gate/up Givens coordinates. The candidate uses **the same gate Mirror code** to construct the router prototype and the expert function. Compare it with a separate small linear router, an independently learned prototype router, the equal-code-count native scalar gain, a hard shared expert, and four untied experts. The model is soft-gradient/hard-Top1 but **computes all 4 experts** in its eager CPU forward implementation; it is *not physically sparse*.

## Completed experiments

- Development: 63 fits, world 99200, 400 steps, selected common learning rate **0.015** solely on validation.
- Primary: 210 fits; worlds **99311–99315**, 2 seeds, 3 teacher families, 7 methods, **1,000 steps**, batch64, role label CE0.15.
- Separately prelocked follow-up: 96 fits, **new worlds 99421–99423**, 2 seeds, 2 teacher families, 4 methods, 4,000 steps, with role CE0.15 versus CE0.0. Measure both 1,000- and 4,000-step checkpoints.

## Main result (Givens-aligned teacher, 5 worlds; world median of paired seed means)

| method | test NMSE ↓ | params | saved bytes |
|---|---:|---:|---:|
| shared | 0.3198 | 421 | 4689 |
| mirror_linear | **0.1460** | 485 | 5321 |
| mirror_tied | 0.2056 | 449 | 4689 |
| gain_tied | 0.2152 | 449 | 4689 |
| independent | **0.0872** | 1573 | 9297 |

The preregistered tied-router mean/worst/route/storage joint gate passed **0/5** aligned worlds. Coupled Mirror is smaller by 632 serialized bytes than decoupled mirror_linear but has **1.405×** paired geometric-mean NMSE (bootstrap descriptive 95% interval 1.235–1.599). The independent-route and unrelated-function generalization gates were **0/5** each.

With 4,000 supervised updates on new worlds, coupled Mirror beat same-count gain-tied in **3/3** aligned worlds but **still lost to mirror_linear in 3/3**. With no teacher role CE, the coupled model's role assignment and NMSE were worse; report permutation-aligned role metrics, not just raw task IDs. No training run experienced global expert occupancy <5% under the primary supervised protocol.

## Verification

14 tests passed. All 63 dev, 210 fresh and 192 new-world follow-up checkpoint payloads (=465 original saved states) passed SHA256 and independent NumPy FP64 output/metric checks. Max discrepancy: primary 1.415e-7, follow-up 5.131e-8. Three complete primary retrains and two follow-up retrains matched saved bytes and checkpoints. Fifteen teacher class-oracle representations matched exactly.

**Scientific limits:** Synthetic teachers (one deliberately Mirror-aligned); explicit teacher role labels for primary; 3–5 independent worlds; eager CPU forward evaluates all experts; not natural-language/real sparse MoE, no GPU/optimized kernel benchmark, no general training-compute/quality advantage. No previously established MA status changes.

## Reproduction & archive

Complete tested source, tests, original frozen protocols, reports and analytic tools: **MSR001_20261017_SOURCE.zip** in this ChatGPT conversation; SHA256 `dd936d6d21a88b11b945591427d2e9a4616547a679d2b7e8ecdfbac73d5d2239`.

All original run records, SHA256-verified serialized .pt model states and logs: **MSR001_20261017_FULL_EVIDENCE.zip**, SHA256 `494c2e7167d38ca071938a510c5c6e43b7c29e77c231f8ef8d7ea1344cd7234a`. The GitHub result summaries are not the full original experiment input data or all 465 state payloads. Sources and original immutable locks are available in the ZIP; do not claim scripts are in this GitHub directory unless separately uploaded.

**Next:** partially-tied routing (Mirror geometry score + rank-1/2 correction), gradient separation, and real independently learned FFN expert routing. **main and canonical MA registry unchanged.**
