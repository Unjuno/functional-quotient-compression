# MA-274 — BOFT logical expert views

Status: SCREENING (protocol frozen)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA21 BOFT.

## H — falsifiable hypothesis

A single physical nonlinear expert plus one shared butterfly angle atom and a scalar code per logical expert can reproduce routed expert functions with less serialized state than independent BOFT experts, while beating hard tying, IA3, and rank-one BatchEnsemble controls on quality-per-byte.

## Mirror insertion

> **Mirror insertion:** this experiment adds an expert address `m_e` at the shared FFN hidden activation interface, scaling a shared butterfly orthogonal atom so four logical expert functions can be expressed without duplicating the FFN weights or a full BOFT table per expert.

- Physical object: shared 16x16 input/output FFN matrices.
- Mirror coordinate: one scalar code per expert multiplying a shared 4-stage x 8-pair butterfly angle atom.
- Native BOFT control: independent 32-angle table per expert.
- Simpler controls: hard tying, IA3 hidden gains, BatchEnsemble rank-one factors.
- Independent upper: four independent FFNs.
- Router: oracle expert IDs are supplied to all methods and are not learned; this isolates expert function recovery. Router cost is reported as metadata, and no routing-quality claim is made.

## Protocol

Four routed experts; input/hidden/output width 16; tanh FFN. Teacher functions share one base FFN and use expert butterfly transforms on the hidden activation, with expert coefficients generated from one shared angle atom. Development worlds 27400/27401; LR {.003,.01}, 1,000 updates, batch 128. Fresh 27402–27404 stay sealed unless dev passes. All models initialize independently; no teacher weights or angle codes are copied.

**PASS (screen only):** Mirror MSE <=1.10x independent, payload <=60% independent, and beats native BOFT and rank-one controls by >=10% MSE at byte-near payload (within 15%) or lower bytes. **FAIL:** misses the quality/byte gate or a simpler control matches at equal/lower bytes.

## Boundaries

Synthetic regression with oracle routes and a deliberately factorized BOFT teacher. This is not a learned-router MoE, language, or near-convergence fixed-byte capacity claim.

## D — FAIL (development screen)

The selector chose LR 0.01 by mean MSE over all methods/worlds. Mirror MSE was 0.0214 and 0.0449 in the two development worlds, versus native BOFT 0.0121 and 0.0358 and independent upper 0.0060 and 0.0040. Payload was 5,430 B for Mirror, 5,624 B for BOFT, 6,703 B for rank-one modulation, and 11,459 B for independent FFNs. Mirror therefore missed the independent-quality gate and was worse than BOFT for only 3.4% fewer bytes. It did beat rank-one in quality while using 19% fewer bytes. Fresh worlds 27402–27404 were not opened.

At the selected LR, Mirror active-compute proxy was 204.8M examples per training screen, below BOFT at 393.2M and independent at 524.3M; measured throughput was only 0.33–0.36M examples/s, far below IA3 (6.35–7.20M) and rank-one (3.58–4.32M). The Python-loop butterfly implementation dominates this eager CPU runtime result.

### C — strongest counter-hypothesis

The factorized scalar-code angle family is exactly aligned with the teacher construction, but joint optimization of the shared atom, codes and FFN may be poorly conditioned. This fixed-budget failure is not evidence of representational impossibility. The independent and native BOFT controls also use different parameterization/optimization paths.

### U — not established

Fresh replication, learned router/load balance, near-convergence frontier, fused runtime, and whether more private angle residuals rescue quality without erasing the byte margin remain untested.

### Fact / interpretation / hypothesis

- **Fact:** 24 development rows cover two worlds, two learning rates and six methods; teacher parameters were not copied to student initialization; oracle expert IDs were supplied equally to all methods. Actual payload bytes and runtime proxies are recorded.
- **Interpretation:** Mirror provides a better quality/byte point than rank-one factors on this task, but does not replace BOFT or independent experts under the preregistered gate.
- **Hypothesis:** The recurring limit may be the shared angle atom/scalar code's optimization and private-state budget; see `FAMILY_BOFT_MIRROR_DIAGNOSTIC.md`.
