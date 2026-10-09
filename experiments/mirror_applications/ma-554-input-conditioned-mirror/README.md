# MA-554 — Input-conditioned Mirror modulation network

Status: **FAIL for Mirror-specific value**  
Branch: `research/ma-554-input-conditioned-mirror-20261009`  
Base commit: `93600071`  
Prior art: PA106 FiLM; PA122 Dynamic Filter Networks

## H

A low-description context-to-Mirror-code generator can reproduce useful sample-conditioned functions with less inference payload and compute than FiLM and generic dynamic-filter generation, while retaining held-out-context quality.

## Frozen screen

The target is a context-conditioned 8×8 operator on Gaussian inputs. A fixed orthogonal basis defines a smooth one-angle rotation orbit; context is two-dimensional and the angle is a fixed smooth function of context. Compare shared static operator, angle-coded Mirror view, direct two-basis coefficient generator, context-generated diagonal FiLM, and generic full-matrix linear dynamic filter. Fit each generator on fixed training contexts and evaluate on held-out contexts. Two seeds; actual deterministic NPZ payloads include shared maps, generator weights, basis, and metadata. Report output nMSE, bytes, operator application and generator MAC proxy, wall time. This is a dev mechanism screen; no fresh contexts are opened after an exact direct-control match.

PASS requires Mirror output nMSE ≤1e-4 and ≥10% fewer bytes than both FiLM and generic dynamic filter, while beating direct two-basis synthesis by ≥10% at equal quality. FAIL if direct coefficients match within 1% bytes/quality or quality fails.

## H / T / D / C / U

- **H:** Input-conditioned low-description views improve quality/byte/compute over existing dynamic generators.
- **T:** Two development seeds, 256 train and 256 held-out contexts/world; 10 result rows.
- **D:** FAIL for Mirror-specific value. Mirror/direct coefficient both 973B and exact outputs (nMSE 0); dynamic filter 1268B/.01958, FiLM 1148B/.21959, static 718B/1.49663.
- **C:** Direct coefficient generation, FiLM, or dynamic filters may match or dominate.
- **U:** Natural inputs, trained Transformer, end-to-end latency on accelerator, adaptation cost.
