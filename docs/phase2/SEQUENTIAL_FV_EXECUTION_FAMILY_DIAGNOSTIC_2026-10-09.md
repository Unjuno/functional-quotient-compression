# Sequential function-vector execution family diagnostic — 2026-10-09

## Decision

Pause the unchanged support-extracted FV / sequential executor lane (MA-541 through MA-544) pending a material redesign. Continue with MA-545, which tests a different activation-space expert-routing insertion rather than another sequential operator-code executor.

## Evidence kept separate

### Facts

- **MA-539** used a learned support-pair encoder, generic learned packet latent, PTP-style slot codes, and a charged lookup upper on arbitrary 32-state permutations. On two development worlds, every learned method had 0/96 exact packets. FV token NLL was lower than generic latent in the selected development setting, but FV payload was 23,784 B versus 21,094 B generic latent. The exact table upper was 100% at 1,052 B. Fresh stayed sealed. See the verified [MA-539 report](../../experiments/mirror_applications/ma-539-packet-function-vector/README.md).
- **MA-540** used a learnable family of eight random affine bijections over 16 states, two-step endpoint-only training, and four held-out ordered pairs. On both dev worlds, support-extracted FV tied recurrence and same-width native operator-code tied recurrence both reached 100% atomic accuracy, held-out pair accuracy, and intermediate-valid paths. FV payload was 27,441 B versus 23,865 B native (1.150x); FV used 41,984 versus 9,216 MAC/packet. A one-shot FV decoder without explicit state passing reached 9.4% and 28.1% exact pair accuracy. Fresh stayed sealed because the byte gate failed. Deterministic replay matched payload hashes and non-timing metrics.
- **SRM003** found poor direct composition in its causal decoder and 100% accuracy in an external two-call counterfactual on its tested models. The counterfactual used extra execution calls and supplied order; it was not a one-forward result.

### Interpretation

The results are consistent with explicit state passing being a useful execution interface in these synthetic tasks. They do not show a benefit specific to function vectors: in MA-539 the generic latent was at least as useful and smaller, while in MA-540 ordinary native operator embeddings matched FV quality with lower storage and compute. MA-540's one-shot control supports the ordered state-passing mechanism, not a claim about FV-specific capacity.

### Hypothesis for the family pause

The repeated attribution problem is that a support-derived FV behaves like an ordinary operator/address code, while its extractor and support state add cost. Another depth coordinate or a private residual on the same executor is unlikely to answer this without a different functional insertion or a task where native identifiers cannot explain the effect.

## Scope

- Defer MA-541 (depth-specific FV composition), MA-542 (attention-head FV compression), MA-543 (dynamic FV state), and MA-544 (FV plus sparse private residual) until a new protocol identifies a falsifiable advantage over ordinary recurrent codes, timestep embeddings, direct head controls, or native private residuals.
- Do not infer that all representation-space Views fail. MA-545 has a separate hypothesis: route among activation-space function vectors as logical experts and compare against weight-space Mirror-MoE and ordinary activation interventions.
- Continue to report actual serialized bytes, state-passing compute, and simple native controls. Keep fresh worlds sealed on every failed development gate.

## Next

MA-545 — function-vector mixture-of-experts without weight experts (P0; PA99). It is the next distinct candidate after this family pause.
