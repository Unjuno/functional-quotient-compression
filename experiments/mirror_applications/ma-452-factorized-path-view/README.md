# MA-452 — Factorized PathNet path × Mirror role address

Status: SCREENING  
Evidence lane: GENERALIZATION/STORAGE/RUNTIME  
Branch: `research/ma-452-factorized-path-view-20261008`  
Base commit: `085e918`  
Prior art: PA80 (PathNet), PA81 (Routing Networks)

## Hypothesis

H: Separate path identity from role address so a compact factorized Mirror codebook can define useful functions at path-role pairs never seen together during meta-training. Direct native conditioning and additive factorized controls test attribution.

**Mirror insertion:** this experiment attaches a learned two-angle role codebook independently of the selected PathNet path, allowing a shared module path and a role never jointly observed during training to compose into a new function.

## Frozen split and controls

There are four two-layer paths and three role addresses. The eight train pairs cover every path and every role, while four path-role pairs are withheld. Conditions are PathNet path-only, factorized Givens Mirror, rank-2 additive basis × role code, full role-vector table, native direct Givens conditioning and independent held-out task fits. The measured task is a small aligned 4D regression mechanism, not a full PathNet or Routing Network reproduction.

Two development worlds (45201, 45202) and the training/evaluation schedule are frozen in `PROTOCOL.json`. Fresh worlds 45211–45213 remain sealed unless all success gates pass. Every module, code, route/role ID, basis and schema byte is charged in actual uncompressed `.npz` inference payloads.

## Results

Facts, interpretation and hypothesis will be recorded after the frozen development run and serialized-payload replay.
