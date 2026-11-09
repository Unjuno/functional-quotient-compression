# MA-755 — Ctrl-Adapter source × backbone bridge views

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

On three held-out source-control/backbone pairs, independent full bridges have zero error at 3,341 B and 192 MACs. A learned two-tower bridge reaches relative operator nRMSE `4.91e-5` at 1,308 B (60.9% less) and 96 MACs (50% less), after 2,500 Adam updates and about 1.80 s fit time. Source-only and backbone-only controls have held-out errors 1.236 and 1.211. The native two-tower control and Mirror have identical payload hash/bytes, error and MACs.

## Interpretation

The synthetic source/backbone mappings are compositional by construction, and a learned low-rank two-tower bridge interpolates the withheld pairings with substantially lower storage and active operations than independent full bridges. This is evidence for ordinary cross-domain factorized bridges on the designed task. It is not Mirror-specific: the native control exactly reproduces the function and payload. Training cost is also nonzero and is not comparable to the oracle independent upper control's zero-update construction.

## Hypothesis result

**FAIL** for Mirror-specific value, despite a favorable synthetic bridge storage/compute signal. Fresh/audit was not opened after exact native alias.

## Frozen protocol / scope

PA193 motivates reusing pretrained control networks through compact bridges into multiple target backbones. Here, 3 source controls and 4 target backbones map 8D features through a shared 2D latent. Three source/backbone pairs are held out while the other nine full teacher operators train source encoders and backbone decoders using Adam (2,500 updates, lr .025). We compare independent full pair matrices, source-only and backbone-only averages, native learned two-tower bridge and Mirror Views. Actual NPZ bytes, held-out operator error, active MAC proxy and fit/evaluation wall time are reported.

The teacher is generated from the same two-tower factorization used by the candidate, so this is an aligned mechanism screen. It does not use pretrained ControlNets, diffusion backbones, images or video. No accelerator runtime, zero-shot transfer to real models, or capacity claim is made.

## H / T / D / C / U

- **H:** Source-control and target-backbone coordinates support held-out bridge transfer with <=.02 error and >=25% fewer bytes/MACs than independent pair bridges.
- **T:** Seed 75501; 3×4 pair grid; 8D input/output; latent 2; 9 train pairs; 3 heldout pairs; Adam 2,500 updates; five controls.
- **D:** **FAIL** for Mirror specificity. Native two-tower reaches `4.91e-5` error at 1,308 B / 96 MACs vs independent 3,341 B / 192 MACs; Mirror exactly aliases native two-tower.
- **C:** The teacher is compositional by construction; ordinary two-tower factorization explains all gains over independent bridges.
- **U:** Real pretrained control/backbone pairs, trained model fidelity, distribution-shifted source/backbone combinations, end-to-end image/video quality and accelerator cost.
