# MA-526 — SAE feature atoms for function-vector interventions

Status: SCREENING; protocol fixed before implementation. Branch: `research/ma-526-sae-feature-mirror-atoms-20261009`.
Prior art: PA102, *SAEs Are Good for Steering — If You Select the Right Features*.

## H — Hypothesis

A 64-atom SAE pool selected from fit-task vectors and eight sparse signed coefficients per function will preserve held-out FV task effect, reduce incremental code bytes, and beat native SAE top-eight and global OMP-8 controls.

## Prior-art delta

PA102 establishes SAE feature steering and that feature selection affects side effects. This experiment measures whether one shared SAE subdictionary can encode many support-derived function interventions. It includes native SAE activations and standard global sparse coding as direct controls. SAE bytes are charged for standalone deployment.

## T — Frozen protocol

See `PROTOCOL.json` and `freeze.json`. Use Pythia-70m layer-3 residual SAE from pinned `Elriggs/pythia-70M-deduped-sae` revision `36c2027509efad69836aa4231999c9b717848ecd`; file SHA-256 `85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4` (4,204,391 B). Reuse MA-516 function-vector tasks/splits. Tasks 0–11 choose the shared 64-feature pool; tasks 12–15 are held out. Encode with eight-step signed pursuit. Compare no intervention, explicit FVs, native SAE top-eight positive activations, global OMP-8 and shared-pool OMP-8. Development seeds 52601/52602; fresh 52611–52613 locked.

Report actual uncompressed NPZ codes and total standalone Pythia+SAE deployment bytes, as well as the incremental state on top of an SAE already resident for other uses. The SAE dictionary is not free in standalone accounting.

## D — Decision

Pending development.

## C — Strongest counter-hypothesis

Sparse coding over SAE atoms is already a native method. A selected shared pool may lose function-relevant directions; if it does not, OMP/top-k explains the representation without Mirror-specific value. The 4.2 MB SAE can dominate deployment storage.

## U — Boundaries

This is a four-task held-out intervention screen on one 70M model and one SAE checkpoint, not a broad SAE steering benchmark.
