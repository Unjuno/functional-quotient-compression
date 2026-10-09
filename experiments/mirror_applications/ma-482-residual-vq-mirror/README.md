# MA-482 — Residual-VQ Mirror code composition

Status: PROMISING, scoped synthetic rate-distortion/storage result. This does not establish general functional capacity or a Mirror-specific advantage.

## Hypothesis and method

H: For synthetic functions composed of known orthogonal low-dimensional residual views, one shared basis plus compact discrete phase addresses can match or beat native residual VQ in quality with fewer serialized inference bytes.

T: 128 vectors in 16 dimensions, four known orthogonal 2D planes with amplitudes `[1,.5,.25,.125]`, stage counts R={1,2,4}, and K={4,8,16,32}. Development worlds 48200-48201 trained native full-vector codebooks. Confirmatory A1 used fresh worlds 48220-48222 and seeds 0-2. Controls were dense vectors, native learned residual VQ, generic explicit coefficient codebooks, and Mirror's deterministic phase-grid decoder. Payloads were serialized with `torch.save`; reported bytes include codes, basis/metadata, scales, and any codebooks.

## Amendment and provenance

The first frozen runner accidentally accessed the synthetic teacher's generating phase when assigning Mirror and generic codes. Those A0 outputs remain in `artifacts/a0_results_exploratory.csv`, `artifacts/a0_fresh_runs_exploratory.jsonl`, and the old-world payloads, but are excluded from all claims. Before A1 fresh evaluation, PROTOCOL.json was amended: phase is now inferred only from the candidate target vector using `atan2`; no tuning or sweep changes were made. A first A1 runner attempt also wrote only the final method row due to an indentation error; it was discarded and rerun after correction. The authoritative A1 CSV contains 432 rows (4 methods × 4 K × 3 stages × 3 worlds × 3 seeds).

## Results

At R=4,K=32 across 9 fresh world/seed cases:

| Method | Mean payload bytes | Mean normalized error |
|---|---:|---:|
| Native learned RVQ | 10,597 | 0.13276 |
| Generic explicit coefficient table | 3,545 | 0.05780 |
| Mirror phase grid | 3,037 | 0.05780 |

Mirror used 28.7% of native RVQ bytes at this point and had lower error under the fixed development codebook budget. Generic coefficients decode exactly the same functions and cost 16.8% more than Mirror, so the distinguishing result is compact deterministic representation of this known teacher structure, not a new function family. Additional stages improve fresh reconstruction error under ablation, while increasing bytes. Full stage/K curves and measured encode/decode timing are in RESULTS_CORE.csv.

## H / T / D / C / U

- H: Shared plane plus stage phase addresses can match native residual-VQ quality at fewer actual bytes on the registered synthetic family.
- T: Frozen sweep, development-only codebook fitting, 3 fresh worlds × 3 seeds, four storage controls, actual serialized payloads, and stage ablations. A1 has no teacher-phase access.
- D: PROMISING, scoped to synthetic orthogonal phase components and this finite learned-RVQ budget.
- C: Known basis, amplitudes, and phase-grid structure make this a highly favorable transform-coded dataset; generic explicit coefficients match quality.
- U: Natural or learned functions, unknown/nonorthogonal bases, equal-byte matched optimization, and deployed throughput remain untested.

## Reproduction

```bash
python experiments/mirror_applications/ma-482-residual-vq-mirror/source/run.py --phase development
python experiments/mirror_applications/ma-482-residual-vq-mirror/source/run.py --phase fresh
python -m unittest discover -s experiments/mirror_applications/ma-482-residual-vq-mirror/tests
```
