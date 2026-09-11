# C4 scale replication: 8M + 1M frozen-protocol verdict (28M Gates 1–7 follow-up)

Branch: `experiment/c4-scale-replication-8m-1m` (base `65af860`).
Lane code = 28M Gates 1–7 code parameterized only
(`--model-dir/--tag/--paid-scalars/--checkpoint-sha/--backbone-*/--skip/--only-cand`);
no method change. Splits identical: calibration [0,64), dev [64,320),
audit [320,1344) on `TinyStories-valid.txt` (SHA verified match).
Metrics implementation unchanged (4000-resample story bootstrap, seed 266272).

## H

The 28M FAIL is not checkpoint/scale-specific: on 8M and 1M, functional
sharing / role-shared codebooks / selected private exceptions do not improve
the matched-byte rate–distortion frontier over the strong activation-aware
non-sharing control in any viable-quality regime.

## T

| | 1M | 8M | 28M |
|---|---|---|---|
| repo@rev | roneneldan/TinyStories-1M@77f1b16 | ...-8M@8612e3b | ...-28M@52dabea |
| ckpt SHA256 match to T266 | True (07f9609…) | True (22c355…) | True (8ddd260…) |
| paid scalars | 3745984 | 19702528 | 51987968 |
| tensors / tied | 108 / yes | 108 / yes | 108 / yes |
| Gate A parity (HF GPT-Neo, FP32 CPU) | 0.0 bit-exact | 0.0 bit-exact | 0.0 bit-exact |
| audit reference NLL (1024) | 2.3391 | 1.6074 | 1.4431 |
| backend | CPU (ref) | CPU (ref) | CPU (ref) |
| tests | 275 pass (265+10 new) | same | 265 |

## D

### 8M — PASS-C4

8M mirrors 28M: no viable-regime sharing/private win.
- B-meta vs act_b4_g128 (matched ~10.51 MB): 1.9757 vs 1.9169, CIs
  non-overlapping → FAIL replicates.
- shareK256_B8 vs vqK256_B8: 7.4460 vs 7.7285 → directional signal replicates
  inside collapse (both PPL 1700+).
- BC_P256 vs backbone: 7.4480 vs 7.4460 → private flat replicates.
- Dev→audit orderings replicate.

### 1M — PASS-C4

1M replicates the general FAIL and extends it:
- Uniform frontier viable only at b8 (2.3442, dNLL CI [0.0042,0.0055]); b6
  shows a tiny rtn<act inversion (recorded, baseline not switched).
- shareK64_B32 vs vqK64_B32: 13.9867 vs 11.3641 → sharing significantly
  WORSE (sign reversed vs 28M/8M), CIs non-overlapping.
- Private monotonically harmful on both backbones
  (BC: 14.06→18.02; AC: 11.43→15.77 across S=0→1024).
- K256-VQ and B-meta structurally impossible on 1M (recorded exclusions:
  K=256 exceeds per-tensor sample counts; cols=64 < supergroup=128).

### Cross-scale — PASS-C4

The FAIL generalizes across the three tested checkpoints; C4-A/C4-B (small-scale
viable sharing gain) are NOT supported anywhere. The collapsed-regime sharing
micro-signal is scale-dependent in sign (28M −0.52, 8M −0.28, 1M +2.62) and
never viable — no scaling law is claimed (one checkpoint per scale).

## C

- C4-A (8M/1M harbor sharing gain): REJECTED (no viable gain at either scale).
- C4-B (scale-dependent viable gain): REJECTED (nowhere viable).
- C4-C (VQ too weak to show viable signal): OPEN — consistent with sec-21
  "stronger base" next lane; not tested here by design.
- C4-D (checkpoint-idiosyncratic): OPEN — one checkpoint per scale; direction
  magnitudes differ, but the viable-regime FAIL is unanimous.
- C1/C2 (28M): corroborated — act-only frontier dominates on all scales.

## U

- Single checkpoint per scale; no training replicas.
- 1M viable regime ≈ b8 only; K256/B-meta untestable there (structural).
- CPU FP32 batch-1 eval; 1024-story audits; same torch env as 28M lane.
- No GPTQ/AWQ/AQLM comparison (out of scope).

## Scale table (audit NLL unless marked *dev; bytes actual)

| row | 1M (3.75M paid) | 8M (19.7M paid) | 28M (52.0M paid) |
|---|---|---|---|
| Control act_b4_g64 | 3.9647 @2.13MB 3.52x | 1.8295 @11.13MB 3.54x | 1.5020 @29.32MB 3.55x |
| Control act_b8_g64 (near-lossless) | 2.3442 @4.00MB 1.87x | 1.6085 @20.97MB 1.88x | 1.4436 @55.29MB 1.88x |
| Role sharing (audit) | 13.9867 vs 11.3641 (K64B32, WORSE) | 7.4460 vs 7.7285 (K256B8, direction ok, collapsed) | 7.2831 vs 7.7992 (K256B8, direction ok, collapsed) |
| Private S64 (*dev) | BC 14.41 / AC 11.70 (harm) | BC 7.467 / AC 7.638 (flat-harm) | BC 7.226 / AC 7.810 (flat-harm) |
| Private S256 (*dev) | BC 14.77 / AC 12.25 (harm) | BC 7.471 / AC 7.683 (flat-harm) | BC 7.226 / AC 7.813 (flat-harm) |
| Private S1024 (*dev) | BC 18.02 / AC 15.77 (worse) | BC 7.538 / AC 7.909 (worse) | BC 7.233 / AC 8.289 (worse) |
| Tensor max|cos| | <=0.062 | <=0.041 | <=0.043 |
| Best viable rate (dNLL<0.1) | b8 1.87x | b8 1.88x | b8 1.88x |
| Sharing ΔNLL / Δbytes | +2.6225 / −60446 B | −0.2825 / −92791 B | −0.5161 / −92748 B |
| Verdict | PASS-C4 | PASS-C4 | (prior FAIL) |

## Measured facts

- `runs/c4-8m-eval/{dev,audit}_FRONTIER.csv`, `runs/c4-1m-eval/...` (per-story
  JSONs alongside; git-excluded, hashes in manifests).
- `runs/c4-manifest/`: preflight, input_provenance, parity_8M/1M, freeze_8M/1M,
  verdict_8M/1M.
- Canonical copies: `experiments/c4-scale/` (this file + CSVs + manifests).

## Inference

The tested sharing family shows no viable-regime gain on any of the three
checkpoints; private support is flat-to-harmful everywhere (steepening at 1M);
the only sharing-direction signal lives in collapse and flips sign at 1M.

## Unsupported claims

No scaling law; no "FQC refuted in general" (only the tested family × three
checkpoints); no viable-rate sharing statement (untestable with this VQ base).

## Next gate (sec 22 trigger met: 1+2+3 hold)

Retire the tested cross-tensor/codebook sharing family from the main line.
Next central hypothesis: task-null / locally quotientable degrees of freedom
— with a viable-rate stronger base (additive/residual quantization or
GPTQ-style compensation) as a separate lane prerequisite (sec 21).
