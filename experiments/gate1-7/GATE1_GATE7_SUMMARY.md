# Gates 1–7: matched-byte FQC sharing ablation on official validation (28M)

Branch: `experiment/gate1-official-validation`. HEAD base: `41e0844`.
All bytes below are ACTUAL final serialized file sizes (header+payload+padding+checksum).

## H (falsifiable hypothesis)

At matched actual final serialized bytes, activation-aware non-sharing codec
+ functional sharing (+ private exceptions / + joint allocation) shows lower
task-functional distortion (token-weighted NLL, with KL as guardrail) than the
activation-aware non-sharing codec alone, on untouched official TinyStories
validation stories.

## T (test conditions)

- Checkpoint: `roneneldan/TinyStories-28M` rev `52dabea`, `pytorch_model.bin`
  SHA256 `8ddd260f…` — VERIFIED identical to the recorded T266 supplied
  checkpoint (hash, size 241547069 B, tokenizer hash all match). Hence the
  "supplied" 28M WAS the upstream revision repackaged.
- Primary corpus: upstream `TinyStories-valid.txt` (21990 stories,
  SHA256 `94e43181…`); secondary: `TinyStoriesV2-GPT4-valid.txt` (not used for selection).
- Splits (preregistered before any eval): calibration stories [0,64) = fitting;
  development [64,320) = selection (256 stories); audit [320,1344) = untouched
  until Gate 7 (1024 stories). Audit files did not exist before the freeze.
- Env: Apple M1 Max 64 GB, macOS 26.6.2, venv Python 3.14.5 / torch 2.14.0 /
  transformers 5.17.0 / numpy 2.5.3, CPU backend (MPS measured slower for this
  regime: 1.72 s vs 0.59 s on 4 smoke stories; CPU/MPS NLL agree to ~3e-7).
- Metrics: token-weighted NLL/PPL, KL(orig||comp), top-1 agreement, actual
  bytes, bits/paid-scalar, ratio vs 16-bit paid baseline (51987968 scalars =
  103975936 B). Story-unit bootstrap (4000 resamples, seed 266272).
- Reproductions: `tests/` 103 + `experiments/t282/tests` 149 + 13 new = **265 passed**.

## D (decision)

**FAIL.** No tested FQC sharing / private / joint candidate beats the strong
non-sharing control at matched final bytes with viable quality. Key audit
(1024 untouched stories, reference NLL 1.4431) head-to-heads, story-bootstrap
CIs non-overlapping:

| pair (matched/nearby bytes) | control NLL | FQC NLL | verdict |
|---|---|---|---|
| act_b4_g128 (27.697 MB) vs B-meta shared-metadata (27.698 MB) | 1.5168 | 1.5395 | sharing significantly **worse** |
| act_b3_g64/g128 vs D-lite joint emb2/attn4/mlp3 (20.5 MB) | 1.7481/1.8439 | 1.8694 | joint significantly **worse** |
| vqK256_B8 (6.708 MB) vs role-shared codebook (6.615 MB) | 7.7992 | 7.2831 | sharing better but **both collapsed** (PPL 1450+) |
| shareK256_B8 vs +256 private rows | 7.2831 | 7.2848 | private flat (replicates dev) |

- Gate 1: HF `GPTNeoForCausalLM` (transformers 5.17, eager, FP32 CPU) logit
  parity vs offline engine: max abs err **0.0** (bit-exact), tokenizer IDs equal.
  RTN control byte-exact PASS; act-refit byte-exact FAIL across torch versions
  (2.10.0+cpu observed vs 2.14.0 here; same byte counts, refit metadata flips)
  with within-env determinism PASS (rebuild-002 == rebuild-001 bitwise).
  Historical act hashes were NOT rewritten.
- Gate 2: 24-control frontier, 1.88x–74.6x. act > rtn at every uniform point
  (e.g. b4_g64 dev 1.1568 vs 1.1942, CIs non-overlapping). Per-tensor VQ without
  row scales collapses beyond ~8x; row scales do not rescue it (B8: 7.73).
- Gate 3: whole-tensor cross-layer sharing ruled out by diagnostic
  (max |cosine| <= 0.043, ~random). Role-shared codebooks: 3/4 better on dev,
  1/4 worse, all inside collapse; audit replicates the one tested pair.
- Gate 4: exact fp16 private rows S in {0,64,256,1024} on two backbones:
  flat-to-harmful (A-backbone P1024: 7.78 -> 8.29). Replicates T266
  non-monotonicity in a new form.
- Gate 5 (D-lite): joint triples J1/J2 strictly worse; J3 ties uniform
  (1.5501 vs 1.5435, directionally worse). No Pareto improvement.
- Dev -> audit orderings replicate throughout: the FAIL is not dev overfit.

## C (rival hypotheses)

- C1 (act-quantization explains all): SUPPORTED as caution — act alone gives the
  viable frontier; no sharing increment found on top.
- C2 (block/metadata structure explains): SUPPORTED — B-meta lands on the
  uniform line; shared-metadata is reparameterization, not quotient gain.
- C3 (dev overfit): REJECTED for these verdicts — audit replicates all directions.
- C4 (checkpoint-specific): OPEN — 28M only; 1M/3M/8M repetition not run.
- C5 (trivial correlation, not quotient): SUPPORTED at tensor granularity
  (orthogonality diagnostic); codebook-level sharing shows only a collapsed-regime
  micro-effect.

## U (key uncertainties)

- 28M single scale; no independent training replicas; no 1M/3M/8M repetition.
- Eval CPU FP32 batch-1; no MPS/CUDA perf claim (one small negative datapoint only).
- No GPTQ/AWQ/AQLM comparison (out of scope for this gate).
- Missing `codec_v2` MLP-tail encoder (decoder-only in tree) blocked testing the
  historical 64x-family recipe; extreme-rate lane used per-tensor VQ instead.
- Audit = 1024 stories (preregistered minimum); 4096 expansion not run.

## Measured facts (artifact files)

- `GATE2_DEV_FRONTIER.csv`: 20 controls + reference, dev-256.
- `GATE7_AUDIT.csv`: 12 frozen candidates, audit-1024. Full per-story JSONs in
  `runs/eval-001/` (git-excluded; hashes in manifests).
- Manifests: `runs/manifest-001/` (environment, provenance, per-gate verdicts).

## Inference (separate from facts)

The tested FQC sharing forms add no rate–distortion value beyond a strong
activation-aware non-sharing codec on 28M official validation. Collapse is
distributed, not row-concentrated (hence private rescue fails). The one
replicable sharing-direction signal lives where models are unusable.

## Unresolved questions

1. Does role-shared-codebook direction hold at 1M/3M/8M and at viable rates
   with a stronger VQ base (e.g. AQ/GPTQ-style base the uniform frontier uses)?
2. Is there a viable-regime sharing axis beyond adjacent metadata (e.g. distant
   meta sharing, cross-tensor scale tying with joint refit)?
3. Would QCO over (module bits x support x codebook scope) jointly find what
   greedy family steps missed? (Not justified on current evidence.)

## Next experiment (if pursued)

Repeat Gates 2–4 on 8M (and 1M) with the frozen lane code to test C4/scale
direction; only then consider a viable-rate VQ base + joint (scope, support,
bits) search. Do NOT re-tune on audit stories [320,1344).
