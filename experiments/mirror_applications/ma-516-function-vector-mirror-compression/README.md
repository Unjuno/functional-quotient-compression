# MA-516 — Pythia function-vector compression

Status: SCREENING; no development model passes run yet.
Branch: research/ma-516-function-vector-mirror-compression-20261008
Prior art: PA99, Function Vectors in Large Language Models.

## H — Hypothesis

On four held-out in-context relation functions in Pythia-70m-deduped, projecting support-extracted function vectors into a rank-4 shared basis will preserve held-out query accuracy within 5 percentage points and mean gold-candidate log-probability within .10 nats of explicit vectors, using <=.50x their actual vector payload bytes in both development worlds.

Mirror insertion: extract one residual-stream function vector per relation task from support demonstrations, fit a centered rank-r basis on task IDs 0–11, and represent all functions (including IDs 12–15) by small codes. Model weights remain frozen.

## T — Frozen protocol

The model is EleutherAI/pythia-70m-deduped at immutable HF revision e93a9faa9c77e5d09219f6c868bfc7a1bd65593c, local weight SHA-256 3da388…a33fd. The weight file is 166,029,852 B; the common config+tokenizer files are included in total deployment bytes. Hidden size is 512, six layers; intervention is at hidden_states[4]. Four relation task directions are held out from basis fitting. Each task uses eight support and eight disjoint query pairs, with eight candidate answers. Function vectors are mean leave-one-out demonstration-minus-query-only activations at the answer delimiter. Ranks {1,2,4,8,12}; dev seeds 51601/51602; fresh 51611–51613 sealed. Full protocol and hashes are in PROTOCOL.json and freeze.json.

Amendment 1 fixed only a rank-1 method-dispatch bug after the initial sweep; those first outputs are preserved under runs/pre_amendment_1_rank1_dispatch_bug/ and the same frozen seeds were rerun. Amendment 2 corrects the support-token count aggregation only; its pre-amendment output is also preserved. The development runner lives in source/run_screen.py. It uses the locally downloaded checkpoint; no checkpoint or weight edits are committed. Python packages (Transformers 4.48.3, tokenizers 0.21.4, safetensors 0.8.0, hub 0.27.1) are in an external temporary target directory.

## D — Decision

Pending development.

## C — Strongest counter-hypothesis

Function vectors may be distributed across unrelated task directions, so rank four can harm task quality. If low-rank reconstruction works, centered native PCA uses exactly the same basis, codes and decoder and will exactly match the Mirror output and storage.

## U — Scope limits

One 70M model, constrained-answer ranking, and 16 relation directions are a small mechanism screen, not a broad language benchmark. Support extraction and query compute are recorded. The candidate-answer set is supplied equally to all methods. A native PCA match is not Mirror-specific evidence.

## Evidence classification

- Facts: pending frozen dev metrics and payload replay.
- Interpretation: compressed FVs only help if the explicit vectors actually improve held-out task behavior and the shared code retains that improvement.
- Hypothesis: an additional structured functional address may improve over native FV/PCA sharing; this experiment tests that narrowly.
