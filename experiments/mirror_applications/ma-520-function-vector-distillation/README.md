# MA-520 — Function-vector distillation into a shared code decoder

Status: SCREENING; protocol frozen before implementation. Branch: `research/ma-520-function-vector-distillation-20261009`.
Prior art: PA99, *Function Vectors in Large Language Models*; see the maintained PA map. MA-516 found useful explicit FVs but its rank-four PCA compression missed the gold-logprob preservation gate and exactly aliased native PCA.

## H — Hypothesis

A shared rank-four decoder trained to distill task function vectors can preserve the causal effect of basis-held-out tasks within 0.10 mean gold-logprob nats and 0.05 accuracy of explicit FVs, use at most half the explicit FV-bank bytes, and beat native rank-four PCA by at least 0.10 gold-logprob nats in both development seeds.

## Prior-art delta

PA99 establishes causally active function vectors and partial vector composition. The tested delta is the trained shared decoder and compact per-function address, compared directly with a native PCA basis and explicit FV storage. A learned low-rank decoder is itself an ordinary shared-basis control; no Mirror-specific claim survives if PCA matches it.

## T — Frozen protocol

See `PROTOCOL.json` and `freeze.json`. Use pinned Pythia-70m, the MA-516 task/split/extraction/evaluator, task IDs 0–11 to fit a rank-four shared linear decoder, and IDs 12–15 as decoder-held-out tasks. Train exactly 2,000 full-batch Adam updates on FV reconstruction loss. Assign held-out codes by least-squares projection onto the frozen decoder. Controls: no intervention, explicit FP32 FVs, explicit int8 FVs, and native centered PCA rank four. Development seeds are 52001/52002; fresh seeds 52011–52013 remain locked.

All actual uncompressed NPZ bytes are charged, including code, decoder, metadata, and quantization scales. Include the pinned model/config/tokenizer common base in total deployment bytes. Report extraction, fit, held-out code assignment, query evaluation, support/query tokens, active multiply proxy, optimizer updates, and causal task scores.

## D — Decision

Pending development.

## C — Strongest counter-hypothesis

A learned shared linear decoder is only a reparameterized low-rank basis. PCA is optimal for squared vector reconstruction on the fitting FVs and may preserve causal task behavior as well or better. The small number of functions and high no-intervention accuracy may also make the quality gate unattainable.

## U — Boundaries

Sixteen relation functions and four decoder-held-out tasks on one 70M model provide a constrained recall screen, not broad language evidence. Held-out codes use teacher FVs during compilation; this cost is reported separately, and no teacher vectors are retained for inference.
