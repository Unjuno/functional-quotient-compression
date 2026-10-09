# MA-545 — Few-shot-routed function-vector experts without weight experts

Status: SCREENING
Evidence lane: LANGUAGE/REPRESENTATION_ROUTING/CAUSAL_QUALITY/ACTUAL_BYTES
Branch: `research/ma-545-fv-residual-moe-20261009`
Base commit: `a37cb993d0a9bef5c7a9bb39e86abca39ed30bf6`

## H — Hypothesis

On 16 held-out Pythia-70M relation tasks, a router that reads few-shot demonstrations can select support-derived residual-stream FV experts with route accuracy >=.75 and preserve oracle-routed FV quality, while using <=.75x the complete same-router rank-one MLP-output expert bank.

## Insertion and prior art

One shared frozen Pythia model receives one support-derived additive layer-3 residual direction per task. A learned affine router sees a few-shot prompt's final query representation and selects one direction. PA99 establishes activation-space function vectors; PA100 establishes contrastive additions; PA101 studies context-dependent activation steering; PA17 is a rank-one shared-weight modulation control. MA-530 failed query-only routing (.477/.516 route accuracy), so this run specifically tests demonstration-context routing.

## Fixed protocol

`PROTOCOL.json` and `freeze.json` are frozen before implementation or model evaluation. Development seeds are 54501/54502. Fresh seeds 54511–54513 stay unopened unless both development seeds pass all preregistered gates.

## T — Execution

Pinned Pythia-70M-deduped (revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`), CPU/five threads, two development seeds (54501, 54502), eight support and eight held-out queries per each of 16 tasks. Each query prompt includes four support demonstrations. The support-derived FV is the mean of eight layer-3 residual differences. A 512-to-16 affine router trained for 1000 Adam updates selects top-1. The rank-one weight control is fit only from support MLP input/FV pairs.

Initial ambient-RNG runs are retained in `results/pre_amendment_1/`. Amendment 1 fixes RNG to each world seed; Amendment 2 embeds the split manifest in every charged NPZ and updates actual serialized byte counts. No fresh labels were accessed.

## Results

Development-only result: both held-out route accuracies are 1.00. Routed FV equals oracle FV in candidate accuracy and gold likelihood. Mean gold-logprob improves over same-prompt ICL by 0.259 and 0.320 nats; accuracy changes are -0.0078 and 0.0000. With the charged manifest, FV router+expert payload is 86,951 B vs 119,943 B rank-one weight expert payload (0.725x). The development thresholds therefore authorize fresh evaluation, but do not establish robust generalization. See `RESULTS_CORE.csv` and per-seed metrics under `results/`.

Fresh seeds remain sealed until the verified dev gate is committed. Data, model and intermediate artifacts remain under `results/`; do not delete negative results.

## Decision

FACT: Development met route, oracle-quality, byte and mean gold-logprob gates; accuracy did not improve.
INTERPRETATION: Context makes routing easy in these tasks; the FV increases candidate likelihood but does not reliably change the selected candidate.
HYPOTHESIS: Whether this is useful beyond the development worlds remains untested.
BOUNDARY: One Pythia-70M checkpoint and 16 relation tasks.
