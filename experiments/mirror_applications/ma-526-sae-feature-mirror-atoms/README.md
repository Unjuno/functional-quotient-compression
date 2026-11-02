# MA-526 — SAE feature bank as Mirror atoms

Status: SCREENING. Evidence lane: LANGUAGE / MECHANISM / STORAGE / RUNTIME.
Base commit: `263bd613` (MA-431 completed branch).
Prior art: PA102, SAE feature steering.

## H — falsifiable hypothesis

For four related but distinct news-topic steering functions in GPT-2 small, a shared SAE feature bank plus per-topic Givens Mirror coordinates can preserve fresh topic-label logit uplift and limit collateral logit drift while using fewer actual serialized code bytes than independent sparse SAE feature lists and a generic shared low-rank feature code.

## Mirror insertion

> **Mirror insertion:** this experiment adds a compact Givens coordinate `m_c` to a shared SAE decoder-feature coefficient vector so that four topic-specific residual-stream steering functions can be expressed without storing an independent feature-coefficient vector for every topic.

- Physical shared object: pinned GPT-2 small plus one pinned layer-8 residual-stream SAE dictionary.
- Exact Mirror point: at the layer-8 SAE coefficient vector; paired feature coefficients are rotated by topic-specific angles before decoding into a residual-stream intervention.
- Persistent `m`: up to eight FP16 Givens angles for each of four topic functions (K/2 active pairs); all angle and feature-index bytes are charged.
- Cheapest controls: explicit sparse SAE feature lists and unconstrained generic low-rank/shared coefficients over the same feature bank.

## Task and data

Use `fancyzhx/ag_news` pinned to dataset revision `eb185aade064a813bc0b7f42de02595523103ca7`. Prompt: `News: {text}\nTopic:`. The four intended next-label functions are `World`, `Sports`, `Business`, and `Sci/Tech`. GPT-2 small is pinned to revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. SAE: `jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs`, revision `21df74db7b9f690cd2403a20485cf75f46330d0d`, layer 8, top-k 32, `d_sae=32768`.

The locked split uses deterministic, stratified row selection: 32 train, 16 development, and 32 fresh official-test examples per label. Fresh row identities and split hashes are recorded before development. Fresh texts and activations remain unopened until the development configuration is frozen.

For each topic `c` and feature `j`, training computes `S[c,j] = mean_i <∇h logP(label_c|x_i), D_j>`, where `h` is the last layer-8 residual and `D_j` is SAE decoder feature `j`. Independent SAE codes use each topic’s top-K positive scores. The shared bank uses the top-K features by total absolute score across topics. Mirror rotates adjacent coefficient pairs in one shared K-vector; generic shared-low-rank uses the same K atoms with rank `K/2` unconstrained task coefficients. Development selects `K∈{4,8,16}` and steering scale `α∈{0.25,0.5,1.0}`; Mirror base coefficients initialize to the pooled train score, and angles initialize to zero before development fitting. The primary endpoint is held-out correct-label sequence log-probability uplift under the corresponding topic code. Secondary endpoints are correct-label ranking accuracy, full-vocabulary KL drift, active SAE features, actual code bytes, and single-thread CPU latency.

## Controls

1. Frozen GPT-2/SAE with no intervention.
2. Independent SAE top-K feature steering per topic (native control).
3. Random features at the same K and coefficient norm.
4. Shared SAE atom bank with generic per-topic low-rank coefficients.
5. Shared SAE atom bank with Givens Mirror angle codes.
6. Dense residual-gradient steering as an unrestricted upper reference.

The topic code is supplied to all intervention methods. This screen does not evaluate a router. For each method, compute the full base model and SAE serialized file bytes once as shared state, then add the actual serialized method-specific code and metadata bytes. Also report incremental code bytes separately.

## Gates

### PASS (scoped natural-language steering bank)

On every fresh topic, Mirror retains at least 90% of the independent top-8 SAE intervention's mean correct-label log-probability uplift, loses no more than 2 percentage points of correct-label ranking accuracy, has no more than 1.10× its mean full-vocabulary KL drift, and uses at least 20% fewer actual incremental code bytes. It must also beat or remain Pareto-incomparable with the matched generic shared low-rank code at byte-near cost (within 10%).

### FAIL

Development fails to find a Mirror configuration satisfying the quality/KL/byte gates, or a generic/shared low-rank control matches or dominates it at byte-near cost. Fresh remains sealed after a development FAIL.

### NOT ESTABLISHED

Tokenizer or SAE semantics cannot be reproduced; split contamination; invalid label-score comparison; payload accounting failure; or inconsistent fresh behavior that cannot decide the registered gate.

## Compute and storage

Count GPT-2 model files, tokenizer files, complete SAE encoder/decoder state, topic code tensors, feature IDs, metadata, and reconstruction configuration. The shared GPT-2+SAE payload is charged once per deployed bank and reported separately from incremental intervention bytes. Report actual file sizes and SHA-256 values. Record text tokens/examples, forward/backward operations, SAE encoding/decoding MAC proxy, train/dev/fresh wall-clock, and single-thread CPU examples/second. No model training is performed; only feature/code selection and development fitting are allowed.

## Boundaries

This tests a 4-topic steering bank on AG News and one GPT-2/SAE layer. Topic IDs are oracle-supplied. It does not establish a general router, broad task capacity, or use on other models/layers. SAE feature semantics and feature-selection side effects are central; success on a feature-aligned target does not imply arbitrary behaviors are compressible.
