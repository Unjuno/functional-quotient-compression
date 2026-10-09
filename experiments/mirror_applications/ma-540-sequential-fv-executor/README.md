# MA-540 — Sequential function-vector executor

Status: SCREENING; preregistered protocol, not yet run.

## H — Falsifiable hypothesis

On held-out ordered compositions from a small noncommutative operator family, a support-extracted function vector (FV) supplied at each step to a single recurrent transition block will preserve exact execution and intermediate-state validity, while matching a same-width native operator-code control at no more than 90% of its actual serialized inference bytes. If FV cannot clear both quality and storage gates, its added address/extractor does not establish a useful Mirror execution advantage.

## T — Task and fixed controls

Each world contains eight random affine bijections over four-bit states. Training includes every atomic mapping and twelve ordered operator pairs; four ordered pairs, including both orders for two pairs, are held out. Every held-out pair is evaluated on all sixteen start states. The system receives per-operator support demonstrations to extract function vectors, then runs the requested operator sequence. Pair endpoint loss is used; no intermediate target supervision is provided.

The candidate unrolls one shared transition block twice, feeding the predicted intermediate distribution into step two. Controls are same-width learned native operator codes with the same tied block, an FV one-shot decoder without explicit state passing, untied depth blocks, an external two-call execution of the candidate block, and exact lookup tables. The exact table is a charged upper reference. External two-call execution is a runtime and execution interface control, not a capacity claim.

Prior-art boundary: PA99 motivates representation-space function vectors; PA72 says tied recurrent blocks already have depth addresses, so the native code/timestep controls are required. SRM003 found poor direct composition but exact two-call replay on its specific task, with extra compute and supplied order. MA-539 found the FV route too weak for arbitrary random permutations from sparse examples. MA-540 switches to a learnable affine function family and tests explicit state passing.

## Mirror insertion and cost

- Physical object: one shared state transition block.
- Coordinate: one support-extracted 16-dimensional code per operator, selected in program order.
- Simpler control: direct learned operator-code bank with the same recurrent block.
- Logical multiplicity: eight operators and held-out two-step programs; operator count alone is not treated as capacity.
- FV deployment charges the pair encoder, support examples, cached vectors, transition model, step embeddings and metadata. The native control charges its learned code bank and all decoder state. The exact table charges all mappings.

## Development and fresh boundary

Development worlds 54001 and 54002; fresh worlds 54011–54013 are sealed. One configuration is fixed: 2,500 AdamW updates, batch 64, learning rate .002, weight decay .0001, one CPU thread. Each method receives a deterministic paired minibatch stream. No settings are selected after observing development results.

Fresh opens only if both development worlds reach >=90% FV exact pair accuracy and >=90% intermediate-valid paths, FV stays within five percentage points of the tied native-code control, and the full actual FV payload is <=90% of the control. Otherwise record a development FAIL and keep fresh sealed.

## D — Decision criteria

PASS requires every fresh world to retain all frozen quality and byte gates. FAIL follows any development gate miss. NOT ESTABLISHED applies only to task leakage, failed deterministic replay, or incomplete payload accounting. Report facts, interpretation, strongest counter-hypothesis, and remaining limits separately.

## C / U

Strong counter-hypothesis: explicit operation codes plus ordinary recurrent state passing already provide the whole mechanism; FV extraction adds cost and no functional freedom. Unconfirmed: natural-language composition, longer sequences, unseen operators, other recurrent blocks, GPU/runtime behavior, and private-parameter thresholds.
