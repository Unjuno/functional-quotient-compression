# MA-540 — Sequential function-vector executor

Status: **FAIL** on the frozen development gate; fresh worlds remain sealed.

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
- FV deployment charges the pair encoder, support examples, transition model, step embeddings and metadata. Codes are reconstructed from this paid state and are recomputed for each forward call; no cached-vector state is included. The native control charges its learned code bank and all decoder state. The exact table charges all mappings.

## Development and fresh boundary

Development worlds 54001 and 54002; fresh worlds 54011–54013 are sealed. One configuration is fixed: 2,500 AdamW updates, batch 64, learning rate .002, weight decay .0001, one CPU thread. Each method receives a deterministic paired minibatch stream. No settings are selected after observing development results.

Fresh opens only if both development worlds reach >=90% FV exact pair accuracy and >=90% intermediate-valid paths, FV stays within five percentage points of the tied native-code control, and the full actual FV payload is <=90% of the control. Otherwise record a development FAIL and keep fresh sealed.

## D — Decision criteria

PASS requires every fresh world to retain all frozen quality and byte gates. FAIL follows any development gate miss. NOT ESTABLISHED applies only to task leakage, failed deterministic replay, or incomplete payload accounting. Report facts, interpretation, strongest counter-hypothesis, and remaining limits separately.

## Reproduction

Environment: Python 3.12 / PyTorch CPU, one Torch intra-op thread. Run the unit checks and the two development worlds with:

```bash
pytest -q experiments/mirror_applications/ma-540-sequential-fv-executor/tests
python experiments/mirror_applications/ma-540-sequential-fv-executor/source/run_experiment.py --world 54001 --out experiments/mirror_applications/ma-540-sequential-fv-executor/runs/dev/world_54001
python experiments/mirror_applications/ma-540-sequential-fv-executor/source/run_experiment.py --world 54002 --out experiments/mirror_applications/ma-540-sequential-fv-executor/runs/dev/world_54002
```

The runner refuses the registered fresh worlds unless both saved development summaries pass the frozen quality and byte gates.

## C / U

Strong counter-hypothesis: explicit operation codes plus ordinary recurrent state passing already provide the whole mechanism; FV extraction adds cost and no functional freedom. Unconfirmed: natural-language composition, longer sequences, unseen operators, other recurrent blocks, GPU/runtime behavior, and private-parameter thresholds.

## Results and decision

**H:** The FV-sequenced shared block should execute held-out ordered pairs accurately and cost <=90% of the equal-width native-code recurrent block.

**T:** Two development worlds (54001, 54002), eight random invertible affine maps over 16 states per world, 64 support pairs, 128 atomic training mappings and 192 ordered-pair training examples per world. Each trained model saw 160,000 sampled examples over 2,500 updates. Evaluation used all 16 start states for each of four held-out orders. Five controls/upper references were retained per world. Fresh worlds were not evaluated.

**FACT:** The support-extracted FV tied executor had 100% atomic and held-out pair accuracy, 100% intermediate-valid path rate, and final NLL 0.000745/0.000622. The same-width native-code tied control also had 100% accuracy/path rate and lower NLL, 0.000493/0.000513. FV payload was 27,441 B; native-code payload was 23,865 B (FV is 1.150x the control). FV inference proxy was 41,984 MAC/packet versus 9,216 for native code. FV training proxy was 475,136 MAC/update versus 442,368. The FV one-shot control, which cannot pass intermediate state, reached only 9.4%/28.1% held-out exact accuracy with NLL 3.934/3.941. Untied depth achieved 100% at 44,523 B; the exact operator table upper achieved 100% at 668 B.

Measured internal-unroll latency for FV was 0.197/0.196 ms per batch-1 packet and external two-call latency was 0.165/0.168 ms; native code was 0.130/0.131 ms internal and 0.107/0.108 ms external. These CPU microbenchmarks are descriptive only. The external path achieved the same exact outputs as internal state passing.

**INTERPRETATION:** Explicit ordered state passing enabled composition in this bounded task; the one-shot control failed. Function vectors worked as operator addresses, but the ordinary learned code bank reproduced the result with smaller payload and less compute. The measured storage/compute frontier therefore favors the native tied control.

**D:** MA-540 **FAIL** for Mirror-specific value and the predeclared quality/byte gate. Fresh stays sealed because the FV payload exceeds the <=0.90x native-code threshold in both development worlds.

**C:** The affine family is deliberately learnable from atomic examples, and the native operator embedding bank supplies all code functionality without support extraction. Explicit recurrence, rather than an FV-specific property, explains the successful composition.

**U:** No natural-language tasks, sequences longer than two steps, unseen operators, GPU runtime, adaptive/private-code boundary, or fresh-world replication. The exact table upper demonstrates this artificial task can be represented compactly outside learned networks.

**Accounting amendment:** The first run undercounted two transition steps in active MACs and omitted operator-ID-order metadata. Initial artifacts are retained in `runs/dev_initial_metrics_undercharged/`; identical development runs were repeated with corrected byte/MAC accounting. No data, model, setting, quality result, or fresh-access rule changed. `PROTOCOL.json` records this amendment.
