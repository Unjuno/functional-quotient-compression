# MA-1162 拡張：最小共有核 Mirror Network 実験ロードマップ（2026-10-18）

**Status: DESIGN / BACKLOG ONLY. This file is not an already run protocol, measured result, or scientific status update.**  
Parent principle: [Mirror-native minimal shared core](../../../docs/phase2/MIRROR_NATIVE_MINIMAL_CORE_PRINCIPLE_2026-10-18.md).  
Historical controls: MA-041, MA-1156..1162, MA-253, MA-691, MA-1163/1164. Reuse their actual implementation, results and failures; never promote aligned-only gains.

> **Mirror insertion:** introduce a small, explicitly priced role/head/depth View code into one shared physical parameter vector/core so that parallel logical Attention/FFN/Cross-Attention functions and sequential depth can be obtained without storing independent full-sized copies for each role.

## Research target and strict interpretation

One physically stored main object S, whose **dimension m is tunable**, supports multiple addressed nonlinear functions. This is not the unsupported claim that a single floating-point scalar can independently train a Transformer. Test both **learning from scratch with shared S** and **mechanically representing a trained baseline** as distinct lanes. Do not mix their accuracy and optimization budgets.

Primary objective: the smallest m and complete paid bytes that recover useful multi-role functionality; then optimize K heads, R depth repeats, activation family, cache sharing and runtime. A one-scalar/high-precision variant remains a distinct theoretical/numerical line; it must not be silently conflated with the m-dimension capacity sweep.

## Priority queue (one experiment at a time)

| Order / owner | Concrete task | Dependency | Controls | Required evidence / stop rule |
|---|---|---|---|---|
| **P0 / MA-1162-M0** | Implement and test a single shared S and reusable on-demand Mirror/nonlinear operator with role ports (self-attn causal, self-attn bidirectional, cross-attn, FFN). Head K/step R explicit | Existing MA-1162 / multihead-fold L0 source | same core without View; null common QK rotation; native equivalent folded Q/output | Causality/bidirectional/cross-memory tests, head/role ablation, gradients and finite numerical equivalence. Reject a dummy cross-attn that ignores source. No task-quality claim. |
| **P0 / MA-1162-M1** | Train the SAME shared-core block for at least two useful roles: genuine content retrieval + ordered/state-dependent nonlinear role; compare m and K | M0 correctness | ordinary tied block, fixed-code Mirror, FiLM/gain, rank1/2 shared-basis, untied attention/FFN quality ceiling | Independent training worlds; at equal bytes and separately equal dense-MAC + wall budget, show useful role accuracy and no collapsed heads. If equal-byte native wins, preserve M0 scientific verdict. |
| **P0 / MA-1162-M2** | Sweep physically learned m versus K horizontal heads versus R sequential depth, one at a time and paired interactions | M1 proves roles are learnable | width-reduced plain FFN, tied-depth Universal Transformer, ordinary multihead GQA/MQA, byte-near low-rank | Quality/byte, quality/FLOP, output/head KL + ablations, Jacobian effective rank/gradient interference, serialized FP16/FP32 and runtime. Stop growth if new head/step worsens strict Pareto frontier twice on DEV. |
| **P1 / MA-1162-M3** | Joint encoder-self + decoder-self + true encoder/decoder cross-attn using S with different masks and logical source/target addresses | M0/M1 roles and stable grad | untied standard encoder–decoder; fully tied non-Mirror; role embedding / FiLM; shared-kernel learned adapters | Train on source-key/value→target-query task with heldout unseen source–query combinations; encoder-output perturbation must alter decoder cross output, source padding and causal masks correct. No free simultaneous dependency bypass. |
| **P1 / MA-691-M4** | Share canonical KV across compatible Mirror multiheads and cross reads; prove safe and unsafe cases | MA-691 exact algebra, MA-253 counterexample, M0/M3 producer contract | materialized View caches, native absorbed Q/output, GQA/MQA, MLA, independent cross-cache | Physical cache pointers/actual allocation, source/version/prefix provenance, RoPE commutation, logits parity, prefill+decode tok/s, dwell; ordinary absorbed native is PRIMARY exact-functional speed control. |
| **P2 / MA-1162-M5** | Add 2/4 logical FFN/MoE roles, nonlinearity/mixed activation, sparse role selection, optional private residuals | M1/M2 | independent MoE, tied+role embeddings/FiLM, simple learned low-rank, top-k router native | Natural/off-orbit functions, held-out composition, retained old-role accuracy, program bytes + router/optimizer, used-head fraction and end-to-end latency. |
| **P2 / MA-1162-M6** | Fused execution without materialized virtual tensors, and controlled CPU/GPU study | M2/M4 identify positive frontier | optimized native GEMM/FlashAttention and fused non-Mirror role kernels | Same hardware, precision, batch/context, cold compile and warm steady-state separated; real resident memory and tokens/s, not theoretical arithmetic alone. |
| **P3 / separate precision study** | Increase bits per scalar at fixed m with same Mirror program | positive m-scale result before cost-heavy arbitrary precision | same-budget FP16/32/64 ordinary and non-Mirror, dyadic scalar fixedpoint | Prove increased useful held-out capacity rather than only numerical representational resolution; optimizer stability and bit-operation runtime. |

This is a **work queue**, not a frozen experiment grid: do NOT open new fresh results until a versioned PROTOCOL, source hashes and splits have been committed.

## Immediate implementable M0 contract

Recommended minimal shapes: one tensor object S as m-vector; shared working feature width 16 or 32; K=2 Mirror heads; R=1 and 2; one stream source length 8, decoder length 6; self bidirectional/causal/cross role masks. Use deterministic fixed View codes first. Then add learned codes but pay their scalars/bytes separately.

Required unit tests before any optimizer:
1. Exactly one principal independently stored physical tensor S; no silent per-role dense trainable weights; if any learned code exists, separately report its parameters.
2. Mirror forward/inverse and nonlinear compositional behavior; invertibility only asserted for Q, not ReLU/tanh.
3. Distinct Q/K relative maps change attention scores; equal orthogonal Q/K map is null gauge.
4. Future-token perturbation does not alter earlier decoder outputs in decoder-self path.
5. Encoder bidirectional state may depend on later encoder tokens; encoder padding is masked.
6. Cross-attn output depends on encoder *content* and on decoder query; zeroing/shuffling source changes output; source mask works.
7. Running K heads concurrently on one state matches sequential evaluation of the same K heads; no K-fold storage of the shared core.
8. Repeating R on changed states matches explicit reference steps, and R=2 is not automatically counted as two independent learned blocks.
9. Shared K/V path passes MA-691 exactness only when source producer/history/version are identical and View maps right-act; intermediate nonlinear altered history must reject cache alias (MA-253).
10. Numeric/finite gradients for S and paid code; actual serialized inference payload bytes and output replay after save/load.
11. The on-demand operator path does not allocate the full virtual d×d weight matrices in its guaranteed no-materialization implementation; instrument allocations where possible.
12. A fixed Mirror program is compared to an ordinary Q/output-folded native implementation with same logits.

**M0 PASS:** all correctness tests pass, physically correct state ownership and no hidden oracle/external role tables. **M0 FAIL:** any missing cross-source dependence, causal leak, invalid cache alias, uncharged trainable matrix, or algebra failure. M0 PASS does NOT prove compression or general learning.

## M1/M2 protocol before freeze (proposals, not already executed)

### Development design

- Use existing MA-1162 Copy/Delay, Content Retrieval, Ordered Transitions, plus hidden-branch negative control; add new cross-source retrieval for M3.
- Develop on new worlds distinct from all previously opened MA-1162 IDs; choose and record RNG seeds after scanning the relevant results directories. Do not recycle 618200-series or multihead-world tests as new fresh evidence.
- Start m=32/128/512, K=1/4, R=1/3 (12 exploratory candidate cells). Hold FP32 across cells and record any layer-width changes separately. Add m=256, K=2 and R=2 only in a newly preregistered follow-up if a specific frontier is located.
- All models must use the same train/validation split, batch, examples seen and optimizer updates for fixed-update plots; add a *separate* equal-training-MAC proxy and measured wall-time evaluation. Use native model-width tuning on development to meet actual serialized bytes; parameter count alone is insufficient.
- For m=32/128/512, capacity might require a learned wider hidden working state than S. The state is RAM, not stored weights; its peak memory and extra operation count still need accounting.

### Audit design

- New independent test worlds >=3 and seeds >=2 per world, frozen *after* development and never used for hyperparameter selection.
- Independent task roles should be evaluated both separately and together; include held-out source/query combinations, new sequences and an off-orbit function family.
- Native controls: standalone full MHA+FFN, same-physical-budget narrow MLP/Transformer, tied/recurrent (Universal Transformer/ALBERT-style), GQA/MQA/MLA, FiLM/gain, rank1/2, ordinary statically folded Q/output, simple sparse/private residual when needed.
- Primary metrics: held-out CE/NLL and task accuracy per role; actual model+code+index serialization (including dtype); current-token/decode and full prefill runtime; peak memory; FLOP/MAC proxy; updates; head attention KL/ablation and gradient-interference spectral diagnostics.
- Mirror versus native at matched bytes AND a second matched-compute comparison. No difference in exposed task IDs, source memory, or extra oracle labels.

### Candidate go/no-go gates (must freeze before fresh)

- **Representation gate:** among 3+ fresh worlds, at least 2 have Mirror >=95% native independent upper-control quality *and* outperform the strongest equal-byte/equal-compute non-Mirror shared control by predeclared metric (e.g. >=1 percentage point accuracy or >=5% relative NLL). For tasks where accuracy saturates, use NLL and OOD held-out roles.
- **Non-collapse:** each required role beats token-local/null baseline; at least two head code ablations have measurable effect, and no role loses >2 percentage points against its permitted role-specific reference.
- **Efficiency:** actual inference bytes (paid S+program+indices) below native of similar quality and no >10% degradation in measured end-to-end runtime for an adoption claim; report quality gains without that as a different mechanism class.
- **Robustness:** cache provenance/masking exact; negative random branch cannot be predicted from unobserved randomness.
- **Boundary:** if a simpler native control matches, classify M0/no Mirror-specific gain rather than optimizing opened fresh data or changing thresholds.

## Mathematical / optimization diagnostics to measure before inventing a new optimizer

1. Rank and singular-value spectrum of finite-sample output Jacobian relative to the physical m-vector (rank at most m for smooth programs). Report **effective** rank, not just theoretical maximum; measure sensitivity and condition number.
2. Task-wise gradient cosine/conflict across Mirror roles and depths. A single S may be unable to update independently in contradictory directions; learned View code adds paid degrees of freedom.
3. Finite-difference versus automatic gradients; activation saturation under repeated R; residual stability and Lipschitz/norm traces.
4. Compare AdamW, matched step budgets and exact same tensor against a simple preconditioned/normalized gradient baseline on *development* only. Only create a new optimizer if gradient geometry explains failure; do not assume optimizer alone can solve a representation barrier.
5. Observe whether adding K or R increases useful held-out function diversity at a fixed m without exceeding byte/compute cost; do not infer infinite Attention heads.

## Execution order and concrete deliverables

**Next actionable coding task: M0 ONLY.** Extend existing \`multihead_fold_20261018/source/multihead.py\` or a new contained module under MA-1162, preserving existing tests. Implement four role ports (encoder-self, decoder-self, cross, nonlinear FFN) sharing one physical S; implement source mask/causal strictness and a simple mixed-head aggregation. Require all 12 M0 tests and save actual payload bytes and full output roundtrip. Prototype may run entirely in CPU container; do not offload to user's local machine.

Then M1 to learn real source-dependent retrieval without role-specific dense weights; then M2 physical dimension scaling with byte-near controls. MA-691 L0 frozen plan may run as its own *separate* experiment stream only after core-correctness boundary is understood; do not mix its cache result with m-scaling causality.

Deliver for each executed stage: \`README.md\`, frozen \`PROTOCOL.json\`, \`source/\`, \`tests/\`, full \`RESULTS_CORE.csv\` with negative rows, \`VERIFICATION.json\` with SHA256 of code/data and environment, Japanese \`REPORT_JA.md\`, Git branch commit/readback. Record source hash, data/split hashes, hardware, precision, thread count, batch, clock control (or lack thereof), measured p50/p90, and unknown calibrated u_c and coverage k when not estimated. Run repository registry integrity check before any **new MA ID** or row edit; design addendum uses existing IDs so the historical ledger is unchanged.

## H / T / D / C / U

**H:** physical-vector dimension m, plus K parallel Mirror roles and R recurrent depth, provide a useful increasing rate–quality frontier against same-byte/same-compute native sharing methods.  
**T:** M0 correctness and function/mask tests -> M1 independent-role training -> M2 small m×K×R development sweep -> sealed fresh evaluation -> M3 source–target/cross-attention.  
**D:** algebra/causal pass in M0; 2/3 fresh paired quality and 0 forbidden cache uses for M1–M4; strong native match counts as Mirror-specific FAIL.  
**C:** tied native recurrence and GQA/MQA/MLA, FiLM or simple low-rank/role embedding already provide the same functions faster or at fewer bytes.  
**U:** task alignment, optimizer convergence, role-code overhead, off-orbit representational deficit, CPU/GPU kernel gap, natural-language transfer and true memory-system costs.

### ERROR CHECK (plan stage)

This roadmap records NO new measured metrics and NO fresh run. It defines roles, dependent computation and controls. Historical negative outcomes and current registry statuses remain untouched. Do not cite a draft gate as if it were already frozen or passed.
