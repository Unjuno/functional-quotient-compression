# MA-1173 — Prospective Cutting of Mirror Virtual MLP Supergraph During Causal Language Training

## 日本語要約

**構造機構PASS・将来評価型刈り取りによる学習改善FAIL・Mirror固有の優位性FAIL。** 3ノード×2方向から18本の仮想MLP経路を生成し、経路をON/OFFしながら2回反復する小さな因果言語モデルを訓練した。経路は18→12本に減り、仮想MLP部分のMACは33.3%減ったが、全モデルの保存量は約0.94%しか減らず、未来8学習ステップを試行して切断を選んでも、ランダム切断・小ゲート切断に対する未見データの正答率改善は0/6。条件付き切断は18回中1回しか採用されなかった。**意味は、切断機構の実装が成功した一方で、切断による将来の規則学習改善は現設定では成立しないこと。** 小さな人工記号列を用いた最初の機構実験であり、自然言語LMの性能や完全な自己設計グラフを実証したものではない。


**2026-10-21 JST — Result: structural mechanism PASS, prospective-learning advantage FAIL, Mirror-specific value FAIL.**  
This is a synthetic token-grammar causal language model, **not real natural-language evidence**. All historical MA-1166..1172 outcomes remain unchanged. Dedicated research branch only, no main merge.

## 1. Research identity and implementation

One global physically learned **288-FP32 coefficient** core and four signed-permutation pages generate **18 ordered 12→12→12 virtual MLP edges** for 3 nodes × all directed connections × 2 Mirror addresses. There are two repeated Loop updates. A separate **causal two-head attention** performs genuine token mixing. Per-edge gate logits and separate Boolean ON/OFF masks control *presence* independently of learned edge weight magnitudes. Cutting an edge removes it from virtual-weight generation and active edge execution; it is not merely multiply-by-zero in a fully executed graph.

Full Mirror model: **1878 independently learned scalars** (including token embedding/attention/head); native tied 12→16→12 FFN control **1876**. All 18×2×12×12 = **5184 generated logical weights** are functions of the shared core and addresses, not independently learned parameters. The Mirror decoder uses Householder→tanh→Householder; exact-address affine and direct-tanh controls are included. The 3 nodes, 2 views, 2 Loop iterations, attention, and output head are NOT optimized or pruned in this first test.

## 2. Frozen protocol and data separation

Full preregistered protocol SHA-256: 

bdfaf85229434d6d3e9df3f0f33e2f640bf65c0cfe8059bde676307e1847da49

Two worlds (balanced query and 90%-first-key shortcut), seeds 31/47/59. Each has independent train/dev/fresh/OOD samples (4096/512/1024/1024), symbolic key-value token grammar of sequence length 12 and vocabulary 24. Training is AdamW lr 0.005, batch32, 260 updates, weighted objective = mean causal token NLL plus 2× last token loss. Native and no-cut controls have matching training schedule and start seeds. Cut events at 80/140/200; each forced event removes 2 edges. Random, weakest gate, candidate lookahead and guarded lookahead are distinct interventions.

At each lookahead event: 4 disjoint randomly sampled two-edge groups, each trained for 8 hypothetical optimizer updates using a cloned model AND cloned AdamW state, plus one uncut reference continuation using the same fixed future training batches; choice made by **dev**, never by fresh. 5×8×3 = **120 additional optimizer updates**, including discarded probes. Corresponding no-cut-extra control uses 380 *real* optimizer updates. Guarded lookahead commits only when predicted future dev improvement vs uncut ≥0.0005. Fresh/OOD opened after all 54 training checkpoints were complete; neither split selected hyperparameters, structural decisions, or checkpoints.

## 3. Fresh/OOD outcomes (six world/seed pairs; medians)

| 方式 | Fresh NLL ↓ | OODキー参照正答率 ↑ | 残存経路 | 実測推論バイト | 全optimizer更新 |
|---|---:|---:|---:|---:|---:|
| Mirror full graph | 0.8977 | 42.24% | 18 | 13668 B | 260 |
| Mirror random cuts | 0.8986 | 42.19% | 12 | 13540 B | 260 |
| Mirror low-gate cuts | 0.8986 | 42.19% | 12 | 13540 B | 260 |
| Mirror lookahead cuts | 0.8986 | 42.19% | 12 | 13540 B | 380 |
| Mirror guarded lookahead | 0.8977 | 42.24% | 18 | 13668 B | 380 |
| Direct tanh lookahead | 0.8985 | 42.14% | 12 | 13540 B | 380 |
| Affine lookahead | 0.8973 | 42.19% | 12 | 13540 B | 380 |
| Native tied FFN | 0.8948 | 41.11% | — | 13287 B | 260 |
| Mirror full graph +120 updates | 0.8811 | 41.80% | 18 | 13668 B | 380 |


- **H_mechanism PASS:** forced interventions cut graph from 18 to 12 live edges. The active virtual-MLP MAC proxy dropped from **10368 to 6912 MAC/token** (33.3%) excluding attention, activations, Mirror decoding and other layers. Actual inference packet changed from **13668 B to 13540 B** (0.94% lower), not a 33% total-byte reduction. The full learned coefficient tensors still exist during training and are not independent virtual copies.
- **H_learning FAIL:** mirror_lookahead had **0/6** OOD query-accuracy wins over random and **0/6** over low-gate, versus a frozen threshold of ≥4/6 paired wins and ≥0.02 absolute gain. Median results tie, not merely lack statistical significance.
- **H_Mirror_specific FAIL:** mirror_lookahead improved OOD accuracy over direct-tanh **1/6** and affine **1/6** pairs, without meeting a strict Pareto quality/runtime/bytes benefit. Native tied FFN and longer ordinary training are substantive alternatives.
- **Guard diagnosis:** only **1/18** candidate interventions passed a +0.0005 future-dev-loss threshold; it mostly left the full graph intact. Across forced Mirror lookahead, **3/18** shortlisted removals had lower post-H dev loss than keeping all edges, despite all forced cuts being applied. **2** cases had worse immediate dev loss but better hypothetical H-step dev loss after cutting. This is a local learning diagnostic, not validated final superiority.
- **Shortcut/OOD:** the shortcut world changed first-key query prevalence from 90% in train to 10% in OOD. Median OOD query accuracy was approximately 25.8% for each route policy, showing the expected distributional robustness improvement did not materialize.

## 4. Hardware/runtime and hidden costs

Hardware AMD EPYC 9V74, CPU PyTorch 2.10.0+cpu, FP32, one thread, unlocked clock; batch-1 forward latency median across six models: full Mirror **425.7 μs**, cut Mirror lookahead **411.0 μs**, tied native FFN **139.5 μs**. Each per-model median came from 15 warmups and 75 timed forward passes. These are eager-CPU estimates, not GPU/fused-kernel claims. Runtime fixed deterministic page/direction buffers **14832 B** and dynamically decoded active graph weights **13824 B** (12 edges), not included in the serialized inference packet bytes above. The serialized packet also assumes access to the external deterministic decoder program; its source/code storage remains separately payable. No claim of free virtualization.

## 5. Verification and limitations

- Pretraining unit tests **6/6 PASS**: all-edge tensor generation, physical parameter parity, causal future-token exclusion, shared-core/addr/gate gradients, hard-cut active-edge execution, deterministic split separation.
- **54/54** training records, **54/54** checkpoint hashes verified; dev stats reproduced within 1e-7 before fresh access; **48/48** graph-model deployment packet readbacks reconstructed the same logits (max abs error ≤1e-6). Full source, result CSV and deployment audits included.
- Only **edge** pruning, not node creation/deletion, Mirror direction count, address topology, Loop count, or output-choice learning. Does not prove arbitrary logical MLP capacity or capacity per physical coefficient.
- Six paired synthetic conditions, no natural language corpus, no GPU test, no confidence claim from 6 trials. Short-lookahead chosen on a repeatedly used development split may overfit that split. Synthetic causal grammar and teacher distribution may underexploit Mirror views. Source program and deterministic buffers are not zero-cost; shared-core parameter updates can interfere across edges.

## 6. H / T / D / C / U

**H:** Prospective structural cuts guide physical coefficient learning more effectively than low-gate or random cuts and generic non-Mirror shared maps.  
**T:** 9 methods × 2 distributions × 3 seeds; frozen train/dev/fresh/OOD, fully counted +120 lookahead optimizer updates, native/affine/direct-tanh controls.  
**D:** Structural edge execution PASS, prospective learning and Mirror-specific predictive benefit FAIL, guard cut 1/18; no adoption.  
**C:** ordinary extra training, sparse regularization and direct nonlinear generated weights may match; this small synthetic task may rely primarily on the causal attention backbone.  
**U:** CPU clock, split variability, finite-horizon counterfactual approximation, repeated dev decisions, learned gate identifiability, and the absence of natural-language evidence. Model and metric outputs are dimensionless for token losses in nats/token, MAC/token and bytes; no mixed-dimension subtraction.

## 7. Scientific next stage

Before expanding to real text or claiming adaptive NAS, explicitly **test whether cutting a high-impact edge changes subsequent gradients and yields more robust held-out rule learning**. Compare physical-core gradient alignment and trained-path causal necessity with an independent high-impact teacher. If that falsifier also fails, change the supergraph operator rather than increasing graph size or hiding CPU cost. Node/view/Loop pruning would be later, independent experiments.
