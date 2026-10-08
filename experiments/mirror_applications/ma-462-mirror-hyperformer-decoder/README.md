# MA-462 — Fixed-embedding Mirror decoder vs HyperFormer-style decoder

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-462-mirror-hyperformer-decoder-20261008`  
Base commit: `d12d62a1902c702e57b09cfaba49d0862d485226`  
Prior art: PA82 and PA118.

## H — hypothesis

With identical fixed task-factor embeddings and a shared frozen backbone/adapter basis, a structured Mirror decoder can generate useful adapters for held-out task-factor combinations with lower full inference bytes than a generic HyperFormer-style decoder, and can beat simple linear/additive decoders when task variation is factor-aligned.

> **Mirror insertion:** this experiment adds `m(a,b)` to the shared adapter-atom coefficient interface so task-specific logical adapters can be generated from two factor coordinates without storing a separate adapter decoder or full adapter for every task.

## Prior-art delta and controls

PA82/PA118 already establish shared task-conditioned hypernetworks that generate adapters. The tested delta is whether a structured Mirror code is smaller or extrapolates better than that generic decoder while using the same fixed conditioning embeddings. The generic hypernetwork is a small coefficient-emitting HyperFormer-style control, not a claim of reproducing the full paper architecture.

Controls are frozen base/no adapter, linear decoder, generic shared hypernetwork, additive factor decoder, flat per-task coefficients, Mirror plus private residual, and independent full adapters.

## Protocol

- A task-agnostic MLP is pretrained on a common regression teacher for 500 updates, then frozen. Adapter atoms attach to its output projection.
- Sixteen tasks form a 4×4 factor grid. Four combinations are held out while all factor values remain represented in training.
- Fixed `E_A` and `E_B` embeddings are generated per world and held identical across every decoder. They are never trained.
- Target task adapter code: `c(α)=α c_view(a,b)+(1−α)c_private(task)`, with α in `(0, 0.25, 0.5, 0.75, 1)`. `c_view` is product-factorized; `c_private` is independent per task.
- Development seeds 4621/4622; fresh seeds 46201/46202/46203. Decoder training is fixed at 300 updates. Mirror/additive latent ranks `(2, 4, 8)` are selected using development only.
- The alpha=1 development gate requires both worlds to be within 10% of HyperFormer, at least 20% better than the best linear/additive control on held-out tasks, and at most 80% of HyperFormer's complete serialized payload. If no rank passes, fresh remains unopened.
- Actual `torch.save` inference bytes include frozen base, adapter atoms, fixed embeddings, all decoder state and metadata. Decoder MACs, adapter application MACs, training proxy, generator wall time and inference wall time are separate.

## Random selection provenance

Draw 4 selected MA-462 from 531 registered P0/UNTESTED candidates with no remote `research/ma-*` branch. Seed, index, pool hash and exact eligible list are in `source/random_draw.json` and `source/selection_pool.csv`.

## Results and decision

### Fact

- The final deterministic development run completed 120 fits across worlds 4621/4622, alpha values 0/0.25/0.5/0.75/1, and ranks 2/4/8 plus fixed controls. Fresh worlds 46201–46203 were not accessed because the development gate failed.
- At alpha=1, every Mirror rank passed the held-out-vs-HyperFormer, held-out-vs-best-linear/additive, and actual serialized payload gates in both worlds. Payloads were 15,349 / 15,477 / 15,733 bytes for ranks 2 / 4 / 8, versus 22,319 bytes for HyperFormer.
- Rank 4 held-out NMSE was 0.0028382 vs 0.0069616 (world 4621) and 0.08162 vs 0.19227 (world 4622). Its seen NMSE was 0.00018283 vs 4.7532e-06, and 0.0005417 vs 2.1815e-05.
- Rank 8 passed all four gate clauses in world 4622, but missed seen-vs-HyperFormer in world 4621 (0.00012859 vs 4.7532e-06); therefore no rank passed both worlds.
- At alpha=0, rank-4 Mirror held-out NMSE was 0.6565 / 0.6249; Mirror plus private residual was 0.0632 / 0.1248. Private coordinates improved this independent-variation boundary at added bytes.
- Full per-row results, serialized hashes/bytes, compute proxies, and wall-clock measurements are in `RESULTS_CORE.csv`. Raw records and gate calculations are in `source/development_raw.json` and `source/development_summary.json`.
- A preliminary development execution was invalidated before reporting because backbone bias initialization was not seed-locked. The initializer was fixed, all final results were regenerated, and a full 120-row replay matched all semantic metrics and payload lengths exactly (maximum float delta 0.0). SHA-256 values are execution-specific because the `torch.save` archive bytes can differ even when tensors and lengths match.

### Interpretation

**D: FAIL** under the preregistered development gate. The compact decoder showed held-out extrapolation within the allowed relative tolerance, and rank 8 passed the full gate on one world, but seen-task fidelity failed the conjunction across worlds. No fresh data was opened and no setting was changed after fresh evaluation.

### Hypothesis

A product-factor Mirror decoder may regularize unseen combinations while a generic HyperFormer fits seen combinations more closely. Private residuals help on task-specific variation by adding paid state. This is a synthetic fixed-budget signal, not a capacity result or a Mirror-specific win over all structured controls.

### H / T / C / U

- **H:** Fixed task embeddings and one shared adapter basis allow a small factor-product Mirror decoder to extrapolate across held-out task combinations with lower complete inference bytes than a generic HyperFormer-style decoder, while retaining its seen-task quality.
- **T:** Frozen shared MLP per world; 4x4 task grid; four held-out pairs; alpha interpolation from independent private codes to factor-generated view codes; 2 development seeds; rank 2/4/8; 300 decoder updates; fixed linear, generic MLP, additive, flat, private-residual, no-adapter and independent controls. All inference payloads were serialized and measured.
- **D:** **FAIL**; none of the ranks met all gate clauses on both development worlds. Fresh remains unopened.
- **C:** The held-out benefit may be ordinary low-capacity regularization. HyperFormer fits observed task combinations more closely; the private-residual model improves task-specific fit by adding paid state. Two synthetic worlds provide limited evidence.
- **U:** Fresh replication, natural task/pretrained model quality, near-convergence capacity, GPU/runtime efficiency, and whether stronger matched structured controls erase the held-out signal remain untested.

BOUNDARY: synthetic decoder mechanism only; no natural-language, pretrained LLM or full HyperFormer++ claim.
