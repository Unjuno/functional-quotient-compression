# MA-553 — Factorized task × layer FiLM Mirror code

Status: **FAIL**  
Branch: `research/ma-553-task-layer-film-mirror-20261009`  
Base: `503f1b4f`  
Prior art: PA106 FiLM; PA118 HyperFormer++

## H — Hypothesis

A small task factor × layer factor Mirror coordinate can recover useful task-layer FiLM affine states for combinations omitted from training, with lower actual bytes than independent task-layer FiLM and better held-out quality per byte than a direct coefficient factorization.

## Insertion

The physical object is a shared hidden representation. Logical task-layer functions use featurewise affine scale and bias. Mirror inserts a rank-one task × layer code into gamma/beta. Controls: independent FiLM table, rank-one direct coefficient table, Mirror-factorized coordinate, and shared no-modulation. HyperFormer++-style shared generator is represented by a byte-counted small MLP mapping concatenated task/layer embeddings to gamma/beta.

## Frozen protocol

Six task IDs × four layer roles. The target affine state is generated from a known rank-2 interaction with noise, then fit on 18/24 combinations and evaluated on six prelocked held-out pairs (2 tasks × 3 layers). Two development seeds, exact least-squares fitting on train pairs, fixed 32-channel representation. Report held-out gamma/beta nMSE, function-output nMSE on 512 Gaussian probes, distinct outputs, actual deterministic NPZ bytes including metadata, multiply-add proxy and wall time. No fresh data are opened if direct factorized coefficients match Mirror within the specificity gate.

PASS requires held-out output nMSE ≤1e-3, at least 20% fewer bytes than the independent FiLM table, and at least 10% lower bytes than byte-near direct factorization or HyperFormer control at matched quality. FAIL if direct coefficient control matches Mirror within 1% quality/bytes or held-out threshold fails. Fresh sealed on failure.

## H / T / D / C / U

- **H:** Task × layer Mirror coordinates generalize to unseen combinations with useful quality and a storage/quality advantage over independent FiLM and native generators.
- **T:** Six-by-four affine bank, two development seeds, 18 observed pairs and six held-out pairs; 12 development result rows.
- **D:** FAIL: independent FiLM exactly reconstructs held-out states (6634B), while Mirror/direct factorization (5204B, exact alias) has output nMSE 4.92793. HyperFormer-like linear generator (3072B) also misses quality at 3.85963.
- **C:** FiLM's native gamma/beta and low-rank task/layer coefficients already provide this function class.
- **U:** Trained Transformer tasks, HyperFormer++ implementation, natural held-out task generalization, GPU latency.
