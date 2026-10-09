# MA-528 — Residual-selected shared SAE basis with behavior codes

Status: **FAIL**. Branch: `research/ma-528-sae-behavior-codes-20261009`. Frozen protocol hash: `7d24c4af3045027882e4b640d380b19aee097c4e2c1ba55d9ada17a549fbe782`. Prior art PA102: SAE feature steering depends strongly on feature choice and side effects.

## H — Hypothesis

A shared 64-atom SAE basis selected by residual reduction on tasks 0–11 plus 16-coefficient behavior codes can preserve held-out explicit-FV behavior within .10 gold-logprob nats and .05 accuracy, use <=.95x global per-task SAE OMP16 bytes, and beat a matched aggregate-activation top64 pool by >=.10 nats.

## T — Frozen execution

Pinned Pythia-70m with the same pretrained layer-3 SAE used by MA-526/527. For each development seed, tasks 0–11 selected the shared dictionary using 64 rounds of stable greedy residual correlation. Each of 16 behaviors, including the four held-out behaviors, received a deterministic 16-term signed OMP code. Held-out tasks contributed no atoms to the basis. Controls were no intervention, explicit FP32 FVs, global SAE OMP16 lists, and aggregate-activation top64 pool plus OMP16. Development seeds: 52801/52802. Fresh 52811–52813 stayed locked.

Serialized inference payloads are uncompressed NPZ. The pool IDs, local indices, coefficients and metadata are charged. The SAE is charged in standalone totals; decoded vectors are not stored. No optimizer updates were used. Support-derived function-vector extraction and sparse coding are reported separately from candidate inference.

## D — Decision

**FAIL; fresh data remains sealed.** Both seeds miss explicit-FV causal quality and actual-byte gates. Residual selection improves over activation-frequency pool selection, but does not beat global SAE sparse lists.

| Seed | Method | Bytes | Accuracy | Gold log-prob | Δ vs explicit FV | Δ vs aggregate pool | Δ vs global OMP16 |
|---|---|---:|---:|---:|---:|---:|---:|
| 52801 | Explicit FV | 34,826 | .2813 | −10.929840 | — | — | — |
| 52801 | Global OMP16 | 3,840 | .2813 | −12.067249 | −1.137409 | — | — |
| 52801 | Aggregate pool64 OMP16 | 3,990 | .2188 | −12.525457 | −1.595617 | — | — |
| 52801 | Residual pool64 OMP16 | 3,990 | .2188 | −12.166717 | −1.236877 | +.358739 | −.099468 |
| 52802 | Explicit FV | 34,826 | .2188 | −10.793003 | — | — | — |
| 52802 | Global OMP16 | 3,840 | .1875 | −11.859687 | −1.066684 | — | — |
| 52802 | Aggregate pool64 OMP16 | 3,990 | .1875 | −12.619658 | −1.826655 | — | — |
| 52802 | Residual pool64 OMP16 | 3,990 | .1563 | −11.949545 | −1.156542 | +.670112 | −.089858 |

Residual pool code bytes are 11.47% of explicit-FV bytes, but are 150 B larger than the direct global OMP16 list. Charging the required 4,204,391 B SAE makes standalone Pythia+SAE+View **4,173,555 B larger** than Pythia+explicit FVs. The residual selector takes .122/.069 s; pooled OMP16 coding takes .0055/.0049 s; global OMP16 takes .0525/.0805 s. Those CPU code-construction times do not include the larger residual-selector operation proxy. Candidate evaluation wall time was 13.64/15.63 s for residual pool, 18.96/16.21 s global OMP and 14.98/15.11 s explicit FV; these include support extraction, use unoptimized CPU code, and do not establish optimized serving throughput.

## Evidence classes

**Facts:** Three tests pass. Deterministic replay reproduced eight paid payload hashes, core metrics, task splits and both pool selections exactly. Fresh seeds were not accessed.

**Interpretation:** Choosing atoms against task-vector residuals helps compared with selecting by aggregate SAE activation, yet the learned pool does not reach explicit-FV behavior and its shared-pool bookkeeping costs more than independent global sparse IDs. This checkpoint's feature dictionary is not a compact drop-in representation for these task FVs.

**Hypothesis:** Broader or task-conditioned feature sets, private residuals, or natural SAE-steering benchmarks may behave differently. They require new protocols and should not inherit a capacity claim from this screen.

## C — Strongest counter-hypothesis

The observed residual-pool gain may reflect a better greedy feature-selection heuristic while global OMP remains the stronger function code: it has better likelihood on both seeds and costs 150 fewer bytes. Most of the quality loss may come from the SAE atom span, not from sharing the pool.

## U — Boundaries

One SAE checkpoint, one 70M causal LM and four held-out relation tasks were tested. This is fixed-support task-code evidence, not near-convergence capacity, broad feature interpretability, or general natural-language steering evidence.
