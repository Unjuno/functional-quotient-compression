# MA-592 results — prompt pool composition

## Fact

The two frozen MA-591 soft-prompt banks were reused without retraining, each with eight article prompts and four held-out 64-token query contexts per task (32 queries/bank). Routing used cosine similarity between mean final hidden states from the frozen Pythia backbone, top-2 selection, and softmax temperature 0.1. `m` is the two selected pool indices plus their two weights. The same-seed replay reproduced titles, routes, NLL values, hit rates and bytes exactly.

| Bank | Method | Mean target NLL | Delta vs oracle | Top-1 route accuracy | Top-2 containment | Persistent bytes |
|---|---|---:|---:|---:|---:|---:|
| 59101 | Oracle prompt | 4.46637 | 0 | 65.6% | 87.5% | pool+keys 77,778 B |
| 59101 | Native L2P top-1 | 4.50260 | +0.03623 | 65.6% | 87.5% | 77,778 B |
| 59101 | Native L2P top-2 | 4.51913 | +0.05276 | 65.6% | 87.5% | 77,778 B |
| 59101 | Mirror top-2 | 4.51913 | +0.05276 | 65.6% | 87.5% | 77,778 B |
| 59102 | Oracle prompt | 4.15049 | 0 | 53.1% | 68.8% | pool+keys 78,802 B |
| 59102 | Native L2P top-1 | 4.18575 | +0.03526 | 53.1% | 68.8% | 78,802 B |
| 59102 | Native L2P top-2 | 4.19582 | +0.04533 | 53.1% | 68.8% | 78,802 B |
| 59102 | Mirror top-2 | 4.19582 | +0.04533 | 53.1% | 68.8% | 78,802 B |

The eight-prompt bank plus serialized FP16 keys cost 77,778/78,802 B. Dynamic top-2 coordinates add 6 B/query (two uint8 indices and two FP16 weights), or 192 B across 32 queries. Mirror top-2 and native L2P weighted top-2 have equal NLL and execute the same weighted sum. Top-2 composition is worse than top-1 by +0.01653/+0.01007 nat/token. For bank 59101, top-2 also exceeds the oracle tolerance (+0.05276 vs +0.05). The best top-1 router is within oracle tolerance in both banks, but routing accuracy is only 65.6%/53.1%.

Key extraction took 0.878/1.058 s; routing all 32 queries took 0.878/1.058 s; mixing took 1.95/2.32 ms. Inference per 32-query bank took 2.17/2.25 s for top-1 and 2.24/2.27 s for weighted top-2. Each path uses eight virtual tokens. Total run wall time was 30.28 s; these CPU timings have no fused retrieval kernel.

## Interpretation

**D: FAIL.** The Mirror coordinate is exactly the native L2P-style top-2 softmax composition and provides no different function. In both seeds, top-2 composition is worse than simpler top-1 routing; one bank misses the oracle NLL gate. The query-conditioned weighted pool also shows substantial routing error. Fresh data remain sealed.

## H / T / D / C / U

**H:** A small query-conditioned coefficient vector over a shared prompt pool can compose task-specific prompts, retaining oracle quality and improving on native top-1 selection without storing per-query private prompts.

**T:** Two MA-591 prompt banks, Pythia-70M hidden-state cosine keys, 32 held-out contexts per bank, oracle/top-1/top-2/Mirror/zero controls, target NLL, routing accuracy/containment, actual pool/key/code bytes, routing/mixing/inference time. Same-seed replay matched all core results.

**D:** FAIL. Top-2 misses the improvement-over-top-1 gate in both banks, one seed misses oracle tolerance, and Mirror exactly aliases native weighted prompt selection. Fresh sealed.

**C:** The two selected prompts are often nearly tied in cosine similarity but encode unrelated article functions; averaging them interferes. Native L2P already implements the weighted view.

**U:** More tasks, learned queries/keys, class-incremental vision retention, alternative routing temperatures, longer contexts, and optimized kernels.

## Fact / Interpretation / Hypothesis

- **Fact:** Native top-2 and Mirror top-2 have identical NLL and byte-identical mechanism; both are worse than top-1.
- **Interpretation:** Prompt-pool composition did not add quality or capacity beyond native query-key retrieval in this screen.
- **Hypothesis:** A task-aware composition objective or diversity regularizer could improve the pool, but must beat native L2P and a simple gate at matched bytes.
