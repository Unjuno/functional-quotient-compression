# MA-783 status

- Status: **FAIL** (development gates; audit unopened)
- Branch: `research/ma-783-unipool-layer-role-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw 13 replay: passed; 555 candidates, seed/index/pool stored in `source/`
- Protocol: frozen before corpus acquisition; two amendments recorded before data
- Development: 12/12 runs completed (six conditions x two seeds; 1,200 updates each)
- Quality/storage gate: failed in seed 78301 on NLL; payload ratio was 0.430x untied
- Mirror-specific gate: failed against byte-near FiLM/depth controls
- Audit: not accessed, as required by the frozen gate
- Measured total training wall time: 915.9 s, single-thread CPU
- Results: see `RESULTS_CORE.csv` and `source/development_summary.json`

## H / T / D / C / U

**H:** Eight per-layer Givens coordinates on one shared global expert bank can recover near-untied layer specialization under a 70% actual-payload cap and outperform byte-near FiLM and depth embedding.

**T:** Four-layer width-64 causal character Transformer; two seeds; 1,200 AdamW updates per condition; six fixed controls; Tiny Shakespeare contiguous 80/10/10 byte split. Development only; audit span unopened.

**D:** **FAIL.** Givens used 612,501 B versus 1,423,166 B untied (0.430x), but missed the NLL margin in seed 78301 and did not beat both byte-near controls by 0.02 nats in either seed.

**C:** Shared expert pooling plus simple native conditioning captures the useful layer specialization; Givens adds bytes without a reliable quality gain.

**U:** Near-convergence capacity, other tasks/worlds, optimized runtime and audit generalization remain unknown.
