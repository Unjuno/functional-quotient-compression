# MA-015 — domain-factorized Mirror experts

Status: **PROMISING**; fresh aligned gates passed 3/3.
Branch: `research/ma-015-domain-factorized-experts-20261007`.
Base commit: `0f875318639af56f7ff6089cddceaa93d305d480`.

## H — Hypothesis

One shared four-expert bank plus a small view per domain can represent twelve domain×expert functions at lower actual payload than twelve independent MLPs.

## T — Test

Three balanced domains, four oracle top-1 experts per domain, 16→32→16 GELU. The aligned teacher shares an expert bank and applies domain-specific Givens views; an independent teacher measures the private-function boundary. Controls are independent experts, tying, scalar gates, and rank-1/rank-2 residuals. Development world 150000 selects LR; fresh worlds stay sealed until both access conditions pass.

## Development result

- Selected LR 0.003 on world 150000.
- Mirror MSE was 6.34e-4 vs 9.22e-4 independent (0.688x); actual payload was 18,727B vs 51,060B (0.367x). Pre-fresh gate passed.
- Rank-2 residual had MSE 6.13e-4 at 23,270B; hard tying used 18,221B at MSE 1.07e-3. Mirror is close in quality to rank-2 but 19% more bytes than hard tie.
- On unrelated domain/expert functions, Mirror MSE was 2.39e-2 vs 9.57e-4 independent; private state required.

## D — Decision: PASS on registered synthetic gate; PROMISING overall

Mirror/full MSE ratios were 0.556, 0.503, and 0.621 across fresh worlds. Actual payload was 18,727B vs 51,060B independent (0.367x; 63.3% fewer bytes). Mirror used 2.8% more bytes than hard tying and reduced MSE by 37.5–60.6%. Compared with rank-2 residual, Mirror was 4.1–6.1% lower MSE and used 19.5% fewer bytes.

Runtime regressed: median eager CPU throughput was 0.517M examples/s vs 0.937M tied and 0.929M independent; median training wall was 2.28s vs 1.10s tied. Active proxy was 1,024 MACs plus about 12 Givens coordinate FLOPs per example (rank-2 uses 1,120 MACs). On unrelated domain/expert functions, Mirror MSE was 0.0234–0.0252 vs 0.00091–0.00097 full independent.

All 36 fresh deterministic rows replayed exactly (timing excluded); eight frozen hashes were verified against commit `7be4b69d5fc2346f68ad5f47920a7c451f40a2fb`. The aligned teacher was deliberately factorized, so this is not natural-domain or capacity evidence.

## C — Strongest counter-hypothesis

The aligned teacher is constructed from the same domain-view structure. Real domains may require independent experts or stronger domain-adaptation controls.

## U — Unconfirmed

Learned routers, natural domains, near-convergence capacity, optimized kernels, and language quality remain untested.

## Fact / interpretation / hypothesis

- **Fact:** all three fresh worlds passed aligned quality/storage; Mirror payload was 0.367x independent and 1.028x hard tying.
- **Interpretation:** a domain coordinate crossed with expert identity recovered the aligned synthetic functions compactly, while unrelated domains needed private weights and eager runtime regressed.
- **Hypothesis:** some trained domain-specific expert pools may share a compact domain orbit; that prevalence remains unknown.
