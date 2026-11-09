# UniPool layer Givens View diagnostic — 2026-11-09

## Evidence

- MA-783: a synthetic routed global pool plus layer Givens codes restores routed operator nRMSE to `1.0e-7` at 2,590 B vs 5,158 B independent. It adds 33% MAC, measured ~1.9× CPU time, and exactly aliases native direct-angle Givens.
- MA-784: on a designed four-function rotation orbit, K=1 plus role angles reaches nRMSE `7.8e-8`, but actual payload saves only 11.4% vs independent K=4 (below the frozen 25% gate), adds 12.5% MAC and ~4.5× CPU time vs K=1 without Views. Native Givens exactly matches the View. K=2 saves 1.8% vs K=4.

## Interpretation

The two consecutive UniPool/Givens screens reproduce standard direct-angle orthogonal parameterization exactly. The K=1 pool can express four orbit members, but measured compression, compute and CPU latency do not establish a Pareto benefit. The prior experiment's stronger apparent pool-size reduction is sensitive to how the full bank compresses on disk.

## Ruling

Pause further proposals whose only mechanism is adding per-layer direct-angle Givens coordinates around a UniPool expert. Resume only with a native-distinct function family or a realistic measured quality/latency gain over direct-angle OFT/Givens at equal complete bytes. This pause does not cover router compression (MA-782), expert merging (MA-785/786), upcycling, or other non-Givens pool mechanisms.

## H / T / D / C / U

- **H:** Small layer Givens Views reduce the physical expert pool needed at target quality.
- **T:** Synthetic fixed-router and one-plane orbit screens, MA-783/784; full source protocols and payloads are in their experiment directories.
- **D:** **FAIL** for Mirror specificity; the pool-size compression gate also fails at 11.4% savings.
- **C:** Direct native Givens/OFT provides the same layer function, while archive compression already exploits orbit structure.
- **U:** Trained MoE quality, realistic routers, fused GPU kernels and heterogeneous experts.
