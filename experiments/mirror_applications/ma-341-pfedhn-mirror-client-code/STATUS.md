# MA-341 — NOT ESTABLISHED

## H

On held-out non-IID Digits clients, a four-angle shared-base Mirror code can approach pFedHN's unseen-client quality after 20 support updates while reducing actual server/client bytes and avoiding domination by byte-near FiLM.

## T

Three fresh client partitions (world seeds 23/47/71) and three model seeds (31/47/59); clients 0–9 train shared systems and clients 10–11 are held out. Compared pFedHN-style 8D code to full 2,410-parameter MLP generation, 4-angle Mirror, 4D FiLM, global shared MLP, and independent local MLP. Shared methods used at most 80 epochs with train-client-only dev early stopping. Held-out adaptation updated only the code, with shared parameters frozen; the independent control initialized a fresh local model and adapted on support. Query data was held for evaluation. Actual safetensors files define payload bytes.

## D

**NOT ESTABLISHED for a Mirror-specific overall advantage.** After amendment and full rerun, the quality gate passes: for every world, both held-out clients are within 2 points of pFedHN after averaging the three model seeds. Mirror server+registered-client payload is 10,792B versus 630,248B for pFedHN; per-client code download is 88B versus 104B. Corrected FiLM reaches mean unseen accuracy 0.9555 versus Mirror 0.9673, with 11,960B combined state and the same 88B/client download. Mirror is faster to train than FiLM (1.78s vs 1.34s), while FiLM code adaptation is faster (7.5ms vs 16.4ms). These tradeoffs do not establish a uniformly better Mirror point.

## C

Small client-conditioned FiLM codes already recover nearly all useful variation; the small pooled accuracy difference can arise from partition noise and does not justify Mirror-specific complexity.

## U

Only two held-out clients per world and one small dataset were used. The pFedHN-style generator is a direct reimplementation rather than a full paper reproduction. Privacy, real-user federation, communication rounds, optimized batch-1 latency and near-convergence capacity remain untested.

## Fact / interpretation / hypothesis

- **Fact:** Initial FiLM bases were zero initialized, so the first run had a degenerate FiLM control. Amendment 1 changed the learned bases to seeded N(0,0.02), added a nonzero-code-gradient test, and reran all worlds/seeds. On the corrected run, unseen accuracy was Mirror 0.9673, pFedHN 0.9610, FiLM 0.9555; Mirror/pFedHN difference was within 2 points for both held-out clients in 3/3 worlds.
- **Fact:** Corrected-run serialized server+registered-client payloads were Mirror 10,792B, FiLM 11,960B, pFedHN 630,248B. Per-client code payload was 88B for Mirror/FiLM and 104B for pFedHN.
- **Interpretation:** Mirror gives a small quality and shared-payload improvement over FiLM, but adaptation is slower. On this screen that is a tradeoff rather than a broad Pareto improvement.
- **Hypothesis:** More heterogeneous client tasks and larger support/query samples may change relative quality, but require a separately frozen experiment.
