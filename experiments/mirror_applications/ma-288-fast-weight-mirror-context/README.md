# MA-288 — Mirror fast-weight programmer context code

## H — Hypothesis

A dynamic context-to-code map can retrieve session functions with less state than explicit fast weights and generalize to held-out context identities.

## T — Conditions

Synthetic context-conditioned rank-2 linear rule, context dimension 8, memory/output dimension 16, 16 complete context identities (12 train, 4 held out), 64 support and 128 query samples/context. Fresh worlds 28810–28812 × seeds 0–2; 63 rows. Compared no memory, fast outer-product update, explicit/static context-ID tables, dynamic rank-2 Mirror code, generic rank-2 linear code, and generic MLP. Payloads charge all decoder/controller parameters. Fresh source/protocol frozen at `4970e0dc`. CPU only.

## D — FAIL for Mirror-specific value; compact dynamic code generalizes

| Method | Held-out NRMSE | Payload B | Update MACs | Query MACs | Fit seconds |
|---|---:|---:|---:|---:|---:|
| no_memory | 1 | 576 | 0 | 0 | 0.000 |
| fast_outer | 0.8907353 | 577 | 98,304 | 65,536 | 0.000 |
| explicit_updates | 1 | 1,096 | 0 | 0 | 0.000 |
| static_id | 1 | 1,089 | 0 | 0 | 0.203 |
| mirror_code | 1.909635e-06 | 264 | 0 | 0 | 0.152 |
| generic_lowrank | 1.746228e-05 | 268 | 0 | 0 | 0.154 |
| generic_mlp | 0.06700484 | 3,348 | 0 | 0 | 0.250 |

Fact: Mirror code reaches held-out NRMSE 2.0e-6 at 264 B; generic linear rank-2 reaches 1.7e-5 at 268 B. Generic MLP is 0.0670 at 3,348 B. Fast outer-product update is 0.891 at 577 B; explicit/static context IDs and no-memory are 1.0.

Interpretation: dynamic context encoding generalizes where static IDs do not, at low state bytes. The generic rank-2 controller matches its quality with nearly identical bytes, so no Mirror-specific advantage is established. The simple outer-product fast-weight estimator is poor on this clean context rule, but is much cheaper.

## C — Strongest counter-hypothesis

The task is generated from a rank-2 linear context rule, exactly matching the generic low-rank decoder. This demonstrates a structured context programmer, not broad memory capacity or real-session retrieval. The reported “update MACs” for learned controllers are not fully instrumented; they are represented by training wall time and should not be compared as exact operation counts.

## U — Unknown

No nonlinear context rules, noisy online updates, continual interference, natural prompt/session memory, or GPU runtime was measured.
