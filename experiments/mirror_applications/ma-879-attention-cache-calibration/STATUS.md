# MA-879 status

Status: PROMISING — Draw32 and update settings frozen before fresh query audit.

## H

Attention-weighted calibration improves functional KV preservation over raw cache-MSE fitting for fixed-size Mirror coordinates.

## T

Four-context causal attention screen, aligned and unrelated KV translators, development worlds 3/7, fresh audit worlds 11/17/23, model seeds 31/47/59.

## D

**PASS aligned calibration gate.** Same-size Mirror attention calibration reduced held-out output MSE relative to raw-cache Mirror in 3/3 audit worlds (pooled 0 vs 1.33e-6; 336 B each), while attention KL/NLL did not regress against independent attention mappers. Weighted ridge was nearly exact at 688 B; basis2 nearly exact at 512 B. Off-orbit Mirror output MSE was 0.319. CPU latency 0.404 ms Mirror vs 0.0087 ms weighted ridge.

## C

The aligned teacher favors Mirror; raw-cache Mirror is already near exact, weighted ridge is exact and much faster, and unrelated maps need independent capacity.

## U

Synthetic mechanism only; no natural LM, autoregressive likelihood or end-to-end cache transfer.
