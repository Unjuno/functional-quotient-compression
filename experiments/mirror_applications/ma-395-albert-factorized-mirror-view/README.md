# MA-395 — ALBERT factorized embedding plus domain Mirror views

Status: **FAIL — low-dimensional Mirror phase missed held-out transfer gates.**
Evidence lane: QUALITY / STORAGE / HELD-OUT TRANSFER / COMPUTE
Base commit: `194aa9d`
Prior art: PA61, ALBERT factorized embeddings and parameter sharing.

## H — hypothesis

A shared ALBERT-style low-dimensional token table plus one cyclic Fourier/Givens phase per domain, applied before projection, should generalize to held-out token-domain pairs, exceed hard sharing, and approach a post-embedding adapter with fewer bytes.

## Mirror insertion

The shared path is `token table (256×8) → domain phase in 8D → shared projection (8×16) → classifier`. The phase uses one domain coordinate to rotate Fourier frequency pairs. Post-embedding rank-4 and full maps are separate controls.

## T — protocol and execution

Protocol frozen before development: two seeds (39501/39502), 128 tokens, eight domains/classes, 1,600 Adam updates, and seeded 25% held-out token-domain pairs. Target is `(token_id mod 8 + domain) mod 8`. Controls were independent factorized token tables, hard-shared factorized table, post-embedding rank-4 adapter, post-embedding full domain map, and Mirror phase. All receive the same domain ID and training pairs. Quality is measured after FP16 reload.

## D — decision

**Fact:** Held-out Mirror accuracy was 0.3227/0.4711, versus post-rank4 0.9801/0.9938 and post-full 1.000/1.000. Mirror held-out NLL was 14.685/5.846 versus rank-4 0.120/0.030 and full map approximately zero. Mirror payload was 6,066 B, below rank-4 8,808 B and full factorized 34,486 B, so the byte threshold passed. On seen pairs Mirror accuracy was 1.000/1.000, so failure is specifically transfer to withheld combinations. Hard shared embeddings scored near zero on held-out pairs. Ten serialized payloads replayed all seen/held-out metrics exactly; four tests passed. Fresh seeds 39511–39513 remain unopened.

**Interpretation:** Applying one phase inside the low-dimensional ALBERT embedding bottleneck compressed state but failed to transfer to unseen token-domain pairs. An ordinary post-embedding rank-4 adapter achieved near-perfect transfer at 1.45× the Mirror bytes, which is a better quality point under this protocol.

**Hypothesis:** The shared token table did not settle into a Fourier-aligned representation, so per-domain phase codes fit observed pairs but extrapolated poorly. The high held-out NLL shows the failure is not just a small accuracy-margin miss.

## C — strongest counter-hypothesis

This synthetic task has an exact cyclic-shift structure that favors Fourier phase views, but the low-dimensional table is learned from scratch under a fixed update budget. A better initialization or longer training could change the result; no such tuning was done after the frozen run.

## U — unresolved

Natural ALBERT transfer, multilingual corpora, learned frequency initialization, longer convergence, embedding-output tying, and deployment latency are untested. No general ALBERT or capacity claim is made.
