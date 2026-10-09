# MA-534 — Role-conditioned logical MLP views over a sparse transcoder

Status: FAIL. The frozen development run tested whether sixteen role-specific Givens coordinates can express useful, input-dependent layer-3 MLP residual functions using a shared selected 16-atom transcoder view.

## H — hypothesis

Across 16 relation tasks, Givens coordinates fit on eight support examples per role will stay within 0.20 gold-candidate log-probability nats and 0.05 accuracy of an explicit per-role mean MLP-delta bank, use at most twice its actual serialized bytes, and beat equal-size pairwise gains and a stronger per-feature gate by at least 0.10 nats on both development seeds.

## T — protocol and execution

Pinned model: EleutherAI Pythia-70M-deduped, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, layer 3. Twelve tasks fit the support-only transcoder and shared atom pool; four disjoint relation tasks are held out for query evaluation. Each of 16 roles gets eight support-derived MLP-output delta examples. The query role ID is supplied externally; routing is out of scope.

The transcoder is a 512→2048 ReLU encoder, global top-32 latent activations, and a 2048→512 decoder, trained for exactly 1000 Adam updates. The selected 16 encoder/decoder atom pairs are the complete paid basis at inference. The view computes ReLU activations on those 16 selected encoder rows directly. This precise boundary is recorded in `IMPLEMENTATION_AMENDMENT_1.json`; the full 2048-feature global top-32 mask is not reconstructed from a partial dictionary.

Controls: no intervention; explicit per-role mean delta; unmodulated selected-feature delta; role-specific pairwise gains; role-specific elementwise gates; role-specific eight-angle Givens views. The three fitted controls use 500 Adam updates per role. Actual uncompressed NPZ payload bytes are authoritative. Compute includes support capture, transcoder training, per-role optimization, and candidate evaluation.

Protocol and amendment were committed before the first model run. Fresh seeds 53411–53413 stay sealed unless both development seeds pass every frozen gate.

## T — execution

Two pinned development seeds (53401, 53402); each seed re-trained the 2048-feature top-32 transcoder for 1000 updates and fit each role-code control for 500 updates on each of 16 roles. The four held-out task roles contributed 32 queries per seed. All six controls were evaluated. Replay reran both seeds from scratch; pool IDs, split manifests, actual payload bytes and every quality metric matched exactly. Total replay wall time was 84.17 s and 94.46 s; per-method evaluation times are in `results/seed_*/metrics.json`. The early implementation errors before persisted metrics are retained in `execution_log/attempts.jsonl`.

## D — decision

**FAIL.** The candidate met explicit mean-bank quality tolerance on both seeds, but failed the actual-byte cap and both Mirror-attribution margins. Fresh seeds 53411–53413 stayed sealed under the frozen rule.

## C — strongest counter-hypothesis

The independent elementwise sparse gate and equal-size pairwise gains explain any benefit. Givens has no robust advantage over either: it is worse than pairwise on both seeds and is worse than elementwise on one seed, while its selected 16-atom basis makes the deployment payload larger than the explicit delta bank.

## U — remaining limits

No learned router, larger model, different layer, natural held-out task family, private residual, or independent dense expert upper was evaluated. Each held-out task has eight queries, so accuracy changes in increments of 1/32. The transcoder feature code's online decoder compute is not included in the reported active-compute proxy as a hardware FLOP measurement; the proxy counts its fixed matrix operations.

## C — strongest counter-hypothesis

With only eight support targets per role, 16 selected decoder directions may fail to reproduce the required MLP-output changes. Pairwise gains or independent per-feature gates may explain any improvement without a Mirror-specific effect.

## U — limits

One small language model, one layer, one support-trained transcoder, 16 synthetic relation tasks, and externally supplied role IDs. This evaluates support-adapted residual behavior, not independent dense expert capacity or learned routing.
