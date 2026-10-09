# MA-534 — Role-conditioned logical MLP views over a sparse transcoder

Status: SCREENING. The frozen development run tests whether sixteen role-specific Givens coordinates can express useful, input-dependent layer-3 MLP residual functions using a shared selected 16-atom transcoder view.

## H — hypothesis

Across 16 relation tasks, Givens coordinates fit on eight support examples per role will stay within 0.20 gold-candidate log-probability nats and 0.05 accuracy of an explicit per-role mean MLP-delta bank, use at most twice its actual serialized bytes, and beat equal-size pairwise gains and a stronger per-feature gate by at least 0.10 nats on both development seeds.

## T — protocol and execution

Pinned model: EleutherAI Pythia-70M-deduped, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, layer 3. Twelve tasks fit the support-only transcoder and shared atom pool; four disjoint relation tasks are held out for query evaluation. Each of 16 roles gets eight support-derived MLP-output delta examples. The query role ID is supplied externally; routing is out of scope.

The transcoder is a 512→2048 ReLU encoder, global top-32 latent activations, and a 2048→512 decoder, trained for exactly 1000 Adam updates. The selected 16 encoder/decoder atom pairs are the complete paid basis at inference. The view computes ReLU activations on those 16 selected encoder rows directly. This precise boundary is recorded in `IMPLEMENTATION_AMENDMENT_1.json`; the full 2048-feature global top-32 mask is not reconstructed from a partial dictionary.

Controls: no intervention; explicit per-role mean delta; unmodulated selected-feature delta; role-specific pairwise gains; role-specific elementwise gates; role-specific eight-angle Givens views. The three fitted controls use 500 Adam updates per role. Actual uncompressed NPZ payload bytes are authoritative. Compute includes support capture, transcoder training, per-role optimization, and candidate evaluation.

Protocol and amendment were committed before the first model run. Fresh seeds 53411–53413 stay sealed unless both development seeds pass every frozen gate.

## D — decision

Pending execution.

## C — strongest counter-hypothesis

With only eight support targets per role, 16 selected decoder directions may fail to reproduce the required MLP-output changes. Pairwise gains or independent per-feature gates may explain any improvement without a Mirror-specific effect.

## U — limits

One small language model, one layer, one support-trained transcoder, 16 synthetic relation tasks, and externally supplied role IDs. This evaluates support-adapted residual behavior, not independent dense expert capacity or learned routing.
