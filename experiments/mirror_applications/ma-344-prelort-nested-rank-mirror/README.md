# MA-344 — PreLort nested ranks with client phase views

Status: **PROMISING for storage/communication versus active client factors; no Mirror-specific gain over ordinary scalar phase**
Dedicated branch: `research/ma-344-prelort-nested-rank-mirror-20261008`
Prior art: PA39, PreLort prefix-nested federated LoRA.

## H — Hypothesis

For rank-heterogeneous clients using nested LoRA ranks 1/2/4, one shared prefix basis plus one phase per active segment can retain client functions while replacing large client-specific A factors. Direct coefficient and ordinary scalar phase controls test whether the storage change is Mirror-specific. An off-orbit client should require private factors.

## T — Test

Synthetic 16×16 linear adapters for 64 clients, with ranks distributed across 1, 2 and 4. Each active segment contributes a rank-one update whose right factor is selected from two shared segment matrices by a client-specific phase. One additional rank-4 client is off-orbit. Methods: independent full client matrices; PreLort-style shared prefix B with each client's active A factors; shared B/A1/A2 plus Mirror phases; direct cos/sin coefficients; an ordinary scalar-phase implementation with identical state; no-private Mirror; and hard sharing. Each client used 512 held-out vectors. Dev seeds 34401/34402; fresh seeds 34411/34412/34413.

All inference states were serialized as deterministic ZIP/NPY FP32 payloads and reloaded before evaluation. The phases and rank assignment are planted, and no optimizer updates were run. This is post-fit representation evidence, not federated training.

## D — Decision

**PROMISING for the narrow PreLort storage/communication comparison.** Across 3/3 fresh worlds, per-rank nMSE was 1.21e-14–1.39e-14. Mirror used a mean 4,713B vs 11,412B for PreLort-style active A factors (58.7% fewer bytes) and 4,713B vs 62,117B for independent full client matrices. Total per-client communication proxy fell from 8,768B to 608B across the bank, scaling with active rank. A shared hard adapter was smaller but had rank-specific nMSE 0.15–0.38.

The direct two-coefficient control used 5,234B, only 9.95% more than Mirror—just short of the frozen 10% Mirror-specific margin. More decisively, the ordinary scalar-phase control had identical bytes, hashes, and quality. Therefore the broad shared-basis/phase approach is promising against client-specific factors, but no Mirror-specific advantage is established. Without private state, the off-orbit task had mean nMSE 0.651; a private rank-4 factor restored near-zero error.

## Fact / Interpretation / Hypothesis

**Fact:** All three fresh worlds retained each rank group at nMSE <1.4e-14. Actual Mirror payload was 4,713B on average versus 11,412B PreLort active factors. The native phase control was byte/hash identical. The off-orbit client needed paid private factors.

**Interpretation:** Sharing per-segment bases and encoding active client variation as phase addresses moves this planted task family to a better storage/communication point than storing each client's active A factors. The result is not specific to Mirror parameterization.

**Hypothesis:** Larger nested-rank banks may retain this advantage, but it needs trained federated tasks and realistic rank distributions. A future Mirror-specific claim needs a better result than ordinary phase coordinates.

## C — Strongest counter-hypothesis

The reduction is explained by shared prefix bases and a known one-dimensional phase manifold. The ordinary scalar phase control is the same inference representation; the exact task orbit was planted.

## U — Unconfirmed

No optimizer steps, trained PreLort, client drift, communication rounds, real data, unseen phase learning, or Transformer/LM quality was tested. The code addresses are oracle-known rather than inferred from client support data.

## Reproduction

```bash
python experiments/mirror_applications/ma-344-prelort-nested-rank-mirror/source/run.py
python -m pytest -q experiments/mirror_applications/ma-344-prelort-nested-rank-mirror/tests
python experiments/mirror_applications/ma-344-prelort-nested-rank-mirror/source/verify.py
```
