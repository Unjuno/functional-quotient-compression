# MA-1055 — N-BEATS basis phase address

## H — hypothesis

A compact additive series-family × calendar-regime phase code on shared N-BEATS seasonal coefficients would improve held-out-role forecasts by at least 2% over hard sharing, a matched additive-gain control, and a common phase control, while costing less role state than flat role phases. The Mirror claim was restricted to factorized address compression: harmonic coefficient rotation is already representable by native N-BEATS coefficients.

## T — execution

**Fact.** Pinned ServiceNow/N-BEATS source `c746a4f13ffc957487e0c3279b182c3030836053`, official `NBeats`, `TrendBasis`, and `SeasonalityBasis`; isolated NumPy 2 fix changes only `np.float` to `float`. The vendored file hash is in `VERIFICATION.json`. CPU only. Six blocks (three degree-2 trend, three seasonality with upstream `harmonics=3` basis layout), two FC layers/block, width 128, Adam, batch 32, 300 updates/model.

Fit and development used four ETTm1 channels (HUFL/HULL/MUFL/MULL), train-only scaling, 96-point horizon and 512-point context. One development seed (105501) was used as a protocol calibration with no grid search. Fresh audit used the final chronological 20% target interval from ETTm2 for the same four channels, 145 non-overlapping windows/channel, seeds 105502 and 105503. ETTm1 was previously used by MA-1050. ETTm2 OT had been used in MA-1053, but the four audit channels were not; this is a held-out channel screen within an already used benchmark family, not an independent-dataset replication.

The phase operation rotates each non-DC forecast `(cos,sin)` pair together before the official basis expansion. Forecast DC, trend, and all backcast coefficients are unchanged. Controls were per-channel official N-BEATS models, one hard-shared N-BEATS, equal-shape series/regime seasonal log-gains, one common phase per frequency band, factorized series+regime phase, and a flat channel×regime phase table. Every checkpoint is an actual `torch.save` payload, including all model state and reconstruction metadata. Seeded audit checkpoints were reloaded and per-channel MASE independently replayed.

### Fact — audit results

Macro MASE across the four audit channels (lower is better):

| Method | Seed 105502 | Seed 105503 | Two-seed mean | Serialized bytes |
|---|---:|---:|---:|---:|
| Native per-channel upper control | 3.700853 | 3.586656 | 3.643755 | 19,893,327 |
| Hard shared | 2.323018 | 2.228537 | 2.275778 | 4,973,697 |
| Additive gain | **2.249147** | 2.313360 | 2.281254 | 4,974,395 |
| Common phase | 2.297236 | **2.161118** | **2.229177** | 4,974,078 |
| Factorized Mirror phase | 2.283816 | 2.244728 | 2.264272 | 4,974,395 |
| Flat role phase | 2.314288 | 2.359255 | 2.336772 | 4,974,078 |

Factorized phase improves on hard sharing by 0.51% on mean MASE, below the preregistered 2% threshold. It is 0.74% better than additive gain on the two-seed mean, but beats that control by more than 2% in only one of two audit seeds. It loses to common phase by 1.57% on mean MASE. Factor phase costs 698 B more than hard sharing and 317 B more than the flat phase control. Its full payload is about 75% smaller than four independent channel models, but hard sharing and the other shared controls are equally small or smaller; this compression is shared-base savings, not Mirror-specific. The fixed-budget independent control is not a quality upper bound: each channel model sees one quarter of the pooled examples.

Development calibration macro MASE was 2.784552 for hard sharing, 2.498602 gain, 2.665565 common phase, 2.692419 factor phase, and 2.650767 flat phase. The development runner initially also emitted ETTm1 audit metrics by mistake. That output is retained in `runs/exploratory_invalidated/`, was not used for selection, and ETTm1 audit was considered exposed; the audit target was moved to ETTm2 channels by documented amendment AM-01 before the locked fresh run.

Compute per fresh seed: each shared model used 300 updates, 9,600 example presentations and roughly 7.55B MAC-proxy units; native used four models, 1,200 updates, 38,400 example presentations and 30.20B proxy units. Shared-control training wall time was 3.4–4.3 seconds/model; native was 14.8–16.1 seconds. This is a fixed-update CPU screen, not a near-convergence capacity comparison.

## D — decision

**FAIL (scoped fixed-budget result).** The factorized phase address did not pass the 2% quality gate, does not beat common phase, and is larger than both hard sharing and flat phase in the stored payload. The common-gain control matched or beat it. No Mirror-specific storage, quality, or compute Pareto improvement was established.

## C — strongest counter-hypothesis

N-BEATS already learns native harmonic coefficients, so shared learned gain/phase adjustments capture the small regime change without needing a compositional series×regime code. The factorized operator changes coefficient coordinates rather than adding function capacity.

## U — remaining uncertainty

Only two fresh seeds and one benchmark family were evaluated; ETTm2 is cross-series evidence but not a new dataset family. The preregistered gate mentions two of three seeds while only two seeds were reserved for audit, so the formal 2/3 replication fraction is unresolved. The conclusion relies on the numeric quality and byte misses and is not a broad N-BEATS result. No capacity claim, near-convergence study, GPU throughput, or extra non-ETT family was tested.

## Verification

`pytest` structural tests pass (official zero-address forward parity, coefficient-pair norm preservation, split interval bounds). `source/verify_report.py` independently reloads all 12 audit checkpoints, validates payload size and SHA-256, confirms frozen source hashes and ETTm2 data hash, then recomputes every channel MASE exactly from serialized state. See `VERIFICATION.json`.
