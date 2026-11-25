# MA-1057 status

Status: **FAIL (scoped)**. Official Moirai forecast, frozen calibration controls, audit metrics, corrected payload accounting, and replay verification are complete.

H — Factorized rate×horizon calibration improves a held frequency×horizon cell over native/additive controls using less state than independent calibrators.

T — Frozen Salesforce/Moirai-1.1-R-small on PEMS-BAY sensor IDs 14–21; fit on three frequency×horizon cells; dev-selected shared patch 32 and ridge λ=.1; audit only on 15-minute×1-hour for seeds 105702 and 105703 (24 windows each).

D — FAIL. Audit CRPS native 1.878/1.788, factorized 7.347/7.201, additive 7.346/7.201, and seasonal naive 1.733/1.733. Factorization is an exact additive-log reparameterization. Mirror complete payload is 55,321,654 B vs additive 55,321,641 B; calibration payloads include the 128 B normalization state required at inference. Seasonal naive costs 388 B without a model checkpoint.

C — Three fit cells poorly cover the held interaction; native Moirai is strong, and seasonal persistence benefits from its 24-hour context.

U — One network, chronology and held cell with eight disjoint sensors; seeds are sampling repeats, not independent worlds. No capacity claim.

A post-audit storage-accounting correction is disclosed in RESULTS.md; exact serialized float32 state was replayed and the decision was unchanged. The independent-cell calibrator is not an upper bound for the unsupported held cell.
