# MA-1057 status

Status: **FAIL (scoped)**. Official Moirai forecast, frozen calibration controls, audit metrics, payloads, and replay verification are complete.

H — Factorized rate×horizon calibration improves a held frequency×horizon cell over native/additive controls using less state than independent calibrators.

T — Frozen Salesforce/Moirai-1.1-R-small on PEMS-BAY sensor IDs 14–21; fit on three frequency×horizon cells; dev-selected shared patch 32 and ridge λ=.1; audit only on 15-minute×1-hour for sampling seeds 105702 and 105703 (24 windows each).

D — FAIL. Audit CRPS native 1.878/1.788, factorized 7.347/7.201, additive 7.346/7.201, and seasonal naive 1.733/1.733. Factorization is an exact additive-log reparameterization and costs 13 more serialized bytes than additive (398 vs 385 B of method state). It fails quality and state gates.

C — Cross-cell calibration fitted on only three cells is misspecified for the held interaction; the shared base forecasts were already good on the held cell. Seasonal persistence exploits the 24-hour context more effectively.

U — Only one network, one held cell, eight disjoint sensors, and stochastic-sampling repeats on one chronology; no independent dataset-family or capacity claim. Full tables, runtime, provenance, and verification are in RESULTS.md and VERIFICATION.json.
