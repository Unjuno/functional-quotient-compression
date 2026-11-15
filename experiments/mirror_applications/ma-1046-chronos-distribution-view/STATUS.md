# MA-1046 status

- Status: SCREENING
- Branch: research/ma-1046-chronos-distribution-mirror-20261115
- Official checkpoint: amazon/chronos-t5-tiny revision 29d808298f1a62493e7b9a5e08529d0d930fa189
- Inference package chronos-forecasting 1.5.3 installed in isolated /tmp/ma1046-env using system CPU PyTorch
- Protocol frozen before downloading or inspecting any forecast evaluation data
- Development: ETTh1 hourly and ETTm1 15-minute; fresh: Chronos zero-shot Weather daily and Australian Electricity 30-minute
- AM-01 corrects ETTm1 benchmark horizon to the PA326 value 24 before any forecast metrics; 96-step daily period remains for MASE. AM-02 corrects WQL aggregation to pool loss numerators and absolute-target denominators before division; CRPS selection and audit remain unchanged. Development completed on 42 windows each from ETTh1 and ETTm1. Frozen final angles: 0.00 rad at 15m and -0.05 rad at 1h; no audit forecasts accessed. AM-03 records the downloaded Weather archive having 3,018 rows (paper table: 3,010); selected audit indices are unchanged. AM-04 fixes TSF preamble parsing. Weather forecasts from seeds 104601–104603 are preserved but invalidated; the final confirmatory seeds are 104604–104606. No Australian forecasts were generated in the stopped attempt.

## Next action

Fetch the pinned development and fresh archives, record checksums/provenance, run development selection only, freeze the two frequency-code values, then run the locked fresh audit.
