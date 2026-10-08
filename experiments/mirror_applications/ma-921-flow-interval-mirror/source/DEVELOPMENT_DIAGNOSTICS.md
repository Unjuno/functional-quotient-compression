# MA-921 development evidence

Canonical development outputs are `development_200_updates.json` and `development_500_updates.json`. Both were produced with the committed `run_experiment.py` code whose hash is in `development_config.json`. An earlier 200-update exploratory file was generated before deterministic reference-map caching; it is not used.

Each condition serializes its complete model into an in-memory torch payload, records the actual `len(payload)` and SHA-256, then loads it and verifies exact tensor equality before evaluation. The all-condition result rows and gate computations are in `RESULTS_CORE.csv` and `verification_summary.json`. Fresh seeds were not run.
