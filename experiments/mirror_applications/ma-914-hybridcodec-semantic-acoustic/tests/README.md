Upstream smoke/unit command:

`/tmp/ma914-venv/bin/pytest -q /tmp/ma914-hybridcodec/tests`

Result: 5 passed, 3 deprecation warnings, 8.13 seconds. Tests exercise synthetic module shapes and checkpoint formats. They do not exercise audio input, full pretrained inference, semantic/acoustic disentanglement, Mirror variants, actual deployment serialization or quality metrics. The first attempt in the shared runtime failed at collection because torchaudio's CUDA 13 wheel did not match CPU torch; the isolated matching CPU pair resolved that import failure.
