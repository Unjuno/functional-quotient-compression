# MA-533 — Native skip-transcoder MLP approximation fidelity

**Result: FAIL under the preregistered raw-output gate.** On 2,048 held-out, non-overlapping Wikitext-2 train-token activation vectors, the pinned SmolLM2-135M layer-8 top-k=128 skip-transcoder had raw relative MSE 0.80601 and mean cosine 0.73807, missing gates <=0.10 and >=0.95. Fresh validation seeds stayed sealed.

The tokenwise L2 direction diagnostic reaches relative MSE 0.08673/cosine 0.95616, but uses each target output norm and is not deployable. A fit-only linear output-norm head makes it deployable but raw MSE is 0.14862, still above gate. Fit-only global RMS normalization performs worse (MSE 3.392).

The closest simple control is a rank-128 affine map: raw MSE 0.81749, cosine 0.47964, at 592,381 incremental bytes. The raw transcoder is only 0.0115 MSE better but costs 341,363,524 bytes (576x the rank control) and 16.2x native-MLP MACs.

Actual standalone serialization removes the replaced native layer-8 MLP weights before charging the replacement: native model 272,437,465 B, rank-128 replacement 267,721,062 B, raw transcoder replacement 608,492,205 B. Transcoder CPU evaluation took 3.320s vs native MLP 0.158s over 2,048 vectors.

See [protocol](PROTOCOL.json), [status](STATUS.md), [results](RESULTS_CORE.csv), and [verification](VERIFICATION.json).
