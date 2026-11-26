# MA-1062 status

Phase: complete; **FAIL** for the registered quality/storage/lookup-latency gate.

The frozen MovieLens ua.test audit was run on two training seeds after AM-01 added the per-coordinate diagonal comparator and removed audit labels/IDs from saved outputs. The first invocation hit a pre-training runner bug; AM-02 records it and preserves the same split, selected epochs and seeds. No model update or metric was produced by the failed attempt.

H=512 Mirror logloss is 0.605938 vs 0.606059 paired-seed mean for the best H=512 diagonal, far short of the 0.002 gate; AUC is 0.716810 vs 0.765202 full DLRM. Payload is 50,595 B vs 187,591 B full (73.0% smaller), but lookup p95 is 5.4–6.6× plain shared hashing. Replay passes for all 18 method/seed payloads; test data labels and IDs are not persisted. See RESULTS.md and VERIFICATION.json.
