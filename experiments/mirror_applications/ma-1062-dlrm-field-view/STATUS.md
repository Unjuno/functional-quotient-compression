# MA-1062 status

Phase: corrected fresh audit is frozen and ready to retry; the first audit command failed before training or metric evaluation. The official `ua.test` numeric rows were loaded into memory after freeze but produced no predictions or metrics.

AM-01 saved only aggregate metrics/logits (no row labels or IDs) and added the per-coordinate diagonal to the strongest-control gate. AM-02 fixed the final-training `eval_rows=None` path. Development results, selected epochs, training seeds, controls, and audit split are unchanged; both amendments are documented in `PROTOCOL.json` and freeze history files.
