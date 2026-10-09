# MA-589 amendments

## Amendment 1 — residual basis indexing

- First registered seed 58901 invocation stopped before metrics serialization; it is excluded.
- Defect: residual basis storage is a Python list of layer arrays, but encoding indexed a layer array with a tuple.
- Correction: added shared residual encode/decode helpers with a unit test and corrected nested indexing through the helper.
- Seeds, data, protocol, and gates are unchanged. Updated source hash is recorded in `FREEZE.json` before rerunning development.
