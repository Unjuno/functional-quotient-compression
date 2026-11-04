# MA-647 — ReLoRA basis rotation schedule

Status: FAIL. Evidence lane: synthetic mechanism/storage/runtime screen.

## H

Can one shared butterfly schedule and a scalar per restart generate diverse rank-4 update subspaces with less storage than independent ReLoRA factors, without losing rank growth?

## T

Four phase-specific rank-4 updates compose a rank-16 matrix. Compared independent ReLoRA, independent BOFT angles, shared-angle scalar Views, native shared-angle BOFT, and a static rank-4 control. Two development seeds; complete payload bytes, rank, error, and operation proxy measured.

## D

The shared schedule reconstructs rank16 and saves 7.6% vs independent ReLoRA, but exactly equals native shared-angle BOFT in bytes and function. Static reuse caps rank at four and loses quality. Fresh stayed sealed.

## Fact / Interpretation / Hypothesis

- **Fact:** 10 rows; tests and replay pass.
- **Interpretation:** The savings are explained by native BOFT sharing.
- **Hypothesis:** Further ReLoRA restart-coordinate variants need to demonstrate a distinct advantage over BOFT or direct block IDs.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
