# MA-582 amendments

## Amendment 1 — cache matrix layout

- Discovered before any registered development result was retained: seed 58201 stopped during calibration.
- Defect: KV arrays are `[batch, head, sequence, dim]`; the frozen encoder reshaped using the head count as the row count.
- Correction: reshape using sequence length (`k.shape[2]`). Added a layout unit test.
- The failed invocation is excluded and produced no metrics artifact. Protocol, gates, data, and seeds are unchanged.
- New source is frozen and hashed in `FREEZE.json` before rerunning the registered development seed.
