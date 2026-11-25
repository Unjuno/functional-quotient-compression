# MA-1059 — Natural time-series View geometry

**Mirror insertion:** this experiment adds a compact series coordinate `m` to the shared temporal forecast interface so that heterogeneous series functions can vary without duplicating a full per-series forecast head.

The test is whether naturally occurring series lie near a shared function family that a low-description View can express, and where sparse private trend state becomes necessary. PA328, PA330 and PA335 define the strongest native controls: Moirai cross-series/frequency conditioning, TimeMixer scale mixing, and N-BEATS basis coefficients.

Before opening fresh data, choose a licensed public dataset and record source hashes, series groups and train/dev/fresh splits. Compare native shared forecast, hard sharing, byte-matched additive/diagonal or low-rank controls, Mirror, and independent per-series heads. Only development may choose code rank and private residual threshold.

Storage is the actual serialized inference payload, including model, bases, coordinates, IDs, normalizers and private residuals. Results are a screening result for one benchmark family; fixed-budget gains are not capacity evidence.
