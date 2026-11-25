# MA-1056 — DLinear with a periodic residual View

## H — hypothesis

On a fixed DLinear linear/trend decomposition, a per-series annual phase code acting on a deterministic harmonic projection of the residual history may lower forecast error for held-out future targets, while costing less than independent per-series harmonic coefficients. The claim must beat a one-scalar-per-series seasonal gain, hard sharing, and tuned seasonal-naive controls at the same context/horizon. The evaluation asks whether the view helps at all against this inexpensive forecast family.

## T — method

Uses the pinned LTSF-Linear `DLinear.py` architecture (Apache-2.0 source) and the public Exchange-Rate dataset pinned by commit/hash in `PROTOCOL.json`. The Mirror insertion is post-DLinear: compute the first annual Fourier component of the official DLinear seasonal residual over the input context, then rotate its `(cos,sin)` coefficient pair by one learned phase scalar per currency. The signal amplitude is input-derived; phase is the only private Mirror coordinate. Controls include DLinear, DLinear plus zero-phase residual, an equal-size per-currency gain, independent per-currency harmonic coefficients, separate DLinear models, last-value and annual seasonal-naive forecasts.

The series are standardized with training-only statistics. Development chooses shared DLinear context and learning rate plus the seasonal-naive blend. Audit targets are chronological future blocks, untouched until the settings and checkpoint payload format are frozen.

## Results

See `RESULTS.md` after locked audit. Every model state, code, configuration, and reconstruction description is serialized and measured in bytes. Quality is macro MASE across currency series and target blocks; compute includes optimizer updates, examples, harmonic-projection proxy and CPU wall time.

## Fact / interpretation / hypothesis

The result is limited to this public daily exchange-rate benchmark and fixed update budget. The input-derived harmonic is a deterministic feature, not an extra parameter-free inference operation; its compute is charged. No capacity claim is made from a fixed-update screen.

Reproduction: `PROTOCOL.json`, `source/`, `tests/`, and `VERIFICATION.json`.
