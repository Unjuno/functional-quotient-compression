# MA-617 — factorized module × position Mirror code

Status: **FAIL** (development-only; fresh sealed). Prior art PA127.

## Fact

Two deterministic worlds evaluated all 27 ordered module triples from a three-module bank; nine triples per world were designated held out. Route-only modules used 2,085 B and had heldout mean nMSE 4.455/1.901. Position-phase Mirror reconstructed every triple with zero error at 2,337 B. The direct sine/cosine coefficient control produced the same functions and the same 2,337 B payload in both worlds. Independent per-triple matrices used 2,661 B. The amended MAC proxy was 29.33 per order for both View codes and 24 for route-only/independent. CPU timings were about 0.87–1.05 ms for Mirror, 0.83–0.88 ms for direct coefficients, and 0.60 ms for independent matrices. No optimizer updates; no fresh data accessed. Eight payload records replayed exactly.

## Interpretation

Factorized position codes express unseen route combinations and save 12.2% serialized bytes versus independent matrices in this small analytic setup. The direct coefficient code has identical quality, complete payload and MAC proxy, so the saving is ordinary shared-basis composition rather than Mirror-specific value. Mirror decode was slower than both direct coefficients and independent execution in these small CPU diagnostics. The frozen 0.90x byte gate against the direct code and 0.50x against independent were both missed.

## H / T / D / C / U

- **H:** a shared phase per composition position can specialize a reusable physical module bank for held-out ordered module compositions.
- **T:** two development seeds; 27 three-module sequences, nine designated held-out; route-only, phase View, direct coefficients and independent matrices.
- **D:** **FAIL** for Mirror-specific value; fresh sealed.
- **C:** phase is an ordinary normalized sine/cosine coefficient pair; archive alignment makes both encodings the same actual byte size.
- **U:** learned router/function discovery, trained nonlinear modules, natural tasks, larger networks and optimized kernels.

## Family ruling

MA-616 and MA-617 both fail Mirror-specific gates because direct coefficients exactly reproduce the View outputs at the same serialized size. Pause unchanged module-route/position Givens variants pending a materially distinct insertion or control.
