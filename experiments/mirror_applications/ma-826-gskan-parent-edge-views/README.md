# MA-826 — GS-KAN parent function with edge Views

## H / T / D / C / U

**H:** Compositional Mirror edge Views over a shared parent improve on native GS-KAN affine transforms in full bytes or function-evaluation frontier while retaining heldout edge quality.

**T:** Synthetic analytic parent `sin(x)+0.2x^3`; 32 edge functions with affine scale/shift; eight edges scored separately; 8,192 samples; seed 82601; no training. Compared independently sampled edge functions, native GS-KAN affine parent, Mirror affine Views, and affine plus unit modulation.

**D: FAIL for Mirror-specific attribution.** Native GS-KAN and Mirror affine payloads are both 1,096 B and have zero heldout error, identical function evaluations, and equivalent outputs. The independent sampled edge table is 8,438 B with nRMSE 8.85e-5. Adding modulation increases Mirror payload to 1,295 B without reducing function evaluations or error.

**C:** The directly relevant GS-KAN parameterization exactly accounts for the shared-function compression. Unit modulation is a redundant extra code.

**U:** Trained spline KANs, learned parent functions, natural graph tasks, fused edge-function kernels and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Actual payload bytes/hashes, heldout errors, number of parent evaluations and replay checks are in `RESULTS_CORE.csv`.
- **Interpretation:** Shared-parent edge functions compress the synthetic bank, but Mirror affine views are the same method as GS-KAN.
- **Hypothesis:** Any further Mirror claim must beat transformed-parent controls through composition, robustness or runtime, not field names or extra modulation state.
