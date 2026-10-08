# Mathematical Formula Stage-0 — Exact Algebra and Constructed Negative Cases

**2026-10-08 / DESIGNED BEFORE RUN / EXECUTED / NOT A TRAINED-MODEL BENEFIT.**

[Preregistered protocol](../../FORMULA_STAGE0_PROTOCOL.json) was committed as `b71c4a1ccfd2696ca9b8def2c46e4b35ca102dac` **before** development or fresh values were opened. Run from the source directory:
```
OPENBLAS_NUM_THREADS=1 python run_formula_audit.py --output ../results
python test_formula_audit.py
```

## Scope and environment

- NumPy 2.3.5, CPU x86-64 Linux, float64 exact matrix/activation controls; no neural network training, no GPU, no throughput claims.
- 12 independent algebraic formula checks F01–F12 on each synthetic world; 3 dev worlds (11,12,13) and 5 frozen fresh worlds (101..105). No hyperparameters tuned.
- Original equations, variables, source docs and limitations: [FORMULA_LEDGER](../../FORMULA_LEDGER.md).

## H / T / D / C / U

- **H:** Under the predeclared finite-dimensional assumptions, folding, weighted reachability, finite linear gradient iteration, simplex support, phase noncancellation, exact canonical KV, PSD coupling majorization, serializer hard budgets, decoder-prerequisite quotient, joint-layout complementarity, activation-weighted linear error and noncommuting execution all satisfy their algebraic contracts, including negative cases.
- **T:** Run 12 formula checks with seeded finite matrices, and explicitly require the counterexamples (wrong cache state, wrong diagonal-only distortion, raw 76-bit false pass, weight-only misranking, noncommuting residual sum). Exact code and paired rows in `source/` and `results/`.
- **D:** All **12/12 checks passed in 5/5 frozen fresh worlds**. Maximum float64 algebraic discrepancy across the recorded 12 columns was `1.4210854715202004e-14` (the eigenvalue-100 numerical comparison); minimum invalid cache reuse difference was `0.033955041907426375`, robustly nonzero. Raw 76 bits needed 88 serialized bits with 5-bit header and byte rounding. Constructed joint-optimal distortion advantage was 5 versus no one-axis move; that **is not a real model compression gain**.
- **C:** The invariances can hold while a low-description m has **no additional function utility or storage gain** over a native linear/shared-basis method; prior SM, SRM and real-digit trials provide precisely such counterexamples.
- **U:** Finite floating point precision, matrix conditioning, domain of invertibility/PSD, finite synthetic samples and no training error. These are deterministic algebraic checks; statistical confidence intervals for natural task generalization are not justified. Preserve each seed and SHA-256. SI: tensor errors and NLL are dimensionless as normalized numbers, serialized bits/bytes and elapsed seconds are distinct axes.

## Exact provenance and execution artifacts

- [Audit source](source/run_formula_audit.py); [2 unit tests](source/test_formula_audit.py).
- [3 development rows](results/dev_raw.csv); [5 fresh rows](results/fresh_raw.csv); [verification manifest](results/VERIFICATION.json).
- SHA-256 of source: `17920fd25d6060155d68964bdf71c25b4ccdc2d722e9f60e071ee3a0b12a4c0c`.
- SHA-256 dev: `08c438cd5830570d252e79b629e20755c8eb54506f5bda283cdd3d273d00eb58`; fresh: `5fba42ae4de8aa54ab821ffb97641a2063d6b5c93b96560c9b52c2518c530e83`.
- Four compound candidate IDs (MA-1171..1174) remain **UNTESTED** for learned task utility. Exact algebra is evidence for implementation premises, not a scientific adoption result.

### Variable/units

See prior defined symbols in [the formula ledger](../../FORMULA_LEDGER.md). Here `epsilon` is dimensionless max absolute numerical discrepancy (real scalar >=0), `S` is actual serialized bits/bytes (nonnegative integer), `L` denotes task loss in nat/token (dimensionless real), `t` is runtime in SI seconds (real >=0). Never infer equal-time benefits or add unlike units.

## Post-publication replay qualification

A second independent **process launch in the same container environment** with `OPENBLAS_NUM_THREADS=1` reproduced both CSV SHA-256 digests byte-for-byte (3 dev / 5 fresh). An intentionally stripped Python environment (`env={PATH,OPENBLAS_NUM_THREADS}`) preserved every algebraic PASS and all five worlds but altered floating-point low bits of several outputs (approximately machine epsilon), so **bitwise reproducibility is conditional on the recorded NumPy/BLAS/process environment**, not guaranteed across arbitrary runtimes. The scientific decision did not change. The exact two original CSVs and their source SHA are immutable above.
