# MCX Stage-0 — measured mathematical cross-domain diagnostics (2026-10-08)

**Preregistered protocol:** [PROTOCOL.json](PROTOCOL.json); original protocol-only commit `ce378160a1c108891d8e989096f2df59f63dc60c` before any dev/fresh checks were run. **Evidence category: synthetic algebra only.** This does NOT run original TabM, Caduceus, DISCO models; no trained task-quality/LLM, GPU or compression claims.

## H / necessary scientific boundaries

1. For reverse-complement operator J with J²=I, an actual RC-equivariant representation H obeys H(Jx)=P H(x). Then orientation View outputs are exactly cheaply derived, but they are **group-orbit equivalent**, not independent useful new capability. A generic nonsymmetrized feature map does not obey this identity.
2. With a rectangular base W and nonzero n in ker(W), y(x)=y(x+n) while input-fast-weight member outputs (x⊙r)W and ((x+n)⊙r)W are generically different. Therefore **TabM's native input fast weights do not generally permit reconstruction from one pre-member final output**. Pure output scaling is exact and cheap.
3. For non-commuting PDE A,B, naive Lie splitting has O(dt²) local error. The native **Strang** symmetric splitting and simple BCH commutator correction reduce that error. The stronger native comparator must remain in every DISCO/Mirror experiment.

## T — frozen pilot

- CPU Python3.13, NumPy2.3.5, SciPy1.17.0, float64, OPENBLAS_NUM_THREADS=1.
- Dev whole worlds 11,12,13; untouched fresh worlds 101..105 (5); 3 time-step values dt=.05,.1,.2 s, 15 fresh rows.
- Original full runner/test source and dev/fresh raw CSV are in user-visible `mirror_crossdomain_stage0_20261008.zip`; this Git folder retains a standalone algebra reference, 5 regression tests and measured summary.
- All **7 original source unit tests passed**. In the five fresh worlds exact equalities' maximum absolute error **1.33226762955e-15**; minimum input-member collision output deviation **0.04284898989** despite identical original logits; generic RC feature parity deviation at least **2.07821366451**.

## D — finite CPU numbers, fresh 5/5

| dt (s) | Median relative Frobenius error: Lie | BCH-corrected Lie | Native Strang | Strang better than corrected |
|---:|---:|---:|---:|---:|
| .05 | 3.46889886e-4 | 1.14592976e-5 | 2.86433248e-6 | 5/5 |
| .10 | 1.39370993e-3 | 9.12304033e-5 | 2.27919404e-5 | 5/5 |
| .20 | 5.57044402e-3 | 7.22547996e-4 | 1.80141051e-4 | 5/5 |

Algebra **PASS**, generic cheap multi-functional output claim **not established**, native-Strang superiority in this fixture **observed**, Mirror-specific storage/runtime/quality **UNTESTED**. No numerical errors were hidden. This row-level evidence is a math audit, not an independent model benchmark.

## C — falsifications

- RC-only symmetry changes orientation of the same model function; 2 Views != 2 independently learned experts.
- For TabM input-side r, cached y=xW can be insufficient; keep x or intermediate and pay genuinely branch-specific compute.
- Operator splitting order matters. A cheaper Mirror code that ignores noncommuting terms can fail when dt increases, even if it reconstructs individual native operators exactly. Native symmetric splitting may dominate.

## U — numerical and statistical uncertainty

Float64 normal rounding ~1e-15 for exact identities; 5 paired synthetic worlds are insufficient for a population 95% statement. dt is time s, A,B units 1/s, exp(dtA) dimensionless. Errors are dimensionless relative Frobenius. No real serialized model byte or runtime measurement was made. Confidence claims beyond exact algebra are **not supported**.

## Original native sources

- [TabM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html).
- [Caduceus](https://proceedings.mlr.press/v235/schiff24a.html).
- [Physics neural-operator splitting](https://proceedings.mlr.press/v306/serrano26a.html).
- [DISCO](https://proceedings.mlr.press/v267/morel25a.html).

### Provenance of original standalone artifact

Original runner SHA256: `3c557463ec0454f5e1770934acd5fba7038ea83a4f1e90cf22c8514a4370dfde`; dev raw `0958afe09cf22c560072e448075e890355173faeadeb8aa2a732dc8bda0ea16d`; fresh raw `e97cd4cb090c9e6d0b67ce0f06e884ff686da520c619966a6ee5fa0502ce4c20`. The algebra reference in this Git folder is independently authored and is not claimed to have the same SHA as the original runner.
