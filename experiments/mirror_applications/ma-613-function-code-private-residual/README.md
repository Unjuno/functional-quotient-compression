# MA-613 — Shared function code basis with sparse private residuals

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / PRIVATE-STATE BOUNDARY
Base commit: `50d9c1f1` (worker-ready baseline plus MA-602–612 evidence)
Prior art: PA128 Neural Interpreters

## Hypothesis

H: Shared signature + Givens function codes remain compact and accurate when only a minority of functions need off-orbit private residuals; actual residual index/value bytes should reveal where the shared function orbit stops being worthwhile.

## Mirror insertion

> **Mirror insertion:** this experiment adds a sparse private residual table to shared signature/phase Mirror codes, so rare functions outside the shared Fourier orbit can be recovered without storing a full function code for every logical function.

## Scope and gate

Deterministic function-code screen across private-function fractions 0–100%. Compare no residual, partial residual sweep, complete residual recovery, and full per-function Fourier code. PASS for sparse recovery requires exact quality with <=25% private functions and <=0.85× the full-code payload. All index and residual values are charged. This is not a trained Neural Interpreter result.

## Findings

### Fact

Across two deterministic task grids (61301, 61302), the full per-function payload was 5,401 B. The shared signature/phase code with exact sparse residuals used 2,969 B at 25% private functions (0.550x full) and 3,353 B at 100% (0.621x). At 25%, omitting half the residuals gave NRMSE2 0.01485; exact residual recovery gave NRMSE2 below 2.5e-15. No optimizer updates or fresh worlds were used. Payload hashes and exact serialized sizes were checked for all 36 records.

### Interpretation

The frozen mechanism gate passes for this analytic function family: up to 25% private exceptions are recoverable with full precision and substantially fewer serialized bytes than independent codes. Sparse residual values and indices are paid. The byte advantage persists at 100% because the shared code structure remains cheaper than ten coefficients per function. Partial recovery trades bytes for error approximately in proportion to unrecovered private functions.

### Hypothesis / limits

This does not establish learned-interpreter capacity or natural task compression: targets and exception locations are known deterministically, the family is Fourier-structured, and there is no learned selector or noisy adaptation. The strongest counter-hypothesis is that the apparent gain is entirely due to a hand-designed shared analytic basis.

**H:** for the specified analytic family, sparse private residuals preserve arbitrary off-orbit exceptions while retaining a payload advantage. **T:** two deterministic grids, five private fractions, zero/half/full residual recovery, full coefficient control. **D:** PROMISING (mechanism/storage only). **C:** hand-designed Fourier structure explains the entire result. **U:** learned code discovery, unknown exception detection, noisy data, natural functions, and independent fresh replication.
