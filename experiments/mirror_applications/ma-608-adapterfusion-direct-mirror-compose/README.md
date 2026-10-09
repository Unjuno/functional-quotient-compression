# MA-608 — Direct Mirror composition versus AdapterFusion

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `4c935606` (worker-ready baseline plus MA-602–607 evidence)
Prior art: PA125 AdapterFusion

## Hypothesis

H: AdapterFusion over shared low-rank source adapters can compose source Views in coefficient space before execution, preserving linear adapter outputs exactly while replacing N adapter forwards with one shared projection; this may not extend through nonlinear bottlenecks.

## Mirror insertion

> **Mirror insertion:** this experiment adds per-source Givens coordinates to a shared adapter basis, then fuses their codes into one input-conditioned effective rank operator before the adapter projection, avoiding repeated source-adapter execution.

Controls: full independent AdapterFusion; shared source adapters executed individually; direct composition of Mirror Givens codes; direct composition of generic 6×6 coefficient matrices. The same fusion gate weights are used in all paths.

## Scope and gate

This is a deterministic algebra/runtime screen with no optimizer updates. PASS for a general AdapterFusion replacement requires exact linear composition (max error <=1e-6), >=25% fewer payload bytes than full independent adapters, >=40% lower per-example compute proxy than AdapterFusion, and nonlinear bottleneck composition NRMSE <=0.05. Any miss is FAIL for the broad replacement claim; linear exactness is reported separately. Runtime basis reconstruction and resident matrix bytes are reported.

## H / T / D / C / U

**H — hypothesis:** compose four shared low-rank adapter Views into one effective rank operator before execution, preserving the linear AdapterFusion result while reducing repeated adapter work; the same operation may be approximate through tanh bottlenecks.

**T — execution:** deterministic CPU algebra screen in two seeded worlds, 4,096 inputs/world, eight 32→6→32 source adapters, input-softmax fusion gate. Compared full independent factors, shared Mirror adapters executed separately, Mirror code fusion before execution, and unrestricted shared 6×6 matrix-code fusion. No optimizer updates. Serialized every inference payload and replayed from saved state. Counted angle-to-matrix setup (120 Givens operations/world) and 1,152 B resident reconstructed matrices separately from payload bytes.

**D — PASS for this scoped mechanism screen:** linear outputs matched within 2.4e-7 max absolute error in both worlds. Mirror payload was 5,529 B versus 15,645 B for full independent adapters (64.7% fewer) and 6,233 B for the unrestricted shared matrix-code control (11.3% fewer). Compute proxy was 1,220 MAC/example versus 2,528 for shared AdapterFusion (51.7% lower); measured CPU latency for 4,096 examples was 1.23–1.27 ms for direct composition versus 4.06–4.17 ms for shared AdapterFusion. The nonlinear tanh stress case had normalized MSE 0.0123/0.00663, below the frozen 0.05 gate. No fresh set was required for this deterministic algebraic experiment.

**C — strongest counter-hypothesis:** the nonlinear result is helped by the small-activation, near-linear regime; stronger nonlinearities or context gates that depend on intermediate adapter representations could prevent collapsing the adapters to one code. Direct generic matrix-code execution uses the same per-example compute, so Mirror's incremental benefit there is storage (704 B) rather than compute.

**U — boundaries:** deterministic low-rank synthetic adapter algebra only, no trained task adapters, optimizer evidence, natural AdapterFusion checkpoint, or language evaluation. The result establishes the algebraic/compute point under shared factors, not general task composition.

## Facts / interpretation / hypothesis

- **Fact:** all six serialized payloads match actual file size and SHA-256; payload reload max error <=2.4e-7; both tests pass.
- **Interpretation:** when source adapters share the same input/output factors and fusion is a softmax over input, weighted adapter outputs can be collapsed into one fused rank-space transform before the output projection. Givens codes store this source family more compactly than full coefficient matrices.
- **Hypothesis:** direct code-space fusion may reduce adapter latency on aligned low-rank banks; robustness to realistic nonlinear bottlenecks and learned AdapterFusion checkpoints remains untested.
