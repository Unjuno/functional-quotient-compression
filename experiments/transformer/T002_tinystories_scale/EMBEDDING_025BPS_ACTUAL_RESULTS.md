# TinyStories actual embedding experiments at ~0.25 bit/scalar

This note records high-value **actual-model** results from container-side experiments on the uploaded TinyStories GPT-Neo scale family (1M / 3M / 8M / 28M checkpoints). These results use manual PyTorch GPT-Neo replay and GPT-2 BPE tokenization reconstructed from each checkpoint package. They are not SmolLM2 results and do not establish 64x feasibility.

## 1. Scale changes embedding geometry, but plain ~0.25-bps PQ still fails functionally

Across the uploaded scale series, the tied embedding fraction falls strongly with model size (approximately 85.9% / 77.7% / 65.3% / 49.5% for 1M / 3M / 8M / 28M). Top-35 embedding variance concentration also falls with size (approximately 0.799 / 0.533 / 0.347 / 0.216).

Despite this geometric change, a uniform weight-space product-quantization treatment near 0.23-0.25 bit/scalar caused large tied input/output functional damage at every tested size. Increasing model size did not make uniform embedding PQ safe by itself.

## 2. Functional root + sparse exceptions is much stronger than diffuse correction

On the 8M model under a hard 0.25-bit/scalar embedding budget, the strongest tested family remains:

`task-signature shared root -> sparse high-leverage token exceptions`

Several diffuse alternatives failed:

- 2-4 bit/token scalar gains applied to all vocabulary rows worsened KL relative to spending those bits on more sparse exceptions;
- coarse block-VQ residual corrections applied to many thousands of rows substantially improved weight NMSE but increased functional KL dramatically;
- calibration-hidden-subspace low-rank corrections applied to many rows remained materially worse than full-row sparse exceptions;
- separating input/output decoded correction roles did not beat applying the same high-fidelity exception to the tied row.

The key boundary is therefore: **better global weight reconstruction is not sufficient; high-leverage rows require relatively accurate correction.**

## 3. A small shared residual dictionary can help, but only modestly

A hierarchical representation was tested on the 8M embedding:

`task-signature root K192 + functional residual dictionary K8 + 4-bit sparse exceptions`

Across four clustering seeds this reduced mean holdout KL by roughly 4% relative to a single K384 root with sparse exceptions. Deeper residual-dictionary stacks did not improve robustly and generally lost too many sparse exceptions.

Thus a shallow shared hierarchy is a useful secondary optimization, not a solution to the 0.25-bps bottleneck.

## 4. Private exception precision has a discrete optimum

With the K192 + K8 hierarchy fixed and total bytes held constant, exception precision showed a strong discrete optimum:

- 2-bit exceptions: catastrophic functional failure;
- 3-bit: materially worse than 4-bit despite correcting more rows;
- **4-bit: best tested tradeoff**;
- 5-6 bit: worse because fewer high-leverage rows can be protected.

Adding multiple scale groups inside each 4-bit exception row did not improve the overall hard-budget tradeoff; scale metadata reduced the number of protected rows.

## 5. Functional-subspace correction alone is not enough for tied embeddings

Encoding only coefficients in calibration-hidden-state principal directions allowed many more vocabulary rows to receive corrections, but remained much worse functionally than sparse full-row 4-bit exceptions. A hybrid that used full-row corrections for calibration input tokens and functional-subspace corrections for many output tokens also remained worse.

This indicates that preserving only the currently observed output subspace is insufficient for the tied input/output embedding under these probes.

## Scientific boundaries

Do not infer that:

- 0.25 bit/scalar embedding compression is impossible for all codec families;
- the current task-signature root or selector is globally optimal;
- these short probes are benchmark-quality language-model evaluation;
- individual embedding conclusions automatically transfer to SmolLM2.

The current evidence is narrower: **for the tested actual TinyStories models, uniform/diffuse embedding correction at ~0.25 bps is weak, while shared structure plus sparse high-fidelity functional exceptions is substantially better but still leaves material task distortion.**
